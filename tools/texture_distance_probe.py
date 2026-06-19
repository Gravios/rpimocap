#!/usr/bin/env python
"""
texture_distance_probe.py
=========================
Standalone diagnostic for the texture-change foreground hypothesis.

Builds a per-pixel background texture model from N background frames,
then for a set of probe frames dumps, side by side:
  * the raw (demosaiced) frame
  * the texture-distance heatmap (blue=no change, red=large change)
  * the thresholded foreground mask

The point is to eyeball, on the exact cam1 frames where the
production bg-sub currently produces nothing, whether texture
distance lights up a clean rat region. If it does, the full
graph-cut (MRF) segmentation is worth building. If bedding
disturbance or specular wobble drowns it, we learn that cheaply.

Does NOT touch the production pipeline. Reuses RatTextureBank's
Gabor kernels so the descriptor matches the rest of the project.

Usage
-----
python texture_distance_probe.py \
    --cam0 cam0_raw.tif --cam1 cam1_raw.tif \
    --bayer-pattern RGGB \
    --bg-frames 60 --bg-start 0 \
    --probe-frames 922 1844 3688 11064 \
    --green-channel \
    --out /tmp/texdist_probe

Threshold method:
  --threshold-method otsu|absolute|percentile (default otsu)
  --abs-thresh 3.0          (z-score units, for 'absolute')
  --threshold-percentile 95 (for 'percentile')
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np


def _to_gray(frame: np.ndarray, use_green: bool) -> np.ndarray:
    """Match the production detector's channel handling: green Bayer
    channel (if requested) else luminance."""
    if frame.ndim == 2:
        return frame
    if use_green and frame.shape[2] == 3:
        return frame[:, :, 1]
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cam0", required=True)
    ap.add_argument("--cam1", required=True)
    ap.add_argument("--bayer-pattern", default="RGGB")
    ap.add_argument("--bg-frames", type=int, default=60,
                    help="Number of frames to build the background "
                         "texture model from.")
    ap.add_argument("--bg-start", type=int, default=0,
                    help="First frame index for background model.")
    ap.add_argument("--bg-stride", type=int, default=10,
                    help="Sample every Nth frame for the bg model "
                         "(spreads samples across the session).")
    ap.add_argument("--probe-frames", type=int, nargs="+",
                    required=True,
                    help="Frame indices to dump distance maps for.")
    ap.add_argument("--green-channel", action="store_true",
                    default=False)
    ap.add_argument("--scales", type=int, nargs="+",
                    default=[5, 9, 13],
                    help="Gabor scales (kernel sizes). Must match if "
                         "comparing to a saved texture bank.")
    ap.add_argument("--n-orientations", type=int, default=4)
    ap.add_argument("--smooth-k", type=int, default=7,
                    help="Box filter on each Gabor response.")
    ap.add_argument("--post-smooth-k", type=int, default=15,
                    help="Box filter on the final distance map "
                         "(crude smoothness-term stand-in).")
    ap.add_argument("--persistence-power", type=float, default=1.0,
                    help="Exponent on the persistence damping. >1 "
                         "suppresses persistent-background pixels "
                         "more aggressively. Default 1.0.")
    ap.add_argument("--max-aspect-ratio", type=float, default=6.0,
                    help="Reject thresholded components more "
                         "elongated than this (the cable is a thin "
                         "streak, aspect 10-30; the rat is compact, "
                         "1.5-3). Default 6.0. Set 0 to disable.")
    ap.add_argument("--min-fill-ratio", type=float, default=0.0,
                    help="Reject components whose area/bbox-area is "
                         "below this (lines fill ~0.1-0.2, blobs "
                         "~0.5+). Complements aspect for diagonal "
                         "cables. Default 0 (disabled).")
    ap.add_argument("--illumination-correct", action="store_true",
                    default=False,
                    help="Build a static shadow/illumination field "
                         "(per-pixel temporal median intensity) and "
                         "flat-field correct every frame by it before "
                         "computing textures. Removes the fixed IR "
                         "falloff so the same texture reads the same "
                         "everywhere in the arena.")
    ap.add_argument("--illumination-blur-sigma", type=float,
                    default=51.0,
                    help="Gaussian sigma to isolate the LOW-FREQUENCY "
                         "illumination falloff from the median field. "
                         "Large (e.g. 51) keeps only the smooth shadow "
                         "gradient, leaving sharp static structures "
                         "(rails) out of the field so they aren't "
                         "divided away. Set 0 for full flat-field "
                         "(removes structure too). Default 51.")
    ap.add_argument("--threshold-method", default="otsu",
                    choices=["otsu", "absolute", "percentile"])
    ap.add_argument("--abs-thresh", type=float, default=3.0)
    ap.add_argument("--threshold-percentile", type=float, default=95.0)
    ap.add_argument("--min-area", type=int, default=1000)
    ap.add_argument("--out", required=True,
                    help="Output directory for diagnostic PNGs.")
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)

    # Lazy imports from the package
    from rpimocap.io.export import TiffCapture
    from rpimocap.detection.rat_texture import build_gabor_kernels
    from rpimocap.detection.texture_distance import (
        dense_gabor_descriptor, BackgroundTextureModel,
        build_persistent_texture_model,
        build_illumination_field, apply_illumination_correction,
        texture_distance_map, threshold_distance_map,
        colorize_distance_map)

    # Build kernels matching RatTextureBank defaults
    orientations = [i * np.pi / args.n_orientations
                    for i in range(args.n_orientations)]
    kernels = build_gabor_kernels(orientations, args.scales)
    n_orient = len(orientations)
    n_scales = len(args.scales)
    print(f"Gabor: {n_orient} orientations × {n_scales} scales "
          f"→ descriptor dim {3 * n_scales} (rotation-invariant)")

    for cam_id, path in ((0, args.cam0), (1, args.cam1)):
        print(f"\n── Cam {cam_id}: {path}")
        cap = TiffCapture(path, bayer_pattern=args.bayer_pattern)

        # ── Build background texture model ────────────────────────
        print(f"  Building bg texture model: {args.bg_frames} frames "
              f"from idx {args.bg_start} stride {args.bg_stride}")
        # Collect frames spread across the session for a robust median
        # background texture model. With the rat in every frame, the
        # per-pixel temporal median rejects the rat as an outlier.
        bg_grays = []
        idx = args.bg_start
        while len(bg_grays) < args.bg_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            bg_grays.append(_to_gray(frame, args.green_channel))
            idx += args.bg_stride
        if len(bg_grays) < 3:
            print(f"  ERROR: only {len(bg_grays)} bg frames collected")
            cap.release()
            continue

        # ── Static shadow / illumination field ────────────────────
        # The per-pixel temporal median of intensity is the static
        # scene illumination (the rat, a moving minority, is rejected).
        # Flat-field correcting each frame by this field makes the
        # texture descriptor illumination-invariant, so the same
        # bedding reads the same whether well-lit or in shadow.
        illum_field = None
        if args.illumination_correct:
            illum_field = build_illumination_field(
                bg_grays, blur_sigma=args.illumination_blur_sigma)
            tgt = float(illum_field.mean())
            print(f"  Illumination field: mean={tgt:.1f} "
                  f"min={float(illum_field.min()):.1f} "
                  f"max={float(illum_field.max()):.1f} "
                  f"(blur_sigma={args.illumination_blur_sigma})")
            # Save the field as a heatmap for inspection
            fld_vis = np.clip(illum_field / (illum_field.max() + 1e-6)
                              * 255, 0, 255).astype(np.uint8)
            cv2.imwrite(
                os.path.join(args.out, f"illumination_cam{cam_id}.png"),
                cv2.applyColorMap(fld_vis, cv2.COLORMAP_INFERNO))
            # Correct the bg frames so the texture model is built on
            # illumination-normalized frames
            bg_grays = [apply_illumination_correction(g, illum_field,
                                                       target_level=tgt)
                        for g in bg_grays]

        model, persistence_map = build_persistent_texture_model(
            bg_grays, kernels, n_orient, n_scales,
            smooth_k=args.smooth_k, rotation_invariant=True)
        print(f"  Persistent model built from {model.n} frames "
              f"(median + MAD), mean shape {model.mean.shape}")

        # Report spread magnitude — high values are where the bg
        # texture is unstable (specular wobble, rat paths).
        std_med = float(np.median(model.std))
        std_p99 = float(np.percentile(model.std, 99))
        pers_med = float(np.median(persistence_map))
        print(f"  bg descriptor spread: median={std_med:.4f} "
              f"p99={std_p99:.4f}")
        print(f"  persistence map: median={pers_med:.3f} "
              f"(1=stable bg, 0=transient)")
        # Save the persistence map as a heatmap for inspection
        pers_vis = (persistence_map * 255).astype(np.uint8)
        pers_heat = cv2.applyColorMap(pers_vis, cv2.COLORMAP_VIRIDIS)
        cv2.imwrite(
            os.path.join(args.out, f"persistence_cam{cam_id}.png"),
            pers_heat)

        # ── Probe frames ──────────────────────────────────────────
        for pf in args.probe_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, pf)
            ok, frame = cap.read()
            if not ok or frame is None:
                print(f"  probe frame {pf}: could not read")
                continue
            gray = _to_gray(frame, args.green_channel)
            # Apply the same illumination correction the model was
            # built with, so the descriptor space matches.
            if illum_field is not None:
                gray = apply_illumination_correction(
                    gray, illum_field,
                    target_level=float(illum_field.mean()))
            dist = texture_distance_map(
                gray, model, kernels, n_orient, n_scales,
                smooth_k=args.smooth_k, rotation_invariant=True,
                persistence_map=persistence_map,
                persistence_power=args.persistence_power,
                post_smooth_k=args.post_smooth_k)
            mask, thr = threshold_distance_map(
                dist, method=args.threshold_method,
                abs_thresh=args.abs_thresh,
                percentile=args.threshold_percentile,
                min_area_px=args.min_area,
                max_aspect_ratio=args.max_aspect_ratio,
                min_fill_ratio=args.min_fill_ratio)

            # Compose a 1×3 panel: raw | heatmap | mask-overlay
            raw_bgr = (frame if frame.ndim == 3
                        else cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
            # Normalize raw for visibility (percentile stretch)
            g = gray.astype(np.float32)
            lo, hi = np.percentile(g, [1, 99])
            raw_vis = np.clip((g - lo) / (hi - lo + 1e-6) * 255,
                              0, 255).astype(np.uint8)
            raw_vis = cv2.cvtColor(raw_vis, cv2.COLOR_GRAY2BGR)

            heat = colorize_distance_map(dist)

            overlay = raw_vis.copy()
            cnts, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, cnts, -1, (0, 255, 0), 2)

            n_fg = int((mask > 0).sum())
            panel = np.hstack([raw_vis, heat, overlay])
            label = (f"cam{cam_id} f{pf}  thr={thr:.2f}  "
                     f"fg={n_fg}px  cc={len(cnts)}")
            cv2.putText(panel, label, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (255, 255, 255), 2)
            out_path = os.path.join(
                args.out, f"texdist_cam{cam_id}_f{pf:06d}.png")
            cv2.imwrite(out_path, panel)
            print(f"  probe f{pf}: thr={thr:.2f} fg={n_fg}px "
                  f"cc={len(cnts)} → {out_path}")

        cap.release()

    print(f"\nDone. Panels in {args.out}/")
    print("Each panel: [ raw | texture-distance heatmap | mask "
          "overlay ]")
    print("Look for: clean rat-shaped red region in the heatmap, "
          "dark (blue) static artifacts.")


if __name__ == "__main__":
    main()
