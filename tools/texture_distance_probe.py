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


def _read_gray_at(cap, idx, use_green):
    """Seek to frame idx and return its gray (or None)."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    return _to_gray(frame, use_green)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cam0", required=True)
    ap.add_argument("--cam1", required=True)
    ap.add_argument("--bayer-pattern", default="RGGB")
    ap.add_argument("--calib", default=None,
                    help="Calibration .npz with dlt_P0/dlt_P1 (or "
                         "P0/P1). When provided, an arena ROI is "
                         "projected per camera and ALL texture-"
                         "distance computation is restricted to "
                         "inside the arena — bright things outside "
                         "(experimenter's hands, room behind the "
                         "acrylic, door reflections) are excluded "
                         "from the rat detector, the illumination "
                         "field, the persistence map, and the "
                         "distance map.")
    ap.add_argument("--roi-pad-px", type=int, default=20,
                    help="Expand the projected arena hull outward by "
                         "this many px (avoids clipping the rat when "
                         "it presses against a wall). Default 20.")
    ap.add_argument("--bg-frames", type=int, default=60,
                    help="Number of frames to build the background "
                         "texture model from.")
    ap.add_argument("--bg-start", type=int, default=0,
                    help="First frame index for background model.")
    ap.add_argument("--bg-stride", type=int, default=10,
                    help="Sample every Nth frame for the bg model "
                         "(spreads samples across the session).")
    ap.add_argument("--probe-frames", type=int, nargs="+",
                    default=[],
                    help="Frame indices to dump distance maps for. "
                         "Optional when --track is used.")
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
    ap.add_argument("--roi-mode", default="box",
                    choices=["box", "floor", "volume"],
                    help="Detection ROI extent. 'box' (legacy) = convex "
                         "hull of all 8 arena corners — ~3.6x the floor "
                         "area, ~73%% of it the through-wall region seen "
                         "past the transparent walls, which locks "
                         "detection onto non-corresponding per-camera "
                         "artifacts. 'floor' = the 4 floor corners only "
                         "(arena footprint). 'volume' = floor + a "
                         "--roi-max-height-mm band (covers a reared "
                         "animal, excludes the upper through-wall "
                         "region). Use floor or volume for stereo.")
    ap.add_argument("--roi-max-height-mm", type=float, default=120.0,
                    help="Height (mm) of the volume band above the floor "
                         "for --roi-mode volume.")
    ap.add_argument("--bg-select", default="stride",
                    choices=["stride", "motion"],
                    help="How to pick background-model frames. 'stride' "
                         "(default) samples at a fixed interval; 'motion' "
                         "picks frames where the rat is actively moving "
                         "(high inter-frame motion in the ROI), so a "
                         "dwelling rat is NOT baked into the median "
                         "background (avoids the persistence hole that "
                         "suppresses the rat).")
    ap.add_argument("--bg-motion-oversample", type=int, default=4,
                    help="With --bg-select motion: scan this many times "
                         "--bg-frames candidates to score for motion.")
    ap.add_argument("--bg-motion-min-pct", type=float, default=40.0,
                    help="With --bg-select motion: discard candidate "
                         "frames below this motion percentile (still "
                         "frames that would contaminate the median).")
    ap.add_argument("--log-descriptor", action="store_true",
                    help="Apply log1p to the Gabor descriptor (variance-"
                         "stabilize the exponential-like background: CoV "
                         "~1, tails 8-16x mean over 1e9 px). Compresses "
                         "the heavy background tail so the z-score is "
                         "better founded. Applied to BOTH the bg model "
                         "and the per-frame descriptor.")
    ap.add_argument("--device", default="cpu",
                    choices=["cpu", "gpu", "auto"],
                    help="Compute the Gabor descriptor + texture "
                         "distance on 'cpu' (OpenCV, default), 'gpu' "
                         "(CuPy — the data-parallel bottleneck port), or "
                         "'auto'. The bg model is uploaded once and kept "
                         "resident. GPU results match the CPU path "
                         "within FP tolerance.")
    ap.add_argument("--persistence-power", type=float, default=1.0,
                    help="Exponent on the persistence damping. >1 "
                         "suppresses persistent-background pixels "
                         "more aggressively. Default 1.0.")
    ap.add_argument("--foreshorten-correct", action="store_true",
                    default=False,
                    help="Down-weight the texture distance where the "
                         "arena floor is steeply foreshortened (the "
                         "pixel footprint is elongated, so the same fur "
                         "reads differently). Requires --calib. Targets "
                         "frame-edge / flank weakness, esp. cam1.")
    ap.add_argument("--foreshorten-max-aniso", type=float, default=3.0,
                    help="Foreshortening (1/cos theta) at which the "
                         "confidence weight reaches 0. Larger = gentler "
                         "(only the most grazing regions suppressed). "
                         "Default 3.0.")
    ap.add_argument("--mask-rat-persistence", action="store_true",
                    default=False,
                    help="When building the persistence map, detect "
                         "and EXCLUDE the rat in each bg frame so the "
                         "rat's frequented/dwell spots aren't marked "
                         "low-persistence (which would SUPPRESS the "
                         "rat there in the distance map). Strongly "
                         "recommended — without it, the rat fights "
                         "its own persistence shadow.")
    ap.add_argument("--persistence-rat-percentile", type=float,
                    default=96.0,
                    help="Intensity percentile for the per-bg-frame "
                         "rat detector used by --mask-rat-persistence.")
    ap.add_argument("--persistence-rat-min-area", type=int,
                    default=1500,
                    help="Min area (px) for the per-bg-frame rat "
                         "detector.")
    ap.add_argument("--persistence-rat-dilate", type=int, default=35,
                    help="Dilation (px) of the per-bg-frame rat mask. "
                         "Should exceed the Gabor kernel footprint "
                         "(~largest scale + smooth-k) so the rat's "
                         "descriptor halo is also excluded. Default "
                         "35.")
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
    ap.add_argument("--suppress-thin-width", type=int, default=0,
                    help="Remove structures narrower than this many "
                         "px (the tether cable) via morphological "
                         "opening, BEFORE component analysis. Unlike "
                         "the aspect/fill filters, this severs the "
                         "cable from the rat WITHIN a merged "
                         "component — needed because the cable "
                         "physically attaches to the headstage. Set "
                         "to ~3x the cable width, well under the "
                         "rat-body width (25-40 typical). Default 0 "
                         "(disabled).")
    # ── Segmentation backend ───────────────────────────────────────
    ap.add_argument("--segment-method", default="threshold",
                    choices=["threshold", "graphcut"],
                    help="How to turn the texture-distance map into a "
                         "mask. 'threshold' = abs/otsu threshold + "
                         "morphology + CC filter (fast, heuristic). "
                         "'graphcut' = minimize a binary MRF energy "
                         "(texture-distance data term + contrast-"
                         "sensitive smoothness) via max-flow — a "
                         "globally-optimal coherent silhouette, the "
                         "first principled step toward the variational "
                         "formulation. Requires PyMaxflow.")
    ap.add_argument("--gc-smooth-weight", type=float, default=2.0,
                    help="graphcut: weight on the smoothness term. "
                         "Larger = smoother/rounder silhouette, "
                         "rejects more noise, fills more holes. Main "
                         "regularization knob. Default 2.0.")
    ap.add_argument("--gc-edge-sigma", type=float, default=10.0,
                    help="graphcut: intensity scale (gray levels) for "
                         "the contrast-sensitive smoothness. Edges "
                         "with |dI| >> this are cheap boundaries. "
                         "Default 10.0.")
    ap.add_argument("--gc-data-scale", type=float, default=1.0,
                    help="graphcut: logistic steepness of the data "
                         "term. Larger = more threshold-like. Default "
                         "1.0.")
    ap.add_argument("--gc-predicted-roi", action="store_true",
                    default=False,
                    help="graphcut (track mode): solve the max-flow "
                         "only inside a box around the Kalman-predicted "
                         "rat instead of the whole frame. Big speed win "
                         "with an identical cut inside the box; falls "
                         "back to full-frame before a track exists.")
    ap.add_argument("--gc-roi-pad-px", type=int, default=120,
                    help="graphcut predicted-ROI: slack (px) added "
                         "around the predicted blob radius. Must exceed "
                         "the per-frame rat displacement. Default 120.")
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
    # ── Tracking mode (contiguous range + Kalman + dynamic shadow) ──
    ap.add_argument("--track", action="store_true", default=False,
                    help="Run a CONTIGUOUS frame range through the "
                         "TextureBlobTracker (Kalman) and dynamic "
                         "shadow model, writing a trajectory-overlay "
                         "montage. Uses --track-start / --track-end / "
                         "--track-step instead of --probe-frames.")
    ap.add_argument("--track-start", type=int, default=0)
    ap.add_argument("--track-end", type=int, default=500)
    ap.add_argument("--track-step", type=int, default=1)
    ap.add_argument("--track-gate-px", type=float, default=120.0,
                    help="Kalman gate radius (px). Candidates farther "
                         "than this from the prediction are rejected.")
    ap.add_argument("--track-max-coast", type=int, default=10,
                    help="Frames the track survives on prediction "
                         "alone during a dropout.")
    ap.add_argument("--dynamic-shadow", action="store_true",
                    default=False,
                    help="Adapt the illumination field over the "
                         "tracked range with a slow EMA, masking out "
                         "the tracked rat so it can't poison the "
                         "field. Tracks slow IR drift + cast shadow.")
    ap.add_argument("--dynamic-shadow-alpha", type=float, default=0.02,
                    help="EMA rate for the dynamic shadow field. "
                         "Larger adapts faster. Default 0.02.")
    ap.add_argument("--track-dump-heatmaps", type=int, default=0,
                    metavar="EVERY_N",
                    help="In track mode, also write a standalone "
                         "3-panel PNG [raw+circle | heatmap | mask] "
                         "every N tracked frames (0 = only the "
                         "montage). Lets you scrub individual frames "
                         "rather than just the 12-cell summary.")
    ap.add_argument("--min-area", type=int, default=1000)
    ap.add_argument("--out", required=True,
                    help="Output directory for diagnostic PNGs.")
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)

    # Lazy imports from the package
    from rpimocap.io.export import TiffCapture
    from rpimocap.detection.rat_texture import build_gabor_kernels
    from rpimocap.detection.segment import (
        arena_roi_mask, arena_roi_corners)
    from rpimocap.detection.texture_distance import (
        dense_gabor_descriptor, BackgroundTextureModel,
        build_persistent_texture_model,
        build_illumination_field, apply_illumination_correction,
        DynamicShadowModel, TextureBlobTracker,
        texture_distance_map, threshold_distance_map,
        graphcut_segment_distance, crop_box_from_prediction,
        colorize_distance_map)

    def _segment(dist, gray_c, roi, crop_box=None):
        """Dispatch to the chosen segmentation backend, returning
        (mask, threshold_or_flow). crop_box (graphcut only) restricts
        the max-flow to a predicted-ROI band."""
        if args.segment_method == "graphcut":
            mask, info = graphcut_segment_distance(
                dist, gray=gray_c, roi_mask=roi,
                fg_thresh=args.abs_thresh,
                data_scale=args.gc_data_scale,
                smooth_weight=args.gc_smooth_weight,
                edge_sigma=args.gc_edge_sigma,
                min_area_px=args.min_area,
                suppress_thin_width=args.suppress_thin_width,
                crop_box=crop_box)
            return mask, info["flow"]
        mask, thr = threshold_distance_map(
            dist, method=args.threshold_method,
            abs_thresh=args.abs_thresh,
            percentile=args.threshold_percentile,
            min_area_px=args.min_area,
            max_aspect_ratio=args.max_aspect_ratio,
            min_fill_ratio=args.min_fill_ratio,
            suppress_thin_width=args.suppress_thin_width)
        return mask, thr

    # Arena corners in mm (matches cli/segment.py _ARENA_CORNERS)
    _ARENA_CORNERS = np.array([
        [-140, -215,   0], [ 140, -215,   0],
        [ 140,  215,   0], [-140,  215,   0],
        [-140, -215, 388], [ 140, -215, 388],
        [ 140,  215, 388], [-140,  215, 388],
    ], dtype=np.float64)
    # Corner subset defining the detection ROI (box / floor / volume).
    _ROI_CORNERS = arena_roi_corners(
        _ARENA_CORNERS, mode=args.roi_mode,
        max_height_mm=args.roi_max_height_mm)
    if args.roi_mode != "box":
        print(f"  ROI mode: {args.roi_mode}"
              + (f" (height band {args.roi_max_height_mm:.0f}mm)"
                 if args.roi_mode == "volume" else "")
              + f" — {len(_ROI_CORNERS)} corners (was 8-corner box hull)")
    # Load DLT projection matrices if a calibration was given
    dlt_P = {0: None, 1: None}
    if args.calib:
        cal = np.load(args.calib)
        dlt_P[0] = cal.get("dlt_P0", cal.get("P0", None))
        dlt_P[1] = cal.get("dlt_P1", cal.get("P1", None))
        if dlt_P[0] is None or dlt_P[1] is None:
            print("  WARNING: calib has no dlt_P0/dlt_P1 — no ROI")
        else:
            print(f"  Loaded DLT matrices from {args.calib} "
                  f"→ arena ROI enabled")

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

        # Per-camera frame count — the two cameras can differ (a sync
        # drop or early stop truncates one). In track mode the loop stops
        # when a camera runs out of frames, so a shorter cam1 produces
        # fewer outputs; and the tail of the longer camera has no stereo
        # partner. Surface it up front.
        try:
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        except Exception:
            n_frames = -1
        print(f"  frames: {n_frames}")
        if args.track and n_frames > 0 and args.track_end > n_frames:
            print(f"  NOTE: --track-end {args.track_end} exceeds cam "
                  f"{cam_id}'s {n_frames} frames; it will stop at "
                  f"{n_frames}. For stereo, use a range both cameras "
                  f"cover (track-end <= the smaller frame count).")

        # ── Build background texture model ────────────────────────
        print(f"  Building bg texture model: {args.bg_frames} frames "
              f"from idx {args.bg_start} stride {args.bg_stride}")
        # Collect frames spread across the session for a robust median
        # background texture model. With the rat in every frame, the
        # per-pixel temporal median rejects the rat as an outlier.
        bg_grays = []
        if args.bg_select == "motion":
            # Motion-aware: pick frames where the rat is moving so a
            # dwelling rat isn't baked into the median background.
            from rpimocap.detection.bg_select import select_active_frames
            # ROI for motion scoring (build from a probe frame's shape)
            probe_g = None
            cap.set(cv2.CAP_PROP_POS_FRAMES, args.bg_start)
            ok, _pf = cap.read()
            if ok and _pf is not None:
                probe_g = _to_gray(_pf, args.green_channel)
            motion_roi = None
            if probe_g is not None and dlt_P[cam_id] is not None:
                motion_roi = arena_roi_mask(
                    dlt_P[cam_id], _ROI_CORNERS, probe_g.shape,
                    pad_px=args.roi_pad_px)
            sel_end = (n_frames if n_frames > 0
                       else args.bg_start + args.bg_frames
                       * args.bg_stride * args.bg_motion_oversample)
            sel_idx, _cand, _motion = select_active_frames(
                cap, args.bg_frames, args.bg_start, sel_end,
                green_channel=args.green_channel, roi_mask=motion_roi,
                oversample=args.bg_motion_oversample,
                min_motion_percentile=args.bg_motion_min_pct)
            print(f"  motion-selected {len(sel_idx)} bg frames "
                  f"(of {len(_cand)} scanned); motion "
                  f"min={_motion.min():.2f} max={_motion.max():.2f} "
                  f"mean={_motion.mean():.2f}")
            for fi in sel_idx:
                g = _read_gray_at(cap, fi, args.green_channel)
                if g is not None:
                    bg_grays.append(g)
        else:
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

        # ── Arena ROI ─────────────────────────────────────────────
        # Restrict ALL texture-distance computation to inside the
        # arena. Without this, bright things outside the acrylic
        # (experimenter's hands, the room, door reflections) get
        # treated as candidate rat / foreground.
        roi = None
        if dlt_P[cam_id] is not None:
            roi = arena_roi_mask(
                dlt_P[cam_id], _ROI_CORNERS,
                bg_grays[0].shape, pad_px=args.roi_pad_px)
            cov = 100.0 * float((roi > 0).sum()) / roi.size
            print(f"  Arena ROI: {cov:.1f}% of frame inside")
            cv2.imwrite(
                os.path.join(args.out, f"arena_roi_cam{cam_id}.png"),
                roi)

        # ── Foreshortening confidence map (static, per-camera) ─────
        # The arena floor (z=0 plane) is seen at different angles across
        # the frame. Where it is grazing, the pixel footprint elongates
        # and the same fur reads differently — a bias separate from
        # radial lens distortion. Build a [0,1] confidence (1 face-on,
        # 0 grazing) to down-weight the texture distance there.
        aniso_w = None
        if args.foreshorten_correct and dlt_P[cam_id] is not None:
            from rpimocap.detection.foreshortening import (
                footprint_anisotropy_plane, anisotropy_weight)
            H, W = bg_grays[0].shape[:2]
            aniso = footprint_anisotropy_plane(
                dlt_P[cam_id],
                plane_point=np.array([0.0, 0.0, 0.0]),
                plane_normal=np.array([0.0, 0.0, 1.0]),
                image_size=(W, H), stride=8)
            aniso_w = anisotropy_weight(
                aniso, max_aniso=args.foreshorten_max_aniso)
            covlo = 100.0 * float((aniso_w < 0.5).sum()) / aniso_w.size
            print(f"  Foreshorten map: aniso "
                  f"min={float(aniso.min()):.2f} "
                  f"max={float(aniso.max()):.2f} "
                  f"mean={float(aniso.mean()):.2f}  |  weight "
                  f"min={float(aniso_w.min()):.2f} "
                  f"max={float(aniso_w.max()):.2f} "
                  f"mean={float(aniso_w.mean()):.2f}  |  "
                  f"{covlo:.1f}% down-weighted <0.5")
            if float(aniso_w.max() - aniso_w.min()) < 0.05:
                print("    (near-uniform → this camera sees the floor "
                      "almost face-on; foreshortening correction is a "
                      "near-no-op here, which is expected, not a bug)")
            # grayscale (raw confidence) AND a colorized map so a subtle
            # gradient is visible even when the weight is near-uniform.
            cv2.imwrite(
                os.path.join(args.out,
                             f"foreshorten_w_cam{cam_id}.png"),
                (aniso_w * 255).astype(np.uint8))
            _fw = aniso_w.copy()
            _lo, _hi = float(_fw.min()), float(_fw.max())
            if _hi - _lo > 1e-6:
                _fw = (_fw - _lo) / (_hi - _lo)      # stretch to full range
            cv2.imwrite(
                os.path.join(args.out,
                             f"foreshorten_w_cam{cam_id}_color.png"),
                cv2.applyColorMap((_fw * 255).astype(np.uint8),
                                  cv2.COLORMAP_VIRIDIS))
        elif args.foreshorten_correct:
            print("  WARNING: --foreshorten-correct needs --calib; "
                  "skipping foreshortening.")

        # ── Static shadow / illumination field ────────────────────
        # The per-pixel temporal median of intensity is the static
        # scene illumination (the rat, a moving minority, is rejected).
        # Flat-field correcting each frame by this field makes the
        # texture descriptor illumination-invariant, so the same
        # bedding reads the same whether well-lit or in shadow.
        illum_field = None
        if args.illumination_correct:
            illum_field = build_illumination_field(
                bg_grays, blur_sigma=args.illumination_blur_sigma,
                roi_mask=roi)
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
            smooth_k=args.smooth_k, rotation_invariant=True,
            mask_rat=args.mask_rat_persistence,
            rat_percentile=args.persistence_rat_percentile,
            rat_min_area_px=args.persistence_rat_min_area,
            rat_dilate_px=args.persistence_rat_dilate,
            roi_mask=roi, log_transform=args.log_descriptor)
        print(f"  Persistent model built from {model.n} frames "
              f"(median + MAD), mean shape {model.mean.shape}")

        # ── Device dispatch for the descriptor + distance ──────────
        # On 'cpu' use the canonical OpenCV path; on 'gpu'/'auto' use the
        # CuPy port, uploading the bg model once (resident). The two
        # match within FP tolerance.
        _dev_model = None
        if args.device != "cpu":
            from rpimocap.detection.gpu_texture import (
                upload_model, texture_distance_device, array_module)
            _xp, _ndi, _on_gpu = array_module(args.device)
            _dev_model = upload_model(model.mean, model.std,
                                      device=args.device)
            print(f"  device: {args.device} "
                  f"(GPU active: {_on_gpu}); bg model resident")

        def _distance(gray_in):
            if args.device == "cpu":
                return texture_distance_map(
                    gray_in, model, kernels, n_orient, n_scales,
                    smooth_k=args.smooth_k, rotation_invariant=True,
                    log_transform=args.log_descriptor,
                    persistence_map=persistence_map,
                    persistence_power=args.persistence_power,
                    anisotropy_weight=aniso_w, roi_mask=roi,
                    post_smooth_k=args.post_smooth_k)
            return texture_distance_device(
                gray_in, _dev_model[0], _dev_model[1], kernels,
                n_orient, n_scales, smooth_k=args.smooth_k,
                rotation_invariant=True,
                log_transform=args.log_descriptor,
                persistence_map=persistence_map,
                persistence_power=args.persistence_power,
                anisotropy_weight=aniso_w, roi_mask=roi,
                post_smooth_k=args.post_smooth_k,
                device=args.device, xp=_xp, ndi=_ndi)

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
            dist = _distance(gray)
            mask, thr = _segment(dist, gray, roi)

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

        # ── Track mode: contiguous range + Kalman + dynamic shadow ─
        if args.track:
            print(f"  Tracking frames {args.track_start}–"
                  f"{args.track_end} step {args.track_step}")
            tracker = TextureBlobTracker(
                gate_px=args.track_gate_px,
                max_coast=args.track_max_coast,
                select="area")
            dsm = None
            if args.dynamic_shadow and illum_field is not None:
                dsm = DynamicShadowModel(
                    illum_field.copy(),
                    alpha=args.dynamic_shadow_alpha,
                    blur_sigma=args.illumination_blur_sigma)
            traj = []          # (frame_idx, cx, cy, r, measured, coasting)
            cands = []         # (frame_idx, cx, cy, area) — all blobs,
            #                    for downstream stereo gating (stereo_track)
            montage_frames = []
            n_meas = n_coast = n_lost = 0
            for fi in range(args.track_start, args.track_end,
                            args.track_step):
                cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                gray = _to_gray(frame, args.green_channel)
                # Illumination correction — dynamic field if enabled,
                # else the static field
                if dsm is not None:
                    gray_c = dsm.correct(
                        gray, target_level=float(illum_field.mean()))
                elif illum_field is not None:
                    gray_c = apply_illumination_correction(
                        gray, illum_field,
                        target_level=float(illum_field.mean()))
                else:
                    gray_c = gray
                dist = _distance(gray_c)
                # Predicted-ROI crop (graphcut only): peek the tracker's
                # prediction (non-mutating) to restrict the max-flow to a
                # band around where the rat is expected, before update().
                crop_box = None
                if (args.gc_predicted_roi
                        and args.segment_method == "graphcut"):
                    crop_box = crop_box_from_prediction(
                        tracker.peek_prediction(), gray_c.shape,
                        pad_px=args.gc_roi_pad_px)
                mask, thr = _segment(dist, gray_c, roi, crop_box=crop_box)
                # Record ALL candidate blob centroids (not just the
                # tracked one) so a downstream stereo pass can gate them
                # against the arena volume (rejects through-the-glass
                # floor patches a single 2-D view can't disambiguate).
                ncc, lbl, stats, cents = cv2.connectedComponentsWithStats(
                    (mask > 0).astype(np.uint8), connectivity=8)
                for ci in range(1, ncc):
                    area = int(stats[ci, cv2.CC_STAT_AREA])
                    if area >= args.min_area:
                        cands.append((fi, float(cents[ci][0]),
                                      float(cents[ci][1]), area))
                res = tracker.update(mask)
                # Update the dynamic shadow, masking the tracked rat
                if dsm is not None:
                    rat_mask = None
                    if res["state"] is not None:
                        cx, cy, r = res["state"]
                        rat_mask = np.zeros(gray.shape, np.uint8)
                        cv2.circle(rat_mask, (int(cx), int(cy)),
                                   int(max(r * 1.5, 20)), 255, -1)
                    dsm.update(gray, update_mask=rat_mask)
                # Tally
                if res["lost"]:
                    n_lost += 1
                elif res["coasting"]:
                    n_coast += 1
                elif res["measured"]:
                    n_meas += 1
                if res["state"] is not None:
                    cx, cy, r = res["state"]
                    traj.append((fi, cx, cy, r,
                                 res["measured"], res["coasting"]))
                # Sample a few montage frames across the range
                want_montage = ((fi - args.track_start)
                                % max(1, (args.track_end
                                          - args.track_start) // 12) == 0)
                want_dump = (args.track_dump_heatmaps > 0
                             and (fi - args.track_start)
                             % args.track_dump_heatmaps == 0)
                if (want_montage or want_dump) and res["state"] is not None:
                    # Panel 1: illumination-corrected raw + tracked
                    # circle (green=measured, orange=coasting)
                    g = gray_c.astype(np.float32)
                    lo, hi = np.percentile(g, [1, 99])
                    vis = np.clip((g - lo) / (hi - lo + 1e-6) * 255,
                                  0, 255).astype(np.uint8)
                    vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
                    cx, cy, r = res["state"]
                    col = ((0, 255, 0) if res["measured"]
                           else (0, 165, 255))   # green meas, orange coast
                    cv2.circle(vis, (int(cx), int(cy)), int(r),
                               col, 2)
                    cv2.putText(vis, f"f{fi}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
                    tag = "MEAS" if res["measured"] else "COAST"
                    cv2.putText(vis, tag, (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
                    # Panel 2: texture-distance heatmap (what the
                    # tracker is actually seeing this frame)
                    heat = colorize_distance_map(dist)
                    cv2.circle(heat, (int(cx), int(cy)), int(r),
                               (255, 255, 255), 2)
                    # Panel 3: thresholded mask overlay on the raw
                    mask_ov = vis.copy()
                    cnts_m, _ = cv2.findContours(
                        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(mask_ov, cnts_m, -1,
                                     (0, 255, 255), 2)
                    cv2.putText(mask_ov,
                                f"fg={int((mask>0).sum())}px "
                                f"cc={len(cnts_m)}",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 255, 255), 2)
                    cell = np.hstack([vis, heat, mask_ov])
                    if want_montage:
                        montage_frames.append(cell)
                    if want_dump:
                        dpath = os.path.join(
                            args.out,
                            f"track_cam{cam_id}_f{fi:06d}.png")
                        cv2.imwrite(dpath, cell)
            # Write trajectory CSV
            traj_csv = os.path.join(
                args.out, f"track_cam{cam_id}.csv")
            with open(traj_csv, "w") as fh:
                fh.write("frame,cx,cy,r,measured,coasting\n")
                for row in traj:
                    fh.write(f"{row[0]},{row[1]:.2f},{row[2]:.2f},"
                             f"{row[3]:.2f},{int(row[4])},"
                             f"{int(row[5])}\n")
            # Write candidates CSV (all blobs per frame) for the stereo
            # gating pass (tools/stereo_gate.py).
            cand_csv = os.path.join(
                args.out, f"candidates_cam{cam_id}.csv")
            with open(cand_csv, "w") as fh:
                fh.write("frame,cx,cy,area\n")
                for row in cands:
                    fh.write(f"{row[0]},{row[1]:.2f},{row[2]:.2f},"
                             f"{row[3]}\n")
            # Write montage. Each cell is now a 3-panel strip
            # [raw+circle | heatmap | mask], so use 2 columns to keep
            # the grid readable.
            if montage_frames:
                cols = 2
                rows_n = (len(montage_frames) + cols - 1) // cols
                h, w = montage_frames[0].shape[:2]
                grid = np.zeros((rows_n * h, cols * w, 3), np.uint8)
                for k, vis in enumerate(montage_frames):
                    rr, cc = divmod(k, cols)
                    grid[rr*h:(rr+1)*h, cc*w:(cc+1)*w] = vis
                mpath = os.path.join(
                    args.out, f"track_montage_cam{cam_id}.png")
                cv2.imwrite(mpath, grid)
            n_total = n_meas + n_coast + n_lost
            print(f"    tracked {len(traj)}/{n_total} frames: "
                  f"measured={n_meas} coast={n_coast} lost={n_lost}, "
                  f"gated_out={tracker.n_gated_out}")
            print(f"    trajectory → {traj_csv}")

        cap.release()

    print(f"\nDone. Panels in {args.out}/")
    print("Each panel: [ raw | texture-distance heatmap | mask "
          "overlay ]")
    print("Look for: clean rat-shaped red region in the heatmap, "
          "dark (blue) static artifacts.")


if __name__ == "__main__":
    main()
