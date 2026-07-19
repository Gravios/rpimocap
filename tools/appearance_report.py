#!/usr/bin/env python3
"""
tools/appearance_report.py
==========================
Runnable diagnostics for the textured appearance model
(:mod:`rpimocap.model.appearance`). Point it at a stereo frame and it prints,
for each camera:

  * the per-camera whitening report (gradient anisotropy + dominant axis) — this
    should reproduce the sensor's near-vertical structure;
  * per-feature d' (rb / coh / grain) of the bootstrapped model, measured
    against an **independent** brightness reference so the colour number is not
    circular;
  * posterior separation (AUC + median posterior on rat vs bedding).

With ``--next-frame`` and ``--fps`` it additionally reports motion-blur coverage
(soft/partial pixel counts) for a range of shutter settings.

This is a *diagnostic*, not a fit: the appearance energy is validated
component-wise here, but until the unsupervised bootstrap lands (two-component
colour fit inside the localisation blob) there is no end-to-end pose fit to run.
The detector is used only as the initialiser — it locates the animal via the
triangulated point; its mask is never used as a segmentation (see
``bootstrap_masks`` for why).

Examples
--------
Raw Bayer TIFF stack (the real data), frame 2716::

    python tools/appearance_report.py \\
        --cam0 strohA-al-RPICAM-20260214/raw/cam0.tif \\
        --cam1 strohA-al-RPICAM-20260214/raw/cam1.tif \\
        --calib calib_from_corners.npz --frame 2716

Demosaiced PNG pair (e.g. the uploaded validation frames)::

    python tools/appearance_report.py \\
        --cam0 frame_002716_cam0.png --cam1 frame_002716_cam1.png \\
        --calib calib_from_corners.npz

Add motion-blur numbers by giving the next frame and the capture rate::

    python tools/appearance_report.py --cam0 c0.tif --cam1 c1.tif \\
        --calib calib_from_corners.npz --frame 2716 \\
        --next-frame 2717 --fps 50
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

from rpimocap.detection.topo_detect import build_floor_mask, detect_stereo
from rpimocap.model.appearance import (AppearanceModel, appearance_energy,
                                       bootstrap_masks, estimate_whitening,
                                       image_features, render_coverage,
                                       roi_from_mask, whitening_report)
from rpimocap.model.mesh_model import (build_rat_mesh,
                                       render_mesh_pose_silhouette)
from rpimocap.model.rat_skeleton import RatPose

# Arena corners in mm — floor z=0 (first four), ceiling z=388 (last four).
# Matches tools/topo_track.py and tools/texture_distance_probe.py.
_ARENA_CORNERS = np.array([
    [-140, -215,   0], [140, -215,   0], [140,  215,   0], [-140,  215,   0],
    [-140, -215, 388], [140, -215, 388], [140,  215, 388], [-140,  215, 388],
], dtype=np.float64)


# --------------------------------------------------------------------------
# frame loading (canonical path = TiffCapture, PNG fallback for probe frames)
# --------------------------------------------------------------------------
def _load_bgr(path: str, frame: int, bayer_pattern: str) -> np.ndarray:
    """Load one BGR frame the same way the tracking pipeline does.

    ``.tif``/``.tiff`` go through :class:`rpimocap.io.export.TiffCapture` (raw
    Bayer demosaic, uint16->uint8 normalisation identical to the rest of the
    pipeline); anything else is read with ``cv2.imread`` (already-demosaiced
    PNG/JPEG), for which ``--frame`` must be 0.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in (".tif", ".tiff"):
        from rpimocap.io.export import TiffCapture
        cap = TiffCapture(path, bayer_pattern=bayer_pattern)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame >= n:
            hint = ""
            if n == 1:
                hint = ("\n  (This TIFF exposes a single frame. Clips written as "
                        "many single-page\n   series read as 1 frame; the "
                        "full-session raw stacks are multi-page and\n   index "
                        "normally. Use --frame 0, or point at the session stack.)")
            sys.exit(f"error: {path} has {n} frame(s); --frame {frame} "
                     f"out of range{hint}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, bgr = cap.read()
        if not ok or bgr is None:
            sys.exit(f"error: cannot read frame {frame} from {path}")
        return bgr
    if frame != 0:
        sys.exit(f"error: {path} is a still image; --frame must be 0 (got {frame})")
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        sys.exit(f"error: cannot read image {path}")
    return bgr


def _load_calib(path: str):
    d = np.load(path)
    for a, b in (("dlt_P0", "dlt_P1"), ("P0", "P1")):
        if a in d and b in d:
            if a == "P0":
                print("warning: using P0/P1 — make sure these are the "
                      "arena-registered matrices, not raw autocalib.",
                      file=sys.stderr)
            return d[a], d[b]
    sys.exit(f"error: {path} has no dlt_P0/dlt_P1 (or P0/P1). "
             f"Use the arena-registered calibration (calib_from_corners.npz).")


# --------------------------------------------------------------------------
# independent reference (brightness) — for NON-CIRCULAR d' on colour
# --------------------------------------------------------------------------
def _bright_reference(gray: np.ndarray, floor: np.ndarray, pct: float = 90.0):
    """Largest bright floor component — an independent stand-in for the rat.

    This is deliberately *not* used anywhere in the model; it exists only to
    score the learned features. Because ``rb = R/B`` never looks at overall
    brightness, judging the colour model against a brightness-defined mask is
    non-circular. (It under-segments — it misses shadowed flanks — so treat the
    recall it implies as a floor, not truth.)
    """
    thr = np.percentile(gray[floor], pct)
    m = (floor & (gray > thr)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab = cv2.connectedComponents(m)
    if n <= 1:
        return m.astype(bool)
    sizes = [(lab == k).sum() for k in range(1, n)]
    return (lab == (1 + int(np.argmax(sizes)))).astype(bool)


def _auc(scores_pos: np.ndarray, scores_neg: np.ndarray) -> float:
    """Mann-Whitney AUC of pos vs neg scores."""
    s = np.concatenate([scores_pos, scores_neg])
    y = np.concatenate([np.ones(len(scores_pos)), np.zeros(len(scores_neg))])
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), float)
    ranks[order] = np.arange(1, len(s) + 1)
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Diagnostics for the textured appearance model.")
    ap.add_argument("--cam0", required=True, help="cam0 .tif stack or still image")
    ap.add_argument("--cam1", required=True, help="cam1 .tif stack or still image")
    ap.add_argument("--calib", required=True,
                    help="npz with dlt_P0/dlt_P1 (calib_from_corners.npz)")
    ap.add_argument("--frame", type=int, default=0,
                    help="frame index into the TIFF stacks (0 for stills)")
    ap.add_argument("--bayer-pattern", default="RGGB")
    ap.add_argument("--roi-mode", default="floor", choices=("floor", "volume"),
                    help="arena ROI footprint (floor recommended for detection)")
    ap.add_argument("--coh-k", type=int, default=32,
                    help="coherence integration window (px); must fit inside "
                         "the animal's narrow dimension (default 32 for ~74px)")
    ap.add_argument("--bright-pct", type=float, default=90.0,
                    help="percentile for the independent brightness reference")
    ap.add_argument("--next-frame", type=int, default=-1,
                    help="if set, a second frame index for motion-blur numbers")
    ap.add_argument("--fps", type=float, default=0.0,
                    help="capture rate for motion blur (30 or 50); required "
                         "with --next-frame")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    P = list(_load_calib(args.calib))
    bgrs = [_load_bgr(args.cam0, args.frame, args.bayer_pattern),
            _load_bgr(args.cam1, args.frame, args.bayer_pattern)]
    Gs = [b[:, :, 1] for b in bgrs]           # green channel = detector input
    shp = Gs[0].shape
    floors = [build_floor_mask(P[c], _ARENA_CORNERS, shp, mode=args.roi_mode)
              for c in (0, 1)]

    # --- initialiser: the detector LOCATES (triangulated point), nothing more.
    rng = np.random.default_rng(args.seed)
    R = detect_stereo(Gs[0], Gs[1], floors[0], floors[1], P[0], P[1], rng=rng)
    print("=" * 70)
    print(f"frame {args.frame}   image {shp[1]}x{shp[0]}")
    print("-" * 70)
    if not R.accepted:
        print("detector: NO consistent stereo pair — cannot place the render "
              "seed. (Appearance bootstrap needs the triangulated point.)")
        return 1
    print(f"detector (initialiser only): 3D point "
          f"({R.point[0]:.0f}, {R.point[1]:.0f}, {R.point[2]:.0f}) mm   "
          f"reproj {R.reproj_err:.1f}px")
    print(f"  detector masks: cam0 {int(R.det0.mask.sum())}px  "
          f"cam1 {int(R.det1.mask.sum())}px  "
          f"(NOT used as segmentation — see bootstrap_masks)")

    mesh = build_rat_mesh()
    # render seed = nominal-scale body at the triangulated point (z clamped so a
    # floor-height point still yields a body sitting on the floor).
    seed_pos = np.array([R.point[0], R.point[1], max(float(R.point[2]), 45.0)])

    dets = [R.det0.mask, R.det1.mask]
    posts, rois, models = [], [], []
    for c in (0, 1):
        G, floor = Gs[c], floors[c] > 0
        # best-yaw nominal render at the point = the localisation seed
        best = None
        for yaw in np.radians(np.arange(0, 360, 30)):
            r = render_mesh_pose_silhouette(
                mesh, RatPose(root_pos=seed_pos, root_rot=np.array([0, 0, yaw]),
                              scale=1.0), P[c], shp) > 0
            ref = _bright_reference(G, floor, args.bright_pct)
            iou = (r & ref).sum() / max((r | ref).sum(), 1)
            if best is None or iou > best[0]:
                best = (iou, yaw, r)
        seed_mask = best[2]
        roi = roi_from_mask(seed_mask, margin=64)
        fg, bg = bootstrap_masks(floor, seed_mask, coh_k=args.coh_k, roi=roi)
        # Whitening is a CAMERA property (the sensor's gradient anisotropy), so
        # estimate it from the whole floor bedding minus just an animal halo —
        # NOT from the coh_k-eroded / ROI-restricted bg used for the histogram
        # model. Two separate concerns:
        #   * the coh_k floor erosion exists to keep coherence *windows* off the
        #     wall (a sampling concern for the histograms);
        #   * it must not touch the whitening covariance, because the near-wall
        #     bedding carries much of the anisotropy. Eroding it biases the
        #     ratio (measured: eig 1.90 -> 1.49) while leaving the axis intact.
        # And the full floor (~347k px) gives a far more stable 2x2 covariance
        # than the local bg (~20k px).
        halo = cv2.dilate(seed_mask.astype(np.uint8),
                          np.ones((41, 41), np.uint8)).astype(bool)
        whiten_bedding = floor & (~halo)
        W = estimate_whitening(G, whiten_bedding)
        feats = image_features(bgrs[c], W, coh_k=args.coh_k)
        model = AppearanceModel.from_masks(feats, fg, bg)
        post = model.posterior_fg(feats)
        posts.append(post); rois.append(roi); models.append(model)

        rep = whitening_report(G, whiten_bedding)
        ref = _bright_reference(G, floor, args.bright_pct)
        bed = floor & (~cv2.dilate(ref.astype(np.uint8),
                                   np.ones((41, 41), np.uint8)).astype(bool))
        # per-feature d' vs the INDEPENDENT reference (non-circular for colour)
        dprime = {}
        for name in model.features:
            a = feats.get(name)[ref]; b = feats.get(name)[bed]
            dprime[name] = (np.median(a) - np.median(b)) / (
                0.5 * (a.std() + b.std()) + 1e-9)
        auc = _auc(post[ref], post[bed])

        print("-" * 70)
        print(f"cam{c}")
        print(f"  whitening : eig_ratio {rep['eig_ratio']:.2f}   "
              f"dominant gradient {rep['dominant_grad_deg']:.0f}deg  "
              f"(=> structure {(rep['dominant_grad_deg'] + 90) % 180:.0f}deg, "
              f"expect ~90 vertical)")
        print(f"  seed      : best-yaw render {int(seed_mask.sum())}px  ->  "
              f"fg {int(fg.sum())}px, bg {int(bg.sum())}px")
        print(f"  d' vs independent bright-rat: " +
              "  ".join(f"{k}={v:+.2f}" for k, v in dprime.items()))
        print(f"  posterior : AUC {auc:.3f}   median on rat "
              f"{np.median(post[ref]):.3f}  on bedding {np.median(post[bed]):.3f}")

    # --- energy at the reference pose (sanity, comparable across cameras) ---
    print("-" * 70)
    for c in (0, 1):
        sil = (render_mesh_pose_silhouette(
            mesh, RatPose(root_pos=seed_pos, root_rot=np.array([0, 0, 0.0]),
                          scale=1.0), P[c], shp) > 0).astype(np.float32)
        e = appearance_energy(sil, posts[c], rois[c])
        print(f"cam{c}: appearance energy at nominal render (yaw 0) = {e:.4f}")

    # --- optional motion blur ---
    if args.next_frame >= 0:
        print("=" * 70)
        if args.fps <= 0:
            print("motion blur: --fps required with --next-frame (use 30 or 50)")
            return 1
        bgr1 = [_load_bgr(args.cam0, args.next_frame, args.bayer_pattern),
                _load_bgr(args.cam1, args.next_frame, args.bayer_pattern)]
        G1 = [b[:, :, 1] for b in bgr1]
        R1 = detect_stereo(G1[0], G1[1], floors[0], floors[1], P[0], P[1],
                           rng=np.random.default_rng(args.seed))
        if not R1.accepted:
            print("motion blur: next frame has no stereo pair; cannot estimate "
                  "displacement.")
            return 1
        disp = float(np.linalg.norm(np.asarray(R1.point) - np.asarray(R.point)))
        speed = disp * args.fps
        print(f"motion blur ({args.fps:.0f} Hz):  point moved {disp:.1f} mm "
              f"between frames  ->  {speed:.0f} mm/s")
        p0 = RatPose(root_pos=seed_pos, root_rot=np.array([0, 0, 0.0]), scale=1.0)
        p1 = RatPose(root_pos=np.array([R1.point[0], R1.point[1], seed_pos[2]]),
                     root_rot=np.array([0, 0, 0.0]), scale=1.0)
        for c in (0, 1):
            sharp = render_coverage(mesh, p0, P[c], shp)
            print(f"  cam{c} (sharp footprint {int(sharp.sum())}px):")
            for sh, lbl in [(None, "full interval"), (1 / 250., "1/250 s"),
                            (1 / 1000., "1/1000 s")]:
                cov = render_coverage(mesh, p0, P[c], shp, pose_next=p1,
                                      fps=args.fps, exposure_s=sh, n_sub=7)
                tot = int((cov > 0).sum())
                partial = int(((cov > 0) & (cov < 1)).sum())
                pct = 100.0 * partial / max(tot, 1)
                print(f"    {lbl:14s}: coverage {int(cov.sum()):7d}  "
                      f"partial {partial:6d}px ({pct:4.1f}% of footprint)")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
