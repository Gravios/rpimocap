#!/usr/bin/env python3
"""
tools/skeleton_overlay.py
=========================
Draw the posed skeleton and mesh silhouette over the real camera image, at
several body scales side by side, so the model's size can be judged against the
animal by eye.

This exists because the model is suspected undersized: the rest bones sum to
141 mm nose-to-TailBase while the skeleton module's own docstring claims a
~230 mm body, and the appearance posterior's main blob measures 1.31x / 1.32x
the nominal-scale render in the two cameras. Those are indirect arguments; this
puts the skeleton on the pixels.

Each panel is one scale. Drawn per panel:
  * the mesh silhouette outline (thin contour),
  * bones as lines and joints as dots (the tail chain in a separate colour),
  * a printed mm scale bar measured through the DLT at the animal's height.

Also prints a numeric scale estimate independent of the eye: the animal's major
axis is measured by PCA on the appearance posterior, and compared with the
model's projected major axis, giving the scale that would match them.

Examples
--------
    python tools/skeleton_overlay.py \\
        --cam0 frame_002716_cam0.png --cam1 frame_002716_cam1.png \\
        --calib calib_from_corners.npz --scales 1.0,1.3,1.6 --out overlays/

    python tools/skeleton_overlay.py \\
        --cam0 raw/cam0_..._raw.tif --cam1 raw/cam1_..._raw.tif \\
        --calib calib/calib_from_corners.npz --frame 1722 --out overlays/
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

from rpimocap.detection.topo_detect import build_floor_mask, detect_stereo
from rpimocap.model.appearance import (AppearanceModel, bootstrap_masks,
                                       estimate_whitening, image_features,
                                       roi_from_mask)
from rpimocap.model.mesh_model import (build_rat_mesh,
                                       render_mesh_pose_silhouette)
from rpimocap.model.rat_skeleton import (RAT23_INDEX, RAT23_JOINTS,
                                         RAT_BONES_EXT, TAIL_JOINTS, RatPose,
                                         forward_kinematics_transforms,
                                         project_pose)

_ARENA_CORNERS = np.array([
    [-140, -215,   0], [140, -215,   0], [140,  215,   0], [-140,  215,   0],
    [-140, -215, 388], [140, -215, 388], [140,  215, 388], [-140,  215, 388],
], dtype=np.float64)

# BGR
_C_SIL = (60, 200, 255)      # silhouette outline  (amber)
_C_BONE = (80, 255, 80)      # body bones          (green)
_C_TAIL = (255, 120, 60)     # tail chain          (blue)
_C_JOINT = (255, 255, 255)
_C_TXT = (255, 255, 255)


def _load_bgr(path: str, frame: int, bayer_pattern: str) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".tif", ".tiff"):
        from rpimocap.io.export import TiffCapture
        cap = TiffCapture(path, bayer_pattern=bayer_pattern)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame >= n:
            sys.exit(f"error: {path} has {n} frame(s); --frame {frame} "
                     f"out of range")
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, bgr = cap.read()
        if not ok or bgr is None:
            sys.exit(f"error: cannot read frame {frame} from {path}")
        return bgr
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        sys.exit(f"error: cannot read image {path}")
    return bgr


def _load_calib(path: str):
    d = np.load(path)
    if "dlt_P0" in d and "dlt_P1" in d:
        return d["dlt_P0"], d["dlt_P1"]
    sys.exit(f"error: {path} has no dlt_P0/dlt_P1")


def _mm_per_px(P: np.ndarray, at_mm: np.ndarray) -> float:
    """Local image scale (mm per pixel) at a 3D point, along arena +x."""
    a = project_pose(np.stack([at_mm, at_mm + np.array([10.0, 0.0, 0.0])]), P)
    return 10.0 / max(float(np.linalg.norm(a[1] - a[0])), 1e-6)


def _draw_overlay(canvas, mesh, pose, P, shape, mm_px, label):
    """Draw silhouette contour + bones + joints for one pose onto ``canvas``."""
    sil = render_mesh_pose_silhouette(mesh, pose, P, shape)
    cnts, _ = cv2.findContours((sil > 0).astype(np.uint8), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(canvas, cnts, -1, _C_SIL, 1, cv2.LINE_AA)

    _, wp = forward_kinematics_transforms(pose)
    for (p, c) in RAT_BONES_EXT:
        a = project_pose(np.stack([wp[p], wp[c]]), P)
        col = _C_TAIL if (c in TAIL_JOINTS or p in TAIL_JOINTS) else _C_BONE
        cv2.line(canvas, tuple(np.round(a[0]).astype(int)),
                 tuple(np.round(a[1]).astype(int)), col, 1, cv2.LINE_AA)
    for name in RAT23_JOINTS:
        q = project_pose(wp[name][None, :], P)[0]
        cv2.circle(canvas, tuple(np.round(q).astype(int)), 2, _C_JOINT, -1,
                   cv2.LINE_AA)
    return sil


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Overlay the skeleton/silhouette on the image at several "
                    "scales, to judge model size against the animal.")
    ap.add_argument("--cam0", required=True)
    ap.add_argument("--cam1", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--bayer-pattern", default="RGGB")
    ap.add_argument("--scales", default="1.0,1.3,1.6",
                    help="comma-separated body scales to draw (default "
                         "1.0,1.3,1.6)")
    ap.add_argument("--yaw", type=float, default=None,
                    help="body yaw in degrees; default = best of a 15deg sweep "
                         "against the appearance posterior")
    ap.add_argument("--no-tail", action="store_true")
    ap.add_argument("--margin", type=int, default=90,
                    help="crop margin in px around the animal (default 90)")
    ap.add_argument("--zoom", type=float, default=2.0,
                    help="upscale factor for legibility (default 2)")
    ap.add_argument("--out", default="overlays",
                    help="output directory for the PNGs")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    scales = [float(s) for s in args.scales.split(",") if s.strip()]
    P = list(_load_calib(args.calib))
    bgrs = [_load_bgr(args.cam0, args.frame, args.bayer_pattern),
            _load_bgr(args.cam1, args.frame, args.bayer_pattern)]
    Gs = [b[:, :, 1] for b in bgrs]
    shp = Gs[0].shape
    floors = [build_floor_mask(P[c], _ARENA_CORNERS, shp, mode="floor") > 0
              for c in (0, 1)]

    R = detect_stereo(Gs[0], Gs[1], floors[0].astype(np.uint8),
                      floors[1].astype(np.uint8), P[0], P[1],
                      rng=np.random.default_rng(args.seed))
    if not R.accepted:
        sys.exit("error: detector found no consistent stereo pair on this frame")
    seed_pos = np.array([R.point[0], R.point[1], max(float(R.point[2]), 45.0)])
    mesh = build_rat_mesh(with_tail=not args.no_tail)
    print(f"frame {args.frame}: 3D point ({seed_pos[0]:.0f}, {seed_pos[1]:.0f},"
          f" {seed_pos[2]:.0f}) mm, reproj {R.reproj_err:.1f} px")

    # appearance posterior per camera (also gives the numeric scale estimate)
    posts = []
    for c in (0, 1):
        s0 = render_mesh_pose_silhouette(
            mesh, RatPose(root_pos=seed_pos, root_rot=np.zeros(3), scale=1.0),
            P[c], shp) > 0
        fg, bg = bootstrap_masks(floors[c], s0, coh_k=32,
                                 roi=roi_from_mask(s0, margin=64))
        halo = cv2.dilate(s0.astype(np.uint8),
                          np.ones((41, 41), np.uint8)).astype(bool)
        W = estimate_whitening(Gs[c], floors[c] & ~halo)
        f = image_features(bgrs[c], W, coh_k=32)
        posts.append(AppearanceModel.from_masks(f, fg, bg).posterior_fg(f))

    # yaw: best over a sweep unless given
    if args.yaw is None:
        from rpimocap.model.appearance import appearance_energy
        # The ROI must be FIXED across the sweep, or the energies are not
        # comparable pose to pose (recomputing it per pose put the optimum
        # 135 deg away from the correct answer). Use the union of all
        # hypotheses as the evaluation window.
        sweep = np.arange(0, 360, 15.0)
        rois = []
        for c in (0, 1):
            acc = np.zeros(shp, bool)
            for y in sweep:
                acc |= render_mesh_pose_silhouette(
                    mesh, RatPose(root_pos=seed_pos,
                                  root_rot=np.array([0, 0, np.radians(y)]),
                                  scale=max(scales)), P[c], shp) > 0
            rois.append(roi_from_mask(acc, margin=64))
        best = None
        for y in sweep:
            pose = RatPose(root_pos=seed_pos,
                           root_rot=np.array([0, 0, np.radians(y)]), scale=1.0)
            e = 0.0
            for c in (0, 1):
                sil = render_mesh_pose_silhouette(mesh, pose, P[c], shp)
                e += appearance_energy((sil > 0).astype(np.float32), posts[c],
                                       rois[c])
            if best is None or e < best[0]:
                best = (e, y)
        yaw = best[1]
        print(f"yaw (best of sweep): {yaw:.0f} deg")
    else:
        yaw = float(args.yaw)
        print(f"yaw (given): {yaw:.0f} deg")

    os.makedirs(args.out, exist_ok=True)
    written = []
    mesh_body = build_rat_mesh(with_tail=False)
    print()
    for c in (0, 1):
        mm_px = _mm_per_px(P[c], seed_pos)
        # --- numeric scale estimates, independent of the drawing ---
        comp = (posts[c] > 0.5) & floors[c]
        n, lab, stats, _ = cv2.connectedComponentsWithStats(
            comp.astype(np.uint8))
        k = 1 + int(np.argmax(stats[1:, 4])) if n > 1 else 0
        blob = lab == k
        ys, xs = np.where(blob)
        pts = np.stack([xs, ys], 1).astype(np.float64); pts -= pts.mean(0)
        ev = np.linalg.eigvalsh(np.cov(pts.T))
        animal_major = 4.0 * np.sqrt(max(ev[-1], 0.0))     # +/-2 sd

        pose1 = RatPose(root_pos=seed_pos,
                        root_rot=np.array([0, 0, np.radians(yaw)]), scale=1.0)
        s1 = render_mesh_pose_silhouette(mesh, pose1, P[c], shp) > 0
        s1b = render_mesh_pose_silhouette(mesh_body, pose1, P[c], shp) > 0
        my, mx = np.where(s1)
        mp = np.stack([mx, my], 1).astype(np.float64); mp -= mp.mean(0)
        model_major = 4.0 * np.sqrt(max(np.linalg.eigvalsh(np.cov(mp.T))[-1], 0.0))

        print(f"cam{c}: {mm_px:.3f} mm/px at the animal")
        print(f"      posterior blob : {int(blob.sum()):6d} px area, "
              f"major axis {animal_major:5.0f} px ({animal_major * mm_px:4.0f} mm)")
        print(f"      model @1.0 body-only : {int(s1b.sum()):6d} px  "
              f"-> AREA scale estimate {np.sqrt(blob.sum() / max(s1b.sum(), 1)):.2f}")
        print(f"      model @1.0 with tail : {int(s1.sum()):6d} px  "
              f"-> AREA scale estimate {np.sqrt(blob.sum() / max(s1.sum(), 1)):.2f}")
        print(f"      major-axis scale estimate (with tail): "
              f"{animal_major / max(model_major, 1e-6):.2f}")

        # --- panels ---
        ys, xs = np.where(s1)
        x0 = max(0, min(xs.min(), np.where(blob)[1].min()) - args.margin); x1 = min(shp[1], max(xs.max(), np.where(blob)[1].max()) + args.margin)
        y0 = max(0, min(ys.min(), np.where(blob)[0].min()) - args.margin)
        y1 = min(shp[0], max(ys.max(), np.where(blob)[0].max()) + args.margin)
        panels = []
        for s in scales:
            canvas = bgrs[c].copy()
            pose = RatPose(root_pos=seed_pos,
                           root_rot=np.array([0, 0, np.radians(yaw)]), scale=s)
            _draw_overlay(canvas, mesh, pose, P[c], shp, mm_px, s)
            crop = canvas[y0:y1, x0:x1]
            crop = cv2.resize(crop, None, fx=args.zoom, fy=args.zoom,
                              interpolation=cv2.INTER_NEAREST)
            # 50 mm scale bar
            bar = int(round(50.0 / mm_px * args.zoom))
            h, w = crop.shape[:2]
            cv2.line(crop, (12, h - 14), (12 + bar, h - 14), _C_TXT, 2)
            cv2.putText(crop, "50 mm", (12, h - 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, _C_TXT, 1, cv2.LINE_AA)
            cv2.putText(crop, f"scale {s:.2f}", (12, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, _C_TXT, 2, cv2.LINE_AA)
            panels.append(crop)
        hmax = max(p.shape[0] for p in panels)
        panels = [cv2.copyMakeBorder(p, 0, hmax - p.shape[0], 0, 6,
                                     cv2.BORDER_CONSTANT, value=(0, 0, 0))
                  for p in panels]
        sheet = np.hstack(panels)
        cv2.putText(sheet, f"cam{c}  frame {args.frame}  yaw {yaw:.0f}deg  "
                           f"(amber=silhouette, green=body, blue=tail)",
                    (12, sheet.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    _C_TXT, 1, cv2.LINE_AA)
        path = os.path.join(args.out, f"overlay_cam{c}_frame{args.frame:06d}.png")
        cv2.imwrite(path, sheet)
        written.append(path)
    print()
    for p in written:
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
