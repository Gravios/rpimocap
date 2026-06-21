#!/usr/bin/env python3
"""
synthetic_pose_demo.py
======================
Generate synthetic rat poses with the anatomically-constrained skeleton
and render them (skeleton drawn) into one or two camera views, so you can
eyeball that the generated poses look rat-like and project sensibly.

This is the visual sanity check for rpimocap.model.rat_skeleton — the
generator that produces zero-noise ground truth for the
detect→triangulate pipeline.

Two modes:
  * --calib CAL.npz : project through the REAL dlt_P0/dlt_P1 and render
    onto the real arena's image size (2028×1080). Use this to confirm
    synthetic poses land where a real rat would.
  * (no calib)      : use two synthetic demo cameras looking at the
    arena, render on a blank canvas. Quick standalone check.

Output: a montage PNG with N sampled poses, each shown in cam0 (and cam1
if available), skeleton bones drawn, keypoints marked.

Example
-------
  python tools/synthetic_pose_demo.py \
      --calib calib/calib_from_corners.npz \
      --n-poses 12 --pose-fraction 0.7 --seed 0 \
      --out "${SUBJECT_DIR}/synth_poses.png"
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

# Allow running as a standalone script from the repo root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rpimocap.model import rat_skeleton as rs


def _demo_camera(cam_pos, look_at, f=1500, cx=1014, cy=540):
    cam_pos = np.asarray(cam_pos, float)
    look_at = np.asarray(look_at, float)
    fwd = look_at - cam_pos
    fwd /= np.linalg.norm(fwd)
    up = np.array([0, 0, 1.0])
    right = np.cross(fwd, up); right /= np.linalg.norm(right)
    up2 = np.cross(right, fwd)
    R = np.vstack([right, -up2, fwd])
    t = -R @ cam_pos
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
    return K @ np.hstack([R, t.reshape(3, 1)])


# Region colors (BGR) for drawing
_REGION_COLOR = {
    "head":      (0, 220, 255),    # yellow
    "trunk":     (0, 255, 0),      # green
    "forelimbs": (255, 160, 0),    # blue-ish
    "hindlimbs": (255, 0, 200),    # magenta
}


def _region_of(name):
    for reg, names in rs.RAT23_REGIONS.items():
        if name in names:
            return reg
    return "trunk"


def render_pose(px: np.ndarray, w: int, h: int,
                title: str = "") -> np.ndarray:
    """Draw a projected (23,2) pose onto a blank canvas."""
    img = np.full((h, w, 3), 30, np.uint8)
    # bones
    for (p, c) in rs.RAT23_BONES:
        a = px[rs.RAT23_INDEX[p]]
        b = px[rs.RAT23_INDEX[c]]
        if a[0] < -1e8 or b[0] < -1e8:
            continue
        col = _REGION_COLOR[_region_of(c)]
        cv2.line(img, (int(a[0]), int(a[1])),
                 (int(b[0]), int(b[1])), col, 2)
    # keypoints
    for name, i in rs.RAT23_INDEX.items():
        x, y = px[i]
        if x < -1e8:
            continue
        col = _REGION_COLOR[_region_of(name)]
        cv2.circle(img, (int(x), int(y)), 4, col, -1)
    if title:
        cv2.putText(img, title, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 2)
    return img


def fit_canvas(px: np.ndarray, w: int, h: int,
               pad: int = 40) -> np.ndarray:
    """Shift+scale projected points to fit a w×h canvas (for the
    no-calib demo, where projected coords aren't tied to a real image
    size)."""
    valid = px[px[:, 0] > -1e8]
    if valid.size == 0:
        return px
    mn = valid.min(axis=0)
    mx = valid.max(axis=0)
    span = np.maximum(mx - mn, 1.0)
    scale = min((w - 2 * pad) / span[0], (h - 2 * pad) / span[1])
    out = px.copy()
    m = px[:, 0] > -1e8
    out[m] = (px[m] - mn) * scale + pad
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calib", default=None,
                    help="Calibration .npz with dlt_P0/dlt_P1. If "
                         "given, project through the real cameras onto "
                         "the real image size.")
    ap.add_argument("--image-w", type=int, default=2028)
    ap.add_argument("--image-h", type=int, default=1080)
    ap.add_argument("--n-poses", type=int, default=12)
    ap.add_argument("--pose-fraction", type=float, default=0.7,
                    help="0=rest, 1=full joint-limit range.")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="Body-size multiplier.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arena-bounds", default="-140,140,-215,215,0,388",
                    help="xmin,xmax,ymin,ymax,zmin,zmax mm.")
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    bounds = tuple(float(v) for v in args.arena_bounds.split(","))

    # Projection matrices
    if args.calib:
        cal = np.load(args.calib)
        P0 = cal.get("dlt_P0", cal.get("P0"))
        P1 = cal.get("dlt_P1", cal.get("P1"))
        if P0 is None:
            raise SystemExit("calib has no dlt_P0/P0")
        w, h = args.image_w, args.image_h
        use_calib = True
        cams = [("cam0", P0)] + ([("cam1", P1)] if P1 is not None
                                 else [])
    else:
        cx = (bounds[0] + bounds[1]) / 2
        cy = (bounds[2] + bounds[3]) / 2
        cz = (bounds[4] + bounds[5]) / 2
        P0 = _demo_camera([cx - 300, cy - 400, 700], [cx, cy, cz])
        P1 = _demo_camera([cx + 300, cy - 400, 700], [cx, cy, cz])
        w, h = 480, 360
        use_calib = False
        cams = [("cam0", P0), ("cam1", P1)]

    rng = np.random.RandomState(args.seed)
    cells = []
    n_valid = 0
    for n in range(args.n_poses):
        # rejection-sample a pose that's fully in the arena
        pose = None
        for _ in range(50):
            cand = rs.sample_pose(
                rng, scale=args.scale, arena_bounds=bounds,
                fraction=args.pose_fraction)
            kp = rs.forward_kinematics(cand)
            if rs.check_arena_containment(kp, bounds):
                pose = cand
                break
        if pose is None:
            pose = cand                 # fall back to last candidate
        else:
            n_valid += 1
        kp3d = rs.forward_kinematics(pose)
        views = []
        for cam_name, P in cams:
            px = rs.project_pose(kp3d, P)
            if not use_calib:
                px = fit_canvas(px, w, h)
            views.append(render_pose(
                px, w, h, title=f"pose{n} {cam_name}"))
        cells.append(np.hstack(views))

    # montage
    cw = cells[0].shape[1]
    ch = cells[0].shape[0]
    cols = args.cols
    rows = (len(cells) + cols - 1) // cols
    grid = np.full((rows * ch, cols * cw, 3), 15, np.uint8)
    for k, cell in enumerate(cells):
        r, c = divmod(k, cols)
        grid[r*ch:(r+1)*ch, c*cw:(c+1)*cw] = cell

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".",
                exist_ok=True)
    cv2.imwrite(args.out, grid)
    print(f"{args.n_poses} poses ({n_valid} fully in-arena) → {args.out}")
    print(f"  projection: {'real calib' if use_calib else 'demo cameras'}"
          f"  fraction={args.pose_fraction} scale={args.scale}")


if __name__ == "__main__":
    main()
