#!/usr/bin/env python3
"""
synthetic_dataset_demo.py
=========================
Generate a synthetic-pose dataset (Phase A) and render a montage of
silhouettes (with skeleton overlay) per camera, so you can eyeball the
body model and the projected poses. Optionally save the dataset.

Uses the real calibration (--calib dlt_P0/dlt_P1) onto the real image
size, or demo cameras if no calib is given.

Example
-------
  python tools/synthetic_dataset_demo.py \
      --calib calib/calib_from_corners.npz \
      --n-poses 12 --pose-fraction 0.6 --seed 0 \
      --save-dir "${SUBJECT_DIR}/synth_ds" \
      --out "${SUBJECT_DIR}/synth_ds_montage.png"
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rpimocap.model import rat_skeleton as rs
from rpimocap.model import synthetic_dataset as sd


def _demo_camera(cam_pos, look_at, f=1500, cx=1014, cy=540):
    cam_pos = np.asarray(cam_pos, float)
    look_at = np.asarray(look_at, float)
    fwd = look_at - cam_pos; fwd /= np.linalg.norm(fwd)
    up = np.array([0, 0, 1.0])
    right = np.cross(fwd, up); right /= np.linalg.norm(right)
    up2 = np.cross(right, fwd)
    R = np.vstack([right, -up2, fwd]); t = -R @ cam_pos
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
    return K @ np.hstack([R, t.reshape(3, 1)])


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calib", default=None)
    ap.add_argument("--image-w", type=int, default=2028)
    ap.add_argument("--image-h", type=int, default=1080)
    ap.add_argument("--n-poses", type=int, default=12)
    ap.add_argument("--pose-fraction", type=float, default=0.6)
    ap.add_argument("--scale-min", type=float, default=0.85)
    ap.add_argument("--scale-max", type=float, default=1.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-workers", type=int, default=1)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--save-dir", default=None,
                    help="If set, save the dataset (manifest + meta).")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    W, H = args.image_w, args.image_h
    if args.calib:
        cal = np.load(args.calib)
        P0 = cal.get("dlt_P0", cal.get("P0"))
        P1 = cal.get("dlt_P1", cal.get("P1"))
        cams = {0: P0}
        if P1 is not None:
            cams[1] = P1
    else:
        cams = {0: _demo_camera([-300, -400, 700], [0, 0, 194]),
                1: _demo_camera([300, -400, 700], [0, 0, 194])}

    ds = sd.generate_dataset(
        args.n_poses, cams, (W, H), seed=args.seed,
        pose_fraction=args.pose_fraction,
        scale_range=(args.scale_min, args.scale_max),
        n_workers=args.n_workers)
    n_valid = sum(s.valid for s in ds.samples)
    print(f"generated {len(ds)} poses ({n_valid} valid) "
          f"for {len(cams)} cameras")

    if args.save_dir:
        ds.save(args.save_dir)
        print(f"saved dataset → {args.save_dir}")

    # render: for each pose, each camera, silhouette + skeleton, cropped
    cell_w, cell_h = 280, 240
    cells = []
    for i, s in enumerate(ds.samples):
        views = []
        for cid, P in cams.items():
            sil = ds.body.silhouette(s.keypoints3d, P, (W, H))
            bgr = cv2.cvtColor(sil, cv2.COLOR_GRAY2BGR)
            for (p, c) in rs.RAT23_BONES:
                a = s.keypoints2d[cid][rs.RAT23_INDEX[p]]
                b = s.keypoints2d[cid][rs.RAT23_INDEX[c]]
                if a[0] > -1e8 and b[0] > -1e8:
                    cv2.line(bgr, (int(a[0]), int(a[1])),
                             (int(b[0]), int(b[1])), (0, 0, 255), 1)
            ys, xs = np.where(sil > 0)
            if len(xs):
                x0, x1 = max(0, xs.min() - 20), xs.max() + 20
                y0, y1 = max(0, ys.min() - 20), ys.max() + 20
                crop = bgr[y0:y1, x0:x1]
            else:
                crop = bgr
            crop = cv2.resize(crop, (cell_w, cell_h))
            cv2.putText(crop, f"p{i} cam{cid}", (6, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            views.append(crop)
        cells.append(np.hstack(views))

    cw = cells[0].shape[1]; ch = cells[0].shape[0]
    cols = args.cols
    rows = (len(cells) + cols - 1) // cols
    grid = np.full((rows * ch, cols * cw, 3), 15, np.uint8)
    for k, cell in enumerate(cells):
        r, c = divmod(k, cols)
        grid[r*ch:(r+1)*ch, c*cw:(c+1)*cw] = cell
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".",
                exist_ok=True)
    cv2.imwrite(args.out, grid)
    print(f"montage → {args.out}")


if __name__ == "__main__":
    main()
