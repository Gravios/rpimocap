#!/usr/bin/env python3
"""
project_corners.py
==================
Overlay the calibration's arena-corner projections onto the actual raw
cam0/cam1 TIFF frames, to confirm the calibration and pixel convention
match the images the detector runs on.

For each of the 8 arena corners this draws:
  * a GREEN circle where the calibration projects the corner
    (dlt_P{0,1} @ arena_xyz), and
  * a RED cross at the clicked pixel from align.csv (px0/px1).

If green lands on the real arena corners in the image (and on the red
crosses), the calibration-to-image mapping is correct and any stereo-gate
failure is a correspondence problem (the detector picking different
physical objects in the two views), NOT a coordinate-convention bug. If
green is shifted / flipped from the real corners, the align.csv clicks
were made on a differently-processed image than the probe produces.

It also back-projects a chosen image point in each camera to the arena
floor (z=0), so you can check whether a detected blob in each view maps
to the same arena location.

Example
-------
  python tools/project_corners.py \
      --cam0 <cam0.tif> --cam1 <cam1.tif> \
      --calib calib/calib_from_corners.npz \
      --align align.csv --frame 900 \
      --out /tmp/corner_overlay
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
from rpimocap.io.export import TiffCapture


def _project(P, X):
    p = P @ np.append(X, 1.0)
    return p[:2] / p[2]


def _floor_intersect(P, u, v):
    """Arena floor (z=0) point whose projection through P is pixel (u,v)."""
    M = np.array([[P[0, 0], P[0, 1], -u],
                  [P[1, 0], P[1, 1], -v],
                  [P[2, 0], P[2, 1], -1.0]])
    rhs = -np.array([P[0, 3], P[1, 3], P[2, 3]])
    xy_s = np.linalg.solve(M, rhs)
    return xy_s[0], xy_s[1]


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cam0", required=True)
    ap.add_argument("--cam1", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--align", required=True,
                    help="align.csv with arena_xyz + px0/px1 clicks")
    ap.add_argument("--frame", type=int, default=900)
    ap.add_argument("--bayer-pattern", default="RGGB")
    ap.add_argument("--probe-floor-px", nargs=4, type=float, default=None,
                    metavar=("U0", "V0", "U1", "V1"),
                    help="Optional: a cam0 pixel (U0,V0) and cam1 pixel "
                         "(U1,V1) to back-project to the arena floor and "
                         "compare (e.g. a detected blob in each view).")
    ap.add_argument("--out", default="/tmp/corner_overlay")
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    cal = np.load(args.calib)
    P = {0: np.asarray(cal.get("dlt_P0", cal.get("P0"))),
         1: np.asarray(cal.get("dlt_P1", cal.get("P1")))}

    rows = list(csv.DictReader(open(args.align)))
    arena = {r["label"]: np.array(
        [float(r["arena_x"]), float(r["arena_y"]), float(r["arena_z"])])
        for r in rows}
    pxk = {0: ("px0_x", "px0_y"), 1: ("px1_x", "px1_y")}
    clicks = {cam: {r["label"]: (float(r[pxk[cam][0]]),
                                 float(r[pxk[cam][1]])) for r in rows}
              for cam in (0, 1)}

    for cam, path in ((0, args.cam0), (1, args.cam1)):
        cap = TiffCapture(path, bayer_pattern=args.bayer_pattern)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if args.frame >= n:
            print(f"cam{cam}: frame {args.frame} >= frame count {n}")
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
        ok, frame = cap.read()
        if not ok or frame is None:
            print(f"cam{cam}: could not read frame {args.frame}")
            continue
        img = (frame if frame.ndim == 3
               else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
        print(f"cam{cam}: frame shape {img.shape}")

        ds = []
        for lbl, X in arena.items():
            u, v = _project(P[cam], X)
            cv2.circle(img, (int(u), int(v)), 12, (0, 255, 0), 3)
            cv2.putText(img, lbl, (int(u) + 14, int(v)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cu, cvv = clicks[cam][lbl]
            cv2.drawMarker(img, (int(cu), int(cvv)), (0, 0, 255),
                           cv2.MARKER_CROSS, 20, 3)
            ds.append(np.hypot(u - cu, v - cvv))
        outp = os.path.join(args.out, f"corners_cam{cam}_f{args.frame}.png")
        cv2.imwrite(outp, img)
        print(f"  green(projected) vs red(clicked) median dist: "
              f"{np.median(ds):.1f}px  → {outp}")

    if args.probe_floor_px:
        u0, v0, u1, v1 = args.probe_floor_px
        x0, y0 = _floor_intersect(P[0], u0, v0)
        x1, y1 = _floor_intersect(P[1], u1, v1)
        print(f"\nfloor back-projection (z=0):")
        print(f"  cam0 px({u0:.0f},{v0:.0f}) → arena X={x0:.0f} Y={y0:.0f}")
        print(f"  cam1 px({u1:.0f},{v1:.0f}) → arena X={x1:.0f} Y={y1:.0f}")
        print(f"  separation on floor: "
              f"{np.hypot(x0 - x1, y0 - y1):.0f} mm "
              f"(small = same arena location = a valid correspondence)")

    print("\nIf GREEN lands on the real arena corners in the image, the "
          "calibration matches the TIFF and any stereo-gate failure is a "
          "correspondence problem, not a coordinate-convention bug.")


if __name__ == "__main__":
    main()
