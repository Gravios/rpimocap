#!/usr/bin/env python3
"""
stereo_gate.py
==============
Stereo-gate the per-frame candidate detections from the two cameras into
a single in-arena 3-D trajectory — rejecting through-the-glass floor
patches and other out-of-arena artifacts that a single 2-D view cannot
disambiguate (ROADMAP Phase 1).

Consumes the candidates_cam{0,1}.csv files the probe writes (frame, cx,
cy, area) plus the calibration, and for each synchronized frame picks the
cam0↔cam1 pairing that is epipolar-consistent AND triangulates to a point
inside the arena volume with low reprojection error. Writes a 3-D
trajectory CSV and reports how many frames had a valid in-arena
detection.

Example
-------
  python tools/stereo_gate.py \
      --candidates-dir strohA-al-RPICAM-20260214/texdist_021722_motion_pct \
      --calib calib/calib_from_corners.npz \
      --out strohA-al-RPICAM-20260214/texdist_021722_motion_pct/stereo_track3d.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rpimocap.reconstruction.stereo_track import gate_trajectory
from rpimocap.reconstruction.epipolar import fundamental_from_projections


def _load_candidates(path):
    by_frame = defaultdict(list)
    with open(path) as fh:
        r = csv.DictReader(fh)
        for row in r:
            by_frame[int(row["frame"])].append(
                (float(row["cx"]), float(row["cy"])))
    return by_frame


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidates-dir", required=True,
                    help="Dir with candidates_cam0.csv / candidates_cam1.csv")
    ap.add_argument("--calib", required=True)
    ap.add_argument("--arena-bounds", default="-140,140,-215,215,0,388",
                    help="xmin,xmax,ymin,ymax,zmin,zmax (mm). Use = "
                         "syntax for negatives: --arena-bounds=\"-140,...\"")
    ap.add_argument("--arena-pad-mm", type=float, default=30.0)
    ap.add_argument("--max-epipolar-px", type=float, default=8.0)
    ap.add_argument("--max-reproj-px", type=float, default=8.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    cal = np.load(args.calib)
    P0 = cal.get("dlt_P0", cal.get("P0"))
    P1 = cal.get("dlt_P1", cal.get("P1"))
    if P0 is None or P1 is None:
        sys.exit("calib lacks dlt_P0/dlt_P1")
    bounds = tuple(float(x) for x in args.arena_bounds.split(","))

    c0 = _load_candidates(
        os.path.join(args.candidates_dir, "candidates_cam0.csv"))
    c1 = _load_candidates(
        os.path.join(args.candidates_dir, "candidates_cam1.csv"))
    common = sorted(set(c0) & set(c1))
    print(f"cam0 frames: {len(c0)}, cam1 frames: {len(c1)}, "
          f"overlapping: {len(common)}")

    F = fundamental_from_projections(np.asarray(P0), np.asarray(P1))
    dets = gate_trajectory(
        c0, c1, np.asarray(P0), np.asarray(P1), F=F,
        arena_bounds=bounds, arena_pad_mm=args.arena_pad_mm,
        max_epipolar_px=args.max_epipolar_px,
        max_reproj_px=args.max_reproj_px)

    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "X", "Y", "Z", "u0", "v0", "u1", "v1",
                    "reproj_err"])
        for fi in sorted(dets):
            d = dets[fi]
            w.writerow([fi, f"{d.point[0]:.2f}", f"{d.point[1]:.2f}",
                        f"{d.point[2]:.2f}", f"{d.uv0[0]:.1f}",
                        f"{d.uv0[1]:.1f}", f"{d.uv1[0]:.1f}",
                        f"{d.uv1[1]:.1f}", f"{d.reproj_err:.2f}"])

    n_ok = len(dets)
    print(f"in-arena stereo detections: {n_ok}/{len(common)} frames "
          f"({100*n_ok/max(len(common),1):.1f}%)")
    if n_ok:
        zs = np.array([dets[f].point[2] for f in dets])
        print(f"  Z range {zs.min():.0f}–{zs.max():.0f} mm "
              f"(floor=0; rat body typically 0–60 mm)")
        print(f"  → {args.out}")
    else:
        print("  no in-arena detections — check calibration / arena "
              "bounds, or whether candidates exist in both cameras.")


if __name__ == "__main__":
    main()
