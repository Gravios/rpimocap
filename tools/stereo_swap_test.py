#!/usr/bin/env python3
"""
stereo_swap_test.py
===================
Test whether the cam0/cam1 assignment is swapped relative to the
calibration — a common cause of a large, CONSTANT epipolar error in the
stereo gate.

When two cameras are mounted facing each other (R diagonal ~ -1), a
cam0<->cam1 label swap between annotation and detection makes every true
correspondence land ~half a frame off its epipolar line — a fixed offset
(e.g. ~454px) that no threshold change can rescue. This tool triangulates
the real candidate centroids BOTH ways (as-is and swapped) and reports the
median epipolar distance + triangulated Z for each, so the swap is
confirmed or ruled out from data.

Reads candidates_cam{0,1}.csv (from the probe) + the calibration; pairs
the largest-area blob per camera per frame (a robust proxy for the rat).

Example
-------
  python tools/stereo_swap_test.py \
      --candidates-dir <out_dir> \
      --calib calib/calib_from_corners.npz
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rpimocap.reconstruction.epipolar import (
    fundamental_from_projections, epipolar_distance)
from rpimocap.reconstruction.triangulate import triangulate_dlt


def _load(path):
    d = defaultdict(list)
    with open(path) as fh:
        for r in csv.DictReader(fh):
            d[int(r["frame"])].append(
                (float(r["cx"]), float(r["cy"]),
                 float(r.get("area", 1.0))))
    return d


def _evaluate(Pa, Pb, ca, cb, frames, z_lo, z_hi):
    F = fundamental_from_projections(Pa, Pb)
    eps, zs = [], []
    for fi in frames:
        a = max(ca[fi], key=lambda t: t[2])[:2]
        b = max(cb[fi], key=lambda t: t[2])[:2]
        eps.append(epipolar_distance(F, a, b))
        zs.append(triangulate_dlt(Pa, Pb, a, b)[2])
    eps, zs = np.array(eps), np.array(zs)
    in_z = 100.0 * ((zs >= z_lo) & (zs <= z_hi)).mean()
    return (float(np.median(eps)), float(np.median(zs)), in_z)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidates-dir", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--z-floor", type=float, default=-30.0,
                    help="Lower Z (mm) counted as plausibly in-arena.")
    ap.add_argument("--z-ceil", type=float, default=418.0,
                    help="Upper Z (mm) counted as plausibly in-arena.")
    args = ap.parse_args(argv)

    cal = np.load(args.calib)
    P0 = np.asarray(cal.get("dlt_P0", cal.get("P0")))
    P1 = np.asarray(cal.get("dlt_P1", cal.get("P1")))
    if P0 is None or P1 is None:
        sys.exit("calib lacks dlt_P0/dlt_P1")

    c0 = _load(os.path.join(args.candidates_dir, "candidates_cam0.csv"))
    c1 = _load(os.path.join(args.candidates_dir, "candidates_cam1.csv"))
    frames = sorted(set(c0) & set(c1))
    print(f"frames with candidates in both cameras: {len(frames)}")
    if not frames:
        return

    e_n, z_n, iz_n = _evaluate(P0, P1, c0, c1, frames,
                               args.z_floor, args.z_ceil)
    e_s, z_s, iz_s = _evaluate(P1, P0, c0, c1, frames,
                               args.z_floor, args.z_ceil)
    print(f"  normal  (cam0->P0, cam1->P1): "
          f"epipolar median={e_n:7.1f}px  Z median={z_n:7.0f}mm  "
          f"Z-in-arena={iz_n:.0f}%")
    print(f"  swapped (cam0->P1, cam1->P0): "
          f"epipolar median={e_s:7.1f}px  Z median={z_s:7.0f}mm  "
          f"Z-in-arena={iz_s:.0f}%")

    print("\nVERDICT:")
    if e_s < 25 and e_s < 0.3 * e_n and args.z_floor <= z_s <= args.z_ceil:
        print("  SWAP DETECTED — the swapped pairing is epipolar-"
              "consistent and in-arena while the normal one is not.")
        print("  Fix: swap which TIFF is --cam0/--cam1 in the probe, OR "
              "swap dlt_P0/dlt_P1 in stereo_gate.py for this dataset.")
    elif e_n < 25 and args.z_floor <= z_n <= args.z_ceil:
        print("  No swap — the normal pairing is already consistent. A "
              "low stereo-gate yield is then a threshold/correspondence "
              "issue (see stereo_diagnose.py).")
    else:
        print("  Neither orientation is epipolar-consistent and in-arena. "
              "Not a simple swap — suspect the pixel convention of the "
              "candidate centroids vs the calibration (flip/transpose/"
              "crop), or recalibrate. Project a known arena corner "
              "through P0 and confirm it lands on that corner in the raw "
              "cam0 TIFF.")


if __name__ == "__main__":
    main()
