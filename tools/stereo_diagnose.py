#!/usr/bin/env python3
"""
stereo_diagnose.py
==================
Diagnose why the stereo gate (tools/stereo_gate.py) accepts few frames.

Given the per-frame candidate centroids the probe writes
(candidates_cam{0,1}.csv) plus the calibration, this measures — with NO
gates applied — the actual epipolar distance, triangulated Z, and
reprojection error for the most plausible cam0↔cam1 pairing each frame
(the largest-area blob in each camera). The distributions tell you which
knob is wrong:

  * epipolar distances mostly >8px but <~40px  → the --max-epipolar-px
    gate is too tight for this calibration's residual error; raise it to
    ~1.5x the median.
  * epipolar distances huge (median >100px)    → the largest blob in the
    two cameras often isn't the same object (correspondence problem), or
    the fundamental matrix / calibration is off.
  * triangulated Z far outside [0,388] mm      → calibration unit/frame
    mismatch or a left/right camera swap, not a threshold issue.

It also reports the best pairing found by an exhaustive epipolar search
(all candidate pairs, not just largest), to separate "gate too tight"
from "wrong correspondence".

Example
-------
  python tools/stereo_diagnose.py \
      --candidates-dir <out_dir> \
      --calib calib/calib_from_corners.npz \
      --arena-bounds="-140,140,-215,215,0,388"
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
from rpimocap.reconstruction.triangulate import (
    triangulate_dlt, reprojection_error)


def _load(path):
    d = defaultdict(list)
    with open(path) as fh:
        for row in csv.DictReader(fh):
            d[int(row["frame"])].append(
                (float(row["cx"]), float(row["cy"]),
                 float(row.get("area", 1.0))))
    return d


def _pct(a, q):
    return float(np.percentile(a, q)) if len(a) else float("nan")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidates-dir", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--arena-bounds", default="-140,140,-215,215,0,388",
                    help="xmin,xmax,ymin,ymax,zmin,zmax (mm). Use = "
                         "syntax for negatives.")
    ap.add_argument("--arena-pad-mm", type=float, default=30.0)
    args = ap.parse_args(argv)

    cal = np.load(args.calib)
    P0 = np.asarray(cal.get("dlt_P0", cal.get("P0")))
    P1 = np.asarray(cal.get("dlt_P1", cal.get("P1")))
    if P0 is None or P1 is None:
        sys.exit("calib lacks dlt_P0/dlt_P1")
    b = tuple(float(x) for x in args.arena_bounds.split(","))
    pad = args.arena_pad_mm
    F = fundamental_from_projections(P0, P1)

    c0 = _load(os.path.join(args.candidates_dir, "candidates_cam0.csv"))
    c1 = _load(os.path.join(args.candidates_dir, "candidates_cam1.csv"))
    frames = sorted(set(c0) & set(c1))
    print(f"frames with candidates in BOTH cameras: {len(frames)}")
    if not frames:
        return
    n0 = np.mean([len(c0[f]) for f in frames])
    n1 = np.mean([len(c1[f]) for f in frames])
    print(f"mean candidates/frame: cam0={n0:.1f} cam1={n1:.1f}")

    # ---- (A) largest-blob pairing (naive correspondence) ----
    eps, zs, res = [], [], []
    for fi in frames:
        p0 = max(c0[fi], key=lambda t: t[2])[:2]
        p1 = max(c1[fi], key=lambda t: t[2])[:2]
        eps.append(epipolar_distance(F, p0, p1))
        X = triangulate_dlt(P0, P1, p0, p1)[:3]
        zs.append(X[2])
        res.append(max(reprojection_error(P0, X, p0),
                       reprojection_error(P1, X, p1)))
    eps, zs, res = map(np.array, (eps, zs, res))
    print("\n[A] LARGEST-blob pairing (per frame), no gates:")
    print(f"  epipolar px: median={np.median(eps):.1f} "
          f"p25={_pct(eps,25):.1f} p75={_pct(eps,75):.1f} "
          f"min={eps.min():.1f} max={eps.max():.1f}")
    print(f"    within  8px: {100*(eps<8).mean():.1f}%   "
          f"20px: {100*(eps<20).mean():.1f}%   "
          f"40px: {100*(eps<40).mean():.1f}%")
    print(f"  triangulated Z mm: median={np.median(zs):.0f} "
          f"p25={_pct(zs,25):.0f} p75={_pct(zs,75):.0f} "
          f"min={zs.min():.0f} max={zs.max():.0f}")
    inz = (zs >= b[4] - pad) & (zs <= b[5] + pad)
    print(f"    Z within arena [{b[4]},{b[5]}]+pad: {100*inz.mean():.1f}%")
    print(f"  reproj px: median={np.median(res):.1f} "
          f"p75={_pct(res,75):.1f}")

    # ---- (B) exhaustive best-epipolar pairing per frame ----
    best_eps, best_z, best_res = [], [], []
    for fi in frames:
        be = np.inf
        bx = None
        for a in c0[fi]:
            for c in c1[fi]:
                e = epipolar_distance(F, a[:2], c[:2])
                if e < be:
                    be = e
                    bx = (a[:2], c[:2])
        best_eps.append(be)
        if bx is not None:
            X = triangulate_dlt(P0, P1, bx[0], bx[1])[:3]
            best_z.append(X[2])
            best_res.append(max(reprojection_error(P0, X, bx[0]),
                                reprojection_error(P1, X, bx[1])))
    best_eps = np.array(best_eps)
    best_z = np.array(best_z)
    best_res = np.array(best_res)
    print("\n[B] BEST-epipolar pairing (exhaustive over all candidate "
          "pairs), no gates:")
    print(f"  epipolar px: median={np.median(best_eps):.1f} "
          f"p75={_pct(best_eps,75):.1f} max={best_eps.max():.1f}")
    print(f"    within  8px: {100*(best_eps<8).mean():.1f}%   "
          f"20px: {100*(best_eps<20).mean():.1f}%")
    inzb = (best_z >= b[4] - pad) & (best_z <= b[5] + pad)
    print(f"  triangulated Z mm: median={np.median(best_z):.0f} "
          f"p25={_pct(best_z,25):.0f} p75={_pct(best_z,75):.0f}")
    print(f"    Z within arena: {100*inzb.mean():.1f}%")
    print(f"  reproj px: median={np.median(best_res):.1f} "
          f"p75={_pct(best_res,75):.1f}")

    # ---- verdict ----
    print("\nVERDICT:")
    if np.median(best_eps) > 60:
        print("  Epipolar distances large even for the BEST pairing → "
              "fundamental matrix / calibration suspect, or no true "
              "correspondence present. Check calib P0/P1 and a known "
              "point's epipolar distance.")
    elif np.median(best_z) < b[4] - 200 or np.median(best_z) > b[5] + 400:
        print("  Triangulated Z far outside the arena → calibration "
              "unit/frame mismatch or a left/right camera swap (try "
              "swapping cam0/cam1).")
    elif (best_eps < 8).mean() < 0.3 and (best_eps < 40).mean() > 0.5:
        thr = max(12.0, 1.5 * float(np.median(best_eps)))
        print(f"  Epipolar gate too tight: best-pairing distances cluster "
              f"above 8px. Re-run stereo_gate.py with "
              f"--max-epipolar-px {thr:.0f} (and --max-reproj-px "
              f"{max(8.0, 1.5*float(np.median(best_res))):.0f}).")
    elif inzb.mean() < 0.3:
        print("  Pairings are epipolar-consistent but triangulate "
              "out-of-arena → check arena bounds units / camera frame, "
              "or candidates are artifacts not the rat.")
    else:
        print("  Best pairings look in-range; if stereo_gate.py still "
              "yields little, loosen --max-epipolar-px / --max-reproj-px "
              "toward the p75 values above.")


if __name__ == "__main__":
    main()
