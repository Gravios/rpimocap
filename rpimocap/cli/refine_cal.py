"""
rpimocap-refine-cal — refine stereo calibration from annotated arena corners
=============================================================================
Uses the 8 (or more) arena corner annotations in an align_points.csv to
jointly optimize K0, K1, R, T and radial distortion coefficients via
nonlinear least-squares (Levenberg-Marquardt / TRF).

The optimisation minimises the reprojection error of the known 3D arena
corners onto the annotated pixel clicks.  With 8 non-coplanar corners you
have:
    observations  : 8 × 2 cams × 2 coords = 32 pixel equations
    parameters    : fx0,fy0,cx0,cy0, k1-k3 × 2 cams = 14 intrinsics
                  + rvec_stereo(3) + tvec_stereo(3) = 6 stereo extrinsics
                  + rvec_cam0_arena(3) + tvec_cam0_arena(3) = 6 arena pose
                  = 26 parameters total (well-constrained with 32 equations)

This is the same optimisation as the 'Refine calibration' button in
rpimocap-align, but accessible without opening the GUI.

Usage
-----
    # Basic — saves autocalib_refined2.npz next to the original
    rpimocap-refine-cal \\
        --calib  autocalib-20260214-021722_refined.npz \\
        --align  align-20260214-021722.csv

    # Explicit output path
    rpimocap-refine-cal \\
        --calib  autocalib_refined.npz \\
        --align  align.csv \\
        --out    autocalib_refined2.npz

    # Also include edge traces for distortion (optional, improves k1-k3)
    rpimocap-refine-cal \\
        --calib  autocalib_refined.npz \\
        --align  align.csv \\
        --edges  align_edges.csv

After running, use the new .npz with rpimocap-segment:
    rpimocap-segment --calib autocalib_refined2.npz ...
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calib",  required=True,
                    help="Input calibration .npz (from rpimocap-autocalib or "
                         "rpimocap-calibrate)")
    ap.add_argument("--align",  required=True,
                    help="align_points.csv from rpimocap-align (must contain "
                         "px0/px1 pixel clicks and arena_xyz coordinates)")
    ap.add_argument("--edges",  default=None,
                    help="Optional edges CSV from rpimocap-align "
                         "(improves k1/k2/k3 distortion estimate)")
    ap.add_argument("--out",    default=None,
                    help="Output .npz path (default: <calib_stem>_refined2.npz)")
    ap.add_argument("--edge-weight", type=float, default=0.3,
                    help="Weight for edge constraints vs corner constraints "
                         "(default: 0.3)")
    ap.add_argument("--fix-principal", action="store_true",
                    help="Keep principal point (cx, cy) fixed during optimisation")
    args = ap.parse_args()

    calib_path = Path(args.calib)
    if not calib_path.exists():
        print(f"ERROR: calibration file not found: {calib_path}")
        sys.exit(1)

    align_path = Path(args.align)
    if not align_path.exists():
        print(f"ERROR: alignment CSV not found: {align_path}")
        sys.exit(1)

    out_path = Path(args.out) if args.out else \
        calib_path.with_name(calib_path.stem + "_refined2.npz")

    from rpimocap.reconstruction.align import (
        load_align_csv, load_edges_csv, refine_calibration_from_arena)

    # Load corner annotations
    corners = load_align_csv(align_path)
    usable  = [p for p in corners
               if p.px0 is not None and p.px1 is not None]

    if len(usable) < 4:
        print(f"ERROR: need >= 4 corners with pixel clicks (have {len(usable)}).")
        print("Re-annotate corners in rpimocap-align to store pixel coordinates.")
        sys.exit(1)

    print(f"rpimocap-refine-cal")
    print(f"  input calib : {calib_path}")
    print(f"  align CSV   : {align_path}  ({len(usable)} corners with pixel clicks)")

    # Load initial calibration for comparison
    cal_before = np.load(calib_path)
    K0b = cal_before["K0"]; K1b = cal_before["K1"]
    print(f"\n  Before optimisation:")
    print(f"    fx0={K0b[0,0]:.1f}  fy0={K0b[1,1]:.1f}  "
          f"cx0={K0b[0,2]:.1f}  cy0={K0b[1,2]:.1f}")
    print(f"    fx1={K1b[0,0]:.1f}  fy1={K1b[1,1]:.1f}  "
          f"cx1={K1b[0,2]:.1f}  cy1={K1b[1,2]:.1f}")
    T_before = cal_before["T"].ravel()
    print(f"    |T| = {np.linalg.norm(T_before):.2f} mm  T={T_before.round(1)}")

    # Load optional edges
    edges = []
    if args.edges:
        edges_path = Path(args.edges)
        if edges_path.exists():
            edges = load_edges_csv(edges_path)
            print(f"  edges CSV   : {edges_path}  ({len(edges)} edges)")
        else:
            print(f"  WARNING: edges CSV not found: {edges_path}, skipping")

    print(f"\n  Optimising (TRF least-squares, 26 parameters) ...")

    try:
        result = refine_calibration_from_arena(
            corners, edges, str(calib_path), str(out_path),
            edge_weight=args.edge_weight,
            fix_principal=args.fix_principal,
            verbose=True)
    except Exception as e:
        print(f"\nERROR: optimisation failed — {e}")
        sys.exit(1)

    K0a = result["K0"]; K1a = result["K1"]
    Ta  = result["T"].ravel() if hasattr(result["T"], "ravel") else result["T"]

    print(f"\n  After optimisation:")
    print(f"    fx0={K0a[0,0]:.1f}  fy0={K0a[1,1]:.1f}  "
          f"cx0={K0a[0,2]:.1f}  cy0={K0a[1,2]:.1f}")
    print(f"    fx1={K1a[0,0]:.1f}  fy1={K1a[1,1]:.1f}  "
          f"cx1={K1a[0,2]:.1f}  cy1={K1a[1,2]:.1f}")
    try:
        T_after = np.load(out_path)["T"].ravel()
        print(f"    |T| = {np.linalg.norm(T_after):.2f} mm  T={T_after.round(1)}")
    except Exception:
        pass

    fx_diff = abs(K0a[0,0] - K1a[0,0]) / max(K0a[0,0], K1a[0,0]) * 100
    print(f"\n  fx discrepancy: before={abs(K0b[0,0]-K1b[0,0])/max(K0b[0,0],K1b[0,0])*100:.1f}%  "
          f"after={fx_diff:.1f}%")
    print(f"  Corner RMSE: {result['cost_before']:.3f} px → {result['cost_after']:.3f} px")
    conv = "converged" if result["converged"] else "WARNING: not converged"
    print(f"  Status: {conv}")
    print(f"\n  Saved: {out_path}")
    print(f"  (dlt_P0/dlt_P1 recomputed from the refined geometry, so "
          f"stereo_gate.py / stereo_diagnose.py use the refinement)")
    print(f"\n  Use with rpimocap-segment:")
    print(f"    rpimocap-segment --calib {out_path} ...")


if __name__ == "__main__":
    main()
