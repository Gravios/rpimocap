"""
rpimocap-calibrate-from-corners — estimate calibration from arena corner annotations
=====================================================================================
Uses the 8 annotated arena corner pixel clicks to estimate projection
matrices P0 and P1 directly via DLT (Direct Linear Transform), bypassing
the autocalibration focal length errors entirely.

This is useful when rpimocap-autocalib gives inconsistent focal lengths
(e.g. fx0=2831 vs fx1=3756, 24% discrepancy) that prevent the bundle
adjustment from converging.

With 8 known 3D corners and their pixel observations the DLT gives:
  - Reprojection RMSE ~5-6 px  (vs ~53 px from autocalib)
  - Kabsch alignment RMSE ~4 mm (vs ~118 mm from autocalib)
  - Correct arena dimensions within 1-2%

The output .npz stores P0 and P1 directly (no R/T decomposition needed).
All rpimocap-segment and rpimocap-preview commands accept this format.

Usage
-----
    rpimocap-calibrate-from-corners \\
        --align  align-20260214-021722.csv \\
        --calib  autocalib-20260214-021722_refined.npz \\
        --out    calib_from_corners.npz
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np


def dlt_camera_matrix(pts3d: np.ndarray, pts2d: np.ndarray) -> np.ndarray:
    """Estimate 3x4 camera matrix P from >=6 point correspondences via DLT."""
    A = []
    for (X,Y,Z),(u,v) in zip(pts3d, pts2d):
        A.append([-X,-Y,-Z,-1, 0,0,0,0, u*X,u*Y,u*Z,u])
        A.append([ 0,0,0,0, -X,-Y,-Z,-1, v*X,v*Y,v*Z,v])
    _, _, Vt = np.linalg.svd(np.array(A, dtype=np.float64))
    P = Vt[-1].reshape(3, 4)
    # Ensure points project to positive depth
    Xh = np.hstack([pts3d, np.ones((len(pts3d),1))]).T
    if (P[2] @ Xh).mean() < 0:
        P = -P
    return P


def reproj_rmse(P: np.ndarray, pts3d: np.ndarray, pts2d: np.ndarray) -> float:
    Xh = np.hstack([pts3d, np.ones((len(pts3d),1))]).T
    proj = P @ Xh; proj = proj[:2] / proj[2]
    return float(np.linalg.norm(proj.T - pts2d, axis=1).mean())


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--align",  required=True,
                    help="align_points.csv with 8 corner px0/px1 clicks")
    ap.add_argument("--calib",  required=True,
                    help="Existing calibration .npz (used for dist0/dist1 if present)")
    ap.add_argument("--out",    default=None,
                    help="Output .npz (default: <calib_stem>_corners.npz)")
    args = ap.parse_args()

    from rpimocap.reconstruction.align import load_align_csv

    align_path = Path(args.align)
    calib_path = Path(args.calib)
    out_path   = Path(args.out) if args.out else \
        calib_path.with_name(calib_path.stem + "_corners.npz")

    corners = load_align_csv(align_path)
    usable  = [p for p in corners
               if p.px0 is not None and p.px1 is not None]

    if len(usable) < 6:
        print(f"ERROR: need >= 6 corners with pixel clicks (have {len(usable)})")
        sys.exit(1)

    arena_pts = np.array([p.arena_xyz for p in usable], dtype=np.float64)
    px0       = np.array([p.px0       for p in usable], dtype=np.float64)
    px1       = np.array([p.px1       for p in usable], dtype=np.float64)

    print(f"rpimocap-calibrate-from-corners")
    print(f"  corners : {len(usable)}  ({', '.join(p.label for p in usable)})")
    print(f"  calib   : {calib_path}")

    # Load existing calibration for dist, image size etc.
    cal = np.load(calib_path)
    dist0 = cal.get("dist0", np.zeros((1,5)))
    dist1 = cal.get("dist1", np.zeros((1,5)))

    print(f"\n  Estimating P0 and P1 via DLT ...")
    P0 = dlt_camera_matrix(arena_pts, px0)
    P1 = dlt_camera_matrix(arena_pts, px1)

    rmse0 = reproj_rmse(P0, arena_pts, px0)
    rmse1 = reproj_rmse(P1, arena_pts, px1)
    print(f"  cam0 reprojection: {rmse0:.2f} px")
    print(f"  cam1 reprojection: {rmse1:.2f} px")

    # Verify triangulation
    from rpimocap.reconstruction.triangulate import triangulate_dlt
    tri_pts = np.array([
        triangulate_dlt(P0, P1, tuple(p0), tuple(p1))[:3]
        for p0, p1 in zip(px0, px1)
    ])
    span = tri_pts.max(0) - tri_pts.min(0)
    expected = arena_pts.max(0) - arena_pts.min(0)
    errs = np.linalg.norm(tri_pts - arena_pts, axis=1)

    print(f"\n  Triangulation check:")
    print(f"    Mean 3D error : {errs.mean():.1f} mm  (max {errs.max():.1f} mm)")
    print(f"    X span: {span[0]:.0f} mm  (expected {expected[0]:.0f} mm)")
    print(f"    Y span: {span[1]:.0f} mm  (expected {expected[1]:.0f} mm)")
    print(f"    Z span: {span[2]:.0f} mm  (expected {expected[2]:.0f} mm)")

    # Kabsch alignment RMSE
    from rpimocap.reconstruction.align import AlignPoint, kabsch_align_from_pixels
    pts_aln = [AlignPoint(rec_xyz=arena_pts[i], arena_xyz=arena_pts[i],
                           label=usable[i].label, px0=px0[i], px1=px1[i])
               for i in range(len(usable))]
    try:
        r = kabsch_align_from_pixels(pts_aln, P0, P1)
        print(f"    Kabsch RMSE   : {r.rmse_mm:.2f} mm")
    except Exception:
        pass

    # Save — store P0/P1 directly plus dist for downstream distortion correction
    # Also store dummy K0/K1/R/T for backward compatibility
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Extract approximate K from P (for display only — not used for projection)
    from rpimocap.cli.refine_cal import main as _  # ensure importable
    def approx_K(P):
        M = P[:,:3]
        K_approx = M / M[2,2]
        return np.array([[K_approx[0,0],0,K_approx[0,2]],
                         [0,K_approx[1,1],K_approx[1,2]],
                         [0,0,1]])

    # Save everything needed
    np.savez(out_path,
             P0=P0, P1=P1,
             K0=cal.get("K0", np.eye(3)),   # keep original K for reference
             K1=cal.get("K1", np.eye(3)),
             R=cal.get("R", np.eye(3)),
             T=cal.get("T", np.zeros((3,1))),
             dist0=dist0, dist1=dist1,
             dlt_P0=P0, dlt_P1=P1,          # explicit DLT matrices
             dlt_rmse0=rmse0, dlt_rmse1=rmse1)

    print(f"\n  Saved: {out_path}")
    print(f"\n  Use with rpimocap-segment:")
    print(f"    rpimocap-segment --calib {out_path} ...")
    print(f"\n  NOTE: this calibration uses DLT projection matrices directly.")
    print(f"  The arena coordinate frame IS the world frame — no --align-points needed.")
    print(f"  Coordinates output by rpimocap-segment will already be in arena mm.")


if __name__ == "__main__":
    main()
