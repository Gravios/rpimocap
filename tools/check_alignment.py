"""
Quick diagnostic: print per-corner triangulation and Kabsch residuals.
Run from the session directory:
  python3 /tmp/check_alignment.py \
      --calib  strohA-al-RPICAM-20260214-021722/autocalib-20260214-021722_refined.npz \
      --align  strohA-al-RPICAM-20260214-021722/align-20260214-021722.csv
"""
import argparse, sys
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--calib", required=True)
ap.add_argument("--align", required=True)
args = ap.parse_args()

sys.path.insert(0, "/home/graboski/software/rpimocap")
from rpimocap.reconstruction.align import load_align_csv, kabsch_align
from rpimocap.reconstruction.triangulate import triangulate_dlt

cal = np.load(args.calib)
K0 = cal["K0"]; K1 = cal["K1"]
R  = cal["R"];  T  = cal["T"].ravel()
P0 = K0 @ np.hstack([np.eye(3), np.zeros((3,1))])
P1 = K1 @ np.hstack([R, T.reshape(3,1)])

print(f"Calibration")
print(f"  fx0={K0[0,0]:.1f}  fy0={K0[1,1]:.1f}  cx0={K0[0,2]:.1f}  cy0={K0[1,2]:.1f}")
print(f"  fx1={K1[0,0]:.1f}  fy1={K1[1,1]:.1f}  cx1={K1[0,2]:.1f}  cy1={K1[1,2]:.1f}")
print(f"  |T| = {np.linalg.norm(T):.2f} mm  (stereo baseline)")
print(f"  T   = {T.round(2)}")
print()

pts = load_align_csv(args.align)
print(f"Alignment CSV: {len(pts)} corners")
print()

# Re-triangulate each corner
retriang = []
print(f"{'Label':12s}  {'px0':>14s}  {'px1':>14s}  {'X_cam':>28s}  {'arena':>22s}  {'depth mm':>9s}")
print("-"*105)
for pt in pts:
    if pt.px0 is None:
        print(f"{pt.label:12s}  NO PIXEL CLICKS STORED")
        continue
    xyz = triangulate_dlt(P0, P1,
                          (float(pt.px0[0]), float(pt.px0[1])),
                          (float(pt.px1[0]), float(pt.px1[1])))[:3]
    depth = xyz[2]
    print(f"{pt.label:12s}  "
          f"px0=({pt.px0[0]:6.1f},{pt.px0[1]:6.1f})  "
          f"px1=({pt.px1[0]:6.1f},{pt.px1[1]:6.1f})  "
          f"cam=({xyz[0]:7.1f},{xyz[1]:7.1f},{xyz[2]:7.1f})  "
          f"arena=({pt.arena_xyz[0]:5.0f},{pt.arena_xyz[1]:5.0f},{pt.arena_xyz[2]:5.0f})  "
          f"depth={depth:8.1f}")
    retriang.append((pt, xyz))

print()
if retriang:
    depths = [xyz[2] for _, xyz in retriang]
    print(f"Depth range: {min(depths):.1f} – {max(depths):.1f} mm  (should be roughly 400–1200 mm for this setup)")
    
    # Check Kabsch residuals per point
    from rpimocap.reconstruction.align import AlignPoint, kabsch_align
    new_pts = [AlignPoint(rec_xyz=xyz, arena_xyz=pt.arena_xyz,
                          label=pt.label, px0=pt.px0, px1=pt.px1)
               for pt, xyz in retriang]
    result = kabsch_align(new_pts)
    print(f"Kabsch RMSE: {result.rmse_mm:.2f} mm  (n={result.n_points})")
    print()
    
    # Per-point residuals
    A = np.stack([p.rec_xyz   for p in new_pts])
    B = np.stack([p.arena_xyz for p in new_pts])
    A_aligned = (result.R @ A.T).T + result.t
    residuals  = np.linalg.norm(A_aligned - B, axis=1)
    print(f"{'Label':12s}  {'residual mm':>12s}  {'status'}")
    print("-"*40)
    for (pt, _), res in zip(retriang, residuals):
        flag = " ← OUTLIER" if res > 20 else ""
        print(f"  {pt.label:12s}  {res:10.1f}{flag}")
