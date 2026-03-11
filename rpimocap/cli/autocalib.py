"""
autocalib.py — Self-calibration from subject motion
=====================================================
Estimates camera intrinsics (focal length) from two synchronised video streams
without a calibration target. Uses:

  1. Cross-view SIFT feature correspondences
  2. Essential matrix constraint (Kruppa equations, shared K)
  3. Arena metric scale refinement
  4. (Optional) Nonlinear bundle adjustment

Writes a calibration .npz compatible with the rest of the reconstruct3d
pipeline, plus a standalone HTML validation report.

Usage
-----
    # Basic: centroid detector, arena bounds, same cameras
    python autocalib.py \\
        --cam0   data/session_cam0.mp4 \\
        --cam1   data/session_cam1.mp4 \\
        --bounds "-300,300,-200,200,0,400" \\
        --out    autocalib.npz \\
        --report autocalib_report.html

    # With nonlinear bundle refinement
    python autocalib.py \\
        --cam0  data/session_cam0.mp4 \\
        --cam1  data/session_cam1.mp4 \\
        --bounds "-300,300,-200,200,0,400" \\
        --refine-bundle \\
        --out   autocalib.npz

    # Cross-validate against existing checkerboard calibration
    python autocalib.py \\
        --cam0   data/session_cam0.mp4 \\
        --cam1   data/session_cam1.mp4 \\
        --bounds "-300,300,-200,200,0,400" \\
        --compare calibration.npz \\
        --out    autocalib.npz

Output .npz format
------------------
Same schema as calibrate.py so the file can be used as a drop-in:

  K0, K1     : (3,3) intrinsic matrices  [both identical, shared K]
  dist0, dist1: (1,5) zero distortion    [autocalib cannot estimate distortion]
  R, T        : stereo extrinsics        [estimated from best E decomposition]
  E, F        : essential and fundamental matrices from best frame pair
  P0, P1      : raw projection matrices  [not rectified — run calibrate.py for that]
  image_size  : (width, height)
  focal_px    : scalar focal length estimate
  method      : 'autocalib_kruppa'

Limitations
-----------
- Only estimates focal length (1 DOF). Assumes principal point = image centre,
  square pixels, zero skew, zero distortion.
- Radial distortion in wide-angle lenses can introduce systematic error.
  Compare with checkerboard calibration if using fisheye or action cameras.
- Accuracy is best when the subject moves through large depth variations
  (front/back of arena) AND when background texture is present for SIFT.
- With only focal length solved, this is a starting point — use as initial
  guess for calibrate.py if a checkerboard can be obtained.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np


def parse_bounds(s: str):
    """Parse ``"xmin,xmax,ymin,ymax,zmin,zmax"`` into a nested tuple of float pairs."""
    vals = [float(x) for x in s.split(",")]
    if len(vals) != 6:
        raise ValueError("--bounds expects 6 comma-separated values")
    return ((vals[0], vals[1]), (vals[2], vals[3]), (vals[4], vals[5]))


def build_stereo_extrinsics(result, estimates, image_size):
    """
    Recover approximate stereo (R, T) from the best essential matrix.
    Returns (R, T, E, F) where T is in the same (unknown absolute) units
    as the reconstruction.
    """
    from rpimocap.calibration.autocalib.kruppa import decompose_essential, cv_triangulate

    K = result.K
    est = max(estimates, key=lambda e: e.quality)
    E = K.T @ est.F @ K
    en = np.linalg.norm(E)
    if en < 1e-10:
        R = np.eye(3)
        T = np.array([1.0, 0.0, 0.0])
        return R, T, est.F * 0, est.F

    E_norm = E / en
    res = decompose_essential(E_norm, est.pts0, est.pts1, K)
    if res is None:
        R = np.eye(3)
        T = np.array([1.0, 0.0, 0.0])
    else:
        R, T = res

    return R, T, E_norm, est.F


def save_npz(result, estimates, image_size, out_path, arena_bounds):
    """Write calibration .npz in the same schema as calibrate.py."""
    from rpimocap.calibration.autocalib.kruppa import make_K

    K = result.K
    dist_zero = np.zeros((1, 5), dtype=np.float64)

    R, T, E, F = build_stereo_extrinsics(result, estimates, image_size)

    # Raw (non-rectified) projection matrices
    P0 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P1 = K @ np.hstack([R, T.reshape(3, 1)])

    w, h = image_size
    np.savez(
        out_path,
        K0=K, dist0=dist_zero,
        K1=K, dist1=dist_zero,   # shared K
        R=R, T=T,
        E=E, F=F,
        P0=P0, P1=P1,
        image_size=np.array([w, h]),
        focal_px=np.array(result.f_px),
        cx=np.array(result.cx_px),
        cy=np.array(result.cy_px),
        f_kruppa=np.array(result.f_kruppa),
        f_metric=np.array(result.f_metric if result.f_metric else result.f_kruppa),
        scale_factor=np.array(result.scale_factor if result.scale_factor else 1.0),
        mean_essential_residual=np.array(result.mean_essential_residual),
        n_estimates=np.array(result.n_estimates_used),
        method=np.array("autocalib_kruppa"),
        # Null rectification maps (identity, image size w×h)
        map0x=np.tile(np.arange(w, dtype=np.float32), (h, 1)),
        map0y=np.tile(np.arange(h, dtype=np.float32).reshape(-1, 1), (1, w)),
        map1x=np.tile(np.arange(w, dtype=np.float32), (h, 1)),
        map1y=np.tile(np.arange(h, dtype=np.float32).reshape(-1, 1), (1, w)),
        R0=np.eye(3), R1=np.eye(3),
        Q=np.eye(4),
    )
    print(f"  Saved .npz → {out_path}")


def compare_with_checkerboard(result, npz_path: str):
    """Load a checkerboard calibration and compare focal lengths."""
    d = np.load(npz_path)
    f_check = float(d["K0"][0, 0])
    f_auto  = result.f_px
    diff_pct = 100 * abs(f_check - f_auto) / f_check
    rms = float(d.get("rms_stereo", d.get("rms0", np.array([0.0]))))
    print(f"\n── Cross-validation vs checkerboard calibration ─────")
    print(f"  f (autocalib)    : {f_auto:.2f} px")
    print(f"  f (checkerboard) : {f_check:.2f} px")
    print(f"  Difference       : {diff_pct:.2f}%")
    print(f"  Checkerboard RMS : {rms:.4f} px")
    if diff_pct < 2.0:
        print("  ✓ EXCELLENT agreement (<2%)")
    elif diff_pct < 5.0:
        print("  ✓ GOOD agreement (<5%)")
    elif diff_pct < 10.0:
        print("  ⚠ MODERATE agreement (<10%) — consider using checkerboard value")
    else:
        print("  ✗ POOR agreement (>10%) — distortion likely significant; "
              "use checkerboard calibration")
    return rms


def main():
    """Entry point for the ``rpimocap-autocalib`` command-line tool."""
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    io = ap.add_argument_group("Input / Output")
    io.add_argument("--cam0",    required=True, help="Camera 0 video file")
    io.add_argument("--cam1",    required=True, help="Camera 1 video file")
    io.add_argument("--bounds",  default=None,
                    help="Arena bounds: xmin,xmax,ymin,ymax,zmin,zmax (mm)")
    io.add_argument("--out",     default="autocalib.npz")
    io.add_argument("--report",  default="autocalib_report.html")
    io.add_argument("--compare", default=None,
                    help="Path to calibrate.py .npz for cross-validation")

    smp = ap.add_argument_group("Sampling")
    smp.add_argument("--sample-interval", type=int, default=60,
                     help="Sample every N frames (default: 60)")
    smp.add_argument("--max-estimates", type=int, default=120,
                     help="Max F estimates to collect (default: 120)")
    smp.add_argument("--warmup", type=int, default=60,
                     help="Skip first N frames (default: 60)")
    smp.add_argument("--start-frame", type=int, default=0)

    feat = ap.add_argument_group("Feature matching")
    feat.add_argument("--n-features", type=int, default=4000,
                      help="Max SIFT keypoints per frame (default: 4000)")
    feat.add_argument("--ratio-thresh", type=float, default=0.72,
                      help="Lowe ratio test threshold (default: 0.72)")
    feat.add_argument("--ransac-thresh", type=float, default=1.5,
                      help="RANSAC inlier distance in px (default: 1.5)")
    feat.add_argument("--min-inliers", type=int, default=80,
                      help="Minimum inliers per F estimate (default: 80)")
    feat.add_argument("--min-quality", type=float, default=0.35,
                      help="Minimum quality score to include in Kruppa solve (default: 0.35)")

    opt = ap.add_argument_group("Optimisation")
    opt.add_argument("--refine-bundle", action="store_true",
                     help="Run nonlinear bundle refinement as final step")
    opt.add_argument("--f-min-factor", type=float, default=0.4,
                     help="Focal length search lower bound as ×diagonal (default: 0.4)")
    opt.add_argument("--f-max-factor", type=float, default=3.5,
                     help="Focal length search upper bound as ×diagonal (default: 3.5)")
    opt.add_argument("--quiet", action="store_true")

    args = ap.parse_args()
    verbose = not args.quiet

    from rpimocap.calibration.autocalib.features import CrossViewMatcher, sample_frame_pairs, filter_estimates
    from rpimocap.calibration.autocalib.kruppa import run_focal_estimation
    from rpimocap.calibration.autocalib.report import generate_report

    arena_bounds = parse_bounds(args.bounds) if args.bounds else None

    # ── Open videos ──────────────────────────────────────────────────────────
    cap0 = cv2.VideoCapture(args.cam0)
    cap1 = cv2.VideoCapture(args.cam1)
    if not cap0.isOpened() or not cap1.isOpened():
        print("ERROR: could not open one or both video files")
        sys.exit(1)

    w  = int(cap0.get(cv2.CAP_PROP_FRAME_WIDTH))
    h  = int(cap0.get(cv2.CAP_PROP_FRAME_HEIGHT))
    image_size = (w, h)
    total = int(cap0.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap0.get(cv2.CAP_PROP_FPS) or 30.0

    print(f"\nreconstruct3d — autocalib")
    print(f"  cam0        : {args.cam0}")
    print(f"  cam1        : {args.cam1}")
    print(f"  resolution  : {w}×{h}  ({total} frames @ {fps:.1f} fps)")
    if arena_bounds:
        (x0,x1),(y0,y1),(z0,z1) = arena_bounds
        print(f"  arena       : X[{x0},{x1}] Y[{y0},{y1}] Z[{z0},{z1}] mm")
    else:
        print("  arena       : not specified (metric refinement disabled)")

    # ── Feature correspondence ───────────────────────────────────────────────
    print(f"\n── Cross-view feature matching ──────────────────────")
    t0 = time.time()
    matcher = CrossViewMatcher(
        n_features=args.n_features,
        ratio_thresh=args.ratio_thresh,
        ransac_thresh=args.ransac_thresh,
        min_inliers=args.min_inliers,
    )
    all_estimates = sample_frame_pairs(
        cap0, cap1, matcher,
        sample_interval_frames=args.sample_interval,
        max_estimates=args.max_estimates,
        start_frame=args.start_frame,
        warmup_frames=args.warmup,
        verbose=verbose,
    )
    cap0.release()
    cap1.release()

    print(f"  Feature matching: {time.time()-t0:.1f}s")

    if len(all_estimates) < 5:
        print("ERROR: fewer than 5 valid F estimates. Possible causes:")
        print("  - Insufficient texture in scene (arena walls, bedding, subject fur)")
        print("  - Cameras not synchronised (subject position mismatch between views)")
        print("  - Very low frame rate or high motion blur")
        print("  - Try --n-features 6000 --ratio-thresh 0.78 --min-inliers 50")
        sys.exit(1)

    # Filter for Kruppa solve
    good_estimates = filter_estimates(
        all_estimates,
        min_quality=args.min_quality,
        top_n=80,
    )
    print(f"\n  Estimates for Kruppa solve: {len(good_estimates)} "
          f"(quality ≥ {args.min_quality})")

    if len(good_estimates) < 10:
        print("WARNING: fewer than 10 quality estimates — result may be unreliable")
        print("  Lowering quality threshold to 0.2 and retrying...")
        good_estimates = filter_estimates(all_estimates, min_quality=0.2)

    # ── Focal estimation ─────────────────────────────────────────────────────
    result = run_focal_estimation(
        good_estimates,
        image_size=image_size,
        arena_bounds=arena_bounds,
        do_bundle=args.refine_bundle,
        verbose=verbose,
    )

    # ── Cross-validation ─────────────────────────────────────────────────────
    calib_rms = None
    if args.compare:
        calib_rms = compare_with_checkerboard(result, args.compare)

    # ── Summary ──────────────────────────────────────────────────────────────
    import math
    diag = math.hypot(w, h)
    fov  = math.degrees(2 * math.atan(diag / 2 / result.f_px))
    print(f"\n── Result ───────────────────────────────────────────")
    print(f"  Focal length  : {result.f_px:.2f} px")
    print(f"  f / diagonal  : {result.f_px/diag:.3f}")
    print(f"  Diagonal FOV  : ≈ {fov:.1f}°")
    print(f"  Principal pt  : ({result.cx_px:.1f}, {result.cy_px:.1f})")
    print(f"  E residual    : {result.mean_essential_residual:.5f}")
    print(f"\n  K =")
    print(f"    [{result.K[0,0]:.2f}   0.00  {result.K[0,2]:.2f}]")
    print(f"    [  0.00  {result.K[1,1]:.2f}  {result.K[1,2]:.2f}]")
    print(f"    [  0.00    0.00    1.00]")

    # ── Save outputs ──────────────────────────────────────────────────────────
    print(f"\n── Saving outputs ───────────────────────────────────")
    save_npz(result, good_estimates, image_size, args.out, arena_bounds)

    generate_report(
        result=result,
        estimates=all_estimates,
        image_size=image_size,
        arena_bounds=arena_bounds,
        output_path=args.report,
        video_paths=(args.cam0, args.cam1),
        calib_rms=calib_rms,
    )

    print(f"\n✓ autocalib.py complete")
    print(f"  Intrinsics   → {args.out}")
    print(f"  Report       → {args.report}")
    print(f"\nNOTE: autocalib estimates focal length only.")
    print(f"  Distortion coefficients are set to zero.")
    print(f"  For lenses with significant distortion, use autocalib.npz as")
    print(f"  an initial guess and refine with calibrate.py --calib-init autocalib.npz")


if __name__ == "__main__":
    main()
