"""
calibrate.py — Stereo camera calibration pipeline
===================================================
Processes checkerboard footage from two synchronized cameras to produce:
  - Per-camera intrinsics (K, dist)
  - Stereo extrinsics (R, T, E, F)
  - Rectification maps and projection matrices (P0, P1)

Usage
-----
    python calibrate.py \\
        --cam0 data/calib_cam0.mp4 \\
        --cam1 data/calib_cam1.mp4 \\
        --pattern 9x6 \\
        --square 25.0 \\
        --out calibration.npz

    # Or from image pairs in two directories:
    python calibrate.py \\
        --cam0 data/calib_cam0/ \\
        --cam1 data/calib_cam1/ \\
        --pattern 9x6 --square 25.0 --out calibration.npz

Notes
-----
- --pattern is inner corners (not squares): a 10×7 board has 9×6 inner corners.
- --square is physical square size in mm. Affects the scale of the 3D reconstruction.
- Synchronized capture is critical: both cameras must show the same board pose
  in paired frames. Use hardware trigger or frame-matched export.
- Aim for 30–60 paired frames with good angular diversity (tilts, rotations).
- Rational distortion model (cv2.CALIB_RATIONAL_MODEL) is used by default,
  which handles wide-angle lenses well. Disable with --no-rational if needed.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


# --------------------------------------------------------------------------- #
#  Corner detection helpers                                                    #
# --------------------------------------------------------------------------- #

CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 1e-4)


def _sources_from_path(src: str):
    """Yield (index, frame) from a video file or image directory."""
    p = Path(src)
    if p.is_dir():
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}
        imgs = sorted(f for f in p.iterdir() if f.suffix.lower() in exts)
        for idx, img_path in enumerate(imgs):
            frame = cv2.imread(str(img_path))
            if frame is not None:
                yield idx, frame
    else:
        cap = cv2.VideoCapture(str(p))
        if not cap.isOpened():
            raise IOError(f"Cannot open: {src}")
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            yield idx, frame
            idx += 1
        cap.release()


def detect_corners_paired(src0: str, src1: str,
                           pattern: tuple[int, int],
                           skip: int = 8,
                           max_pairs: int = 80,
                           verbose: bool = True
                           ) -> tuple[list, list, tuple[int, int]]:
    """
    Iterate both sources frame-by-frame, detect checkerboard corners in both,
    and collect pairs where detection succeeds in BOTH cameras.

    Returns
    -------
    corners0, corners1 : list of corner arrays (subpixel refined)
    image_size         : (width, height)
    """
    cols, rows = pattern
    gen0 = _sources_from_path(src0)
    gen1 = _sources_from_path(src1)

    corners0, corners1 = [], []
    image_size: Optional[tuple] = None
    frame_idx = -1

    for (i0, f0), (i1, f1) in zip(gen0, gen1):
        frame_idx += 1
        if frame_idx % skip != 0:
            continue
        if len(corners0) >= max_pairs:
            break

        g0 = cv2.cvtColor(f0, cv2.COLOR_BGR2GRAY)
        g1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
        if image_size is None:
            image_size = (g0.shape[1], g0.shape[0])

        ok0, c0 = cv2.findChessboardCorners(g0, (cols, rows), None)
        ok1, c1 = cv2.findChessboardCorners(g1, (cols, rows), None)

        if ok0 and ok1:
            c0 = cv2.cornerSubPix(g0, c0, (11, 11), (-1, -1), CRITERIA)
            c1 = cv2.cornerSubPix(g1, c1, (11, 11), (-1, -1), CRITERIA)
            corners0.append(c0)
            corners1.append(c1)
            if verbose:
                n = len(corners0)
                print(f"  pair {n:3d}  (frame {frame_idx})")

    if image_size is None:
        raise RuntimeError("No frames could be read from one or both sources.")
    print(f"  Total paired detections: {len(corners0)}")
    return corners0, image_size, corners1


# --------------------------------------------------------------------------- #
#  Calibration                                                                 #
# --------------------------------------------------------------------------- #

def _object_points(pattern: tuple[int, int], square_mm: float, n: int):
    cols, rows = pattern
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_mm
    return [objp] * n


def calibrate_intrinsics(obj_pts, img_pts, image_size,
                          rational: bool = True) -> tuple:
    """Per-camera intrinsic calibration. Returns (K, dist, rms)."""
    flags = cv2.CALIB_RATIONAL_MODEL if rational else 0
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_pts, img_pts, image_size, None, None, flags=flags
    )
    return K, dist, rms


def calibrate_stereo(obj_pts, img_pts0, img_pts1,
                     K0, dist0, K1, dist1,
                     image_size, rational: bool = True) -> tuple:
    """
    Stereo calibration with fixed intrinsics.
    Returns (R, T, E, F, rms).
    """
    flags = cv2.CALIB_FIX_INTRINSIC
    if rational:
        flags |= cv2.CALIB_RATIONAL_MODEL
    rms, K0_, d0_, K1_, d1_, R, T, E, F = cv2.stereoCalibrate(
        obj_pts, img_pts0, img_pts1,
        K0, dist0, K1, dist1,
        image_size,
        criteria=CRITERIA,
        flags=flags
    )
    return R, T, E, F, rms


def stereo_rectify(K0, dist0, K1, dist1, R, T, image_size):
    """
    Compute rectification transforms and projection matrices.
    Returns (P0, P1, R0, R1, Q, maps0, maps1).
    """
    R0, R1, P0, P1, Q, roi0, roi1 = cv2.stereoRectify(
        K0, dist0, K1, dist1, image_size, R, T,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
    )
    maps0 = cv2.initUndistortRectifyMap(K0, dist0, R0, P0, image_size, cv2.CV_32FC1)
    maps1 = cv2.initUndistortRectifyMap(K1, dist1, R1, P1, image_size, cv2.CV_32FC1)
    return P0, P1, R0, R1, Q, maps0, maps1


# --------------------------------------------------------------------------- #
#  Validation                                                                  #
# --------------------------------------------------------------------------- #

def validate_epipolar(corners0, corners1, F, max_dist: float = 2.0):
    """
    Report the mean symmetric epipolar distance across all point pairs.
    Values < 1 px indicate a good fundamental matrix.
    """
    pts0 = np.concatenate([c.reshape(-1, 2) for c in corners0])
    pts1 = np.concatenate([c.reshape(-1, 2) for c in corners1])
    ones = np.ones((len(pts0), 1))
    pts0h = np.hstack([pts0, ones])
    pts1h = np.hstack([pts1, ones])
    lines1 = (F @ pts0h.T).T
    lines0 = (F.T @ pts1h.T).T

    def dist(pts, lines):
        """Compute signed line-to-point distance for each row."""
        return np.abs((pts * lines[:, :2]).sum(1) + lines[:, 2]) / \
               np.sqrt(lines[:, 0]**2 + lines[:, 1]**2)

    d = (dist(pts0h, lines0) + dist(pts1h, lines1)) / 2
    print(f"  Epipolar distance: mean={d.mean():.3f} px, max={d.max():.3f} px")
    outliers = (d > max_dist).sum()
    if outliers:
        print(f"  WARNING: {outliers} point pairs exceed {max_dist}px epipolar distance")
    return float(d.mean())


# --------------------------------------------------------------------------- #
#  CLI                                                                         #
# --------------------------------------------------------------------------- #

def main():
    """Entry point for the ``rpimocap-calibrate`` command-line tool."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cam0", required=True,
                    help="Camera 0 source: video file or image directory")
    ap.add_argument("--cam1", required=True,
                    help="Camera 1 source: video file or image directory")
    ap.add_argument("--pattern", default="9x6",
                    help="Inner corner count as WxH, e.g. 9x6")
    ap.add_argument("--square", type=float, default=25.0,
                    help="Physical square size in mm (default: 25)")
    ap.add_argument("--skip", type=int, default=8,
                    help="Sample every N frames from the source (default: 8)")
    ap.add_argument("--max-pairs", type=int, default=80,
                    help="Maximum number of paired detections (default: 80)")
    ap.add_argument("--no-rational", action="store_true",
                    help="Use standard distortion model instead of rational")
    ap.add_argument("--out", default="calibration.npz",
                    help="Output .npz path (default: calibration.npz)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    pattern = tuple(int(x) for x in args.pattern.lower().split("x"))
    rational = not args.no_rational
    verbose = not args.quiet

    print(f"Checkerboard: {pattern[0]}×{pattern[1]} inner corners, "
          f"{args.square:.1f} mm squares")
    print(f"Distortion model: {'rational' if rational else 'standard'}\n")

    print("── Paired corner detection ──────────────────────────")
    corners0, image_size, corners1 = detect_corners_paired(
        args.cam0, args.cam1, pattern,
        skip=args.skip, max_pairs=args.max_pairs, verbose=verbose
    )
    if len(corners0) < 10:
        print("ERROR: fewer than 10 paired frames detected. "
              "Check pattern size, lighting, and synchronisation.")
        sys.exit(1)

    obj_pts = _object_points(pattern, args.square, len(corners0))

    print("\n── Camera 0 intrinsics ──────────────────────────────")
    K0, dist0, rms0 = calibrate_intrinsics(obj_pts, corners0, image_size, rational)
    print(f"  RMS reprojection error: {rms0:.4f} px")

    print("\n── Camera 1 intrinsics ──────────────────────────────")
    K1, dist1, rms1 = calibrate_intrinsics(obj_pts, corners1, image_size, rational)
    print(f"  RMS reprojection error: {rms1:.4f} px")

    print("\n── Stereo calibration ───────────────────────────────")
    R, T, E, F, rms_stereo = calibrate_stereo(
        obj_pts, corners0, corners1, K0, dist0, K1, dist1, image_size, rational
    )
    print(f"  RMS stereo reprojection error: {rms_stereo:.4f} px")
    print(f"  Baseline |T|: {np.linalg.norm(T):.2f} mm")

    print("\n── Epipolar validation ──────────────────────────────")
    epi_dist = validate_epipolar(corners0, corners1, F)

    print("\n── Stereo rectification ─────────────────────────────")
    P0, P1, R0, R1, Q, maps0, maps1 = stereo_rectify(
        K0, dist0, K1, dist1, R, T, image_size
    )
    print(f"  Focal length in rectified frame: {P0[0,0]:.2f} px")

    out = args.out
    np.savez(
        out,
        # Intrinsics
        K0=K0, dist0=dist0,
        K1=K1, dist1=dist1,
        # Extrinsics
        R=R, T=T, E=E, F=F,
        # Projection matrices (after rectification)
        P0=P0, P1=P1,
        # Rectification transforms
        R0=R0, R1=R1, Q=Q,
        # Undistort+rectify maps (float32, for cv2.remap)
        map0x=maps0[0], map0y=maps0[1],
        map1x=maps1[0], map1y=maps1[1],
        # Metadata
        image_size=np.array(image_size),
        pattern=np.array(pattern),
        square_mm=np.array(args.square),
        rms0=np.array(rms0),
        rms1=np.array(rms1),
        rms_stereo=np.array(rms_stereo),
        epi_dist=np.array(epi_dist),
    )

    print(f"\n✓ Calibration saved → {out}")
    print(f"\nSummary")
    print(f"  Cam0 intrinsic RMS : {rms0:.4f} px")
    print(f"  Cam1 intrinsic RMS : {rms1:.4f} px")
    print(f"  Stereo RMS         : {rms_stereo:.4f} px")
    print(f"  Epipolar dist (sym): {epi_dist:.4f} px")
    print(f"  Baseline           : {np.linalg.norm(T):.2f} mm")

    if rms_stereo > 1.5:
        print("\nWARNING: Stereo RMS > 1.5px. Consider:")
        print("  - More calibration frames with diverse angles")
        print("  - Ensuring both cameras see the board at the same time")
        print("  - Checking that the board is flat and undistorted")


if __name__ == "__main__":
    main()
