"""
triangulate.py — DLT triangulation and 3D trajectory utilities
==============================================================
Core function: given matched 2D keypoints from two calibrated camera views
(characterised by 3×4 projection matrices P0, P1), recover 3D positions via
the Direct Linear Transform (SVD solution).

Also provides:
  - reprojection_error         : pixel-space round-trip error
  - triangulate_keypoints      : batch triangulation with filtering
  - smooth_trajectory          : temporal Gaussian smoothing per landmark
  - fill_trajectory_gaps       : linear interpolation of missing frames
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# --------------------------------------------------------------------------- #
#  Data class                                                                  #
# --------------------------------------------------------------------------- #

@dataclass
class Point3D:
    name: str
    xyz: np.ndarray        # (3,) world coordinates (same units as calibration)
    confidence: float = 1.0
    reprojection_error: float = 0.0

    def as_list(self) -> list:
        return self.xyz.tolist()


# --------------------------------------------------------------------------- #
#  Core triangulation                                                          #
# --------------------------------------------------------------------------- #

def triangulate_dlt(P0: np.ndarray, P1: np.ndarray,
                    pt0: tuple[float, float],
                    pt1: tuple[float, float]) -> np.ndarray:
    """
    Triangulate a single point pair using the Direct Linear Transform.

    Solves the homogeneous system  A X = 0  where A is built from the
    cross-product of each observation with its projection row:
        x × (P X) = 0

    Parameters
    ----------
    P0, P1 : (3, 4) projection matrices
    pt0    : (x, y) pixel coordinate in camera 0
    pt1    : (x, y) pixel coordinate in camera 1

    Returns
    -------
    (4,) homogeneous 3D point, Euclidean normalised so X[3] == 1
    """
    x0, y0 = pt0
    x1, y1 = pt1
    A = np.array([
        x0 * P0[2] - P0[0],
        y0 * P0[2] - P0[1],
        x1 * P1[2] - P1[0],
        y1 * P1[2] - P1[1],
    ], dtype=np.float64)
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    return X / X[3]


def reprojection_error(P: np.ndarray,
                       X: np.ndarray,
                       pt: tuple[float, float]) -> float:
    """
    Euclidean distance (pixels) between an observed point and the
    reprojection of its triangulated 3D estimate.
    """
    Xh = X if len(X) == 4 else np.append(X, 1.0)
    proj = P @ Xh
    proj = proj[:2] / proj[2]
    return float(np.linalg.norm(proj - np.array(pt)))


# --------------------------------------------------------------------------- #
#  Batch triangulation with confidence filtering                               #
# --------------------------------------------------------------------------- #

def triangulate_keypoints(
    P0: np.ndarray,
    P1: np.ndarray,
    result0,          # Pose2DResult or list[Keypoint2D]
    result1,
    min_confidence: float = 0.3,
    max_reprojection_px: float = 20.0,
) -> list[Point3D]:
    """
    Triangulate all mutually visible, high-confidence keypoints from two views.

    Skips landmarks whose minimum visibility across both cameras is below
    ``min_confidence``, and discards results with mean reprojection error
    above ``max_reprojection_px``.

    Parameters
    ----------
    P0, P1             : (3, 4) projection matrices for camera 0 and 1
    result0, result1   : Pose2DResult objects (or any iterable of Keypoint2D)
    min_confidence     : minimum per-landmark confidence to attempt triangulation
    max_reprojection_px: discard triangulations with higher mean reprojection error

    Returns
    -------
    list of Point3D
    """
    kps0 = result0.by_name() if hasattr(result0, "by_name") else {k.name: k for k in result0}
    kps1 = result1.by_name() if hasattr(result1, "by_name") else {k.name: k for k in result1}

    results = []
    for name, kp0 in kps0.items():
        kp1 = kps1.get(name)
        if kp1 is None:
            continue
        conf = min(kp0.confidence, kp1.confidence)
        if conf < min_confidence:
            continue

        X = triangulate_dlt(P0, P1, (kp0.x, kp0.y), (kp1.x, kp1.y))
        err0 = reprojection_error(P0, X, (kp0.x, kp0.y))
        err1 = reprojection_error(P1, X, (kp1.x, kp1.y))
        err = (err0 + err1) / 2.0

        if err > max_reprojection_px:
            continue

        results.append(Point3D(
            name=name,
            xyz=X[:3].copy(),
            confidence=conf,
            reprojection_error=err,
        ))
    return results


# --------------------------------------------------------------------------- #
#  Trajectory analysis                                                         #
# --------------------------------------------------------------------------- #

def build_trajectory_dict(
    frames: list[list[Point3D]],
    all_names: Optional[list[str]] = None,
) -> dict[str, np.ndarray]:
    """
    Convert a per-frame list of Point3D lists into per-landmark trajectory arrays.

    Returns
    -------
    dict mapping landmark name → (n_frames, 3) array with NaN for missing frames
    """
    if all_names is None:
        all_names = sorted({p.name for frame in frames for p in frame})

    n = len(frames)
    traj = {name: np.full((n, 3), np.nan) for name in all_names}
    for f_idx, frame in enumerate(frames):
        for pt in frame:
            if pt.name in traj:
                traj[pt.name][f_idx] = pt.xyz
    return traj


def smooth_trajectory(
    frames: list[list[Point3D]],
    sigma: float = 1.5,
) -> list[list[Point3D]]:
    """
    Gaussian temporal smoothing of 3D keypoint trajectories.

    NaN frames are excluded from the kernel and the smoothed value is
    computed only from available neighbours, preventing NaN propagation.

    Parameters
    ----------
    frames : per-frame list of Point3D
    sigma  : Gaussian standard deviation in frames

    Returns
    -------
    Smoothed per-frame list (same structure as input)
    """
    from scipy.ndimage import gaussian_filter1d

    all_names = sorted({p.name for frame in frames for p in frame})
    conf_dict = {name: np.zeros(len(frames)) for name in all_names}
    err_dict = {name: np.zeros(len(frames)) for name in all_names}
    for f_idx, frame in enumerate(frames):
        for pt in frame:
            if pt.name in conf_dict:
                conf_dict[pt.name][f_idx] = pt.confidence
                err_dict[pt.name][f_idx] = pt.reprojection_error

    traj = build_trajectory_dict(frames, all_names)
    smoothed_traj: dict[str, np.ndarray] = {}

    for name, xyz in traj.items():
        valid = ~np.isnan(xyz[:, 0])
        if not valid.any():
            smoothed_traj[name] = xyz
            continue
        out = xyz.copy()
        for axis in range(3):
            col = xyz[:, axis].copy()
            col[~valid] = 0.0
            # Weight sum: numerator = smoothed values, denominator = smoothed mask
            sm_num = gaussian_filter1d(col, sigma)
            sm_den = gaussian_filter1d(valid.astype(float), sigma)
            with np.errstate(invalid='ignore'):
                out[:, axis] = np.where(sm_den > 1e-6, sm_num / sm_den, np.nan)
        smoothed_traj[name] = out

    # Reconstruct frame list
    result = []
    for f_idx, frame in enumerate(frames):
        existing_names = {p.name for p in frame}
        new_frame = []
        for name in existing_names:
            xyz = smoothed_traj[name][f_idx]
            if not np.isnan(xyz).any():
                orig = next((p for p in frame if p.name == name), None)
                new_frame.append(Point3D(
                    name=name,
                    xyz=xyz,
                    confidence=orig.confidence if orig else 0.0,
                    reprojection_error=orig.reprojection_error if orig else 0.0,
                ))
        result.append(new_frame)
    return result


def fill_trajectory_gaps(
    frames: list[list[Point3D]],
    max_gap: int = 10,
) -> list[list[Point3D]]:
    """
    Fill short gaps in trajectories using linear interpolation.

    Parameters
    ----------
    frames  : per-frame list of Point3D
    max_gap : maximum consecutive missing frames to interpolate (longer gaps
              are left as NaN)
    """
    all_names = sorted({p.name for frame in frames for p in frame})
    traj = build_trajectory_dict(frames, all_names)
    conf_d = {name: np.zeros(len(frames)) for name in all_names}
    err_d = {name: np.zeros(len(frames)) for name in all_names}
    for f_idx, frame in enumerate(frames):
        for pt in frame:
            if pt.name in conf_d:
                conf_d[pt.name][f_idx] = pt.confidence
                err_d[pt.name][f_idx] = pt.reprojection_error

    for name, xyz in traj.items():
        valid_idx = np.where(~np.isnan(xyz[:, 0]))[0]
        if len(valid_idx) < 2:
            continue
        for i in range(len(valid_idx) - 1):
            a, b = valid_idx[i], valid_idx[i + 1]
            gap = b - a - 1
            if 0 < gap <= max_gap:
                for g in range(1, gap + 1):
                    t = g / (gap + 1)
                    traj[name][a + g] = (1 - t) * xyz[a] + t * xyz[b]
                    conf_d[name][a + g] = min(conf_d[name][a], conf_d[name][b]) * 0.8

    # Reconstruct
    result = []
    for f_idx, frame in enumerate(frames):
        existing_names = {p.name for p in frame}
        new_frame = list(frame)  # keep originals
        for name in all_names:
            if name in existing_names:
                continue
            xyz = traj[name][f_idx]
            if not np.isnan(xyz).any():
                new_frame.append(Point3D(
                    name=name,
                    xyz=xyz,
                    confidence=conf_d[name][f_idx],
                    reprojection_error=0.0,
                ))
        result.append(new_frame)
    return result


# --------------------------------------------------------------------------- #
#  Diagnostics                                                                 #
# --------------------------------------------------------------------------- #

def trajectory_stats(frames: list[list[Point3D]]) -> dict:
    """Print and return per-landmark detection statistics."""
    n = len(frames)
    all_names = sorted({p.name for frame in frames for p in frame})
    stats = {}
    for name in all_names:
        found = sum(1 for frame in frames if any(p.name == name for p in frame))
        errs = [p.reprojection_error
                for frame in frames
                for p in frame
                if p.name == name and p.reprojection_error > 0]
        stats[name] = {
            "detection_rate": found / n if n else 0.0,
            "n_detected": found,
            "mean_repr_err": float(np.mean(errs)) if errs else 0.0,
            "max_repr_err": float(np.max(errs)) if errs else 0.0,
        }
    return stats
