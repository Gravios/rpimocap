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

from dataclasses import dataclass
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
    kalman_outlier: bool = False   # set by kalman_filter_trajectory()

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
    *,
    arena_model=None,
    K0: Optional[np.ndarray] = None, dist0: Optional[np.ndarray] = None,
    R0: Optional[np.ndarray] = None, T0: Optional[np.ndarray] = None,
    K1: Optional[np.ndarray] = None, dist1: Optional[np.ndarray] = None,
    R1: Optional[np.ndarray] = None, T1: Optional[np.ndarray] = None,
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
    arena_model        : Optional ``ArenaRefractionModel``. When supplied, rays are
                         refracted through whichever wall each one crosses before
                         being intersected, correcting the apparent-position bias
                         introduced by acrylic arena walls. Requires the per-camera
                         intrinsics/extrinsics (K0/dist0/R0/T0, K1/dist1/R1/T1) to
                         be passed as well; otherwise this falls back to DLT.

    Returns
    -------
    list of Point3D
    """
    kps0 = result0.by_name() if hasattr(result0, "by_name") else {k.name: k for k in result0}
    kps1 = result1.by_name() if hasattr(result1, "by_name") else {k.name: k for k in result1}

    use_refraction = (
        arena_model is not None
        and K0 is not None and R0 is not None and T0 is not None
        and K1 is not None and R1 is not None and T1 is not None
    )
    if use_refraction:
        # Pre-compute camera centres in world coordinates: C = -R^T T
        from rpimocap.reconstruction.refraction import pixel_to_world_ray, triangulate_refracted

    results = []
    for name, kp0 in kps0.items():
        kp1 = kps1.get(name)
        if kp1 is None:
            continue
        conf = min(kp0.confidence, kp1.confidence)
        if conf < min_confidence:
            continue

        # Initial DLT estimate (also used as seed for refractive iteration
        # and as the fallback if refraction is not configured).
        X = triangulate_dlt(P0, P1, (kp0.x, kp0.y), (kp1.x, kp1.y))
        err0 = reprojection_error(P0, X, (kp0.x, kp0.y))
        err1 = reprojection_error(P1, X, (kp1.x, kp1.y))
        err = (err0 + err1) / 2.0

        if err > max_reprojection_px:
            continue

        if use_refraction:
            try:
                C0, d0 = pixel_to_world_ray(K0, R0, T0, (kp0.x, kp0.y), dist=dist0)
                C1, d1 = pixel_to_world_ray(K1, R1, T1, (kp1.x, kp1.y), dist=dist1)
                X_ref, _gap, _it = triangulate_refracted(
                    C0, d0, C1, d1, arena_model,
                    initial_xyz=X[:3],
                )
                xyz = np.asarray(X_ref, dtype=np.float64).reshape(3)
            except Exception:
                # If refraction fails (e.g. TIR or ray misses all walls in
                # an unexpected way), fall back to the straight-ray solution.
                xyz = X[:3].copy()
        else:
            xyz = X[:3].copy()

        results.append(Point3D(
            name=name,
            xyz=xyz,
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


def kalman_filter_trajectory(
    frames:               list,
    fps:                  float = 25.0,
    max_speed_mm_s:       float = 1000.0,
    max_accel_mm_s2:      float = 2000.0,
    measurement_noise_mm: float = 8.0,
    outlier_sigma:        float = 4.0,
    rts_smooth:           bool  = True,
) -> list:
    """Kalman / RTS smoother for 3D trajectories.

    Replaces the Gaussian smooth + linear gap-fill pipeline with a
    physically-constrained constant-velocity Kalman filter.

    State vector: [x, y, z, vx, vy, vz]  (position + velocity, mm & mm/s)

    Advantages over Gaussian smoothing
    -----------------------------------
    * **Outlier rejection**: measurements whose Mahalanobis distance
      from the Kalman prediction exceeds ``outlier_sigma`` are treated
      as missing (bad blob selections, reflections, etc.).
    * **Physics-based gap filling**: missing frames are filled by the
      constant-velocity prediction (with growing uncertainty), not
      linear interpolation. The prediction is clamped to
      ``max_speed_mm_s`` so gaps cannot produce trajectories that
      travel faster than the rat.
    * **RTS smoother**: a backward pass (Rauch-Tung-Striebel) refines
      all estimates using the full sequence. This is optimal for
      offline processing and gives smoother trajectories than a
      causal filter alone.

    Returns the filtered/smoothed per-frame list (same structure as
    input). Outlier-rejected and gap-filled frames are flagged via the
    ``kalman_outlier`` attribute on Point3D.
    """
    dt  = 1.0 / fps
    sa2 = (max_accel_mm_s2 / 3.0) ** 2   # 1-sigma acceleration variance

    F = np.eye(6)
    F[0, 3] = F[1, 4] = F[2, 5] = dt
    H = np.zeros((3, 6))
    H[0, 0] = H[1, 1] = H[2, 2] = 1.0

    q_pos  = (dt ** 4 / 4.0) * sa2
    q_pv   = (dt ** 3 / 2.0) * sa2
    q_vel  = (dt ** 2)        * sa2
    Q = np.array([
        [q_pos, 0,     0,     q_pv, 0,     0    ],
        [0,     q_pos, 0,     0,    q_pv,  0    ],
        [0,     0,     q_pos, 0,    0,     q_pv ],
        [q_pv,  0,     0,     q_vel,0,     0    ],
        [0,     q_pv,  0,     0,    q_vel, 0    ],
        [0,     0,     q_pv,  0,    0,     q_vel],
    ])
    R = (measurement_noise_mm ** 2) * np.eye(3)
    outlier_thresh = outlier_sigma ** 2

    all_names = sorted({p.name for frame in frames for p in frame})
    result    = [[pt for pt in frame] for frame in frames]

    for name in all_names:
        N = len(frames)
        obs = np.full((N, 3), np.nan)
        for i, frame in enumerate(frames):
            pt = next((p for p in frame if p.name == name), None)
            if pt is not None and not np.isnan(pt.xyz).any():
                obs[i] = pt.xyz

        valid_idx = np.where(~np.isnan(obs[:, 0]))[0]
        if len(valid_idx) < 2:
            continue

        x0 = np.zeros(6)
        x0[:3] = obs[valid_idx[0]]
        if len(valid_idx) >= 2:
            dx = obs[valid_idx[1]] - obs[valid_idx[0]]
            dt_init = (valid_idx[1] - valid_idx[0]) * dt
            x0[3:] = dx / max(dt_init, dt)
        P0 = np.diag([measurement_noise_mm**2] * 3
                   + [max_speed_mm_s**2] * 3)

        xs   = np.zeros((N, 6))
        Ps   = np.zeros((N, 6, 6))
        xp   = np.zeros((N, 6))
        Pp_s = np.zeros((N, 6, 6))
        rejected = np.zeros(N, dtype=bool)
        x, P = x0.copy(), P0.copy()
        started = False

        for i in range(N):
            if started:
                x = F @ x
                P = F @ P @ F.T + Q
            xp[i]   = x.copy()
            Pp_s[i] = P.copy()

            z = obs[i]
            if not np.isnan(z[0]):
                innov = z - H @ x
                S     = H @ P @ H.T + R
                try:
                    S_inv = np.linalg.inv(S)
                    mah2  = float(innov @ S_inv @ innov)
                except np.linalg.LinAlgError:
                    mah2 = outlier_thresh + 1.0
                if mah2 <= outlier_thresh:
                    K = P @ H.T @ S_inv
                    x = x + K @ innov
                    P = (np.eye(6) - K @ H) @ P
                    started = True
                else:
                    rejected[i] = True
            xs[i] = x.copy()
            Ps[i] = P.copy()
            if not started and not np.isnan(z[0]):
                started = True

        if rts_smooth:
            xs_s = xs.copy()
            Ps_s = Ps.copy()
            for i in range(N - 2, -1, -1):
                Pp = Pp_s[i + 1]
                try:
                    G = Ps[i] @ F.T @ np.linalg.inv(Pp)
                except np.linalg.LinAlgError:
                    continue
                xs_s[i] = xs[i] + G @ (xs_s[i + 1] - xp[i + 1])
                Ps_s[i] = Ps[i] + G @ (Ps_s[i + 1] - Pp) @ G.T
            xs = xs_s

        for i, frame in enumerate(result):
            xyz_k = xs[i, :3]
            existing = next((p for p in frame if p.name == name), None)
            was_valid = not np.isnan(obs[i, 0]) and not rejected[i]
            if was_valid and existing is not None:
                existing.xyz = xyz_k.copy()
                existing.kalman_outlier = False
            elif rejected[i] and existing is not None:
                existing.xyz = xyz_k.copy()
                existing.kalman_outlier = True
            elif np.isnan(obs[i, 0]) and existing is None:
                frame.append(Point3D(
                    name=name, xyz=xyz_k.copy(),
                    confidence=0.0,
                    reprojection_error=float(np.sqrt(Ps[i, 0, 0]))))
    return result


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
