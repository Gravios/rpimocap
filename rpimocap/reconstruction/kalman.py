"""
kalman.py — 3D constant-velocity Kalman tracker for triangulated trajectories
=============================================================================
A single-target 6-state Kalman filter operating directly in arena/world
coordinates (mm). Designed for post-processing of triangulated 3D
positions, but can also be applied online inside the per-frame tracking
loop.

State / observation model
-------------------------
State vector x ∈ ℝ⁶:    [x, y, z, vx, vy, vz]      (mm, mm/s)
Observation z ∈ ℝ³:     triangulated 3D position    (mm)

Constant-velocity transition over Δt:
    F = [[I₃   Δt·I₃],
         [0₃   I₃  ]]
    H = [I₃ 0₃]

Process noise Q is built from a continuous white-acceleration model with
spectral density σ_a² (mm²/s³); measurement noise R is diagonal with the
triangulation RMSE σ_z² (mm²).

Outlier rejection
-----------------
A measurement whose Mahalanobis distance exceeds ``mahalanobis_gate``
(default = 5σ) is treated as a missed detection — the predict step
still runs but the update step is skipped. This is the trajectory-level
defence against the detector accidentally locking onto a wall reflection
or a stray bedding blob: the prediction (informed by the last few frames
of clean tracking) is far from the spurious detection, so the spurious
detection is rejected.

For a rat at 25 fps with realistic kinematics
(max ~1000 mm/s, max accel ~2000 mm/s²) the default σ_a = 1500 mm/s^1.5
and σ_z = 5 mm are sensible starting points; the typical Mahalanobis
threshold of 5σ then rejects detections more than ~60 mm from the
predicted position, which is well outside the per-frame motion budget
of 40 mm/frame.

Usage
-----
Offline post-process (recommended first integration):

    from rpimocap.reconstruction.kalman import smooth_trajectory_kalman

    # traj: (n_frames, 3) array of triangulated XYZ; NaN for missing
    filtered, info = smooth_trajectory_kalman(
        traj, dt=1.0/25.0, sigma_a=1500.0, sigma_z=5.0,
        mahalanobis_gate=5.0)

Online use inside a tracking loop:

    tracker = KalmanTracker3D(dt=1.0/25.0, sigma_a=1500.0, sigma_z=5.0)
    for frame in frames:
        xyz_meas = triangulate(frame)              # may be NaN
        accepted = tracker.step(xyz_meas)          # bool
        xyz_filt = tracker.x[:3]                   # smoothed position
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class KalmanInfo:
    """Per-frame diagnostics from a Kalman pass."""
    n_observations:   int = 0
    n_accepted:       int = 0
    n_rejected:       int = 0     # measurement outside Mahalanobis gate
    n_missing:        int = 0     # NaN measurement (no detection)
    mahalanobis:      list = field(default_factory=list)  # per-frame distance


class KalmanTracker3D:
    """Single-target 3D constant-velocity Kalman filter.

    Parameters
    ----------
    dt                : timestep between frames (s); for 25 fps use 1/25.
    sigma_a           : RMS acceleration in mm/s² (process noise scale).
                        ~1500 for a rat (covers normal locomotion); higher
                        for more agile animals or faster cameras.
    sigma_z           : measurement noise σ in mm. Set to your
                        triangulation RMSE (typically 2–6 mm for the
                        rpimocap stereo rig).
    mahalanobis_gate  : reject measurements whose Mahalanobis distance
                        to the prediction exceeds this many σ. Use
                        ``float("inf")`` to disable outlier rejection.
    initial_state     : optional (6,) array to seed the filter; if None
                        the first valid observation seeds [x, 0, 0].
    """

    def __init__(
        self,
        dt:               float = 1.0 / 25.0,
        sigma_a:          float = 1500.0,
        sigma_z:          float = 5.0,
        mahalanobis_gate: float = 5.0,
        initial_state:    "Optional[np.ndarray]" = None,
    ):
        self.dt = float(dt)
        self.sigma_a = float(sigma_a)
        self.sigma_z = float(sigma_z)
        self.gate = float(mahalanobis_gate)

        # Constant-velocity F, observation H
        I3 = np.eye(3)
        Z3 = np.zeros((3, 3))
        self.F = np.block([[I3, dt * I3], [Z3, I3]])
        self.H = np.hstack([I3, Z3])

        # Continuous white-acceleration process noise Q
        # ∫ G G^T  with  G = [Δt²/2·I₃; Δt·I₃]  scaled by σ_a².
        q11 = (dt**4) / 4.0
        q12 = (dt**3) / 2.0
        q22 = (dt**2)
        self.Q = (sigma_a ** 2) * np.block([
            [q11 * I3, q12 * I3],
            [q12 * I3, q22 * I3],
        ])
        self.R = (sigma_z ** 2) * I3

        # State and covariance
        if initial_state is None:
            self.x = np.zeros(6, dtype=np.float64)
            self.initialised = False
        else:
            self.x = np.asarray(initial_state, dtype=np.float64).reshape(6).copy()
            self.initialised = True
        # Large initial uncertainty so the first valid observation dominates
        self.P = np.diag([1e4, 1e4, 1e4, 1e4, 1e4, 1e4])
        self._last_mahal: float = 0.0

    # ------------------------------------------------------------------ #

    def predict(self) -> None:
        """Run the time-update step (predict)."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z: np.ndarray) -> bool:
        """Run the measurement-update step.

        Returns True if the measurement was accepted, False if it was
        rejected by the Mahalanobis gate (in which case the state stays
        at its predicted value).
        """
        z = np.asarray(z, dtype=np.float64).reshape(3)
        # Predicted measurement and innovation
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        # Mahalanobis distance d² = yᵀ S⁻¹ y
        try:
            Sinv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return False
        d2 = float(y @ Sinv @ y)
        self._last_mahal = float(np.sqrt(max(d2, 0.0)))
        if d2 > self.gate ** 2:
            return False
        K = self.P @ self.H.T @ Sinv
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P
        return True

    def step(self, z: "Optional[np.ndarray]") -> bool:
        """One-shot predict+update.

        Parameters
        ----------
        z : (3,) measurement or None / array containing NaNs to indicate
            a missing detection. When missing, only the predict step
            runs and the function returns False.

        Returns
        -------
        bool — True if the measurement was accepted, False if it was
        missing or rejected.
        """
        if z is None or np.any(np.isnan(np.asarray(z))):
            if self.initialised:
                self.predict()
            return False
        if not self.initialised:
            # First valid observation seeds position; velocity = 0
            z = np.asarray(z, dtype=np.float64).reshape(3)
            self.x[:3] = z
            self.x[3:] = 0.0
            self.P = np.diag([self.sigma_z ** 2] * 3 + [1e4] * 3)
            self.initialised = True
            return True
        self.predict()
        return self.update(z)


# --------------------------------------------------------------------------- #
#  Convenience: smooth a full trajectory array                                #
# --------------------------------------------------------------------------- #

def smooth_trajectory_kalman(
    xyz:              np.ndarray,
    *,
    dt:               float = 1.0 / 25.0,
    sigma_a:          float = 1500.0,
    sigma_z:          float = 5.0,
    mahalanobis_gate: float = 5.0,
    fill_predictions: bool  = True,
) -> tuple[np.ndarray, KalmanInfo]:
    """Apply a 3D Kalman filter to a per-frame trajectory.

    Parameters
    ----------
    xyz              : (n_frames, 3) array of triangulated positions
                       (mm). NaN entries mark missing detections.
    dt               : timestep between frames in seconds.
    sigma_a, sigma_z : process and measurement noise (see KalmanTracker3D).
    mahalanobis_gate : outlier-rejection threshold in σ.
    fill_predictions : if True, missing or rejected frames are filled
                       with the Kalman prediction; if False they are
                       written as NaN (in which case the filter still
                       improves the surrounding frames).

    Returns
    -------
    (filtered, info) where filtered has the same shape as xyz and info
    carries per-pass diagnostics.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"xyz must have shape (n, 3); got {xyz.shape}")
    out = np.full_like(xyz, np.nan)
    info = KalmanInfo()
    tracker = KalmanTracker3D(dt=dt, sigma_a=sigma_a, sigma_z=sigma_z,
                              mahalanobis_gate=mahalanobis_gate)
    for i, z in enumerate(xyz):
        if np.any(np.isnan(z)):
            info.n_missing += 1
            accepted = tracker.step(None)
        else:
            info.n_observations += 1
            accepted = tracker.step(z)
            if accepted:
                info.n_accepted += 1
            else:
                info.n_rejected += 1
        info.mahalanobis.append(tracker._last_mahal)
        if accepted:
            out[i] = tracker.x[:3]
        elif fill_predictions and tracker.initialised:
            out[i] = tracker.x[:3]
        # else: leave NaN
    return out, info


def smooth_trajectory_dict_kalman(
    traj: dict,
    *,
    dt:               float = 1.0 / 25.0,
    sigma_a:          float = 1500.0,
    sigma_z:          float = 5.0,
    mahalanobis_gate: float = 5.0,
    fill_predictions: bool  = True,
) -> tuple[dict, dict]:
    """Apply a 3D Kalman filter to a per-landmark trajectory dict.

    Parameters
    ----------
    traj : dict mapping landmark name → (n_frames, 3) array, as built
           by ``build_trajectory_dict``. NaN marks missing frames.

    Returns
    -------
    (filtered_traj, info_dict) — same keys as ``traj``; info_dict maps
    landmark name → KalmanInfo for that landmark.
    """
    filtered = {}
    info = {}
    for name, arr in traj.items():
        f, i = smooth_trajectory_kalman(
            arr, dt=dt, sigma_a=sigma_a, sigma_z=sigma_z,
            mahalanobis_gate=mahalanobis_gate,
            fill_predictions=fill_predictions)
        filtered[name] = f
        info[name] = i
    return filtered, info


__all__ = [
    "KalmanInfo",
    "KalmanTracker3D",
    "smooth_trajectory_kalman",
    "smooth_trajectory_dict_kalman",
]
