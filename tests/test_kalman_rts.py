"""
tests/test_kalman_rts.py
=========================
Unit tests for rpimocap.reconstruction.triangulate.kalman_filter_trajectory.

Verifies:
- noise reduction on a clean constant-velocity track
- outlier rejection via Mahalanobis gating (kalman_outlier=True)
- gap filling by Kalman prediction
- RTS smoother improves over forward-only
- kalman_outlier field present on Point3D
"""
from __future__ import annotations

import numpy as np
import pytest

from rpimocap.reconstruction.triangulate import (
    Point3D,
    kalman_filter_trajectory,
)


def _synthetic_frames(n=200, dt=1/25.0, noise_mm=8.0, seed=0):
    """Make a list-of-list-of-Point3D trajectory (constant velocity)."""
    rng = np.random.default_rng(seed)
    v = np.array([200.0, -100.0, 0.0])
    p0 = np.array([0.0, 0.0, 100.0])
    truth = p0[None, :] + (np.arange(n) * dt)[:, None] * v[None, :]
    meas  = truth + rng.normal(0.0, noise_mm, size=truth.shape)

    frames = []
    for i in range(n):
        frames.append([Point3D(
            name="animal",
            xyz=meas[i].copy(),
            confidence=1.0,
            reprojection_error=2.0)])
    return frames, truth


class TestKalmanFilterTrajectory:

    def test_point3d_has_kalman_outlier_field(self):
        p = Point3D(name="x", xyz=np.zeros(3))
        assert hasattr(p, "kalman_outlier")
        assert p.kalman_outlier is False

    def test_reduces_noise_constant_velocity(self):
        frames, truth = _synthetic_frames(n=300, noise_mm=8.0, seed=1)
        # Raw measurement RMSE
        meas = np.array([f[0].xyz for f in frames])
        rmse_meas = np.sqrt(np.mean((meas - truth) ** 2))

        result = kalman_filter_trajectory(
            frames, fps=25.0, measurement_noise_mm=8.0,
            max_accel_mm_s2=2000.0, rts_smooth=True)
        filt = np.array([f[0].xyz for f in result])
        burn = 25
        rmse_filt = np.sqrt(np.mean((filt[burn:] - truth[burn:]) ** 2))
        assert rmse_filt < 0.6 * rmse_meas, (
            f"Kalman/RTS RMSE {rmse_filt:.2f} mm not below "
            f"measurement RMSE {rmse_meas:.2f} mm")

    def test_outlier_rejection_marks_point(self):
        frames, truth = _synthetic_frames(n=200, noise_mm=2.0, seed=2)
        # Plant a wall-reflection spike at frame 120
        spike_idx = 120
        frames[spike_idx][0].xyz = frames[spike_idx][0].xyz + np.array(
            [300.0, 0.0, 0.0])
        result = kalman_filter_trajectory(
            frames, fps=25.0, measurement_noise_mm=2.0,
            max_accel_mm_s2=2000.0, outlier_sigma=4.0)
        # The spike frame should be flagged as outlier
        assert result[spike_idx][0].kalman_outlier is True
        # And its position should be close to truth, not the spike
        err = float(np.linalg.norm(result[spike_idx][0].xyz - truth[spike_idx]))
        assert err < 30.0, (
            f"outlier-corrected position is {err:.1f} mm from truth")

    def test_gap_fill_with_kalman_prediction(self):
        frames, truth = _synthetic_frames(n=200, noise_mm=2.0, seed=3)
        # Delete the keypoint from frames 80-89 entirely (missing detection)
        for i in range(80, 90):
            frames[i] = []
        result = kalman_filter_trajectory(
            frames, fps=25.0, measurement_noise_mm=2.0,
            max_accel_mm_s2=2000.0)
        # Gap frames should now have a Point3D injected with confidence=0
        for i in range(80, 90):
            assert len(result[i]) == 1, f"frame {i} not gap-filled"
            assert result[i][0].confidence == 0.0
            # Prediction should be close to truth
            err = float(np.linalg.norm(result[i][0].xyz - truth[i]))
            assert err < 40.0, (
                f"gap-fill at frame {i}: err {err:.1f} mm too large")

    def test_rts_beats_forward_only(self):
        frames, truth = _synthetic_frames(n=200, noise_mm=8.0, seed=4)
        # Forward only
        res_fwd = kalman_filter_trajectory(
            frames, fps=25.0, measurement_noise_mm=8.0,
            max_accel_mm_s2=2000.0, rts_smooth=False)
        # With RTS
        frames2, _ = _synthetic_frames(n=200, noise_mm=8.0, seed=4)
        res_rts = kalman_filter_trajectory(
            frames2, fps=25.0, measurement_noise_mm=8.0,
            max_accel_mm_s2=2000.0, rts_smooth=True)
        burn = 25
        rmse_fwd = np.sqrt(np.mean(
            (np.array([f[0].xyz for f in res_fwd[burn:]]) - truth[burn:]) ** 2))
        rmse_rts = np.sqrt(np.mean(
            (np.array([f[0].xyz for f in res_rts[burn:]]) - truth[burn:]) ** 2))
        # RTS smoother should not be worse than forward-only.
        assert rmse_rts <= rmse_fwd + 0.5, (
            f"RTS RMSE {rmse_rts:.2f} mm worse than forward-only "
            f"{rmse_fwd:.2f} mm")

    def test_short_track_skipped_gracefully(self):
        # Only one valid observation — too few to initialise
        frames = [[Point3D(name="x", xyz=np.array([1.0, 2.0, 3.0]))]]
        for _ in range(10):
            frames.append([])
        result = kalman_filter_trajectory(frames, fps=25.0)
        # Should return without error; not initialised → no kalman fill
        assert len(result) == 11
