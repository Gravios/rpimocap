"""
tests/test_kalman.py
=====================
Unit tests for rpimocap.reconstruction.kalman.

Verifies:
- noise rejection on a noisy constant-velocity track
- gap filling by prediction
- outlier rejection via the Mahalanobis gate
- dict-level API matches per-array results
"""
from __future__ import annotations

import numpy as np
import pytest

from rpimocap.reconstruction.kalman import (
    KalmanInfo,
    KalmanTracker3D,
    smooth_trajectory_dict_kalman,
    smooth_trajectory_kalman,
)


def _synthetic_track(n=200, dt=1/25.0, seed=0, noise_sigma=5.0):
    """Constant-velocity ground truth plus i.i.d. Gaussian measurement noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) * dt
    v = np.array([200.0, -100.0, 0.0])           # mm/s
    p0 = np.array([0.0, 0.0, 100.0])
    truth = p0[None, :] + t[:, None] * v[None, :]
    meas = truth + rng.normal(0.0, noise_sigma, size=truth.shape)
    return truth, meas


class TestKalmanTracker3D:

    def test_reduces_noise_on_constant_velocity(self):
        truth, meas = _synthetic_track(n=300, noise_sigma=5.0, seed=1)
        # Use a tighter process model (sigma_a=300) appropriate for a
        # constant-velocity ground truth; the default sigma_a=1500 covers
        # realistic rat dynamics but is too slack to halve noise on a
        # purely constant-velocity track.
        filt, info = smooth_trajectory_kalman(
            meas, dt=1/25.0, sigma_a=300.0, sigma_z=5.0,
            mahalanobis_gate=8.0)
        burn = 25
        rmse_meas = np.sqrt(np.mean((meas[burn:] - truth[burn:]) ** 2))
        rmse_filt = np.sqrt(np.mean((filt[burn:] - truth[burn:]) ** 2))
        # A reasonable Kalman pass should reduce RMSE meaningfully; we
        # require at least 40 % reduction to allow for seed-to-seed
        # variation while still catching a regression.
        assert rmse_filt < 0.6 * rmse_meas, (
            f"Kalman RMSE {rmse_filt:.2f} mm not significantly below "
            f"measurement RMSE {rmse_meas:.2f} mm")
        assert info.n_accepted >= 290

    def test_fills_short_gaps_by_prediction(self):
        truth, meas = _synthetic_track(n=200, noise_sigma=2.0, seed=2)
        # Introduce a 10-frame gap from idx 80–89
        meas[80:90] = np.nan
        filt, info = smooth_trajectory_kalman(
            meas, dt=1/25.0, sigma_a=1500.0, sigma_z=2.0,
            mahalanobis_gate=8.0, fill_predictions=True)
        assert info.n_missing == 10
        # No NaNs remain in the gap region
        assert not np.any(np.isnan(filt[80:90]))
        # Predicted positions during the gap should track truth well
        err = np.linalg.norm(filt[80:90] - truth[80:90], axis=1)
        assert err.max() < 30.0, f"gap prediction max err {err.max():.1f} mm too large"

    def test_outlier_rejection_via_mahalanobis_gate(self):
        truth, meas = _synthetic_track(n=200, noise_sigma=2.0, seed=3)
        # Spike a single frame far from the trajectory (e.g. wall reflection)
        meas[120] = truth[120] + np.array([300.0, 0.0, 0.0])
        filt, info = smooth_trajectory_kalman(
            meas, dt=1/25.0, sigma_a=1500.0, sigma_z=2.0,
            mahalanobis_gate=5.0)
        assert info.n_rejected >= 1, "expected to reject at least the spike"
        # Filtered position at the spike should be near truth, not the spike
        err_spike = np.linalg.norm(filt[120] - truth[120])
        assert err_spike < 30.0, (
            f"filter followed the outlier; err={err_spike:.1f} mm")

    def test_disabled_gate_admits_all_finite_observations(self):
        _, meas = _synthetic_track(n=100, noise_sigma=2.0, seed=4)
        meas[50] = meas[50] + np.array([1000.0, 0.0, 0.0])
        _, info = smooth_trajectory_kalman(
            meas, dt=1/25.0, sigma_a=1500.0, sigma_z=2.0,
            mahalanobis_gate=float("inf"))
        assert info.n_rejected == 0
        assert info.n_accepted == 100

    def test_step_interface(self):
        truth, meas = _synthetic_track(n=50, noise_sigma=3.0, seed=5)
        tr = KalmanTracker3D(dt=1/25.0, sigma_a=1500.0, sigma_z=3.0)
        for i, z in enumerate(meas):
            tr.step(z)
        # After 50 frames the velocity estimate should be close to truth
        v_true = (truth[-1] - truth[0]) / ((len(truth) - 1) * (1/25.0))
        err_v = np.linalg.norm(tr.x[3:] - v_true)
        assert err_v < 50.0, f"velocity error {err_v:.1f} mm/s too large"

    def test_step_initial_missing_observation(self):
        """Missing observations before the first valid one must not crash."""
        tr = KalmanTracker3D(dt=1/25.0)
        # First three frames have no detection
        assert tr.step(None) is False
        assert tr.step(np.array([np.nan, np.nan, np.nan])) is False
        # First real measurement seeds the filter
        assert tr.step(np.array([10.0, 20.0, 30.0])) is True
        np.testing.assert_allclose(tr.x[:3], [10.0, 20.0, 30.0])


class TestDictAPI:

    def test_matches_per_array_result(self):
        _, m_nose = _synthetic_track(n=100, noise_sigma=2.0, seed=10)
        _, m_tail = _synthetic_track(n=100, noise_sigma=2.0, seed=11)
        traj = {"nose": m_nose, "tail": m_tail}
        filt_dict, info_dict = smooth_trajectory_dict_kalman(
            traj, dt=1/25.0, sigma_a=1500.0, sigma_z=2.0)
        # Per-array call must produce the same numbers
        for name, arr in traj.items():
            f, _ = smooth_trajectory_kalman(
                arr, dt=1/25.0, sigma_a=1500.0, sigma_z=2.0)
            np.testing.assert_allclose(filt_dict[name], f)
        assert set(info_dict.keys()) == {"nose", "tail"}
        for info in info_dict.values():
            assert isinstance(info, KalmanInfo)
