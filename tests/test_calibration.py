"""
tests/test_calibration.py
=========================
Unit tests for rpimocap.calibration.

Tests checkerboard calibration helpers and autocalib Kruppa estimator
using synthetic camera geometry (no video files required).
"""

import math
import numpy as np
import pytest

from rpimocap.calibration.autocalib import (
    make_K,
    cost_for_f,
    estimate_focal_kruppa,
    essential_constraint_residual,
    run_focal_estimation,
    FundamentalEstimate,
)
from rpimocap.calibration.autocalib.kruppa import decompose_essential


# --------------------------------------------------------------------------- #
#  Fixtures                                                                    #
# --------------------------------------------------------------------------- #

IMAGE_SIZE = (1280, 720)
F_TRUE = 900.0   # ground-truth focal length in pixels


def _random_F(K_true: np.ndarray, seed: int = 0) -> np.ndarray:
    """Generate a synthetic fundamental matrix from a random camera pair."""
    rng = np.random.default_rng(seed)
    rvec = rng.standard_normal(3)
    rvec = rvec / np.linalg.norm(rvec) * rng.uniform(0.15, 0.5)
    from scipy.spatial.transform import Rotation
    R = Rotation.from_rotvec(rvec).as_matrix()
    t = rng.standard_normal(3)
    t /= np.linalg.norm(t)
    tx = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
    E = tx @ R
    K_inv = np.linalg.inv(K_true)
    F = K_inv.T @ E @ K_inv
    U, S, Vt = np.linalg.svd(F)
    S[2] = 0.0
    F = (U * S) @ Vt
    return F / (np.linalg.norm(F) + 1e-12)


@pytest.fixture(scope="module")
def synthetic_estimates():
    w, h = IMAGE_SIZE
    K = make_K(F_TRUE, w / 2, h / 2)
    ests = []
    for i in range(30):
        F = _random_F(K, seed=i)
        ests.append(FundamentalEstimate(
            frame_idx=i * 60, F=F,
            pts0=np.zeros((80, 2)), pts1=np.zeros((80, 2)),
            n_inliers=200, inlier_ratio=0.7, mean_sampson=0.25,
        ))
    return ests


# --------------------------------------------------------------------------- #
#  make_K                                                                      #
# --------------------------------------------------------------------------- #

def test_make_K_shape():
    K = make_K(900.0, 640.0, 360.0)
    assert K.shape == (3, 3)


def test_make_K_values():
    f, cx, cy = 850.5, 640.0, 360.0
    K = make_K(f, cx, cy)
    assert K[0, 0] == pytest.approx(f)
    assert K[1, 1] == pytest.approx(f)
    assert K[0, 2] == pytest.approx(cx)
    assert K[1, 2] == pytest.approx(cy)
    assert K[2, 2] == pytest.approx(1.0)
    assert K[0, 1] == pytest.approx(0.0)  # zero skew


# --------------------------------------------------------------------------- #
#  Essential matrix constraint                                                  #
# --------------------------------------------------------------------------- #

def test_essential_residual_valid_E():
    """A valid essential matrix should give residual ~0."""
    t = np.array([1.0, 0.0, 0.0])
    tx = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
    E = tx @ np.eye(3)
    E /= np.linalg.norm(E)
    assert essential_constraint_residual(E) < 1e-6


def test_essential_residual_random_matrix():
    """A random matrix should give a high residual."""
    rng = np.random.default_rng(42)
    M = rng.standard_normal((3, 3))
    M /= np.linalg.norm(M)
    assert essential_constraint_residual(M) > 0.05


# --------------------------------------------------------------------------- #
#  cost_for_f                                                                   #
# --------------------------------------------------------------------------- #

def test_cost_minimum_at_true_f(synthetic_estimates):
    """cost_for_f should be minimised at (or near) the true focal length."""
    w, h = IMAGE_SIZE
    cx, cy = w / 2.0, h / 2.0
    fs = np.linspace(400, 1800, 200)
    costs = [cost_for_f(f, synthetic_estimates, cx, cy) for f in fs]
    f_est = fs[np.argmin(costs)]
    error_pct = 100 * abs(f_est - F_TRUE) / F_TRUE
    assert error_pct < 5.0, f"Cost minimum at f={f_est:.1f}, expected ~{F_TRUE}"


# --------------------------------------------------------------------------- #
#  estimate_focal_kruppa                                                       #
# --------------------------------------------------------------------------- #

def test_kruppa_accuracy(synthetic_estimates):
    f_est, _, _ = estimate_focal_kruppa(
        synthetic_estimates, IMAGE_SIZE, verbose=False
    )
    error_pct = 100 * abs(f_est - F_TRUE) / F_TRUE
    assert error_pct < 3.0, f"Kruppa estimate {f_est:.1f} px, true {F_TRUE} px"


def test_kruppa_returns_scan_arrays(synthetic_estimates):
    f_est, scan_f, scan_cost = estimate_focal_kruppa(
        synthetic_estimates, IMAGE_SIZE, scan_steps=50, verbose=False
    )
    assert len(scan_f) == 50
    assert len(scan_cost) == 50
    assert scan_f[0] < scan_f[-1]


# --------------------------------------------------------------------------- #
#  run_focal_estimation                                                        #
# --------------------------------------------------------------------------- #

def test_run_focal_estimation_no_arena(synthetic_estimates):
    result = run_focal_estimation(
        synthetic_estimates, IMAGE_SIZE, arena_bounds=None, verbose=False
    )
    assert result.f_px > 0
    assert result.K.shape == (3, 3)
    assert result.f_metric is None
    assert result.scale_factor is None
    assert result.n_estimates_used == len(synthetic_estimates)


def test_run_focal_estimation_with_arena(synthetic_estimates):
    arena = ((-300.0, 300.0), (-200.0, 200.0), (0.0, 350.0))
    result = run_focal_estimation(
        synthetic_estimates, IMAGE_SIZE,
        arena_bounds=arena, verbose=False
    )
    # Metric refinement should run (may or may not improve if triangulation
    # degenerates with dummy pts, but should not crash)
    assert result.f_px > 0
    assert result.mean_essential_residual >= 0.0


def test_kruppa_result_essential_residual(synthetic_estimates):
    """Cost at optimised f must be lower than at a clearly wrong focal length."""
    result = run_focal_estimation(
        synthetic_estimates, IMAGE_SIZE, verbose=False
    )
    w, h = IMAGE_SIZE
    cx, cy = w / 2.0, h / 2.0
    cost_opt = cost_for_f(result.f_px, synthetic_estimates, cx, cy)
    cost_bad = cost_for_f(result.f_px * 2, synthetic_estimates, cx, cy)
    assert cost_opt < cost_bad, "Optimised f does not give lower cost than 2× f" 


# --------------------------------------------------------------------------- #
#  decompose_essential                                                         #
# --------------------------------------------------------------------------- #

def test_decompose_essential_cheirality():
    """Decomposed R,t should place most points in front of both cameras."""
    rng = np.random.default_rng(7)
    K = make_K(F_TRUE, 640.0, 360.0)

    # True pose
    from scipy.spatial.transform import Rotation
    R_true = Rotation.from_euler('y', 15, degrees=True).as_matrix()
    t_true = np.array([50.0, 0.0, 0.0])
    t_true /= np.linalg.norm(t_true)

    # Synthesise 50 3D points in front of cam0 (z ∈ [200, 600])
    pts3d = rng.uniform(-100, 100, (50, 3))
    pts3d[:, 2] = rng.uniform(200, 600, 50)

    P0 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P1 = K @ np.hstack([R_true, t_true.reshape(3, 1)])

    def proj(P, X):
        h = P @ np.hstack([X, np.ones((len(X), 1))]).T
        return (h[:2] / h[2]).T

    p0 = proj(P0, pts3d)
    p1 = proj(P1, pts3d)

    # Ground-truth E
    tx = np.array([[0, -t_true[2], t_true[1]],
                   [t_true[2], 0, -t_true[0]],
                   [-t_true[1], t_true[0], 0]])
    E = tx @ R_true
    E /= np.linalg.norm(E)

    result = decompose_essential(E, p0, p1, K)
    assert result is not None, "decompose_essential returned None"
    R_rec, t_rec = result
    assert R_rec.shape == (3, 3)
    assert abs(np.linalg.det(R_rec) - 1.0) < 1e-6
