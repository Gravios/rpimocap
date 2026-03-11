"""
tests/test_autocalib.py
========================
Integration tests for rpimocap.calibration.autocalib.

Tests feature dataclass construction, quality scoring, filtering logic,
and report generation (no video I/O required).
"""

import math
import numpy as np
import pytest
import tempfile
import os

from rpimocap.calibration.autocalib import (
    FundamentalEstimate,
    CrossViewMatcher,
    filter_estimates,
    sampson_distance,
    make_K,
    KruppaResult,
    generate_report,
    run_focal_estimation,
)


# --------------------------------------------------------------------------- #
#  FundamentalEstimate                                                         #
# --------------------------------------------------------------------------- #

class TestFundamentalEstimate:

    def _make_est(self, n_inliers=200, inlier_ratio=0.7, mean_sampson=0.3,
                  frame_idx=0):
        F = np.eye(3)
        F[2, 2] = 0.0  # rank-deficient placeholder
        return FundamentalEstimate(
            frame_idx=frame_idx, F=F,
            pts0=np.zeros((n_inliers, 2)),
            pts1=np.zeros((n_inliers, 2)),
            n_inliers=n_inliers,
            inlier_ratio=inlier_ratio,
            mean_sampson=mean_sampson,
        )

    def test_quality_is_in_unit_interval(self):
        est = self._make_est()
        assert 0.0 <= est.quality <= 1.0

    def test_higher_inliers_gives_higher_quality(self):
        lo = self._make_est(n_inliers=50)
        hi = self._make_est(n_inliers=300)
        assert hi.quality > lo.quality

    def test_lower_sampson_gives_higher_quality(self):
        good = self._make_est(mean_sampson=0.1)
        bad  = self._make_est(mean_sampson=2.0)
        assert good.quality > bad.quality


# --------------------------------------------------------------------------- #
#  sampson_distance                                                            #
# --------------------------------------------------------------------------- #

class TestSampsonDistance:

    def test_returns_array_of_correct_length(self):
        F = np.array([[0,0,0],[0,0,-1],[0,1,0]], dtype=float)
        pts0 = np.random.default_rng(0).standard_normal((20, 2)) + 320
        pts1 = pts0 + np.random.default_rng(1).standard_normal((20, 2))
        d = sampson_distance(F, pts0, pts1)
        assert d.shape == (20,)

    def test_zero_distance_for_degenerate_F(self):
        F = np.zeros((3, 3))
        pts = np.ones((5, 2)) * 100
        d = sampson_distance(F, pts, pts)
        assert (d == 0).all()


# --------------------------------------------------------------------------- #
#  filter_estimates                                                            #
# --------------------------------------------------------------------------- #

class TestFilterEstimates:

    def _make_pool(self, n=40):
        rng = np.random.default_rng(0)
        pool = []
        for i in range(n):
            ni  = int(rng.integers(40, 300))
            ir  = float(rng.uniform(0.2, 0.9))
            ms  = float(rng.uniform(0.1, 3.0))
            F   = np.eye(3)
            pool.append(FundamentalEstimate(
                frame_idx=i * 30, F=F,
                pts0=np.zeros((ni, 2)), pts1=np.zeros((ni, 2)),
                n_inliers=ni, inlier_ratio=ir, mean_sampson=ms,
            ))
        return pool

    def test_filter_removes_low_quality(self):
        pool = self._make_pool(40)
        filtered = filter_estimates(pool, min_quality=0.5)
        assert all(e.quality >= 0.5 for e in filtered)

    def test_top_n_cap(self):
        pool = self._make_pool(40)
        filtered = filter_estimates(pool, min_quality=0.0, top_n=10)
        assert len(filtered) <= 10

    def test_empty_input(self):
        assert filter_estimates([], min_quality=0.3) == []


# --------------------------------------------------------------------------- #
#  Report generation                                                           #
# --------------------------------------------------------------------------- #

class TestReportGeneration:

    def _make_result(self):
        w, h = 1280, 720
        f = 900.0
        scan_f = np.linspace(400, 1800, 50)
        scan_c = np.exp(-((scan_f - f)**2) / 50000)
        scan_c = scan_c.max() - scan_c   # inverted: minimum at f
        return KruppaResult(
            f_px=f, cx_px=w/2, cy_px=h/2,
            K=make_K(f, w/2, h/2),
            f_search_vals=scan_f, f_search_costs=scan_c,
            f_kruppa=f, f_metric=None,
            n_estimates_used=30,
            mean_essential_residual=0.018,
            scale_factor=None,
        )

    def _make_estimates(self, n=20):
        ests = []
        rng = np.random.default_rng(0)
        for i in range(n):
            ests.append(FundamentalEstimate(
                frame_idx=i*60, F=np.eye(3),
                pts0=np.zeros((100, 2)), pts1=np.zeros((100, 2)),
                n_inliers=int(rng.integers(80, 250)),
                inlier_ratio=float(rng.uniform(0.4, 0.8)),
                mean_sampson=float(rng.uniform(0.2, 1.5)),
            ))
        return ests

    def test_report_creates_html_file(self):
        result = self._make_result()
        ests   = self._make_estimates()
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = f.name
        try:
            generate_report(result, ests, (1280, 720), None, path)
            assert os.path.exists(path)
            size = os.path.getsize(path)
            assert size > 5000, f"Report too small: {size} bytes"
        finally:
            os.unlink(path)

    def test_report_contains_focal_length(self):
        result = self._make_result()
        ests   = self._make_estimates()
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False,
                                          mode="w+") as f:
            path = f.name
        try:
            generate_report(result, ests, (1280, 720), None, path)
            html = open(path).read()
            assert "900.00" in html or "900" in html
        finally:
            os.unlink(path)


# --------------------------------------------------------------------------- #
#  run_focal_estimation (package-level integration)                           #
# --------------------------------------------------------------------------- #

class TestRunFocalEstimation:

    def _synthetic_estimates(self, f_true=900.0, n=25, image_size=(1280, 720)):
        w, h = image_size
        K = make_K(f_true, w/2, h/2)
        K_inv = np.linalg.inv(K)
        rng = np.random.default_rng(99)
        ests = []
        for i in range(n):
            rvec = rng.standard_normal(3)
            rvec = rvec / np.linalg.norm(rvec) * rng.uniform(0.1, 0.4)
            from scipy.spatial.transform import Rotation
            R = Rotation.from_rotvec(rvec).as_matrix()
            t = rng.standard_normal(3)
            t /= np.linalg.norm(t)
            tx = np.array([[0,-t[2],t[1]],[t[2],0,-t[0]],[-t[1],t[0],0]])
            E  = tx @ R
            F  = K_inv.T @ E @ K_inv
            U, S, Vt = np.linalg.svd(F)
            S[2] = 0
            F = (U * S) @ Vt
            F /= np.linalg.norm(F) + 1e-12
            ests.append(FundamentalEstimate(
                frame_idx=i*60, F=F,
                pts0=np.zeros((100,2)), pts1=np.zeros((100,2)),
                n_inliers=150, inlier_ratio=0.65, mean_sampson=0.3,
            ))
        return ests

    def test_result_focal_length_within_5pct(self):
        f_true = 900.0
        ests   = self._synthetic_estimates(f_true)
        result = run_focal_estimation(ests, (1280, 720), verbose=False)
        err_pct = 100 * abs(result.f_px - f_true) / f_true
        assert err_pct < 5.0, f"f={result.f_px:.1f}, true={f_true}, err={err_pct:.1f}%"

    def test_result_K_is_valid_camera_matrix(self):
        ests   = self._synthetic_estimates()
        result = run_focal_estimation(ests, (1280, 720), verbose=False)
        K = result.K
        assert K.shape == (3, 3)
        assert K[2, 2] == pytest.approx(1.0)
        assert K[0, 1] == pytest.approx(0.0)   # zero skew
        assert K[0, 0] > 0 and K[1, 1] > 0
