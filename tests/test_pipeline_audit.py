"""
tests/test_pipeline_audit.py
==============================
Regression tests for the two bugs uncovered in the segmentation
pipeline audit:

1. bg-adapt block crashed silently every frame because it called
   .astype() on a ForegroundResult object (not its .mask attribute).
   --bg-adapt-alpha was a no-op since the feature shipped.

2. KalmanTracker3D had no reset() method. SegmentTracker.track_sequence
   was zeroing .x and .initialised manually but leaving .P (covariance)
   from a previous run, so a second sequence inherited an over-confident
   filter that under-weighted new measurements.
"""
from __future__ import annotations

import numpy as np
import pytest


class TestBgAdaptUsesMaskNotResult:
    """Verify the bg-adapt block actually runs (vs silently swallowing
    AttributeError every frame as it did before)."""

    def test_background_model_update_accepts_mask_from_fg_result(self):
        """The fix routes fg.mask (a uint8 numpy array) through cv2.dilate.
        Confirm that the chain compiles and produces a bool mask."""
        import cv2
        from rpimocap.detection.segment import (
            BackgroundModel, ForegroundDetector)

        # Build a tiny detector to get a real ForegroundResult
        bg0 = np.full((100, 100), 100, dtype=np.float32)
        bg1 = np.full((100, 100), 100, dtype=np.float32)
        bg  = BackgroundModel(bg0, bg1, method="median")
        det = ForegroundDetector(bg, threshold=10, min_area_px=10)

        # Build a synthetic frame with a high-contrast blob to ensure
        # bg-sub triggers a foreground mask.
        frame = np.full((100, 100, 3), 100, dtype=np.uint8)
        cv2.circle(frame, (50, 50), 15, (200, 200, 200), -1)

        fg = det.detect(frame, 0)
        assert fg.mask is not None
        assert fg.mask.dtype == np.uint8

        # Reproduce the bg-adapt dilate step against fg.mask (the fix)
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))
        m = cv2.dilate(fg.mask.astype(np.uint8), kern).astype(bool)
        assert m.dtype == bool
        assert m.shape == (100, 100)

        # And feed it through BackgroundModel.update — no exception.
        bg.update(frame, frame, mask0=m, mask1=m, alpha=0.99)

    def test_old_code_path_would_have_crashed(self):
        """The ForegroundResult itself does NOT have .astype.
        This locks in why the bug was silent: AttributeError is exactly
        what the broad except in the bg-adapt block was catching."""
        import numpy as np
        from rpimocap.detection.segment import ForegroundResult

        r = ForegroundResult(
            mask=np.zeros((10, 10), np.uint8),
            blobs=[], frame_gray=np.zeros((10, 10), np.uint8),
            n_blobs=0)
        with pytest.raises(AttributeError):
            r.astype(np.uint8)


class TestKalmanReset:
    """A second sequence must start from the same fresh covariance the
    first one did — otherwise the converged P from the first sequence
    makes the filter ignore its own first valid observation."""

    def test_kalman_has_reset_method(self):
        from rpimocap.reconstruction.kalman import KalmanTracker3D
        kf = KalmanTracker3D()
        assert hasattr(kf, "reset")
        assert callable(kf.reset)

    def test_reset_restores_initial_state(self):
        """Step the filter until P shrinks, then reset, confirm
        P is back to its construction-time large-uncertainty diag."""
        from rpimocap.reconstruction.kalman import KalmanTracker3D
        kf = KalmanTracker3D(dt=1/25.0, sigma_a=2000.0, sigma_z=5.0)
        P_initial = kf.P.copy()

        # Run a clean constant-velocity trajectory; P should shrink.
        rng = np.random.default_rng(0)
        for i in range(100):
            z = np.array([float(i), 0.0, 100.0]) + rng.normal(0, 1, 3)
            kf.step(z)
        assert kf.initialised
        assert kf.P[0, 0] < P_initial[0, 0]   # converged

        # Reset
        kf.reset()
        assert kf.initialised is False
        np.testing.assert_allclose(kf.P, P_initial)
        np.testing.assert_allclose(kf.x, np.zeros(6))

    def test_reset_with_initial_state(self):
        from rpimocap.reconstruction.kalman import KalmanTracker3D
        kf = KalmanTracker3D()
        seed = np.array([1, 2, 3, 0, 0, 0], dtype=np.float64)
        kf.reset(seed)
        assert kf.initialised is True
        np.testing.assert_allclose(kf.x, seed)

    def test_two_sequences_independent(self):
        """The whole point of having reset(): a second sequence's first
        valid observation should produce essentially the same posterior
        as the first sequence's first valid observation did. Without
        reset() the converged P from sequence 1 would crush sequence 2's
        first observation."""
        from rpimocap.reconstruction.kalman import KalmanTracker3D

        kf = KalmanTracker3D(dt=1/25.0, sigma_a=2000.0, sigma_z=5.0)
        # Sequence 1 — long, well-converged
        for i in range(200):
            kf.step(np.array([float(i), 0.0, 100.0]))
        assert kf.P[0, 0] < 100.0   # converged

        # Without reset: sequence 2's first step would barely move x[:3]
        # toward the new measurement. With reset: it snaps to it.
        kf.reset()
        kf.step(np.array([500.0, 500.0, 500.0]))
        # First step bootstraps state directly to measurement
        np.testing.assert_allclose(kf.x[:3], np.array([500., 500., 500.]),
                                   atol=1e-6)
