"""
tests/test_luminance_correction.py
===================================
Per-frame multiplicative luminance correction on the background
inside ForegroundDetector.detect().

Each frame, estimate g such that g*bg ≈ frame over the non-animal
pixels of the arena ROI, then use g*bg as the effective background.
This handles fast IR illumination drift (single-frame fluctuations)
that --bg-adapt-alpha is too slow to track.

Key correctness properties:
  1. With luminance_correct=False, the detector behavior is
     identical to the pre-patch behavior (back-compat).
  2. When the frame is bg * g_true (no animal), the estimated g
     equals g_true to within numerical noise.
  3. With an animal-shaped bright outlier present, the median
     ratio still recovers the global g (animal is rejected by clip).
  4. The original BackgroundModel is NOT mutated by the correction.
  5. g is exposed per camera via self._last_g for diagnostics.
"""
from __future__ import annotations

import numpy as np

from rpimocap.detection.segment import (
    BackgroundModel, ForegroundDetector)


def _empty_bg(shape=(180, 240), bg_value=100):
    """Build a flat background — uniform intensity = bg_value."""
    return BackgroundModel(
        bg0=np.full(shape, bg_value, dtype=np.float32),
        bg1=np.full(shape, bg_value, dtype=np.float32),
    )


def _structured_bg(shape=(180, 240), seed=0):
    """A bg with spatial variation — more realistic than uniform."""
    rng = np.random.RandomState(seed)
    bg = rng.uniform(40, 200, size=shape).astype(np.float32)
    # Smooth to imitate a real low-frequency bg
    import cv2
    bg = cv2.GaussianBlur(bg, (9, 9), sigmaX=3)
    return BackgroundModel(bg0=bg.copy(), bg1=bg.copy())


class TestLuminanceCorrectionOff:

    def test_disabled_is_back_compat(self):
        """With luminance_correct=False, the detector behavior is
        unchanged. The g cache stays at 1.0."""
        det = ForegroundDetector(_empty_bg(), threshold=30,
                                  min_area_px=10, morph_k=3,
                                  luminance_correct=False)
        # Frame = bg + bright blob
        frame = np.full((180, 240), 100, dtype=np.uint8)
        frame[80:100, 100:120] = 200
        det.detect(frame, cam=0)
        assert det._last_g[0] == 1.0


class TestLuminanceCorrectionRecoversG:

    def test_uniform_brighter_frame_g_above_one(self):
        """If the frame is uniformly 1.5x brighter than bg, the
        estimated g should be ~1.5 (no animal to throw it off)."""
        det = ForegroundDetector(_empty_bg(bg_value=100),
                                  threshold=30, min_area_px=10,
                                  morph_k=3,
                                  luminance_correct=True)
        frame = np.full((180, 240), 150, dtype=np.uint8)  # 1.5x bg
        det.detect(frame, cam=0)
        assert abs(det._last_g[0] - 1.5) < 0.01, (
            f"expected g ≈ 1.5, got {det._last_g[0]:.4f}")

    def test_uniform_dimmer_frame_g_below_one(self):
        det = ForegroundDetector(_empty_bg(bg_value=120),
                                  threshold=30, min_area_px=10,
                                  morph_k=3,
                                  luminance_correct=True)
        # 0.83x bg
        frame = np.full((180, 240), 100, dtype=np.uint8)
        det.detect(frame, cam=0)
        expected = 100.0 / 120.0
        assert abs(det._last_g[0] - expected) < 0.01

    def test_g_is_one_when_frame_equals_bg(self):
        det = ForegroundDetector(_empty_bg(bg_value=100),
                                  threshold=30, min_area_px=10,
                                  morph_k=3,
                                  luminance_correct=True)
        frame = np.full((180, 240), 100, dtype=np.uint8)
        det.detect(frame, cam=0)
        assert abs(det._last_g[0] - 1.0) < 0.01


class TestLuminanceCorrectionRobustToAnimal:

    def test_animal_outlier_rejected_by_clip(self):
        """A bright animal-shaped blob covers ~5% of pixels at 5x
        bg intensity. With the default clip range [0.5, 2.0], the
        rat pixels are excluded from the median, so the recovered
        g should reflect the surrounding (non-animal) pixels only."""
        det = ForegroundDetector(_empty_bg(bg_value=100),
                                  threshold=30, min_area_px=10,
                                  morph_k=3,
                                  luminance_correct=True)
        # Frame: 1.2x brighter than bg globally, plus a bright
        # blob at 250 (way above the clip ceiling of 100*2 = 200).
        frame = np.full((180, 240), 120, dtype=np.uint8)
        frame[60:120, 80:160] = 250   # animal-shaped outlier
        det.detect(frame, cam=0)
        # Should recover g ≈ 1.2 from the non-animal pixels
        assert abs(det._last_g[0] - 1.2) < 0.02, (
            f"expected g ≈ 1.2 (with animal pixels rejected), "
            f"got {det._last_g[0]:.4f}")

    def test_animal_outlier_dim_rejected_by_clip(self):
        """A dim outlier (below the 0.5*bg clip floor) is also
        rejected. Tests the lo-clip branch."""
        det = ForegroundDetector(_empty_bg(bg_value=200),
                                  threshold=30, min_area_px=10,
                                  morph_k=3,
                                  luminance_correct=True)
        # 1.1x global, plus a very dark blob (40 = 0.2 * 200)
        frame = np.full((180, 240), 220, dtype=np.uint8)
        frame[60:120, 80:160] = 40
        det.detect(frame, cam=0)
        assert abs(det._last_g[0] - 1.1) < 0.02


class TestLuminanceCorrectionImproveDetection:
    """Functional test: with a globally-drifted frame, the rat is
    only detectable when luminance correction is enabled."""

    def test_drift_breaks_naive_bgsub_then_correction_fixes_it(self):
        bg_val = 100
        det_off = ForegroundDetector(_empty_bg(bg_value=bg_val),
                                      threshold=30,    # absolute thr
                                      min_area_px=10, morph_k=3,
                                      luminance_correct=False)
        det_on  = ForegroundDetector(_empty_bg(bg_value=bg_val),
                                      threshold=30,
                                      min_area_px=10, morph_k=3,
                                      luminance_correct=True)
        # Simulate IR illumination drift: whole frame uniformly 50%
        # brighter than bg. The rat is a blob 60 grey-levels brighter
        # than the (now drifted) local background.
        global_g = 1.5
        bright_frame_no_rat = int(bg_val * global_g)   # 150
        bright_frame_rat    = bright_frame_no_rat + 60  # 210
        frame = np.full((180, 240), bright_frame_no_rat,
                         dtype=np.uint8)
        frame[60:120, 80:160] = bright_frame_rat

        r_off = det_off.detect(frame.copy(), cam=0)
        r_on  = det_on.detect(frame.copy(), cam=0)

        # WITHOUT correction: the diff is (frame - bg) = 50 in the
        # bg pixels, 110 in rat pixels. Threshold of 30 fires on
        # EVERY pixel — the whole frame is "foreground". So the
        # labelled CC is huge, max_area filter (default None or
        # 25k) might reject it, or it becomes a single giant blob.
        n_fg_off = int((r_off.mask > 0).sum())

        # WITH correction: g ≈ 1.5, corrected bg ≈ 150. Diff is
        # ~0 in bg pixels, ~60 in rat pixels. Only the rat blob
        # exceeds threshold.
        n_fg_on = int((r_on.mask > 0).sum())

        # The corrected detector should produce a much smaller
        # foreground area — only the rat, not the entire frame.
        assert n_fg_on < n_fg_off, (
            f"luminance correction should shrink the foreground "
            f"area; got n_fg_on={n_fg_on}, n_fg_off={n_fg_off}")
        # And the corrected mask should still pick up the rat blob
        # (somewhere in the 60:120 × 80:160 region)
        assert r_on.mask[60:120, 80:160].sum() > 0


class TestLuminanceCorrectionImmutability:

    def test_bg_model_not_mutated(self):
        """The BackgroundModel's bg0/bg1 arrays must not be
        modified by per-frame correction. The correction is local
        to each detect() call."""
        bg_model = _empty_bg(bg_value=100)
        bg_before = bg_model.bg0.copy()
        det = ForegroundDetector(bg_model, threshold=30,
                                  min_area_px=10, morph_k=3,
                                  luminance_correct=True)
        frame = np.full((180, 240), 180, dtype=np.uint8)  # 1.8x
        det.detect(frame, cam=0)
        np.testing.assert_array_equal(bg_model.bg0, bg_before)


class TestLuminanceCorrectionStructuredBg:
    """Verify correction works on a non-uniform bg with real
    spatial variation — more realistic than the flat-bg cases."""

    def test_g_recovery_with_textured_bg(self):
        bg_model = _structured_bg(shape=(180, 240), seed=42)
        det = ForegroundDetector(bg_model, threshold=30,
                                  min_area_px=10, morph_k=3,
                                  luminance_correct=True)
        # Build a frame that's exactly 1.3x the bg (clipped at 255)
        g_true = 1.3
        frame = np.clip(bg_model.bg0 * g_true, 0, 255).astype(np.uint8)
        det.detect(frame, cam=0)
        # The clip at 255 affects the brightest bg pixels — but
        # they should still be in the median's neighborhood.
        assert abs(det._last_g[0] - g_true) < 0.05, (
            f"expected g ≈ {g_true}, got {det._last_g[0]:.4f}")


class TestLuminanceCorrectionPerCameraIndependent:

    def test_g_independent_per_camera(self):
        bg = BackgroundModel(
            bg0=np.full((180, 240), 100, dtype=np.float32),
            bg1=np.full((180, 240), 100, dtype=np.float32))
        det = ForegroundDetector(bg, threshold=30,
                                  min_area_px=10, morph_k=3,
                                  luminance_correct=True)
        # cam 0 sees a 1.4x bright frame
        det.detect(np.full((180, 240), 140, dtype=np.uint8), cam=0)
        # cam 1 sees a 0.8x dim frame
        det.detect(np.full((180, 240), 80, dtype=np.uint8), cam=1)
        assert abs(det._last_g[0] - 1.4) < 0.01
        assert abs(det._last_g[1] - 0.8) < 0.01
