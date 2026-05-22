"""
tests/test_bg_gabor_persistence.py
====================================
Regression tests for caching the Gabor bedding-energy model in bg.npz.

Why these exist
---------------
A bug shipped in v0.5.0: --texture-suppress at BG-build time computed
the Gabor model on a transient ForegroundDetector instance but never
persisted it. bg.npz only contained bg0/bg1/method, so at tracking
time --gabor-refine was a silent no-op unless the user also passed
--texture-suppress to every tracking run.

These tests verify the now-correct contract:

  1. Without compute_gabor(), bg.npz contains only bg0/bg1/method
     (backward compat with older files).
  2. After compute_gabor(), bg.npz also contains bg_gabor0/bg_gabor1.
  3. Loading a Gabor-cached bg.npz restores the model on the
     BackgroundModel.
  4. ForegroundDetector with a Gabor-cached BackgroundModel exposes
     the energy map at _bg_gabor regardless of the texture_suppress
     flag — so --gabor-refine works without --texture-suppress at
     tracking time.
  5. Older bg.npz files (no bg_gabor*) load without error.
"""
from __future__ import annotations

import numpy as np
import pytest


def _fake_bedding(h=200, w=200, seed=0):
    rng = np.random.default_rng(seed)
    return (60 + 30 * rng.standard_normal((h, w))).astype(np.float32)


class TestBackgroundModelGaborCaching:

    def test_default_save_omits_gabor(self, tmp_path):
        from rpimocap.detection.segment import BackgroundModel
        bg = BackgroundModel(_fake_bedding(), _fake_bedding(seed=1),
                             method="median")
        bg.save(tmp_path / "bg.npz")
        d = np.load(tmp_path / "bg.npz")
        assert "bg0" in d.files
        assert "bg1" in d.files
        assert "method" in d.files
        assert "bg_gabor0" not in d.files
        assert "bg_gabor1" not in d.files

    def test_compute_gabor_populates_fields(self):
        from rpimocap.detection.segment import BackgroundModel
        bg = BackgroundModel(_fake_bedding(), _fake_bedding(seed=1))
        assert bg.bg_gabor0 is None
        assert bg.bg_gabor1 is None
        bg.compute_gabor(lambdas=(8.0, 12.0), n_orientations=4)
        assert bg.bg_gabor0 is not None
        assert bg.bg_gabor1 is not None
        assert bg.bg_gabor0.shape == bg.bg0.shape
        assert bg.bg_gabor1.shape == bg.bg1.shape
        # Normalised to [0, 1] via the 99th percentile
        assert 0.0 <= bg.bg_gabor0.min()
        assert bg.bg_gabor0.max() <= 2.0   # 99-percentile-normalised, modest overshoot OK

    def test_save_and_load_roundtrip(self, tmp_path):
        from rpimocap.detection.segment import BackgroundModel
        bg = BackgroundModel(_fake_bedding(), _fake_bedding(seed=1))
        bg.compute_gabor(lambdas=(8.0,), n_orientations=4)
        bg.save(tmp_path / "bg.npz")

        d = np.load(tmp_path / "bg.npz")
        assert "bg_gabor0" in d.files
        assert "bg_gabor1" in d.files

        bg2 = BackgroundModel.from_npz(tmp_path / "bg.npz")
        assert bg2.bg_gabor0 is not None
        assert bg2.bg_gabor1 is not None
        np.testing.assert_allclose(bg2.bg_gabor0, bg.bg_gabor0)
        np.testing.assert_allclose(bg2.bg_gabor1, bg.bg_gabor1)

    def test_legacy_npz_loads_without_gabor(self, tmp_path):
        """An older bg.npz with only bg0/bg1/method must still load."""
        from rpimocap.detection.segment import BackgroundModel
        np.savez_compressed(
            tmp_path / "old_bg.npz",
            bg0=_fake_bedding(),
            bg1=_fake_bedding(seed=1),
            method=np.array("median"))
        bg = BackgroundModel.from_npz(tmp_path / "old_bg.npz")
        assert bg.bg_gabor0 is None
        assert bg.bg_gabor1 is None


class TestForegroundDetectorUsesCachedGabor:

    def test_detector_reuses_cached_gabor_without_texture_suppress(self):
        """If the BackgroundModel was loaded with a cached Gabor model,
        ForegroundDetector exposes it on _bg_gabor regardless of the
        texture_suppress flag — so --gabor-refine works at tracking
        time without re-passing --texture-suppress."""
        from rpimocap.detection.segment import (
            BackgroundModel, ForegroundDetector)
        bg = BackgroundModel(_fake_bedding(), _fake_bedding(seed=1))
        bg.compute_gabor(lambdas=(8.0,), n_orientations=4)

        # texture_suppress=False (default) — formerly this would have
        # left _bg_gabor empty. With caching, the detector uses what
        # the BG already brings.
        det = ForegroundDetector(bg, threshold=20, min_area_px=50,
                                 texture_suppress=False)
        assert 0 in det._bg_gabor
        assert 1 in det._bg_gabor
        assert det._bg_gabor[0].shape == bg.bg0.shape
        # texture_alpha must remain 0 (since suppression was off);
        # the cached map is for gabor_body_contour, not for diff
        # suppression.
        assert det._texture_alpha == 0.0

    def test_detector_computes_gabor_when_not_cached(self):
        """Older bg.npz path: no cached Gabor + texture_suppress=True
        → detector computes the model itself, unchanged behaviour."""
        from rpimocap.detection.segment import (
            BackgroundModel, ForegroundDetector)
        bg = BackgroundModel(_fake_bedding(), _fake_bedding(seed=1))
        # No compute_gabor() — bg.bg_gabor0/1 are None
        det = ForegroundDetector(bg, threshold=20, min_area_px=50,
                                 texture_suppress=True)
        assert 0 in det._bg_gabor
        assert 1 in det._bg_gabor

    def test_detector_no_gabor_when_neither(self):
        """No cache and texture_suppress=False → _bg_gabor is empty
        (default behaviour unchanged for users who never touch the
        feature)."""
        from rpimocap.detection.segment import (
            BackgroundModel, ForegroundDetector)
        bg = BackgroundModel(_fake_bedding(), _fake_bedding(seed=1))
        det = ForegroundDetector(bg, threshold=20, min_area_px=50,
                                 texture_suppress=False)
        assert det._bg_gabor == {}
