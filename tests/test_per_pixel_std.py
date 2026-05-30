"""
tests/test_per_pixel_std.py
============================
Per-pixel std-based background subtraction. Captures the user's
intuition: trouble regions (cable mount, headstage glints, acrylic
wall edges) have high but predictable variance in the bg sample.
A global threshold catches normal pixel-noise variation there as
'foreground'. A per-pixel σ model lets the threshold be μ + k·σ
locally, so trouble regions need a larger excursion to count.

Also verifies online adaptation: σ² updates with EMA on (frame - μ)²
on the same non-foreground-mask condition as μ does.
"""
from __future__ import annotations

import numpy as np
import pytest

from rpimocap.detection.segment import (
    BackgroundModel, ForegroundDetector)


class TestPerPixelStdInModel:

    def test_model_accepts_and_persists_std(self, tmp_path):
        """std0/std1 round-trip through save/load."""
        bg0 = np.full((40, 60), 50.0, dtype=np.float32)
        bg1 = np.full((40, 60), 50.0, dtype=np.float32)
        s0  = np.full((40, 60), 2.0, dtype=np.float32)
        s1  = np.full((40, 60), 3.0, dtype=np.float32)
        m = BackgroundModel(bg0, bg1, std0=s0, std1=s1)
        m.save(tmp_path / "bg.npz")
        m2 = BackgroundModel.from_npz(tmp_path / "bg.npz")
        assert m2.std0 is not None
        assert m2.std1 is not None
        assert np.allclose(m2.std0, 2.0)
        assert np.allclose(m2.std1, 3.0)

    def test_old_npz_without_std_loads_fine(self, tmp_path):
        """Backward compat: an npz lacking std0/std1 still loads,
        std fields default to None."""
        bg0 = np.full((40, 60), 50.0, dtype=np.float32)
        bg1 = np.full((40, 60), 50.0, dtype=np.float32)
        np.savez_compressed(tmp_path / "old.npz",
                            bg0=bg0, bg1=bg1, method="median")
        m = BackgroundModel.from_npz(tmp_path / "old.npz")
        assert m.std0 is None
        assert m.std1 is None


class TestMahalanobisDetection:

    def _bg_with_two_regions(self):
        """BG with a 'quiet' region (low σ) and a 'trouble' region
        (high σ) — simulates cable-mount vs floor."""
        shape = (60, 120)
        bg0 = np.full(shape, 50.0, dtype=np.float32)
        bg1 = np.full(shape, 50.0, dtype=np.float32)
        std0 = np.full(shape, 1.0, dtype=np.float32)   # quiet
        std0[10:50, 60:110] = 20.0                      # trouble (high σ)
        std1 = np.full(shape, 1.0, dtype=np.float32)
        std1[10:50, 60:110] = 20.0
        return BackgroundModel(bg0, bg1, std0=std0, std1=std1)

    def test_trouble_region_suppressed_at_normal_variation(self):
        """A frame where every pixel deviates by ~10 intensity units
        from the bg mean: in the quiet region (σ=1) this is 10σ — a
        clear foreground. In the trouble region (σ=20) this is 0.5σ
        — normal noise, NOT foreground. The Mahalanobis detector
        catches only the quiet region; the absolute detector with
        threshold=8 would flag both."""
        bg = self._bg_with_two_regions()

        # frame: bg + 10 everywhere
        frame = (bg.bg0 + 10).astype(np.uint8)

        # Absolute mode with threshold=8: flags everything
        det_abs = ForegroundDetector(bg, threshold=8,
                                       min_area_px=10, morph_k=3,
                                       mahalanobis_k=0.0)
        r_abs = det_abs.detect(frame, cam=0)
        # quiet region: should be foreground
        quiet_abs   = r_abs.mask[10:50, 10:50].sum() / 255
        # trouble region: also flagged (the bug)
        trouble_abs = r_abs.mask[10:50, 60:110].sum() / 255
        assert quiet_abs > 0
        assert trouble_abs > 0, "absolute mode flags trouble region too"

        # Mahalanobis mode with k=3: trouble region (σ=20) needs
        # a 60-unit excursion. The 10-unit excursion fails the gate.
        det_mah = ForegroundDetector(bg, threshold=8,
                                       min_area_px=10, morph_k=3,
                                       mahalanobis_k=3.0,
                                       sigma_floor=1.0)
        r_mah = det_mah.detect(frame, cam=0)
        quiet_mah   = r_mah.mask[10:50, 10:50].sum() / 255
        trouble_mah = r_mah.mask[10:50, 60:110].sum() / 255

        assert quiet_mah > 0, (
            "quiet region (10σ at σ=1) should still be foreground")
        # At the σ-boundary the morph operations can bleed a few
        # quiet-region pixels into the trouble region; allow <5% of
        # the trouble area's pixel count.
        trouble_area = (50 - 10) * (110 - 60)   # 2000
        assert trouble_mah < 0.05 * trouble_area, (
            f"trouble region (0.5σ at σ=20) should be effectively "
            f"empty; got {trouble_mah} pixels of {trouble_area}")

    def test_real_event_still_detected_in_trouble_region(self):
        """An ACTUAL bright event in the trouble region — large
        deviation from bg — must still be detected by Mahalanobis
        mode. The gate is supposed to filter noise, not signal."""
        bg = self._bg_with_two_regions()
        # frame: huge bright blob inside the trouble region (e.g.
        # rat moves into the cable-mount area)
        frame = bg.bg0.copy().astype(np.uint8)
        frame[20:40, 75:95] = 200  # 150-unit excursion = 7.5σ

        det = ForegroundDetector(bg, threshold=8,
                                  min_area_px=10, morph_k=3,
                                  mahalanobis_k=3.0, sigma_floor=1.0)
        r = det.detect(frame, cam=0)
        # The bright blob should pass: 150 / 20 = 7.5σ > k=3
        assert r.n_blobs >= 1, (
            "real bright event (7.5σ) should be detected even in "
            "trouble region")

    def test_falls_back_to_absolute_when_no_std(self):
        """An old bg.npz has no std → mahalanobis_k flag is a silent
        no-op, absolute threshold applies as before."""
        bg = BackgroundModel(
            bg0=np.full((40, 60), 50, dtype=np.float32),
            bg1=np.full((40, 60), 50, dtype=np.float32),
        )
        assert bg.std0 is None
        frame = (bg.bg0 + 10).astype(np.uint8)
        det = ForegroundDetector(bg, threshold=8,
                                  min_area_px=10, morph_k=3,
                                  mahalanobis_k=3.0)
        r = det.detect(frame, cam=0)
        # Absolute threshold of 8 against deviation of 10 → foreground
        assert r.mask.any()


class TestOnlineStdAdaptation:

    def test_sigma_increases_in_noisy_region(self):
        """When bg-adapt is updating, σ² should track new variance
        in pixels that suddenly become more noisy (e.g. cable starts
        swinging). EMA on squared residuals does this."""
        shape = (40, 60)
        bg0 = np.full(shape, 50.0, dtype=np.float32)
        bg1 = np.full(shape, 50.0, dtype=np.float32)
        std0 = np.full(shape, 1.0, dtype=np.float32)
        std1 = np.full(shape, 1.0, dtype=np.float32)
        bg = BackgroundModel(bg0, bg1, std0=std0, std1=std1)

        rng = np.random.default_rng(42)
        # 50 frames with σ=10 noise: σ² should rise toward 100
        for _ in range(50):
            f0 = bg.bg0 + rng.normal(0, 10, shape).astype(np.float32)
            f1 = bg.bg1 + rng.normal(0, 10, shape).astype(np.float32)
            bg.update(f0, f1, alpha=0.9)
        # σ should have risen significantly above the initial 1.0
        assert bg.std0.mean() > 3.0, (
            f"σ should rise from 1.0 toward 10 in a noisy stream; "
            f"got mean {bg.std0.mean():.2f}")

    def test_sigma_does_not_change_at_animal_pixels(self):
        """Animal-mask pixels are excluded from both μ AND σ²
        updates, so the rat doesn't bake its noise into the bg σ."""
        shape = (40, 60)
        bg = BackgroundModel(
            bg0=np.full(shape, 50.0, dtype=np.float32),
            bg1=np.full(shape, 50.0, dtype=np.float32),
            std0=np.full(shape, 1.0, dtype=np.float32),
            std1=np.full(shape, 1.0, dtype=np.float32),
        )
        # Animal mask: top-left 20x30 area is the rat — exclude from updates
        animal_mask = np.zeros(shape, dtype=bool)
        animal_mask[:20, :30] = True
        rng = np.random.default_rng(1)
        for _ in range(50):
            f0 = bg.bg0 + rng.normal(0, 10, shape).astype(np.float32)
            f1 = bg.bg1 + rng.normal(0, 10, shape).astype(np.float32)
            bg.update(f0, f1,
                      mask0=animal_mask, mask1=animal_mask, alpha=0.9)
        # σ in the animal region should still be ~1.0 (never updated)
        animal_sigma = bg.std0[:20, :30].mean()
        non_animal_sigma = bg.std0[25:, 35:].mean()
        assert animal_sigma < 1.5
        assert non_animal_sigma > 3.0, (
            f"non-animal region σ should rise; got {non_animal_sigma:.2f}")
