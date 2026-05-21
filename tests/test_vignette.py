"""
tests/test_vignette.py
=======================
Unit tests for rpimocap.detection.vignette.

Verifies that:
  - apply_flat_field flattens a synthetic radial darkening
  - synthesize_flat_field recovers a known vignette profile
  - load_flat_field round-trips PNG and NPZ files
  - degenerate flats are caught with a clean ValueError
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from rpimocap.detection.vignette import (
    apply_flat_field,
    load_flat_field,
    synthesize_flat_field,
)


def _synthetic_vignette(h=240, w=320, falloff=0.5):
    """Build a known radial darkening pattern; mean 1 after normalisation."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    diag2  = cx**2 + cy**2
    r2 = ((xx - cx) ** 2 + (yy - cy) ** 2) / diag2
    ff = 1.0 - falloff * r2
    return ff / ff.mean()


class TestApplyFlatField:

    def test_flattens_darkened_uniform_scene(self):
        ff = _synthetic_vignette(falloff=0.5)
        # A uniformly grey scene attenuated by the vignette
        scene = (128 * ff).astype(np.uint8)
        corrected = apply_flat_field(scene, ff, clip=False)
        # After correction the image should be nearly uniform
        rng = corrected.max() - corrected.min()
        assert rng < 5.0, (
            f"corrected uniform scene has range {rng:.2f}, expected ~0")

    def test_preserves_global_brightness(self):
        ff = _synthetic_vignette(falloff=0.4)
        scene = np.full((240, 320), 100.0, dtype=np.float32) * ff
        corrected = apply_flat_field(scene, ff, clip=False)
        # Mean is preserved exactly
        np.testing.assert_allclose(corrected.mean(), scene.mean(), rtol=1e-3)

    def test_clip_returns_uint8(self):
        ff = _synthetic_vignette(falloff=0.3)
        scene = (200 * ff).astype(np.uint8)
        corrected = apply_flat_field(scene, ff, clip=True)
        assert corrected.dtype == np.uint8
        assert corrected.min() >= 0
        assert corrected.max() <= 255

    def test_shape_mismatch_raises(self):
        ff = _synthetic_vignette(falloff=0.3)
        wrong = np.zeros((100, 100), dtype=np.uint8)
        with pytest.raises(ValueError):
            apply_flat_field(wrong, ff)

    def test_three_channel_passthrough(self):
        ff = _synthetic_vignette(falloff=0.3)
        scene = np.dstack([
            (128 * ff).astype(np.uint8),
            (100 * ff).astype(np.uint8),
            (80  * ff).astype(np.uint8)])
        corrected = apply_flat_field(scene, ff, clip=False)
        # Each channel should be flatter (range smaller) than original
        for ch in range(3):
            r_in  = scene[..., ch].astype(float).max() - scene[..., ch].astype(float).min()
            r_out = corrected[..., ch].max() - corrected[..., ch].min()
            assert r_out < 0.5 * r_in


class TestSynthesizeFlatField:

    def test_recovers_known_vignette(self):
        truth = _synthetic_vignette(falloff=0.6, h=240, w=320)
        # Background as a uniform scene under that vignette
        bg = (180.0 * truth).astype(np.float32)
        ff = synthesize_flat_field(bg, poly_order=4, downsample=4)
        assert ff.shape == bg.shape
        np.testing.assert_allclose(ff.mean(), 1.0, atol=1e-2)
        # The fit should be close to the ground truth shape
        # (after normalisation)
        err = np.abs(ff - truth)
        assert err.max() < 0.02, f"vignette fit max err {err.max():.4f} too large"

    def test_rejects_non_2d_input(self):
        with pytest.raises(ValueError):
            synthesize_flat_field(np.zeros((10, 10, 3), dtype=np.uint8))


class TestLoadFlatField:

    def test_round_trip_npz(self, tmp_path):
        truth = _synthetic_vignette(falloff=0.4, h=120, w=160)
        path  = tmp_path / "flat.npz"
        np.savez(path, flat=truth)
        ff = load_flat_field(path)
        np.testing.assert_allclose(ff.mean(), 1.0, atol=1e-6)
        np.testing.assert_allclose(ff, truth / truth.mean(), atol=1e-5)

    def test_round_trip_png(self, tmp_path):
        truth = _synthetic_vignette(falloff=0.4, h=120, w=160)
        # Scale to uint8 for PNG round-trip; the test tolerates the
        # quantisation loss as long as the gross profile survives.
        img = np.clip(truth * 100.0, 0, 255).astype(np.uint8)
        path = tmp_path / "flat.png"
        cv2.imwrite(str(path), img)
        ff = load_flat_field(path)
        np.testing.assert_allclose(ff.mean(), 1.0, atol=1e-2)
        # Sample a few points: centre should be brightest, corners darkest
        assert ff[60, 80] > ff[0, 0]
        assert ff[60, 80] > ff[-1, -1]

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_flat_field(tmp_path / "does_not_exist.npz")

    def test_zero_mean_flat_rejected(self, tmp_path):
        zero = np.zeros((50, 50), dtype=np.float32)
        path = tmp_path / "zero.npz"
        np.savez(path, flat=zero)
        with pytest.raises(ValueError):
            load_flat_field(path)
