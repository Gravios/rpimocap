"""Tests for the coarse + intensity background-model additions (0077).

``illumination_intensity`` supplies the illumination-flattened intensity
channel prepended to the pooled Gabor descriptor by ``session_stats
--intensity``. These tests cover the flattening behaviour and that the
channel is a genuine outlier signal for a bright object on a graded
background (its intended role), plus the descriptor-augmentation shape.
"""
import numpy as np

from rpimocap.detection.texture_distance import (
    illumination_intensity, dense_gabor_descriptor)
from rpimocap.detection.rat_texture import build_gabor_kernels


def _graded_bedding_with_blob(H=400, W=500, seed=0):
    """A left-right illumination gradient over noisy 'bedding', plus a
    bright blob (the 'rat')."""
    rng = np.random.default_rng(seed)
    grad = np.linspace(40, 150, W)[None, :].repeat(H, 0)      # IR falloff
    bedding = grad + rng.normal(0, 8, (H, W))                 # granular texture
    img = bedding.copy()
    img[180:250, 220:300] += 55                               # bright blob
    return np.clip(img, 0, 255).astype(np.float32), (285, 260, 215)  # (x1,x0? -> blob box)


class TestIlluminationIntensity:

    def test_shape_and_dtype(self):
        g = np.full((64, 80), 100, np.float32)
        I = illumination_intensity(g, illum_sigma=31.0, smooth_k=7)
        assert I.shape == (64, 80)
        assert I.dtype == np.float32
        assert np.isfinite(I).all()

    def test_flattens_illumination_gradient(self):
        # a pure gradient (no texture) should come out roughly flat
        W = 400
        grad = np.linspace(30, 160, W)[None, :].repeat(200, 0).astype(np.float32)
        I = illumination_intensity(grad, illum_sigma=81.0, smooth_k=7)
        # coefficient of variation drops sharply after flattening
        cv_before = grad.std() / grad.mean()
        cv_after = I.std() / I.mean()
        assert cv_after < cv_before * 0.4

    def test_bright_blob_is_an_outlier_after_flattening(self):
        img, _ = _graded_bedding_with_blob()
        I = illumination_intensity(img, illum_sigma=81.0, smooth_k=31)
        blob = I[190:240, 230:290].mean()
        # bedding excluding the blob region
        m = np.ones(I.shape, bool); m[170:260, 210:310] = False
        mu, sd = I[m].mean(), I[m].std()
        # the blob sits well above the flattened bedding distribution
        assert (blob - mu) / sd > 3.0

    def test_smooth_k_zero_disables_smoothing(self):
        g = np.full((40, 40), 90, np.float32) + 1.0
        I0 = illumination_intensity(g, illum_sigma=21.0, smooth_k=0)
        assert I0.shape == (40, 40)


class TestIntensityAugmentedDescriptor:

    def test_prepending_intensity_adds_one_channel(self):
        img, _ = _graded_bedding_with_blob()
        kernels = build_gabor_kernels([o * np.pi / 8 for o in range(8)], [9, 17])
        desc = np.asarray(dense_gabor_descriptor(img, kernels, 8, 2, smooth_k=7))
        ich = illumination_intensity(img, smooth_k=7)
        aug = np.concatenate([ich[None], desc], axis=0)
        assert aug.shape[0] == desc.shape[0] + 1
        assert aug.shape[1:] == desc.shape[1:]
        # channel 0 is the intensity map
        assert np.allclose(aug[0], ich)
