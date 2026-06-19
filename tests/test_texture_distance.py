"""
tests/test_texture_distance.py
==============================
Texture-change foreground detection (diagnostic module). Validates
the dense descriptor, the background texture model (Welford mean +
std), and the distance map that should light up where texture
changes while staying dark on static features.
"""
from __future__ import annotations

import numpy as np
import cv2

from rpimocap.detection.rat_texture import build_gabor_kernels
from rpimocap.detection.texture_distance import (
    dense_gabor_descriptor, BackgroundTextureModel,
    build_background_texture_model, texture_distance_map,
    threshold_distance_map, colorize_distance_map)


N_ORIENT = 4
SCALES = [5, 9, 13]
ORIENTATIONS = [i * np.pi / N_ORIENT for i in range(N_ORIENT)]
KERNELS = build_gabor_kernels(ORIENTATIONS, SCALES)
N_SCALES = len(SCALES)


def _textured_bg(shape=(120, 160), rng_seed=0):
    """A textured background: fine random noise (mimics bedding)."""
    rng = np.random.RandomState(rng_seed)
    # Bedding-like: medium-frequency texture
    base = rng.randint(70, 110, shape).astype(np.uint8)
    return cv2.GaussianBlur(base, (3, 3), 0)


def _smooth_bright_patch(frame, box=(40, 90, 50, 110), intensity=200):
    """Paint a smooth bright patch (mimics fur: bright + low local
    texture compared to bedding)."""
    f = frame.copy()
    y0, y1, x0, x1 = box
    f[y0:y1, x0:x1] = intensity
    return f


class TestDenseDescriptor:

    def test_shape_rotation_invariant(self):
        frame = _textured_bg()
        desc = dense_gabor_descriptor(
            frame, KERNELS, N_ORIENT, N_SCALES,
            smooth_k=7, rotation_invariant=True)
        # 3 features per scale
        assert desc.shape == (3 * N_SCALES, frame.shape[0],
                              frame.shape[1])
        assert desc.dtype == np.float32

    def test_shape_directional(self):
        frame = _textured_bg()
        desc = dense_gabor_descriptor(
            frame, KERNELS, N_ORIENT, N_SCALES,
            smooth_k=7, rotation_invariant=False)
        assert desc.shape == (N_ORIENT * N_SCALES, frame.shape[0],
                              frame.shape[1])

    def test_descriptor_differs_texture_vs_smooth(self):
        """Different surface types produce measurably different
        descriptors — this difference is what drives the distance
        map. (Direction of the difference depends on the specific
        textures; what matters for foreground detection is that a
        material change registers as a descriptor change.)"""
        frame = _textured_bg()
        frame = _smooth_bright_patch(frame, box=(20, 100, 30, 130),
                                      intensity=200)
        desc = dense_gabor_descriptor(
            frame, KERNELS, N_ORIENT, N_SCALES,
            smooth_k=7, rotation_invariant=True)
        smooth_region = desc[:, 50:70, 65:95].mean(axis=(1, 2))
        textured_region = desc[:, 5:15, 5:25].mean(axis=(1, 2))
        # The two descriptors must be meaningfully different
        rel_diff = (np.abs(smooth_region - textured_region).mean()
                    / (textured_region.mean() + 1e-6))
        assert rel_diff > 0.2, (
            f"descriptors should differ by >20%; got {rel_diff:.2%}")


class TestBackgroundTextureModel:

    def test_welford_mean_matches_numpy(self):
        """The Welford accumulator's mean should match np.mean over
        the same frames."""
        rng = np.random.RandomState(1)
        frames = [_textured_bg(rng_seed=s) for s in range(8)]
        descs = [dense_gabor_descriptor(
            f, KERNELS, N_ORIENT, N_SCALES, smooth_k=7) for f in frames]
        model = BackgroundTextureModel()
        for d in descs:
            model.accumulate(d)
        model.finalize()
        expected_mean = np.mean(np.stack(descs, axis=0), axis=0)
        assert np.allclose(model.mean, expected_mean, atol=1e-3)

    def test_welford_std_matches_numpy(self):
        rng = np.random.RandomState(2)
        frames = [_textured_bg(rng_seed=s) for s in range(8)]
        descs = [dense_gabor_descriptor(
            f, KERNELS, N_ORIENT, N_SCALES, smooth_k=7) for f in frames]
        model = BackgroundTextureModel()
        for d in descs:
            model.accumulate(d)
        model.finalize(std_floor=0.0)
        expected_std = np.std(np.stack(descs, axis=0), axis=0, ddof=1)
        assert np.allclose(model.std, expected_std, atol=1e-3)

    def test_finalize_requires_frames(self):
        model = BackgroundTextureModel()
        try:
            model.finalize()
            assert False, "should have raised"
        except RuntimeError:
            pass

    def test_save_load_roundtrip(self, tmp_path):
        frames = [_textured_bg(rng_seed=s) for s in range(5)]
        model = build_background_texture_model(
            frames, KERNELS, N_ORIENT, N_SCALES, smooth_k=7)
        p = str(tmp_path / "texmodel.npz")
        model.save(p)
        loaded = BackgroundTextureModel.load(p)
        assert np.allclose(loaded.mean, model.mean)
        assert np.allclose(loaded.std, model.std)
        assert loaded.n == model.n


class TestTextureDistanceMap:

    def test_static_background_low_distance(self):
        """A frame identical (in texture distribution) to the
        background should produce a low distance everywhere."""
        frames = [_textured_bg(rng_seed=s) for s in range(10)]
        model = build_background_texture_model(
            frames, KERNELS, N_ORIENT, N_SCALES, smooth_k=7)
        # A NEW background-like frame (same distribution, unseen seed)
        test_frame = _textured_bg(rng_seed=99)
        dist = texture_distance_map(
            test_frame, model, KERNELS, N_ORIENT, N_SCALES,
            smooth_k=7, post_smooth_k=9)
        # Distance should be modest everywhere (same texture family)
        assert float(np.percentile(dist, 95)) < 5.0

    def test_texture_change_lights_up(self):
        """Adding a smooth bright patch (fur-like) to a textured
        background should produce HIGH distance in that region and
        low distance elsewhere."""
        bg_frames = [_textured_bg(rng_seed=s) for s in range(10)]
        model = build_background_texture_model(
            bg_frames, KERNELS, N_ORIENT, N_SCALES, smooth_k=7)
        # Test frame: background + smooth bright patch
        test_frame = _smooth_bright_patch(
            _textured_bg(rng_seed=50), box=(40, 90, 50, 110))
        dist = texture_distance_map(
            test_frame, model, KERNELS, N_ORIENT, N_SCALES,
            smooth_k=7, post_smooth_k=9)
        # Distance inside the patch should exceed distance in an
        # untouched background corner
        patch_dist = float(dist[55:75, 65:95].mean())
        corner_dist = float(dist[5:25, 5:35].mean())
        assert patch_dist > corner_dist, (
            f"texture-changed patch {patch_dist:.2f} should exceed "
            f"untouched corner {corner_dist:.2f}")

    def test_roi_mask_zeroes_outside(self):
        frames = [_textured_bg(rng_seed=s) for s in range(5)]
        model = build_background_texture_model(
            frames, KERNELS, N_ORIENT, N_SCALES, smooth_k=7)
        test_frame = _textured_bg(rng_seed=7)
        roi = np.zeros(test_frame.shape, dtype=np.uint8)
        roi[30:90, 40:120] = 255
        dist = texture_distance_map(
            test_frame, model, KERNELS, N_ORIENT, N_SCALES,
            smooth_k=7, roi_mask=roi, post_smooth_k=0)
        # Outside ROI must be exactly 0
        assert float(dist[:20, :20].max()) == 0.0


class TestThresholdDistanceMap:

    def test_threshold_isolates_changed_region(self):
        bg_frames = [_textured_bg(rng_seed=s) for s in range(10)]
        model = build_background_texture_model(
            bg_frames, KERNELS, N_ORIENT, N_SCALES, smooth_k=7)
        test_frame = _smooth_bright_patch(
            _textured_bg(rng_seed=50), box=(40, 90, 50, 110))
        dist = texture_distance_map(
            test_frame, model, KERNELS, N_ORIENT, N_SCALES,
            smooth_k=7, post_smooth_k=9)
        mask, thr = threshold_distance_map(
            dist, method="otsu", min_area_px=100, morph_close_k=5)
        # Some foreground should be found, near the patch
        assert int((mask > 0).sum()) > 0
        # The patch centre should be flagged
        assert mask[65, 80] > 0

    def test_empty_distance_returns_empty(self):
        dist = np.zeros((100, 120), dtype=np.float32)
        mask, thr = threshold_distance_map(dist)
        assert int((mask > 0).sum()) == 0
        assert thr == 0.0


class TestColorize:

    def test_colorize_shape(self):
        dist = np.random.RandomState(0).rand(80, 100).astype(np.float32)
        heat = colorize_distance_map(dist)
        assert heat.shape == (80, 100, 3)
        assert heat.dtype == np.uint8
