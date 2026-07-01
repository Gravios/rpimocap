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
        # single-layer: 3 features per scale
        desc = dense_gabor_descriptor(
            frame, KERNELS, N_ORIENT, N_SCALES,
            smooth_k=7, rotation_invariant=True, second_layer=False)
        assert desc.shape == (3 * N_SCALES, frame.shape[0],
                              frame.shape[1])
        assert desc.dtype == np.float32
        # default (second_layer=True) appends 2 pooled channels per
        # (layer1_scale x second_layer_scale); default second bank = 2 scales
        desc2 = dense_gabor_descriptor(
            frame, KERNELS, N_ORIENT, N_SCALES,
            smooth_k=7, rotation_invariant=True)
        assert desc2.shape[0] == 3 * N_SCALES + 2 * N_SCALES * 2
        # first channels identical to single-layer (layer-2 only appends)
        assert np.allclose(desc2[:3 * N_SCALES], desc)

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

    def test_log_transform_is_log1p(self):
        """log_transform=True returns log1p of the untransformed
        descriptor."""
        frame = _textured_bg()
        d = dense_gabor_descriptor(
            frame, KERNELS, N_ORIENT, N_SCALES, smooth_k=7)
        d_log = dense_gabor_descriptor(
            frame, KERNELS, N_ORIENT, N_SCALES, smooth_k=7,
            log_transform=True)
        assert np.allclose(d_log, np.log1p(d), atol=1e-5)
        assert d_log.shape == d.shape

    def test_log_transform_compresses_tail(self):
        """The log transform compresses the heavy descriptor tail (the
        whole point — the background is exponential-like)."""
        frame = _textured_bg()
        d = dense_gabor_descriptor(
            frame, KERNELS, N_ORIENT, N_SCALES, smooth_k=7)
        d_log = dense_gabor_descriptor(
            frame, KERNELS, N_ORIENT, N_SCALES, smooth_k=7,
            log_transform=True)
        # ratio of 99.9th pct to median shrinks under log
        def tailiness(x):
            return (np.percentile(x, 99.9)
                    / (np.median(x) + 1e-6))
        assert tailiness(d_log) < tailiness(d)


    def test_second_layer_channel_count(self):
        """Default second_layer=True appends 2 pooled channels per
        (layer1_scale x second_layer_scale); default 2 second scales."""
        frame = _textured_bg()
        d1 = dense_gabor_descriptor(
            frame, KERNELS, N_ORIENT, N_SCALES, second_layer=False)
        d2 = dense_gabor_descriptor(
            frame, KERNELS, N_ORIENT, N_SCALES, second_layer=True)
        assert d1.shape[0] == 3 * N_SCALES
        assert d2.shape[0] == 3 * N_SCALES + 2 * N_SCALES * 2
        assert np.allclose(d2[:3 * N_SCALES], d1)     # layer-2 appends

    def test_second_layer_default_on(self):
        """Opt-out: the second layer is ON unless disabled."""
        frame = _textured_bg()
        default = dense_gabor_descriptor(
            frame, KERNELS, N_ORIENT, N_SCALES)
        assert default.shape[0] == 3 * N_SCALES + 2 * N_SCALES * 2

    def test_second_layer_scales_configurable(self):
        frame = _textured_bg()
        d = dense_gabor_descriptor(
            frame, KERNELS, N_ORIENT, N_SCALES, second_layer=True,
            second_layer_scales=(9, 17, 25))
        assert d.shape[0] == 3 * N_SCALES + 2 * N_SCALES * 3


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


# ────────────────────────────────────────────────────────────────────
#  Persistent (median) model + persistence gating + shape filters
# ────────────────────────────────────────────────────────────────────


from rpimocap.detection.texture_distance import (
    build_persistent_texture_model)


def _frame_moving_rat(seed, rat_xy, shape=(160, 220)):
    """Textured bg + static rail + a filled-ellipse rat at rat_xy."""
    rng = np.random.RandomState(seed)
    f = rng.randint(70, 110, shape).astype(np.uint8)
    f = cv2.GaussianBlur(f, (3, 3), 0)
    f[8:24, 100:120] = 235                       # static rail
    cv2.ellipse(f, rat_xy, (40, 28), 10, 0, 360, 205, -1)
    return f


class TestPersistentModel:

    def test_median_model_builds(self):
        positions = [(50, 80), (100, 60), (160, 110), (190, 70),
                     (120, 90), (70, 120), (170, 50), (140, 100)]
        frames = [_frame_moving_rat(s, positions[s % len(positions)])
                  for s in range(16)]
        model, pers = build_persistent_texture_model(
            frames, KERNELS, N_ORIENT, N_SCALES, smooth_k=7,
            second_layer=False)
        assert model.mean.shape == (3 * N_SCALES,) + frames[0].shape
        assert pers.shape == frames[0].shape
        # persistence in [0, 1]
        assert float(pers.min()) >= 0.0
        assert float(pers.max()) <= 1.0

    def test_median_rejects_moving_rat(self):
        """Because the rat is in a different place each frame, the
        per-pixel median should reflect BACKGROUND, not fur. A pixel
        the rat only occasionally covers should have a background-like
        median descriptor."""
        positions = [(50, 80), (100, 60), (160, 110), (190, 70),
                     (120, 90), (70, 120), (170, 50), (140, 100)]
        frames = [_frame_moving_rat(s, positions[s % len(positions)])
                  for s in range(16)]
        model, pers = build_persistent_texture_model(
            frames, KERNELS, N_ORIENT, N_SCALES, smooth_k=7)
        # Build a pure-background reference (no rat) and compare the
        # median model to it at a pixel the rat sometimes covers
        bg_only = _frame_moving_rat(999, (-100, -100))   # rat offscreen
        from rpimocap.detection.texture_distance import (
            dense_gabor_descriptor)
        bg_desc = dense_gabor_descriptor(
            bg_only, KERNELS, N_ORIENT, N_SCALES, smooth_k=7)
        # At a sometimes-covered pixel, the median model should be
        # closer to background than to fur. Compare descriptor norms.
        y, x = 90, 120
        median_vec = model.mean[:, y, x]
        bg_vec = bg_desc[:, y, x]
        # Median should be reasonably close to the background vector
        rel = (np.linalg.norm(median_vec - bg_vec)
               / (np.linalg.norm(bg_vec) + 1e-6))
        assert rel < 1.0, (
            f"median model should resemble background at "
            f"sometimes-covered pixel; rel diff {rel:.2f}")


class TestPersistenceGating:

    def test_gating_suppresses_static_structure(self):
        """A static rail that the median imperfectly models should
        have LOWER distance with persistence gating than without."""
        positions = [(50, 80), (100, 60), (160, 110), (190, 70),
                     (120, 90), (70, 120), (170, 50), (140, 100)]
        frames = [_frame_moving_rat(s, positions[s % len(positions)])
                  for s in range(16)]
        model, pers = build_persistent_texture_model(
            frames, KERNELS, N_ORIENT, N_SCALES, smooth_k=7)
        test = _frame_moving_rat(999, (110, 80))
        dist_no = texture_distance_map(
            test, model, KERNELS, N_ORIENT, N_SCALES,
            smooth_k=7, post_smooth_k=9)
        dist_gate = texture_distance_map(
            test, model, KERNELS, N_ORIENT, N_SCALES,
            smooth_k=7, persistence_map=pers, persistence_power=2.0,
            post_smooth_k=9)
        rail_no = float(dist_no[8:24, 100:120].mean())
        rail_gate = float(dist_gate[8:24, 100:120].mean())
        assert rail_gate <= rail_no, (
            f"gating should not increase rail distance; "
            f"no={rail_no:.2f} gate={rail_gate:.2f}")


class TestAnisotropyGating:

    def test_uniform_weight_scales_distance(self):
        """A uniform anisotropy weight of w multiplies the distance by
        w everywhere."""
        frames = [_textured_bg(rng_seed=s) for s in range(6)]
        model = build_background_texture_model(
            frames, KERNELS, N_ORIENT, N_SCALES, smooth_k=7)
        test = _smooth_bright_patch(
            _textured_bg(rng_seed=33), box=(40, 90, 50, 110))
        base = texture_distance_map(
            test, model, KERNELS, N_ORIENT, N_SCALES,
            smooth_k=7, post_smooth_k=0)
        half = texture_distance_map(
            test, model, KERNELS, N_ORIENT, N_SCALES, smooth_k=7,
            anisotropy_weight=np.full(test.shape, 0.5, np.float32),
            post_smooth_k=0)
        m = base > 1e-6
        ratio = float(np.mean(half[m] / base[m]))
        assert abs(ratio - 0.5) < 1e-4

    def test_weight_one_is_noop(self):
        frames = [_textured_bg(rng_seed=s) for s in range(6)]
        model = build_background_texture_model(
            frames, KERNELS, N_ORIENT, N_SCALES, smooth_k=7)
        test = _textured_bg(rng_seed=44)
        base = texture_distance_map(
            test, model, KERNELS, N_ORIENT, N_SCALES,
            smooth_k=7, post_smooth_k=0)
        gated = texture_distance_map(
            test, model, KERNELS, N_ORIENT, N_SCALES, smooth_k=7,
            anisotropy_weight=np.ones(test.shape, np.float32),
            post_smooth_k=0)
        assert np.allclose(base, gated, atol=1e-5)

    def test_gradient_weight_suppresses_grazing_side(self):
        """A weight that decreases toward one edge suppresses the
        distance there more — the grazing/foreshortened region."""
        frames = [_textured_bg(rng_seed=s) for s in range(6)]
        model = build_background_texture_model(
            frames, KERNELS, N_ORIENT, N_SCALES, smooth_k=7)
        test = _smooth_bright_patch(
            _textured_bg(rng_seed=55), box=(20, 100, 10, 140))
        H, W = test.shape
        # weight 0 at left edge → 1 at right edge
        w = np.linspace(0, 1, W, dtype=np.float32)[None, :].repeat(H, 0)
        gated = texture_distance_map(
            test, model, KERNELS, N_ORIENT, N_SCALES, smooth_k=7,
            anisotropy_weight=w, post_smooth_k=0)
        assert gated[:, :10].mean() < gated[:, -10:].mean()


class TestShapeFilters:

    def test_aspect_filter_rejects_isolated_line(self):
        """An isolated thin line (cable) is rejected by the aspect
        filter while a compact blob (rat) survives."""
        d = np.zeros((200, 300), np.float32)
        cv2.ellipse(d, (80, 100), (45, 35), 0, 0, 360, 10.0, -1)
        cv2.line(d, (180, 40), (280, 180), 10.0, 5)
        m_no, _ = threshold_distance_map(
            d, method="absolute", abs_thresh=3.0, min_area_px=200,
            max_aspect_ratio=0, morph_close_k=3)
        m_yes, _ = threshold_distance_map(
            d, method="absolute", abs_thresh=3.0, min_area_px=200,
            max_aspect_ratio=6.0, morph_close_k=3)
        # Blob survives both
        assert m_yes[100, 80] > 0
        # Cable present without filter, gone with it
        assert m_no[110, 230] > 0
        assert m_yes[110, 230] == 0

    def test_fill_ratio_filter(self):
        """min_fill_ratio rejects a sparse diagonal line whose bbox
        is mostly empty."""
        d = np.zeros((200, 300), np.float32)
        cv2.ellipse(d, (80, 100), (45, 35), 0, 0, 360, 10.0, -1)
        cv2.line(d, (180, 40), (280, 180), 10.0, 4)
        m, _ = threshold_distance_map(
            d, method="absolute", abs_thresh=3.0, min_area_px=200,
            min_fill_ratio=0.35, morph_close_k=1)
        # Blob (fills its bbox well) survives
        assert m[100, 80] > 0
        # Diagonal line (sparse bbox) rejected
        assert m[110, 230] == 0


# ────────────────────────────────────────────────────────────────────
#  Static shadow / illumination model
# ────────────────────────────────────────────────────────────────────


from rpimocap.detection.texture_distance import (
    build_illumination_field, apply_illumination_correction)


def _gradient_frame(seed, rat_xy=None, shape=(160, 300)):
    """Uniform texture under a left-bright / right-dark illumination
    gradient (mimics IR falloff). Optional moving rat."""
    H, W = shape
    rng = np.random.RandomState(seed)
    tex = rng.randint(70, 110, shape).astype(np.float32)
    tex = cv2.GaussianBlur(tex, (3, 3), 0)
    grad = np.linspace(1.8, 0.5, W)[None, :].repeat(H, axis=0)
    f = np.clip(tex * grad, 0, 255).astype(np.uint8)
    if rat_xy is not None:
        cv2.circle(f, rat_xy, 25, 200, -1)
    return f


class TestIlluminationField:

    def test_field_builds_and_positive(self):
        frames = [_gradient_frame(s, rat_xy=(30 + s * 20, 80))
                  for s in range(12)]
        field = build_illumination_field(frames, blur_sigma=0)
        assert field.shape == frames[0].shape
        assert float(field.min()) >= 1.0     # floored positive

    def test_field_captures_gradient(self):
        """The illumination field should be brighter on the lit (left)
        side than the shadowed (right) side."""
        frames = [_gradient_frame(s, rat_xy=(30 + s * 20, 80))
                  for s in range(12)]
        field = build_illumination_field(frames, blur_sigma=21)
        left = float(field[:, :50].mean())
        right = float(field[:, -50:].mean())
        assert left > right, (
            f"field should be brighter on lit side; "
            f"left={left:.1f} right={right:.1f}")

    def test_median_rejects_moving_rat(self):
        """The bright rat circle moves each frame, so the median
        field should NOT show a persistent bright rat blob — the
        field at any pixel reflects background illumination."""
        frames = [_gradient_frame(s, rat_xy=(30 + s * 20, 80))
                  for s in range(12)]
        field = build_illumination_field(frames, blur_sigma=0)
        # The rat (intensity 200) passes through many positions along
        # y=80. The median there should be well below 200 (background
        # illumination level), since the rat is a minority at each x.
        assert float(field[80, 150]) < 180

    def test_requires_min_frames(self):
        try:
            build_illumination_field([_gradient_frame(0)], blur_sigma=0)
            assert False, "should raise on too few frames"
        except RuntimeError:
            pass


class TestIlluminationCorrection:

    def test_correction_equalizes_descriptor(self):
        """The same texture under different illumination produces
        different descriptors before correction, matching descriptors
        after."""
        # Build field from background-only gradient frames
        frames = [_gradient_frame(s, rat_xy=(30 + s * 20, 80))
                  for s in range(12)]
        field = build_illumination_field(frames, blur_sigma=0)
        test = _gradient_frame(999)        # no rat, pure gradient
        # Before
        desc_pre = dense_gabor_descriptor(
            test, KERNELS, N_ORIENT, N_SCALES, smooth_k=7)
        left_pre = desc_pre[:, 70:90, 30:70].mean()
        right_pre = desc_pre[:, 70:90, 230:270].mean()
        ratio_pre = abs(left_pre / (right_pre + 1e-6) - 1.0)
        # After
        corr = apply_illumination_correction(test, field)
        desc_post = dense_gabor_descriptor(
            corr, KERNELS, N_ORIENT, N_SCALES, smooth_k=7)
        left_post = desc_post[:, 70:90, 30:70].mean()
        right_post = desc_post[:, 70:90, 230:270].mean()
        ratio_post = abs(left_post / (right_post + 1e-6) - 1.0)
        # Correction should bring the lit/shadow descriptor ratio
        # closer to 1.0
        assert ratio_post < ratio_pre, (
            f"correction should equalize descriptors; "
            f"pre-imbalance={ratio_pre:.2f} post={ratio_post:.2f}")

    def test_correction_preserves_brightness(self):
        """Correcting a frame that equals the field yields ~the field
        mean everywhere (no gross brightness change)."""
        frames = [_gradient_frame(s) for s in range(8)]
        field = build_illumination_field(frames, blur_sigma=0)
        corr = apply_illumination_correction(field, field)
        # Where frame == field, corrected ≈ target_level (field mean)
        assert abs(float(corr.mean()) - float(field.mean())) < 10

    def test_correction_output_uint8(self):
        frames = [_gradient_frame(s) for s in range(6)]
        field = build_illumination_field(frames, blur_sigma=11)
        corr = apply_illumination_correction(
            _gradient_frame(1), field)
        assert corr.dtype == np.uint8
        assert corr.min() >= 0 and corr.max() <= 255


# ────────────────────────────────────────────────────────────────────
#  Dynamic shadow model
# ────────────────────────────────────────────────────────────────────


from rpimocap.detection.texture_distance import (
    DynamicShadowModel, TextureBlobTracker)


class TestDynamicShadowModel:

    def test_tracks_brightness_drift(self):
        """The field should follow a slow brightness drift."""
        H, W = 100, 160
        dsm = DynamicShadowModel(
            np.full((H, W), 100, np.float32), alpha=0.2, blur_sigma=0)
        for t in range(30):
            frame = np.full((H, W), 100 + t * 1.0, np.float32)
            dsm.update(frame)
        field = dsm.get_field()
        # Drift reached ~129; field should have climbed well above 100
        assert float(np.median(field)) > 115

    def test_masked_rat_excluded(self):
        """Pixels under the update_mask (rat) must NOT pull the field
        toward the rat's brightness."""
        H, W = 100, 160
        dsm = DynamicShadowModel(
            np.full((H, W), 100, np.float32), alpha=0.3, blur_sigma=0)
        for t in range(20):
            frame = np.full((H, W), 100, np.float32)
            # A bright rat that does NOT move (worst case for poisoning)
            cv2.circle(frame, (80, 50), 15, 250, -1)
            m = np.zeros((H, W), np.uint8)
            cv2.circle(m, (80, 50), 18, 255, -1)
            dsm.update(frame, update_mask=m)
        field = dsm.get_field()
        # Despite a static bright rat for 20 frames, the field at the
        # rat location stays at background (mask excluded it)
        assert float(field[50, 80]) < 150, (
            f"masked rat should not poison field; "
            f"got {float(field[50, 80]):.1f}")

    def test_correct_uses_current_field(self):
        H, W = 80, 120
        dsm = DynamicShadowModel(
            np.full((H, W), 100, np.float32), alpha=0.1, blur_sigma=0)
        corrected = dsm.correct(np.full((H, W), 100, np.uint8))
        assert corrected.shape == (H, W)
        assert corrected.dtype == np.uint8

    def test_field_stays_positive(self):
        H, W = 80, 120
        dsm = DynamicShadowModel(
            np.full((H, W), 100, np.float32), alpha=0.5, floor=1.0)
        # Feed near-zero frames; field must not drop below floor
        for _ in range(10):
            dsm.update(np.zeros((H, W), np.float32))
        assert float(dsm.get_field().min()) >= 1.0


# ────────────────────────────────────────────────────────────────────
#  Texture blob tracker (Kalman)
# ────────────────────────────────────────────────────────────────────


def _circle_mask(blobs, shape=(200, 400)):
    m = np.zeros(shape, np.uint8)
    for (cx, cy, r) in blobs:
        cv2.circle(m, (cx, cy), r, 255, -1)
    return m


class TestTextureBlobTracker:

    def test_geometry_prefers_rat_over_diagonal_cable(self):
        """Higher-order geometry: a DIAGONAL cable+headstage composite of
        LARGER area than the rat (worst case for bbox fill-ratio) must
        still be demoted below the compact convex rat."""
        import cv2
        H, W = 460, 620
        m = np.zeros((H, W), np.uint8)
        cv2.ellipse(m, (140, 230), (40, 30), 20, 0, 360, 255, -1)  # rotated rat
        cv2.circle(m, (330, 120), 30, 255, -1)                     # headstage
        cv2.line(m, (330, 120), (560, 420), 255, 11)               # diagonal cable
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
        # area picks the (larger) cable
        assert TextureBlobTracker(select="area").update(m)["state"][0] > 250
        # geometry picks the rat
        assert TextureBlobTracker(
            select="geometry").update(m)["state"][0] < 250

    def test_max_elongation_rejects_cable(self):
        import cv2
        H, W = 460, 620
        m = np.zeros((H, W), np.uint8)
        cv2.ellipse(m, (140, 230), (40, 30), 20, 0, 360, 255, -1)
        cv2.circle(m, (330, 120), 30, 255, -1)
        cv2.line(m, (330, 120), (560, 420), 255, 11)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
        r = TextureBlobTracker(select="geometry",
                               max_elongation=6.0).update(m)
        assert r["state"][0] < 250

    def test_min_solidity_rejects_cable(self):
        import cv2
        H, W = 460, 620
        m = np.zeros((H, W), np.uint8)
        cv2.ellipse(m, (140, 230), (40, 30), 20, 0, 360, 255, -1)
        cv2.circle(m, (330, 120), 30, 255, -1)
        cv2.line(m, (330, 120), (560, 420), 255, 11)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
        r = TextureBlobTracker(select="geometry",
                               min_solidity=0.6).update(m)
        assert r["state"][0] < 250

    def test_geom_features_values(self):
        """Elongation ~1 and solidity ~1 for a compact disc; elongation
        large for a thin line."""
        import cv2
        m = np.zeros((300, 300), np.uint8)
        cv2.circle(m, (150, 150), 40, 255, -1)
        n, lab, st, _ = cv2.connectedComponentsWithStats(m)
        _, sol, el = TextureBlobTracker._geom_features(lab, 1, st[1])
        assert el < 1.2 and sol > 0.9
        m2 = np.zeros((300, 300), np.uint8)
        cv2.line(m2, (30, 150), (270, 150), 255, 6)
        n2, lab2, st2, _ = cv2.connectedComponentsWithStats(m2)
        _, _, el2 = TextureBlobTracker._geom_features(lab2, 1, st2[1])
        assert el2 > 8.0

    def test_compactness_prefers_rat_over_larger_cable(self):
        """The core cable fix: a LARGER but sparse cable+headstage
        composite must NOT out-compete the compact rat under the default
        compactness selection (it does under 'area')."""
        import cv2
        H, W = 400, 600
        m = np.zeros((H, W), np.uint8)
        # compact rat (smaller area) at x=130
        cv2.ellipse(m, (130, 200), (38, 28), 0, 0, 360, 255, -1)
        # cable+headstage composite (LARGER area) at x~360, sparse
        cv2.circle(m, (330, 120), 28, 255, -1)
        pts = np.array([[330, 120], [380, 190], [360, 260],
                        [420, 330], [400, 380]], np.int32)
        cv2.polylines(m, [pts], False, 255, 10)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))

        # area selection picks the cable (the bug)
        rc = TextureBlobTracker(select="area").update(m)
        assert rc["state"][0] > 250            # cable side

        # compactness selection picks the rat (the fix)
        rf = TextureBlobTracker(select="compactness",
                                fill_power=1.5).update(m)
        assert rf["state"][0] < 250            # rat side

    def test_min_fill_rejects_sparse_cable(self):
        import cv2
        H, W = 400, 600
        m = np.zeros((H, W), np.uint8)
        cv2.ellipse(m, (130, 200), (38, 28), 0, 0, 360, 255, -1)
        cv2.circle(m, (330, 120), 28, 255, -1)
        pts = np.array([[330, 120], [380, 190], [360, 260],
                        [420, 330], [400, 380]], np.int32)
        cv2.polylines(m, [pts], False, 255, 10)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
        r = TextureBlobTracker(select="compactness",
                               min_fill=0.3).update(m)
        assert r["state"][0] < 250             # only the rat survives

    def test_area_mode_unchanged(self):
        """Legacy 'area' selection still picks the largest blob."""
        import cv2
        m = np.zeros((300, 500), np.uint8)
        cv2.circle(m, (120, 150), 20, 255, -1)     # small
        cv2.circle(m, (350, 150), 40, 255, -1)     # large
        r = TextureBlobTracker(select="area").update(m)
        assert abs(r["state"][0] - 350) < 10

    def test_initializes_from_first_detection(self):
        trk = TextureBlobTracker()
        r = trk.update(_circle_mask([(100, 100, 30)]))
        assert r["measured"] is True
        assert r["state"] is not None
        cx, cy, _ = r["state"]
        assert abs(cx - 100) < 5 and abs(cy - 100) < 5

    def test_no_detection_no_init(self):
        trk = TextureBlobTracker()
        r = trk.update(np.zeros((200, 400), np.uint8))
        assert r["lost"] is True
        assert r["state"] is None

    def test_tracks_moving_blob(self):
        trk = TextureBlobTracker(gate_px=80)
        last = None
        for t in range(8):
            r = trk.update(_circle_mask([(40 + t * 25, 100, 30)]))
            assert r["measured"] is True
            last = r["state"]
        # Final position should be near the last blob center
        assert abs(last[0] - (40 + 7 * 25)) < 20

    def test_peek_prediction_none_before_init(self):
        trk = TextureBlobTracker()
        assert trk.peek_prediction() is None

    def test_peek_prediction_does_not_mutate(self):
        """peek_prediction must not advance the Kalman state — calling
        it repeatedly leaves statePost unchanged."""
        trk = TextureBlobTracker(gate_px=80)
        for t in range(3):
            trk.update(_circle_mask([(40 + t * 25, 100, 30)]))
        before = trk._kf.statePost.copy()
        for _ in range(5):
            trk.peek_prediction()
        assert np.allclose(before, trk._kf.statePost)

    def test_peek_matches_predict_value(self):
        trk = TextureBlobTracker(gate_px=80)
        for t in range(3):
            trk.update(_circle_mask([(40 + t * 25, 100, 30)]))
        peek = trk.peek_prediction()
        pred = trk.predict()          # this one advances
        assert np.allclose(peek, pred, atol=1e-4)

    def test_peeking_does_not_change_track(self):
        """A tracker that peeks every frame produces the same states as
        one that doesn't (no double-advance)."""
        a = TextureBlobTracker(gate_px=80)
        b = TextureBlobTracker(gate_px=80)
        for t in range(8):
            m = _circle_mask([(40 + t * 25, 100, 30)])
            a.peek_prediction(); a.peek_prediction()
            ra = a.update(m)
            rb = b.update(m)
            if ra["state"] and rb["state"]:
                assert np.allclose(ra["state"], rb["state"])

    def test_gates_out_distractor(self):
        """A blob far from the prediction is rejected; the tracker
        coasts instead of jumping to it."""
        trk = TextureBlobTracker(gate_px=80, max_coast=5)
        for t in range(5):
            trk.update(_circle_mask([(40 + t * 25, 100, 30)]))
        # Now: only a distractor far away
        r = trk.update(_circle_mask([(370, 30, 18)]))
        assert r["coasting"] is True, "distractor should be gated out"
        assert trk.n_gated_out >= 1

    def test_picks_rat_over_distractor(self):
        """With both rat and distractor present, the in-gate rat is
        chosen over the out-of-gate distractor."""
        trk = TextureBlobTracker(gate_px=80, select="area")
        for t in range(5):
            trk.update(_circle_mask([(40 + t * 25, 100, 30)]))
        r = trk.update(_circle_mask(
            [(40 + 5 * 25, 100, 30), (370, 30, 18)]))
        assert r["measured"] is True
        # Tracked position should be near the rat, not the distractor
        assert abs(r["state"][0] - (40 + 5 * 25)) < 30

    def test_coast_then_lost(self):
        """After max_coast missed frames, the track is declared
        lost."""
        trk = TextureBlobTracker(gate_px=50, max_coast=3)
        trk.update(_circle_mask([(100, 100, 30)]))
        # Feed only far distractors → coast until lost
        lost_seen = False
        for _ in range(6):
            r = trk.update(_circle_mask([(380, 20, 15)]))
            if r["lost"]:
                lost_seen = True
                break
        assert lost_seen, "track should eventually be declared lost"


# ────────────────────────────────────────────────────────────────────
#  Rat-masked persistence (rat doesn't suppress its own dwell spots)
# ────────────────────────────────────────────────────────────────────


from rpimocap.detection.texture_distance import (
    _detect_rat_mask_intensity)


def _fixed_bg_with_rat(rat_xy, bg=None, shape=(160, 260)):
    """A FIXED textured background (static bedding) with a bright rat
    circle at rat_xy. The background is identical across frames; only
    the rat moves — the realistic case."""
    if bg is None:
        rng = np.random.RandomState(42)
        bg = cv2.GaussianBlur(
            rng.randint(70, 110, shape).astype(np.uint8), (3, 3), 0)
    f = bg.copy()
    cv2.circle(f, rat_xy, 28, 210, -1)
    return f


class TestRatDetectorForPersistence:

    def test_detects_bright_rat(self):
        f = _fixed_bg_with_rat((150, 80))
        m = _detect_rat_mask_intensity(
            f, percentile=95, min_area_px=500, dilate_px=10)
        assert m[80, 150] > 0           # rat center detected

    def test_no_rat_returns_empty(self):
        rng = np.random.RandomState(0)
        # Uniform dim frame, no bright blob
        f = cv2.GaussianBlur(
            rng.randint(70, 110, (160, 260)).astype(np.uint8),
            (3, 3), 0)
        m = _detect_rat_mask_intensity(
            f, percentile=99, min_area_px=5000, dilate_px=5)
        assert int((m > 0).sum()) == 0


class TestRatMaskedPersistence:

    def _dwell_session(self):
        """Rat dwells at one spot for half the frames, moves for the
        rest — over a FIXED background."""
        rng = np.random.RandomState(42)
        bg = cv2.GaussianBlur(
            rng.randint(70, 110, (160, 260)).astype(np.uint8),
            (3, 3), 0)
        dwell = (150, 80)
        positions = ([dwell] * 10 +
                     [(40, 60), (220, 120), (90, 130), (200, 50),
                      (120, 100), (60, 90), (180, 140), (100, 60),
                      (230, 100), (70, 120)])
        return [_fixed_bg_with_rat(p, bg=bg) for p in positions], dwell

    def test_masking_rescues_dwell_persistence(self):
        """Without masking, the rat's dwell spot is marked
        low-persistence (it would be suppressed). With masking, the
        dwell spot is correctly recognized as static background."""
        frames, dwell = self._dwell_session()
        _, pers_no = build_persistent_texture_model(
            frames, KERNELS, N_ORIENT, N_SCALES, smooth_k=7,
            mask_rat=False)
        _, pers_yes = build_persistent_texture_model(
            frames, KERNELS, N_ORIENT, N_SCALES, smooth_k=7,
            mask_rat=True, rat_percentile=95, rat_min_area_px=500,
            rat_dilate_px=40)
        x, y = dwell
        # Masking should raise persistence at the dwell spot
        assert float(pers_yes[y, x]) > float(pers_no[y, x])
        # And bring it close to the static-background level
        assert float(pers_yes[y, x]) > 0.8

    def test_masking_preserves_static_corner(self):
        """A never-visited background corner stays high-persistence
        whether or not masking is on."""
        frames, _ = self._dwell_session()
        _, pers_yes = build_persistent_texture_model(
            frames, KERNELS, N_ORIENT, N_SCALES, smooth_k=7,
            mask_rat=True, rat_percentile=95, rat_min_area_px=500,
            rat_dilate_px=40)
        assert float(pers_yes[20, 20]) > 0.8

    def test_masked_median_is_background(self):
        """With masking, the model's median descriptor at the dwell
        spot reflects BACKGROUND, not the rat — verified by comparing
        to a pure-background descriptor there."""
        frames, dwell = self._dwell_session()
        model, _ = build_persistent_texture_model(
            frames, KERNELS, N_ORIENT, N_SCALES, smooth_k=7,
            mask_rat=True, rat_percentile=95, rat_min_area_px=500,
            rat_dilate_px=40)
        # Pure-background reference at the dwell pixel
        rng = np.random.RandomState(42)
        bg = cv2.GaussianBlur(
            rng.randint(70, 110, (160, 260)).astype(np.uint8),
            (3, 3), 0)
        bg_desc = dense_gabor_descriptor(
            bg, KERNELS, N_ORIENT, N_SCALES, smooth_k=7)
        x, y = dwell
        rel = (np.linalg.norm(model.mean[:, y, x] - bg_desc[:, y, x])
               / (np.linalg.norm(bg_desc[:, y, x]) + 1e-6))
        assert rel < 0.5, (
            f"masked median should match background at dwell; "
            f"rel diff {rel:.2f}")


# ────────────────────────────────────────────────────────────────────
#  Arena ROI gating (only compute inside the arena)
# ────────────────────────────────────────────────────────────────────


class TestArenaROIGating:

    def _frame_with_outside_distractor(self):
        """Rat inside a central arena ROI, plus a BIGGER bright
        distractor (hand/reflection) outside it."""
        rng = np.random.RandomState(0)
        f = rng.randint(70, 110, (200, 300)).astype(np.uint8)
        cv2.circle(f, (150, 100), 30, 180, -1)      # rat in arena
        f[5:70, 5:90] = 240                         # big hand outside
        roi = np.zeros((200, 300), np.uint8)
        cv2.rectangle(roi, (70, 40), (240, 170), 255, -1)
        return f, roi

    def test_roi_switches_detection_to_rat(self):
        """Without ROI the detector grabs the bigger outside-arena
        distractor; with ROI it grabs the rat inside."""
        f, roi = self._frame_with_outside_distractor()
        m_no = _detect_rat_mask_intensity(
            f, percentile=85, min_area_px=500, dilate_px=5)
        m_roi = _detect_rat_mask_intensity(
            f, percentile=85, min_area_px=500, dilate_px=5,
            roi_mask=roi)
        # Without ROI: detection is on the hand, not the rat
        assert m_no[35, 45] > 0 and m_no[100, 150] == 0
        # With ROI: detection is on the rat, not the hand
        assert m_roi[100, 150] > 0 and m_roi[35, 45] == 0

    def test_roi_mask_stays_inside(self):
        """The returned mask never extends outside the ROI even after
        dilation."""
        f, roi = self._frame_with_outside_distractor()
        m = _detect_rat_mask_intensity(
            f, percentile=85, min_area_px=500, dilate_px=40,
            roi_mask=roi)
        # No mask pixels outside the ROI
        assert int(m[(roi == 0)].sum()) == 0

    def test_persistence_zeroed_outside_roi(self):
        """build_persistent_texture_model zeroes persistence outside
        the arena."""
        rng = np.random.RandomState(1)
        bg = cv2.GaussianBlur(
            rng.randint(70, 110, (160, 260)).astype(np.uint8),
            (3, 3), 0)
        frames = []
        for cx in (60, 120, 180, 90, 150, 200, 70, 130):
            f = bg.copy()
            cv2.circle(f, (cx, 80), 25, 205, -1)
            frames.append(f)
        roi = np.zeros((160, 260), np.uint8)
        cv2.rectangle(roi, (40, 30), (220, 130), 255, -1)
        _, pers = build_persistent_texture_model(
            frames, KERNELS, N_ORIENT, N_SCALES, smooth_k=7,
            roi_mask=roi)
        # Persistence is exactly 0 outside the ROI
        assert float(pers[(roi == 0)].max()) == 0.0
        # And non-zero somewhere inside
        assert float(pers[(roi > 0)].max()) > 0.0

    def test_distance_map_zeroed_outside_roi(self):
        """texture_distance_map zeroes distance outside the ROI."""
        rng = np.random.RandomState(2)
        bg = cv2.GaussianBlur(
            rng.randint(70, 110, (160, 260)).astype(np.uint8),
            (3, 3), 0)
        frames = [bg.copy() for _ in range(6)]
        model = build_background_texture_model(
            frames, KERNELS, N_ORIENT, N_SCALES, smooth_k=7)
        roi = np.zeros((160, 260), np.uint8)
        cv2.rectangle(roi, (40, 30), (220, 130), 255, -1)
        test = bg.copy()
        cv2.circle(test, (130, 80), 25, 205, -1)
        dist = texture_distance_map(
            test, model, KERNELS, N_ORIENT, N_SCALES, smooth_k=7,
            roi_mask=roi, post_smooth_k=0)
        assert float(dist[(roi == 0)].max()) == 0.0


# ────────────────────────────────────────────────────────────────────
#  Cable suppression (thin-structure removal within a component)
# ────────────────────────────────────────────────────────────────────


from rpimocap.detection.texture_distance import suppress_thin_structures


class TestSuppressThinStructures:

    def _rat_with_attached_cable(self):
        """A compact rat body with a thin cable attached — ONE
        connected component (the failure case for aspect filters)."""
        d = np.zeros((300, 400), np.float32)
        cv2.ellipse(d, (150, 150), (60, 45), 0, 0, 360, 10.0, -1)
        cv2.line(d, (205, 140), (360, 90), 10.0, 6)   # attached cable
        return (d > 3).astype(np.uint8) * 255

    def test_removes_attached_cable_keeps_rat(self):
        mask = self._rat_with_attached_cable()
        out = suppress_thin_structures(mask, min_width_px=30)
        # Rat body center kept
        assert out[150, 150] > 0
        # Cable far end removed
        assert out[90, 355] == 0

    def test_aspect_filter_alone_fails_on_fused(self):
        """Demonstrates WHY width suppression is needed: the aspect
        filter passes the fused rat+cable component (cable survives),
        while suppression removes it."""
        d = np.zeros((300, 400), np.float32)
        cv2.ellipse(d, (150, 150), (60, 45), 0, 0, 360, 10.0, -1)
        cv2.line(d, (205, 140), (360, 90), 10.0, 6)
        out_aspect, _ = threshold_distance_map(
            d, method="absolute", abs_thresh=3.0, min_area_px=500,
            max_aspect_ratio=6.0, morph_close_k=1)
        out_supp, _ = threshold_distance_map(
            d, method="absolute", abs_thresh=3.0, min_area_px=500,
            suppress_thin_width=30, morph_close_k=1)
        # Aspect filter leaves the cable; suppression removes it
        assert out_aspect[90, 355] > 0       # cable survives aspect
        assert out_supp[90, 355] == 0        # cable removed by width
        assert out_supp[150, 150] > 0        # rat kept

    def test_empty_mask_returns_empty(self):
        out = suppress_thin_structures(
            np.zeros((100, 100), np.uint8), min_width_px=20)
        assert int((out > 0).sum()) == 0

    def test_all_thin_returns_empty(self):
        """A mask that is ONLY a thin line (no thick body) erodes to
        nothing → empty result, not the line."""
        m = np.zeros((200, 300), np.uint8)
        cv2.line(m, (20, 100), (280, 100), 255, 5)
        out = suppress_thin_structures(m, min_width_px=30)
        assert int((out > 0).sum()) == 0

    def test_thick_blob_survives(self):
        """A purely thick blob (no thin parts) is preserved."""
        m = np.zeros((200, 200), np.uint8)
        cv2.circle(m, (100, 100), 50, 255, -1)
        out = suppress_thin_structures(m, min_width_px=30)
        # Most of the blob survives
        assert int((out > 0).sum()) > 0.7 * int((m > 0).sum())


# ────────────────────────────────────────────────────────────────────
#  Graph-cut (MRF) segmentation
# ────────────────────────────────────────────────────────────────────

import pytest

try:
    import maxflow as _maxflow
    _HAVE_MAXFLOW = True
except ImportError:
    _HAVE_MAXFLOW = False

from rpimocap.detection.texture_distance import graphcut_segment_distance


@pytest.mark.skipif(not _HAVE_MAXFLOW, reason="PyMaxflow not installed")
class TestGraphCutSegment:

    def _fragmented_blob(self):
        """A rat blob with internal holes + scattered noise specks +
        a faint cable — the realistic messy distance map."""
        rng = np.random.RandomState(0)
        H, W = 300, 400
        dist = np.zeros((H, W), np.float32)
        cv2.ellipse(dist, (180, 150), (55, 40), 0, 0, 360, 6.0, -1)
        for _ in range(15):                       # holes
            cx, cy = rng.randint(130, 230), rng.randint(115, 185)
            cv2.circle(dist, (cx, cy), rng.randint(4, 9), 1.5, -1)
        for _ in range(40):                       # noise specks
            cx, cy = rng.randint(0, W), rng.randint(0, H)
            cv2.circle(dist, (cx, cy), rng.randint(2, 5),
                       rng.uniform(4.5, 7), -1)
        gray = np.full((H, W), 90, np.uint8)
        cv2.ellipse(gray, (180, 150), (55, 40), 0, 0, 360, 180, -1)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        return dist, gray

    def test_runs_and_returns_mask(self):
        dist, gray = self._fragmented_blob()
        mask, info = graphcut_segment_distance(
            dist, gray=gray, fg_thresh=4.0, smooth_weight=3.0,
            min_area_px=800)
        assert mask.shape == dist.shape
        assert mask.dtype == np.uint8
        assert info["fg_px"] > 0
        assert "flow" in info

    def test_smoothness_rejects_noise_without_area_filter(self):
        """Higher smoothness weight rejects isolated noise specks even
        with NO minimum-area filter — the energy does it, not a size
        hack."""
        dist, gray = self._fragmented_blob()

        def noise_specks(m):
            n, _, s, _ = cv2.connectedComponentsWithStats(
                (m > 0).astype(np.uint8))
            cnt = 0
            for i in range(1, n):
                cx = s[i, cv2.CC_STAT_LEFT] + s[i, cv2.CC_STAT_WIDTH] // 2
                cy = s[i, cv2.CC_STAT_TOP] + s[i, cv2.CC_STAT_HEIGHT] // 2
                if not (130 < cx < 230 and 115 < cy < 185):
                    cnt += 1
            return cnt

        m_lo, _ = graphcut_segment_distance(
            dist, gray=gray, fg_thresh=4.0, smooth_weight=1.0,
            min_area_px=1)
        m_hi, _ = graphcut_segment_distance(
            dist, gray=gray, fg_thresh=4.0, smooth_weight=6.0,
            min_area_px=1)
        # More smoothing → fewer stray specks
        assert noise_specks(m_hi) < noise_specks(m_lo)
        assert noise_specks(m_hi) <= 2

    def test_smoothness_fills_holes(self):
        """Higher smoothness fills internal holes in the rat blob."""
        dist, gray = self._fragmented_blob()

        def rat_fill(m):
            region = m[115:185, 130:230]
            return float((region > 0).sum()) / region.size

        m_lo, _ = graphcut_segment_distance(
            dist, gray=gray, fg_thresh=4.0, smooth_weight=1.0,
            min_area_px=1)
        m_hi, _ = graphcut_segment_distance(
            dist, gray=gray, fg_thresh=4.0, smooth_weight=12.0,
            min_area_px=1)
        assert rat_fill(m_hi) > rat_fill(m_lo)
        assert rat_fill(m_hi) > 0.85

    def test_roi_clamp_forces_background(self):
        """A high-distance blob outside the ROI is never labeled fg;
        the rat inside is."""
        H, W = 200, 300
        dist = np.zeros((H, W), np.float32)
        cv2.circle(dist, (150, 100), 50, 6.0, -1)
        cv2.circle(dist, (20, 20), 12, 7.0, -1)        # outside-ROI blob
        gray = np.full((H, W), 90, np.uint8)
        cv2.circle(gray, (150, 100), 45, 180, -1)
        roi = np.zeros((H, W), np.uint8)
        cv2.rectangle(roi, (60, 40), (240, 160), 255, -1)
        mask, _ = graphcut_segment_distance(
            dist, gray=gray, roi_mask=roi, fg_thresh=3.0,
            smooth_weight=4.0, min_area_px=50)
        assert mask[20, 20] == 0          # outside ROI forced bg
        assert mask[100, 150] > 0         # rat inside kept

    def test_plain_potts_without_gray(self):
        """Runs without a gray image (plain Potts smoothness)."""
        dist, _ = self._fragmented_blob()
        mask, info = graphcut_segment_distance(
            dist, gray=None, fg_thresh=4.0, smooth_weight=3.0,
            min_area_px=800)
        assert info["fg_px"] > 0

    def test_crop_box_matches_full_inside_box(self):
        """The predicted-ROI crop must produce an identical cut inside
        the box (the blob fully contained) — same result, smaller
        graph."""
        from rpimocap.detection.texture_distance import (
            crop_box_from_prediction)
        rng = np.random.RandomState(0)
        H, W = 400, 600
        dist = np.zeros((H, W), np.float32)
        gray = np.full((H, W), 90, np.uint8)
        yy, xx = np.mgrid[0:H, 0:W]
        blob = ((xx - 420) ** 2 + (yy - 200) ** 2) < 50 ** 2
        dist[blob] = 8.0
        gray[blob] = 200
        dist += rng.rand(H, W).astype(np.float32) * 0.5
        m_full, _ = graphcut_segment_distance(
            dist, gray, fg_thresh=4.0, smooth_weight=2.0, min_area_px=200)
        box = crop_box_from_prediction((420.0, 200.0, 50.0), (H, W),
                                       pad_px=120)
        m_crop, _ = graphcut_segment_distance(
            dist, gray, fg_thresh=4.0, smooth_weight=2.0, min_area_px=200,
            crop_box=box)
        assert m_crop.shape == m_full.shape       # full-size output
        inter = ((m_full > 0) & (m_crop > 0)).sum()
        union = ((m_full > 0) | (m_crop > 0)).sum()
        assert inter / max(union, 1) > 0.98       # identical cut

    def test_crop_box_outside_is_background(self):
        """Everything outside the crop box is background."""
        from rpimocap.detection.texture_distance import (
            crop_box_from_prediction)
        H, W = 300, 400
        dist = np.zeros((H, W), np.float32)
        cv2.circle(dist, (200, 150), 40, 8.0, -1)
        cv2.circle(dist, (40, 40), 30, 8.0, -1)    # blob outside the box
        gray = np.full((H, W), 90, np.uint8)
        box = crop_box_from_prediction((200.0, 150.0, 40.0), (H, W),
                                       pad_px=60)
        mask, _ = graphcut_segment_distance(
            dist, gray, fg_thresh=4.0, smooth_weight=2.0, min_area_px=50,
            crop_box=box)
        assert mask[40, 40] == 0                    # outside box → bg
        assert mask[150, 200] > 0                   # inside box → kept


class TestCropBoxFromPrediction:

    def test_none_prediction_returns_none(self):
        from rpimocap.detection.texture_distance import (
            crop_box_from_prediction)
        assert crop_box_from_prediction(None, (400, 600)) is None

    def test_box_centered_and_clamped(self):
        from rpimocap.detection.texture_distance import (
            crop_box_from_prediction)
        box = crop_box_from_prediction((300.0, 200.0, 40.0), (400, 600),
                                       pad_px=100)
        x0, y0, x1, y1 = box
        assert 0 <= x0 < 300 < x1 <= 600
        assert 0 <= y0 < 200 < y1 <= 400
        # half-size ≈ r + pad = 140
        assert abs((x1 - x0) / 2 - 140) < 2

    def test_box_clamps_at_border(self):
        from rpimocap.detection.texture_distance import (
            crop_box_from_prediction)
        box = crop_box_from_prediction((5.0, 5.0, 40.0), (400, 600),
                                       pad_px=120)
        x0, y0, x1, y1 = box
        assert x0 == 0 and y0 == 0
        assert x1 <= 600 and y1 <= 400
