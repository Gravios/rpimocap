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
            frames, KERNELS, N_ORIENT, N_SCALES, smooth_k=7)
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
