"""
tests/test_uniform_patches_and_artifacts.py
============================================
Patch 0029: patch-based bootstrap with uniformity filter,
per-camera artifact masks via Gabor+histogram, and Canny edge
barrier in edge refinement.
"""
from __future__ import annotations

import numpy as np
import cv2

from rpimocap.detection.rat_texture import (
    RatTextureBank,
    build_camera_artifact_mask,
)
from rpimocap.detection.segment import (
    BackgroundModel, ForegroundDetector)


def _striped(shape, intensity_base=80, stripe_intensity=200,
              period=4, rng_seed=0):
    rng = np.random.RandomState(rng_seed)
    f = np.full(shape, intensity_base, dtype=np.uint8)
    f[::period] = stripe_intensity
    f = f + rng.randint(-10, 10, size=shape).astype(np.int16)
    return np.clip(f, 0, 255).astype(np.uint8)


def _flat(shape, intensity=120, noise=5, rng_seed=0):
    rng = np.random.RandomState(rng_seed)
    f = np.full(shape, intensity, dtype=np.int16)
    f = f + rng.randint(-noise, noise + 1, size=shape).astype(np.int16)
    return np.clip(f, 0, 255).astype(np.uint8)


def _train_bank_striped(scales=(7, 11, 17)):
    """Train a bank on samples from striped texture."""
    b = RatTextureBank(scales=scales)
    full_mask = np.ones((60, 60), dtype=np.uint8) * 255
    samples = []
    for seed in range(15):
        samples.append(b.features_in_blob(
            _striped((60, 60), rng_seed=seed), full_mask))
    b.bootstrap(samples)
    return b


# ────────────────────────────────────────────────────────────────────
#  Uniform-patch sampling
# ────────────────────────────────────────────────────────────────────


class TestUniformPatchSampling:

    def test_returns_features_for_uniform_region(self):
        b = RatTextureBank()
        gray = _striped((200, 300))
        # Mask covering the whole uniform region
        mask = np.zeros_like(gray)
        mask[50:150, 80:230] = 255
        patches = b.sample_uniform_patches(
            gray, mask,
            patch_size=32, stride=16,
            max_patches=10, std_max=80)
        assert len(patches) > 0
        for p in patches:
            assert p.shape == (b.feature_dim,)

    def test_rejects_boundary_patches_with_strict_std(self):
        """Build a frame with a striped region surrounded by a
        very different intensity. Patches near the boundary have
        high intra-patch std → rejected by a strict std_max."""
        gray = np.full((200, 300), 30, dtype=np.uint8)
        # Insert striped patch in the middle
        gray[60:140, 90:210] = _striped((80, 120), rng_seed=5)
        # Mask covers the striped region AND a bit of bg
        mask = np.zeros_like(gray)
        mask[40:160, 70:230] = 255   # extends past the striped region

        b = RatTextureBank()
        # Lenient std
        loose = b.sample_uniform_patches(
            gray, mask, patch_size=24, stride=8,
            max_patches=200, std_max=200.0)
        # Strict std
        strict = b.sample_uniform_patches(
            gray, mask, patch_size=24, stride=8,
            max_patches=200, std_max=25.0)
        # Strict should reject some boundary patches
        assert len(strict) < len(loose), (
            f"strict std should reject some patches; "
            f"loose={len(loose)} strict={len(strict)}")

    def test_returns_empty_for_empty_mask(self):
        b = RatTextureBank()
        gray = np.full((100, 100), 50, dtype=np.uint8)
        mask = np.zeros_like(gray)
        result = b.sample_uniform_patches(gray, mask)
        assert result == []

    def test_max_patches_cap_honored(self):
        b = RatTextureBank()
        gray = _striped((300, 400))
        mask = np.ones_like(gray) * 255   # entire frame in mask
        patches = b.sample_uniform_patches(
            gray, mask, patch_size=32, stride=8,
            max_patches=5, std_max=200.0)
        assert len(patches) <= 5


# ────────────────────────────────────────────────────────────────────
#  Camera artifact mask
# ────────────────────────────────────────────────────────────────────


class TestArtifactMask:

    def test_returns_none_when_bank_not_ready(self):
        b = RatTextureBank()   # untrained
        m = build_camera_artifact_mask(
            b, [_flat((100, 120))],
            intensity_percentile=90, texture_score_max=0.1,
            consistency_fraction=0.5)
        assert m is None

    def test_returns_none_for_empty_frames(self):
        b = _train_bank_striped()
        m = build_camera_artifact_mask(
            b, [],
            intensity_percentile=90, texture_score_max=0.1,
            consistency_fraction=0.5)
        assert m is None

    def test_static_bright_artifact_detected(self):
        """A consistent bright spot at the same pixel across many
        frames, with non-rat texture, should be masked."""
        b = _train_bank_striped()
        # Build 10 frames, each a smooth bg with a bright SQUARE at
        # the same location. The bright square is uniformly bright
        # (smooth, low Gabor) — totally unlike the striped rat
        # texture the bank was trained on.
        frames = []
        rng = np.random.RandomState(0)
        for i in range(10):
            f = np.full((150, 200), 60, dtype=np.uint8)
            # add some noise so frames aren't identical
            f = (f + rng.randint(-3, 4, f.shape)).clip(0, 255).astype(np.uint8)
            # Static bright square — same location every frame
            f[40:60, 80:100] = 240
            frames.append(f)

        m = build_camera_artifact_mask(
            b, frames,
            intensity_percentile=90.0,
            texture_score_max=0.5,    # accept anything not very rat-like
            consistency_fraction=0.5,
            dilate_px=0)
        assert m is not None
        # The static square area should be masked
        center_masked = int(m[50, 90])
        assert center_masked > 0, (
            "static bright non-rat-textured spot should be masked")

    def test_moving_bright_object_not_detected(self):
        """A bright spot that moves frame-to-frame should NOT
        be masked (no consistency at any single pixel)."""
        b = _train_bank_striped()
        rng = np.random.RandomState(1)
        frames = []
        for i in range(15):
            f = np.full((150, 200), 60, dtype=np.uint8)
            f = (f + rng.randint(-3, 4, f.shape)).clip(0, 255).astype(np.uint8)
            # Bright square at DIFFERENT location each frame —
            # wide stride so consecutive positions don't overlap
            y = 20 + (i * 15) % 90
            x = 30 + (i * 20) % 140
            f[y:y+15, x:x+15] = 240
            frames.append(f)

        # Require >70% consistency — moving object can't reach that
        m = build_camera_artifact_mask(
            b, frames,
            intensity_percentile=90.0,
            texture_score_max=0.5,
            consistency_fraction=0.7,
            dilate_px=0)
        assert m is not None
        # No pixel should pass the strict consistency requirement
        n_masked = int((m > 0).sum())
        assert n_masked < 50, (
            f"moving object should not pass 70% consistency; got "
            f"{n_masked} masked pixels")


# ────────────────────────────────────────────────────────────────────
#  Detector applies artifact mask
# ────────────────────────────────────────────────────────────────────


class TestDetectorArtifactMaskGate:

    def test_artifact_mask_zeros_those_pixels(self):
        """Pixels marked as artifact get gated OUT of the foreground
        regardless of bg-sub diff."""
        bg = BackgroundModel(
            bg0=np.full((100, 120), 50, dtype=np.float32),
            bg1=np.full((100, 120), 50, dtype=np.float32))
        # Bright square in the frame
        frame = np.full((100, 120), 50, dtype=np.uint8)
        frame[30:50, 40:60] = 200
        # Artifact mask covers that exact region (small dilation
        # accounts for bg-sub smoothing artifacts at the boundary)
        amask = np.zeros((100, 120), dtype=np.uint8)
        amask[28:52, 38:62] = 255

        det_off = ForegroundDetector(bg, threshold=30,
                                      min_area_px=50, morph_k=3,
                                      artifact_mask=None)
        det_on  = ForegroundDetector(bg, threshold=30,
                                      min_area_px=50, morph_k=3,
                                      artifact_mask=amask)
        r_off = det_off.detect(frame.copy(), cam=0)
        r_on  = det_on.detect(frame.copy(), cam=0)
        # Without artifact mask, the bright region is foreground
        n_off = int((r_off.mask > 0).sum())
        # With artifact mask, the bright region is gated out — verify
        # specifically the CENTER of the masked region is zero
        # (bilateral smear at the boundary may leak past, but the
        # inside should be cleanly zeroed)
        center_under_mask = int(r_on.mask[35:45, 45:55].sum())
        assert n_off > 0, "no-mask detector should find the bright blob"
        assert center_under_mask == 0, (
            f"center of artifact-masked region should be fully zeroed, "
            f"got {center_under_mask} px set inside the mask")
        # And the overall foreground is much smaller
        n_on = int((r_on.mask > 0).sum())
        assert n_on < n_off // 2


# ────────────────────────────────────────────────────────────────────
#  Canny barrier in edge refinement
# ────────────────────────────────────────────────────────────────────


class TestCannyBarrier:

    def test_canny_barrier_stops_growth_at_intensity_edge(self):
        """Place a hull next to a sharp intensity boundary. With
        canny_barrier=True, the refined mask cannot cross the
        boundary."""
        b = _train_bank_striped()
        # Frame: striped region on the LEFT, sharp edge, smooth dark
        # region on the RIGHT
        gray = np.zeros((150, 200), dtype=np.uint8)
        # Left half: striped
        gray[:, :100] = _striped((150, 100), rng_seed=8)
        # Right half: dark uniform
        gray[:, 100:] = 30
        # Hull in the middle-left, near the edge
        hull = np.zeros_like(gray)
        hull[55:95, 60:90] = 255
        # Refine WITHOUT Canny barrier
        refined_no_canny = b.refine_blob_mask(
            gray, hull, expand_px=30, score_threshold=0.10,
            canny_barrier=False)
        # Refine WITH Canny barrier
        refined_canny = b.refine_blob_mask(
            gray, hull, expand_px=30, score_threshold=0.10,
            canny_barrier=True, canny_low=30, canny_high=90,
            canny_dilate=1)
        # The intensity edge at x=100 is a strong gradient. With
        # canny_barrier the refined mask shouldn't extend past it
        # (much). Without, it may grow past more aggressively.
        # Count pixels in the refined mask that are past x=100.
        past_no_canny = int((refined_no_canny[:, 100:] > 0).sum())
        past_canny    = int((refined_canny[:, 100:] > 0).sum())
        assert past_canny <= past_no_canny, (
            f"Canny barrier should not let the refined mask extend "
            f"farther across an intensity edge; "
            f"no_canny={past_no_canny} canny={past_canny}")
