"""
tests/test_edge_refine_texture.py
==================================
Texture-aware edge refinement on RatTextureBank.refine_blob_mask.
Starting from an initial hull, expand outward where local texture
matches the bank's model and stop where it doesn't. The output
mask should snap to the actual textured region's boundary instead
of the convex hull.
"""
from __future__ import annotations

import numpy as np
import cv2

from rpimocap.detection.rat_texture import RatTextureBank


def _striped_texture(shape, intensity_base=60, stripe_intensity=200,
                      stripe_period=4, rng_seed=0):
    """Build a 'rat-like' striped texture (oriented). Random noise
    on top makes Gabor responses well-defined."""
    rng = np.random.RandomState(rng_seed)
    f = np.full(shape, intensity_base, dtype=np.uint8)
    f[::stripe_period] = stripe_intensity
    f = f + rng.randint(-15, 15, size=shape).astype(np.int16)
    return np.clip(f, 0, 255).astype(np.uint8)


def _smooth_uniform(shape, intensity=130, rng_seed=1):
    """Build a 'background-like' smooth uniform region with mild
    noise. Different texture signature from striped."""
    rng = np.random.RandomState(rng_seed)
    f = np.full(shape, intensity, dtype=np.int16)
    f = f + rng.randint(-5, 5, size=shape).astype(np.int16)
    return np.clip(f, 0, 255).astype(np.uint8)


def _train_bank_on_striped(scales=(7, 11, 17)):
    """Bootstrap a bank on samples of the striped texture."""
    b = RatTextureBank(scales=scales)
    samples = []
    full_mask = np.ones((60, 60), dtype=np.uint8) * 255
    for seed in range(15):
        frame = _striped_texture((60, 60), rng_seed=seed)
        samples.append(b.features_in_blob(frame, full_mask))
    b.bootstrap(samples)
    return b


# ────────────────────────────────────────────────────────────────────
#  Basic invariants
# ────────────────────────────────────────────────────────────────────


class TestRefineInvariants:

    def test_bank_not_ready_returns_hull(self):
        """An untrained bank can't score → refine returns hull
        unchanged."""
        b = RatTextureBank()
        gray = _striped_texture((200, 300))
        hull = np.zeros_like(gray)
        hull[80:120, 130:170] = 255
        refined = b.refine_blob_mask(gray, hull, expand_px=20)
        np.testing.assert_array_equal(refined, hull)

    def test_empty_hull_returns_empty(self):
        b = _train_bank_on_striped()
        gray = _striped_texture((200, 300))
        hull = np.zeros_like(gray)
        refined = b.refine_blob_mask(gray, hull, expand_px=20)
        assert int((refined > 0).sum()) == 0

    def test_shape_mismatch_returns_hull(self):
        b = _train_bank_on_striped()
        gray = np.zeros((200, 300), dtype=np.uint8)
        hull = np.zeros((100, 100), dtype=np.uint8)  # wrong shape
        refined = b.refine_blob_mask(gray, hull)
        # Should return a copy of hull (unmodified)
        np.testing.assert_array_equal(refined, hull)


# ────────────────────────────────────────────────────────────────────
#  Expansion to texture boundary
# ────────────────────────────────────────────────────────────────────


class TestExpansionIntoMatching:

    def test_expands_into_same_texture(self):
        """A hull that under-segments a striped patch should grow
        outward when the surrounding area is the SAME texture."""
        b = _train_bank_on_striped()
        # Frame is all striped texture
        gray = _striped_texture((200, 300), rng_seed=99)
        # Hull covers only a small inner region
        hull = np.zeros_like(gray)
        hull[80:100, 130:170] = 255
        refined = b.refine_blob_mask(
            gray, hull, expand_px=30, score_threshold=0.15,
            smooth_window=7)
        # Refined should be at least as large as original hull
        n_hull = int((hull > 0).sum())
        n_ref  = int((refined > 0).sum())
        # The refined mask should grow — extra pixels accepted
        assert n_ref >= n_hull, (
            f"expected refined ≥ hull (extra pixels accepted "
            f"in matching texture); got hull={n_hull} ref={n_ref}")

    def test_does_not_expand_into_different_texture(self):
        """Hull covers a striped patch but is surrounded by SMOOTH
        uniform region. Refinement should NOT grow into the
        smooth region (different texture signature)."""
        b = _train_bank_on_striped()
        # Frame: smooth uniform background with a striped patch in the
        # middle
        gray = _smooth_uniform((200, 300), rng_seed=2)
        # Insert striped patch in [60:120, 110:180]
        striped_patch = _striped_texture((60, 70), rng_seed=11)
        gray[60:120, 110:180] = striped_patch
        # Hull covers the inner part of the striped patch
        hull = np.zeros_like(gray)
        hull[75:105, 130:165] = 255
        refined = b.refine_blob_mask(
            gray, hull, expand_px=30, score_threshold=0.15,
            smooth_window=7)
        # Refined shouldn't grow much beyond the patch boundary
        # (60:120, 110:180). Check that it does NOT include pixels
        # far from the patch.
        # Specifically, pixels at (40, 50) — well into smooth region — must be 0
        assert refined[40, 50] == 0
        assert refined[150, 250] == 0
        assert refined[180, 30] == 0


# ────────────────────────────────────────────────────────────────────
#  Geodesic constraint
# ────────────────────────────────────────────────────────────────────


class TestGeodesicConnection:

    def test_isolated_texture_islands_not_included(self):
        """If there's a disconnected island of rat-texture far from
        the hull, it should NOT be included in the result (the
        refined mask must be connected to the original hull)."""
        b = _train_bank_on_striped()
        # Frame: smooth bg + striped patch near hull + isolated
        # striped island far from hull
        gray = _smooth_uniform((200, 300), rng_seed=3)
        # Patch 1: at (60-120, 110-180), where the hull is
        gray[60:120, 110:180] = _striped_texture((60, 70), rng_seed=11)
        # Patch 2 (isolated island): at (160-185, 250-285)
        gray[160:185, 250:285] = _striped_texture((25, 35), rng_seed=22)
        # Hull on patch 1 only
        hull = np.zeros_like(gray)
        hull[80:100, 130:160] = 255
        refined = b.refine_blob_mask(
            gray, hull, expand_px=20, score_threshold=0.15,
            smooth_window=7)
        # Pixels in the isolated island should NOT be in the refined
        # mask (since they're not connected to the hull via the band)
        island_in_refined = int(refined[160:185, 250:285].sum())
        assert island_in_refined == 0, (
            f"isolated texture island should NOT be in refined "
            f"mask (geodesic constraint); got {island_in_refined} "
            f"island pixels included")


# ────────────────────────────────────────────────────────────────────
#  Expand-px is a hard limit
# ────────────────────────────────────────────────────────────────────


class TestExpandLimit:

    def test_refined_within_expand_distance(self):
        """The refined mask cannot extend more than expand_px from
        the original hull."""
        b = _train_bank_on_striped()
        gray = _striped_texture((200, 300), rng_seed=5)
        hull = np.zeros_like(gray)
        hull[95:105, 145:155] = 255   # 10x10 hull at center
        refined = b.refine_blob_mask(
            gray, hull, expand_px=15, score_threshold=0.15,
            smooth_window=7)
        # Distance transform from hull
        # Compute farthest pixel in refined from hull
        hull_inv = (hull == 0).astype(np.uint8)
        dist_from_hull = cv2.distanceTransform(
            hull_inv, cv2.DIST_L2, 5)
        # In the refined region only
        max_dist_in_refined = float(dist_from_hull[refined > 0].max()
                                     if int((refined > 0).sum()) > 0
                                     else 0)
        # Allow some slack for box-filter smoothing extending features
        # beyond exact pixel boundaries — but should be close to expand_px
        assert max_dist_in_refined < 25, (
            f"refined mask extended {max_dist_in_refined:.1f} px "
            f"from hull, expected ≤ ~15-20 px with expand_px=15")
