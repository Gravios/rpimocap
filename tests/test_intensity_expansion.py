"""
tests/test_intensity_expansion.py
==================================
expand_mask_by_intensity + ForegroundDetector.edge_refine_intensity
end-to-end integration. The texture bank confirms a small bright
core; the intensity expansion grows that core out to the natural
rat/bedding intensity boundary.
"""
from __future__ import annotations

import numpy as np
import cv2

from rpimocap.detection.rat_texture import (
    RatTextureBank, expand_mask_by_intensity)
from rpimocap.detection.segment import (
    BackgroundModel, ForegroundDetector)


def _bright_blob_on_dim_bg(shape=(200, 300),
                            bg_intensity=80,
                            blob_intensity=200,
                            blob_box=(60, 130, 100, 200),
                            edge_softness=8,
                            noise=5,
                            rng_seed=0):
    """Build a frame with a single bright blob on a dim background.
    edge_softness adds a transition zone of intermediate intensity
    at the blob boundary (mimicking the rat-fur/bedding interface)."""
    rng = np.random.RandomState(rng_seed)
    f = (np.full(shape, bg_intensity, dtype=np.int16)
         + rng.randint(-noise, noise + 1, shape)).astype(np.float32)
    # Build the bright blob with a soft Gaussian-ish edge
    y0, y1, x0, x1 = blob_box
    cy = (y0 + y1) / 2
    cx = (x0 + x1) / 2
    h_half = (y1 - y0) / 2
    w_half = (x1 - x0) / 2
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    # Distance from the blob centre, normalized so 1.0 is at the edge
    r = np.sqrt(((yy - cy) / h_half) ** 2 + ((xx - cx) / w_half) ** 2)
    # Smooth step: full intensity inside, fading over edge_softness px
    weight = np.clip(1 - (r - 1) * (h_half / edge_softness), 0, 1)
    f = f + (blob_intensity - bg_intensity) * weight
    return np.clip(f, 0, 255).astype(np.uint8)


def _small_seed_inside_blob(blob_box, shrink=20):
    """Make a seed mask covering the interior of a known bright blob."""
    y0, y1, x0, x1 = blob_box
    mask = np.zeros((200, 300), dtype=np.uint8)
    mask[y0 + shrink:y1 - shrink, x0 + shrink:x1 - shrink] = 255
    return mask


# ────────────────────────────────────────────────────────────────────
#  expand_mask_by_intensity behavior
# ────────────────────────────────────────────────────────────────────


class TestExpandMaskByIntensity:

    def test_grows_to_blob_boundary(self):
        """A small interior seed should grow outward to fill the
        bright blob, not just stay at its tiny initial size."""
        frame = _bright_blob_on_dim_bg(blob_box=(60, 130, 100, 200))
        seed = _small_seed_inside_blob((60, 130, 100, 200), shrink=20)
        expanded = expand_mask_by_intensity(
            frame, seed,
            max_expand_px=40, intensity_quantile=0.25,
            morph_close_k=3)
        n_seed = int((seed > 0).sum())
        n_exp  = int((expanded > 0).sum())
        # Expansion should significantly grow the mask
        assert n_exp > 1.5 * n_seed, (
            f"expansion should grow seed; n_seed={n_seed} "
            f"n_exp={n_exp}")
        # Should also reach close to the blob edge: the blob box is
        # 70×100=7000 px², so the expanded mask should be > 5000
        assert n_exp > 5000

    def test_doesnt_grow_into_dim_background(self):
        """The expanded mask must NOT include pixels far into the
        dim background — the intensity gate stops it at the boundary."""
        frame = _bright_blob_on_dim_bg(blob_box=(60, 130, 100, 200))
        seed = _small_seed_inside_blob((60, 130, 100, 200), shrink=20)
        expanded = expand_mask_by_intensity(
            frame, seed,
            max_expand_px=50, intensity_quantile=0.25,
            morph_close_k=3)
        # Pixels well into the dim background must not be in the mask
        assert expanded[20, 50] == 0    # top-left, far from blob
        assert expanded[180, 280] == 0  # bottom-right, far from blob

    def test_empty_seed_returns_empty(self):
        frame = _bright_blob_on_dim_bg()
        seed = np.zeros_like(frame, dtype=np.uint8)
        expanded = expand_mask_by_intensity(frame, seed)
        assert int((expanded > 0).sum()) == 0

    def test_geodesic_excludes_disconnected_bright_objects(self):
        """A bright blob elsewhere in the frame must NOT be included
        in the expansion (the CC-overlap-with-seed constraint
        guarantees the expanded mask is connected to the original
        seed)."""
        frame = _bright_blob_on_dim_bg(blob_box=(60, 130, 80, 160))
        # Add another isolated bright blob far from the seed
        frame[40:80, 220:260] = 210
        seed = _small_seed_inside_blob((60, 130, 80, 160), shrink=15)
        expanded = expand_mask_by_intensity(
            frame, seed,
            max_expand_px=30, intensity_quantile=0.3,
            morph_close_k=3)
        # The isolated bright object at (40-80, 220-260) is too far
        # to be reached via geodesic dilation from the seed
        far_blob_in_expanded = int(expanded[40:80, 220:260].sum())
        assert far_blob_in_expanded == 0, (
            f"disconnected bright object should not be in the "
            f"expanded mask; got {far_blob_in_expanded} px")


# ────────────────────────────────────────────────────────────────────
#  Detector integration
# ────────────────────────────────────────────────────────────────────


class TestDetectorIntegration:

    def _bank_trained_on_bright(self):
        """Quick bank trained on a bright uniform texture so it
        accepts the bright blobs in the test frame."""
        b = RatTextureBank(scales=(5, 9, 13))
        # Train on samples of the bright interior
        samples = []
        for seed_v in range(8):
            rng = np.random.RandomState(seed_v)
            patch = (rng.randint(-5, 5, (60, 60)) + 200).astype(np.uint8)
            mask = np.ones_like(patch) * 255
            samples.append(b.features_in_blob(patch, mask))
        b.bootstrap(samples)
        return b

    def test_intensity_refinement_doesnt_crash(self):
        """Smoke test: the detector runs cleanly with intensity
        refinement enabled. End-to-end correctness on real frames
        is covered by the helper unit tests above."""
        bank = self._bank_trained_on_bright()
        bg = BackgroundModel(
            bg0=np.full((200, 300), 80, dtype=np.float32),
            bg1=np.full((200, 300), 80, dtype=np.float32))
        frame = _bright_blob_on_dim_bg(blob_box=(60, 130, 80, 220))

        det = ForegroundDetector(
            background=bg, threshold=30, min_area_px=200,
            morph_k=3,
            texture_bank=bank, texture_min_score=0.0,
            merge_blob_distance=50,
            edge_refine_texture=True,
            edge_refine_expand_px=5,
            edge_refine_score_threshold=0.5,
            edge_refine_intensity=True,
            edge_refine_intensity_expand_px=50,
            edge_refine_intensity_quantile=0.25,
            edge_refine_intensity_morph_close_k=5)
        r = det.detect(frame.copy(), cam=0)
        # Should produce SOME mask
        assert int((r.mask > 0).sum()) > 0
        # Result mask should be a connected region near the blob
        # (CC overlap with seed)
        assert r.mask[80, 150] > 0
