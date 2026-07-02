"""Tests for Laplacian-domain edge persistence + persistence fusion.

The property under test: edge persistence is a stable-vs-transient map in
the LoG domain — perfectly static structure (rails/frame) and flat regions
read persistent (~1), a swept object reads transient (~0). Fusion is a
probabilistic-OR that is a no-op at weight 0 and never lowers persistence.
"""
import numpy as np

from rpimocap.detection.texture_distance import (
    build_edge_persistence, combine_persistence)


def _synthetic_sequence(H=120, W=160, T=30, seed=0):
    """Static rails present every frame + a bright square sweeping across."""
    rng = np.random.default_rng(seed)
    frames = []
    for t in range(T):
        img = np.full((H, W), 90, np.float32)
        img += rng.normal(0, 1.5, (H, W)).astype(np.float32)
        img[:, 40:42] = 220          # static vertical rail
        img[30:32, :] = 220          # static horizontal rail
        cx = 10 + t * 4              # moving object
        img[74:88, cx:cx + 14] = 240
        frames.append(np.clip(img, 0, 255).astype(np.uint8))
    return frames


class TestEdgePersistence:

    def test_static_structure_reads_persistent(self):
        frames = _synthetic_sequence()
        roi = np.ones((120, 160), np.uint8) * 255
        ep = build_edge_persistence(frames, log_sigma=2.0, roi_mask=roi)
        static_edge = float(np.mean([ep[31, 100], ep[60, 41], ep[31, 41]]))
        flat = float(np.mean([ep[10, 120], ep[110, 10]]))
        swept = float(ep[74:88, 20:150].mean())
        assert static_edge > 0.75
        assert flat > 0.75
        assert swept < 0.55
        # the discriminating property: static structure >> moving object
        assert (static_edge - swept) > 0.35

    def test_shape_and_range(self):
        frames = _synthetic_sequence(H=64, W=48, T=12)
        ep = build_edge_persistence(frames, log_sigma=1.5)
        assert ep.shape == (64, 48)
        assert ep.dtype == np.float32
        assert ep.min() >= 0.0 and ep.max() <= 1.0

    def test_roi_zeroes_outside(self):
        frames = _synthetic_sequence()
        roi = np.zeros((120, 160), np.uint8)
        roi[20:100, 20:140] = 255
        ep = build_edge_persistence(frames, log_sigma=2.0, roi_mask=roi)
        assert ep[roi == 0].max() == 0.0
        assert ep[roi > 0].max() > 0.0

    def test_raw_laplacian_when_sigma_zero(self):
        frames = _synthetic_sequence(T=10)
        ep = build_edge_persistence(frames, log_sigma=0.0)
        assert ep.shape == (120, 160)
        assert np.isfinite(ep).all()

    def test_accepts_color_frames_via_green(self):
        frames = [np.stack([f, f, f], axis=-1)
                  for f in _synthetic_sequence(T=8)]
        ep = build_edge_persistence(frames, log_sigma=1.5)
        assert ep.shape == (120, 160)


class TestCombinePersistence:

    def test_weight_zero_is_noop(self):
        rng = np.random.default_rng(1)
        tex = rng.random((40, 50)).astype(np.float32)
        edge = rng.random((40, 50)).astype(np.float32)
        out = combine_persistence(tex, edge, edge_weight=0.0)
        assert np.allclose(out, tex)

    def test_fusion_never_reduces_persistence(self):
        rng = np.random.default_rng(2)
        tex = rng.random((40, 50)).astype(np.float32)
        edge = rng.random((40, 50)).astype(np.float32)
        out = combine_persistence(tex, edge, edge_weight=1.0)
        assert (out >= tex - 1e-6).all()
        assert out.max() <= 1.0 + 1e-6

    def test_fusion_monotonic_in_edge(self):
        tex = np.full((3, 3), 0.4, np.float32)
        lo = np.full((3, 3), 0.1, np.float32)
        hi = np.full((3, 3), 0.9, np.float32)
        assert (combine_persistence(tex, hi, 1.0)
                > combine_persistence(tex, lo, 1.0)).all()
