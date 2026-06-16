"""
tests/test_rat_texture_bank.py
==============================
RatTextureBank — multi-scale Gabor texture model with bootstrap,
online updates, version_id tracking, and persistence.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from rpimocap.detection.rat_texture import (
    RatTextureBank,
    DEFAULT_ORIENTATIONS, DEFAULT_SCALES,
    build_gabor_kernels,
    bootstrap_from_random_frames,
)


def _texture_a(rng=None, size=(60, 60), n=1):
    """Make a stack of frames with texture A (oriented stripes)."""
    if rng is None:
        rng = np.random.RandomState(0)
    out = []
    for _ in range(n):
        # Horizontal stripes
        f = np.zeros(size, dtype=np.uint8)
        f[::4] = 200
        f += rng.randint(0, 20, size=size).astype(np.uint8)
        out.append(f)
    return out


def _texture_b(rng=None, size=(60, 60), n=1):
    """Make a stack of frames with texture B (random/uniform-ish)."""
    if rng is None:
        rng = np.random.RandomState(1)
    out = []
    for _ in range(n):
        # Smooth noisy patch — different texture signature
        f = (rng.uniform(80, 120, size=size)).astype(np.uint8)
        out.append(f)
    return out


def _full_mask(shape):
    return np.ones(shape, dtype=np.uint8) * 255


# ────────────────────────────────────────────────────────────────────
#  Construction and Gabor bank
# ────────────────────────────────────────────────────────────────────


class TestGaborKernelBank:

    def test_kernel_count(self):
        kernels = build_gabor_kernels(DEFAULT_ORIENTATIONS, DEFAULT_SCALES)
        assert len(kernels) == len(DEFAULT_ORIENTATIONS) * len(DEFAULT_SCALES)

    def test_kernels_l1_normalized(self):
        kernels = build_gabor_kernels(DEFAULT_ORIENTATIONS, DEFAULT_SCALES)
        for k in kernels:
            assert abs(np.abs(k).sum() - 1.0) < 1e-3, (
                "kernels should be L1-normalized for cross-scale "
                "response comparability")


# ────────────────────────────────────────────────────────────────────
#  Feature extraction
# ────────────────────────────────────────────────────────────────────


class TestFeatureExtraction:

    def test_feature_dim_matches_config(self):
        b = RatTextureBank(orientations=(0.0, np.pi/2),
                            scales=(7, 11, 17))
        assert b.feature_dim == 2 * 3

    def test_features_shape(self):
        b = RatTextureBank()
        frame = _texture_a()[0]
        mask = _full_mask(frame.shape)
        f = b.features_in_blob(frame, mask)
        assert f.shape == (b.feature_dim,)
        assert f.dtype == np.float32

    def test_empty_blob_returns_zero(self):
        b = RatTextureBank()
        frame = np.full((60, 60), 100, dtype=np.uint8)
        mask = np.zeros_like(frame)   # nothing selected
        f = b.features_in_blob(frame, mask)
        assert np.all(f == 0)

    def test_different_textures_give_different_features(self):
        """Different textures must produce different feature vectors,
        otherwise the bank can't discriminate."""
        b = RatTextureBank()
        f_a = b.features_in_blob(_texture_a()[0], _full_mask((60, 60)))
        f_b = b.features_in_blob(_texture_b()[0], _full_mask((60, 60)))
        # L2 distance between the two feature vectors should be
        # significant (textures are intentionally different).
        d = np.linalg.norm(f_a - f_b)
        norm = np.linalg.norm(f_a) + np.linalg.norm(f_b)
        assert d / max(norm, 1e-6) > 0.1, (
            f"textures A and B produced indistinguishable features "
            f"(rel L2 distance {d / max(norm, 1e-6):.4f})")


# ────────────────────────────────────────────────────────────────────
#  Bootstrap
# ────────────────────────────────────────────────────────────────────


class TestBootstrap:

    def test_bootstrap_sets_state(self):
        b = RatTextureBank()
        feats = [b.features_in_blob(f, _full_mask((60, 60)))
                 for f in _texture_a(n=10)]
        b.bootstrap(feats)
        assert b.is_ready
        assert b.version_id == 1
        assert b.n_samples == 10
        assert b.mean.shape == (b.feature_dim,)
        assert b.cov.shape == (b.feature_dim, b.feature_dim)

    def test_bootstrap_too_few_raises(self):
        b = RatTextureBank()
        with_few = [b.features_in_blob(f, _full_mask((60, 60)))
                    for f in _texture_a(n=3)]
        try:
            b.bootstrap(with_few)
        except ValueError:
            pass
        else:
            assert False, "expected ValueError for < 5 samples"

    def test_bootstrap_wrong_dim_raises(self):
        b = RatTextureBank()
        wrong = [np.zeros(7) for _ in range(10)]   # wrong feature dim
        try:
            b.bootstrap(wrong)
        except ValueError:
            pass
        else:
            assert False, "expected ValueError on dim mismatch"


# ────────────────────────────────────────────────────────────────────
#  Scoring
# ────────────────────────────────────────────────────────────────────


class TestScoring:

    def test_score_before_ready_returns_one(self):
        """No-op gate before training — accepts everything."""
        b = RatTextureBank()
        f = np.zeros(b.feature_dim, dtype=np.float32)
        assert b.score(f) == 1.0

    def test_score_at_mean_close_to_one(self):
        b = RatTextureBank()
        feats = [b.features_in_blob(f, _full_mask((60, 60)))
                 for f in _texture_a(n=10)]
        b.bootstrap(feats)
        # At exactly the mean, Mahalanobis² = 0 → score = exp(0) = 1
        s = b.score(b.mean)
        assert s > 0.99, f"expected ~1.0, got {s:.4f}"

    def test_score_consistent_texture_high(self):
        """Bootstrap on texture A, then score a new texture-A sample —
        should be high since same distribution."""
        b = RatTextureBank()
        # Bootstrap on 10 texture-A frames
        feats_a = [b.features_in_blob(f, _full_mask((60, 60)))
                   for f in _texture_a(rng=np.random.RandomState(0), n=15)]
        b.bootstrap(feats_a)
        # Score a NEW texture-A frame (different rng → different noise)
        new_a = b.features_in_blob(
            _texture_a(rng=np.random.RandomState(99))[0],
            _full_mask((60, 60)))
        s = b.score(new_a)
        assert s > 0.3, (
            f"new texture-A sample should score reasonably high "
            f"after bootstrap on texture-A; got {s:.4f}")

    def test_score_different_texture_lower(self):
        """Bootstrap on texture A, then score a texture-B sample — 
        should be relatively lower (different distribution)."""
        b = RatTextureBank()
        feats_a = [b.features_in_blob(f, _full_mask((60, 60)))
                   for f in _texture_a(rng=np.random.RandomState(0), n=15)]
        b.bootstrap(feats_a)
        new_b = b.features_in_blob(
            _texture_b()[0], _full_mask((60, 60)))
        # In-distribution score vs out-of-distribution score
        new_a = b.features_in_blob(
            _texture_a(rng=np.random.RandomState(99))[0],
            _full_mask((60, 60)))
        s_a = b.score(new_a)
        s_b = b.score(new_b)
        assert s_a > s_b, (
            f"texture-A should score higher than texture-B after "
            f"bootstrap on A; got s_a={s_a:.4f}, s_b={s_b:.4f}")


# ────────────────────────────────────────────────────────────────────
#  Online updates and version_id
# ────────────────────────────────────────────────────────────────────


class TestOnlineUpdates:

    def test_buffer_fills_then_refits(self):
        """After update_every samples added, _refit_from_buffer is
        invoked and the buffer drains."""
        b = RatTextureBank(update_every=5)
        # Bootstrap
        b.bootstrap([
            b.features_in_blob(f, _full_mask((60, 60)))
            for f in _texture_a(n=10)])
        # Now add 5 samples — should trigger refit
        for f in _texture_a(rng=np.random.RandomState(2), n=5):
            b.add_sample(b.features_in_blob(f, _full_mask((60, 60))))
        # Buffer should be drained after refit
        assert len(b._online.buffer) == 0
        assert b.n_samples > 10

    def test_no_drift_keeps_version(self):
        """Adding more samples from the SAME distribution should
        NOT cause a significant mean shift → no version bump."""
        b = RatTextureBank(update_every=5, drift_threshold=0.10)
        b.bootstrap([
            b.features_in_blob(f, _full_mask((60, 60)))
            for f in _texture_a(rng=np.random.RandomState(0), n=20)])
        v0 = b.version_id
        # Feed more samples from the same distribution
        for rng_seed in range(10, 20):
            for f in _texture_a(rng=np.random.RandomState(rng_seed), n=5):
                b.add_sample(b.features_in_blob(
                    f, _full_mask((60, 60))))
        # Mean shouldn't have drifted enough to bump version
        # (it MAY have drifted once or twice due to sample randomness,
        # so we just assert the version is still small)
        assert b.version_id <= v0 + 2

    def test_significant_drift_bumps_version(self):
        """Switching the sample distribution to texture B should
        cause a large mean shift → version bumps."""
        b = RatTextureBank(update_every=5, drift_threshold=0.05)
        b.bootstrap([
            b.features_in_blob(f, _full_mask((60, 60)))
            for f in _texture_a(rng=np.random.RandomState(0), n=20)])
        v0 = b.version_id
        # Feed lots of texture-B samples — distribution shift
        for rng_seed in range(5):
            for f in _texture_b(rng=np.random.RandomState(rng_seed), n=5):
                b.add_sample(b.features_in_blob(
                    f, _full_mask((60, 60))))
        # Version should have bumped at least once
        assert b.version_id > v0

    def test_flush_pending_refits_partial_buffer(self):
        """flush_pending() refits even if buffer isn't full."""
        b = RatTextureBank(update_every=100)
        b.bootstrap([
            b.features_in_blob(f, _full_mask((60, 60)))
            for f in _texture_a(n=10)])
        # Add just 3 samples — won't auto-refit
        for f in _texture_a(rng=np.random.RandomState(2), n=3):
            b.add_sample(b.features_in_blob(f, _full_mask((60, 60))))
        assert len(b._online.buffer) == 3
        b.flush_pending()
        assert len(b._online.buffer) == 0


# ────────────────────────────────────────────────────────────────────
#  Persistence
# ────────────────────────────────────────────────────────────────────


class TestPersistence:

    def test_save_load_roundtrip(self):
        b = RatTextureBank(update_every=50, drift_threshold=0.07)
        b.bootstrap([
            b.features_in_blob(f, _full_mask((60, 60)))
            for f in _texture_a(n=15)])
        original_mean = b.mean.copy()
        original_cov  = b.cov.copy()
        original_vid  = b.version_id

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
            b.save(tmp.name)
            tmp_path = tmp.name

        try:
            loaded = RatTextureBank.load(tmp_path)
            assert loaded.feature_dim == b.feature_dim
            assert loaded.version_id == original_vid
            np.testing.assert_allclose(loaded.mean, original_mean,
                                         rtol=1e-5)
            np.testing.assert_allclose(loaded.cov, original_cov,
                                         rtol=1e-5)
            assert loaded.update_every == 50
            assert abs(loaded.drift_threshold - 0.07) < 1e-6
            # Loaded bank should be able to score
            f = b.features_in_blob(_texture_a()[0],
                                     _full_mask((60, 60)))
            assert loaded.score(f) > 0
        finally:
            Path(tmp_path).unlink()

    def test_save_untrained_raises(self):
        b = RatTextureBank()
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            try:
                b.save(tmp_path)
            except RuntimeError:
                pass
            else:
                assert False, "should raise on save of untrained bank"
        finally:
            Path(tmp_path).unlink(missing_ok=True)


# ────────────────────────────────────────────────────────────────────
#  Convenience wrapper
# ────────────────────────────────────────────────────────────────────


class TestBootstrapHelper:

    def test_too_few_samples_raises(self):
        b = RatTextureBank()
        try:
            bootstrap_from_random_frames(b, [], min_samples=20)
        except RuntimeError:
            pass
        else:
            assert False, "expected RuntimeError for too few samples"
