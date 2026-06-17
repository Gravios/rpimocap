"""
tests/test_rotation_invariance.py
==================================
Rotation-invariant Gabor features for RatTextureBank, and the
save/load roundtrip preserving the rotation_invariant flag.

The motivating problem: with legacy directional features the bank's
learned feature vector drifts as the rat rotates (a dominant
orientation moves between Gabor bins). Pooling per-pixel across
orientation (max, mean, std) before spatial averaging makes the
feature vector invariant to the texture's global orientation.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import cv2

from rpimocap.detection.rat_texture import RatTextureBank


# ────────────────────────────────────────────────────────────────────
#  Helpers
# ────────────────────────────────────────────────────────────────────


def _oriented_stripes(shape=(120, 120),
                       angle_deg=0.0,
                       period=10,
                       base=80,
                       stripe=200,
                       noise=8,
                       rng_seed=0) -> np.ndarray:
    """Create a stripe pattern at the given orientation.

    Rotating the pattern lets us test rotation invariance — the
    features computed on the same stripes at different angles
    should be (close to) equal for a rotation-invariant model and
    very different for a directional one.
    """
    rng = np.random.RandomState(rng_seed)
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = w / 2, h / 2
    a = np.deg2rad(angle_deg)
    u = (xx - cx) * np.cos(a) + (yy - cy) * np.sin(a)
    pattern = ((u // (period / 2)) % 2).astype(np.float32)
    img = base + (stripe - base) * pattern
    img = img + rng.randn(*shape) * noise
    return np.clip(img, 0, 255).astype(np.uint8)


def _full_mask(shape, margin=10):
    """Mask covering the centre of the frame (avoid rotation
    border artifacts when comparing features)."""
    h, w = shape
    m = np.zeros(shape, dtype=np.uint8)
    m[margin:h - margin, margin:w - margin] = 255
    return m


# ────────────────────────────────────────────────────────────────────
#  Rotation invariance
# ────────────────────────────────────────────────────────────────────


class TestRotationInvariance:

    def test_features_constant_under_rotation_rotinv(self):
        """With rotation_invariant=True, the same striped texture
        rotated to different angles produces nearly-identical
        feature vectors (within tolerance for filter boundary
        effects)."""
        bank = RatTextureBank(rotation_invariant=True,
                                scales=(7, 11, 17))
        feats_by_angle = []
        for angle in (0, 30, 45, 60, 90, 120, 150):
            img = _oriented_stripes(angle_deg=angle, rng_seed=42)
            mask = _full_mask(img.shape)
            f = bank.features_in_blob(img, mask)
            feats_by_angle.append(f)
        feats_arr = np.stack(feats_by_angle, axis=0)  # (N_angles, D)
        # Per-feature coefficient of variation across angles
        f_mean = feats_arr.mean(axis=0)
        f_std  = feats_arr.std(axis=0)
        # Avoid divide by zero on near-zero features
        cv = f_std / (f_mean + 1e-6)
        # For a truly rotation-invariant texture statistic, CV
        # should be small (well under 0.2). Loose threshold to
        # tolerate stripe-period interaction with filter bandwidth.
        assert cv.max() < 0.25, (
            f"rotation-invariant features should vary little with "
            f"angle; max CV = {cv.max():.3f}, per-feature CV = "
            f"{cv.round(3).tolist()}")

    def test_features_vary_with_rotation_legacy(self):
        """In legacy mode, the same texture rotated produces
        significantly different feature vectors — the contrast
        with the rotation-invariant test above."""
        bank = RatTextureBank(rotation_invariant=False,
                                scales=(7, 11, 17))
        feats_by_angle = []
        for angle in (0, 45, 90):
            img = _oriented_stripes(angle_deg=angle, rng_seed=42)
            mask = _full_mask(img.shape)
            f = bank.features_in_blob(img, mask)
            feats_by_angle.append(f)
        feats_arr = np.stack(feats_by_angle, axis=0)
        f_mean = feats_arr.mean(axis=0)
        f_std  = feats_arr.std(axis=0)
        cv = f_std / (f_mean + 1e-6)
        # Legacy features SHOULD vary across angle since each
        # feature is tied to a specific Gabor orientation. We
        # expect at least one feature to have CV > 0.2.
        assert cv.max() > 0.2, (
            f"legacy directional features should vary with rotation; "
            f"max CV = {cv.max():.3f}")


# ────────────────────────────────────────────────────────────────────
#  Feature dimension
# ────────────────────────────────────────────────────────────────────


class TestFeatureDim:

    def test_rotinv_feature_dim(self):
        b = RatTextureBank(orientations=(0.0, np.pi/4, np.pi/2),
                            scales=(7, 11, 17, 23),
                            rotation_invariant=True)
        # 3 stats × 4 scales = 12 (independent of n_orient)
        assert b.feature_dim == 12

    def test_legacy_feature_dim(self):
        b = RatTextureBank(orientations=(0.0, np.pi/4, np.pi/2),
                            scales=(7, 11, 17, 23),
                            rotation_invariant=False)
        # n_orient × n_scales = 3 × 4 = 12
        assert b.feature_dim == 12

    def test_default_is_rotation_invariant(self):
        """The constructor's default behaviour is the rotation-
        invariant mode."""
        b = RatTextureBank(scales=(7, 11, 17))
        assert b.rotation_invariant is True
        assert b.feature_dim == 3 * 3


# ────────────────────────────────────────────────────────────────────
#  Discrimination: isotropic vs oriented textures
# ────────────────────────────────────────────────────────────────────


class TestDiscrimination:

    def test_isotropic_vs_oriented_have_different_std_feature(self):
        """Isotropic noise (bedding-like) has near-zero per-pixel
        std across orientations. Strongly oriented stripes
        (fur-grain-like) have higher std. The std feature should
        therefore separate them."""
        bank = RatTextureBank(rotation_invariant=True,
                                scales=(7, 11, 17))
        # Isotropic: random noise of same intensity range as stripes
        rng = np.random.RandomState(0)
        iso = (np.full((120, 120), 140, dtype=np.int16) +
                rng.randint(-40, 40, (120, 120))).astype(np.uint8)
        iso = np.clip(iso, 0, 255).astype(np.uint8)
        # Oriented
        ori = _oriented_stripes(angle_deg=30, rng_seed=1)

        mask = _full_mask(iso.shape)
        f_iso = bank.features_in_blob(iso, mask)
        f_ori = bank.features_in_blob(ori, mask)

        # std-across-orientation features are at indices 2, 5, 8
        # (the 3rd of every 3-tuple per scale)
        std_iso = f_iso[2::3]
        std_ori = f_ori[2::3]
        # Mean of std-features should be larger for oriented texture
        assert std_ori.mean() > std_iso.mean(), (
            f"oriented texture should have larger std-across-"
            f"orientation features than isotropic noise; got "
            f"oriented={std_ori.mean():.3f} "
            f"isotropic={std_iso.mean():.3f}")


# ────────────────────────────────────────────────────────────────────
#  Save / load preserves the flag
# ────────────────────────────────────────────────────────────────────


class TestSaveLoadPreservesFlag:

    def _train_and_save_load(self, rotation_invariant: bool):
        """Train a small bank, save, load, verify the flag round-trips
        and feature_dim matches."""
        bank = RatTextureBank(rotation_invariant=rotation_invariant,
                                scales=(7, 11, 17))
        # Generate a handful of feature samples from oriented patches
        samples = []
        for seed in range(8):
            img = _oriented_stripes(angle_deg=seed * 20, rng_seed=seed)
            mask = _full_mask(img.shape)
            samples.append(bank.features_in_blob(img, mask))
        bank.bootstrap(samples)
        assert bank.is_ready

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "bank.npz"
            bank.save(p)
            loaded = RatTextureBank.load(p)
            assert loaded.rotation_invariant == rotation_invariant
            assert loaded.feature_dim == bank.feature_dim
            np.testing.assert_allclose(loaded.mean, bank.mean, atol=1e-5)
            np.testing.assert_allclose(loaded.cov,  bank.cov,  atol=1e-5)

    def test_roundtrip_rotinv_true(self):
        self._train_and_save_load(rotation_invariant=True)

    def test_roundtrip_rotinv_false(self):
        self._train_and_save_load(rotation_invariant=False)

    def test_legacy_npz_without_flag_loads_as_legacy(self):
        """An older saved bank (no rotation_invariant key in the
        file) must load as legacy (False) for backward-compat."""
        # Build a legacy bank, save, then strip the flag from the
        # saved file to simulate an older save.
        bank = RatTextureBank(rotation_invariant=False,
                                scales=(7, 11, 17))
        samples = [bank.features_in_blob(_oriented_stripes(
            angle_deg=a, rng_seed=i), _full_mask((120, 120)))
            for i, a in enumerate((0, 30, 60, 90, 120, 150))]
        bank.bootstrap(samples)
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "legacy_bank.npz"
            bank.save(p)
            # Re-save WITHOUT the flag, mimicking an older bank
            data = dict(np.load(p))
            del data["rotation_invariant"]
            np.savez(p, **data)
            loaded = RatTextureBank.load(p)
            assert loaded.rotation_invariant is False
            assert loaded.feature_dim == len(bank.orientations) \
                                          * len(bank.scales)


# ────────────────────────────────────────────────────────────────────
#  refine_blob_mask works in both modes
# ────────────────────────────────────────────────────────────────────


class TestRefineBlobMaskBothModes:

    def test_refine_runs_in_rotinv_mode(self):
        bank = RatTextureBank(rotation_invariant=True,
                                scales=(7, 11, 17))
        samples = [bank.features_in_blob(_oriented_stripes(
            angle_deg=a, rng_seed=i), _full_mask((120, 120)))
            for i, a in enumerate((0, 30, 60, 90, 120))]
        bank.bootstrap(samples)
        # A frame with matching texture; small hull seed
        frame = _oriented_stripes((180, 240), angle_deg=45, rng_seed=99)
        hull = np.zeros_like(frame)
        hull[80:100, 110:140] = 255
        refined = bank.refine_blob_mask(
            frame, hull, expand_px=25, score_threshold=0.10,
            smooth_window=5)
        # Should at least preserve the hull and may expand into
        # matching texture
        assert int((refined > 0).sum()) >= int((hull > 0).sum())

    def test_refine_runs_in_legacy_mode(self):
        bank = RatTextureBank(rotation_invariant=False,
                                scales=(7, 11, 17))
        samples = [bank.features_in_blob(_oriented_stripes(
            angle_deg=a, rng_seed=i), _full_mask((120, 120)))
            for i, a in enumerate((0, 15, 30, 45, 60, 75, 90))]
        bank.bootstrap(samples)
        frame = _oriented_stripes((180, 240), angle_deg=0, rng_seed=99)
        hull = np.zeros_like(frame)
        hull[80:100, 110:140] = 255
        refined = bank.refine_blob_mask(
            frame, hull, expand_px=25, score_threshold=0.10,
            smooth_window=5)
        assert int((refined > 0).sum()) >= int((hull > 0).sum())
