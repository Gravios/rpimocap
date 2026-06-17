"""
tests/test_intensity_floor_offset.py
====================================
expand_mask_by_intensity with intensity_floor_offset — verify that
the offset lets the threshold reach below the seed's quantile so
the mask can grow into dimmer rat-edge pixels.
"""
from __future__ import annotations

import numpy as np

from rpimocap.detection.rat_texture import expand_mask_by_intensity


def _build_frame_with_bright_core_and_dim_halo():
    """Build a frame where:
    - background: ~80 intensity
    - rat boundary fur (halo around core): ~170
    - rat core (bright interior): ~210-220
    A small seed inside the rat core has a 25th percentile of ~210.
    Without offset, the threshold (~210) excludes the halo (~170).
    With offset 40, the threshold drops to ~170 and the halo is
    included.
    """
    rng = np.random.RandomState(0)
    shape = (200, 300)
    f = (np.full(shape, 80, dtype=np.int16)
         + rng.randint(-3, 3, shape)).astype(np.int16)
    # Halo (dim fur): a big rectangle of intensity 170
    f[60:140, 80:220] = 170 + rng.randint(-3, 3, (80, 140))
    # Bright core: a smaller rectangle inside, intensity 215
    f[80:120, 130:170] = 215 + rng.randint(-3, 3, (40, 40))
    return np.clip(f, 0, 255).astype(np.uint8)


class TestFloorOffset:

    def test_no_offset_keeps_threshold_at_seed_quantile(self):
        """With offset=0 (back-compat default), the threshold equals
        the seed's 25th percentile. Halo pixels (intensity ~170) are
        BELOW that threshold (~215) → halo NOT included → expanded
        mask stays inside the bright core."""
        frame = _build_frame_with_bright_core_and_dim_halo()
        # Seed inside the bright core (intensity ~215)
        seed = np.zeros_like(frame, dtype=np.uint8)
        seed[90:110, 140:160] = 255
        expanded = expand_mask_by_intensity(
            frame, seed,
            max_expand_px=60,
            intensity_quantile=0.25,
            intensity_floor_offset=0.0,    # no offset
            morph_close_k=3)
        # Halo pixel at (70, 100) is intensity ~170 — well below
        # threshold (~215), so excluded
        assert expanded[70, 100] == 0

    def test_offset_50_lets_halo_into_expansion(self):
        """With offset=50, the effective threshold drops by 50 units.
        Seed quantile ~215, threshold becomes ~165 — halo pixels at
        ~170 NOW pass the gate, and the geodesic constraint connects
        them to the seed through the bright core."""
        frame = _build_frame_with_bright_core_and_dim_halo()
        seed = np.zeros_like(frame, dtype=np.uint8)
        seed[90:110, 140:160] = 255
        expanded = expand_mask_by_intensity(
            frame, seed,
            max_expand_px=60,
            intensity_quantile=0.25,
            intensity_floor_offset=50.0,   # threshold ~215 - 50 = ~165
            morph_close_k=3)
        # Halo pixel at (70, 100) is in the bright halo region
        # (rectangle 60:140 × 80:220, intensity ~170)
        assert expanded[70, 100] > 0, (
            "halo pixel should be in expanded mask with offset=50")
        # And the expanded area is significantly larger than seed
        assert int((expanded > 0).sum()) > 2 * int((seed > 0).sum())

    def test_offset_doesnt_admit_background(self):
        """The offset can't reach so low that background pixels
        (intensity ~80) pass the gate."""
        frame = _build_frame_with_bright_core_and_dim_halo()
        seed = np.zeros_like(frame, dtype=np.uint8)
        seed[90:110, 140:160] = 255
        expanded = expand_mask_by_intensity(
            frame, seed,
            max_expand_px=60,
            intensity_quantile=0.25,
            intensity_floor_offset=50.0,
            morph_close_k=3)
        # Background pixel at (20, 20) is intensity ~80, well below
        # threshold (~165) → excluded
        assert expanded[20, 20] == 0
        # Background pixel at (180, 280) — also excluded
        assert expanded[180, 280] == 0
