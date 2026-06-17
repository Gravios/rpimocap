"""
tests/test_intensity_rat_finder.py
====================================
find_rat_seed_by_intensity — the intensity-based rat finder used
during bootstrap. It works because the rat is the brightest object
in the arena under IR illumination.
"""
from __future__ import annotations

import numpy as np

from rpimocap.detection.rat_texture import find_rat_seed_by_intensity


def _frame_with_bright_object(shape=(200, 300),
                                base=80, object_intensity=220,
                                obj_box=(50, 80, 130, 170),
                                noise=10,
                                rng_seed=0):
    """Make a frame with a single bright square against a dim,
    noisy background. obj_box = (y0, y1, x0, x1)."""
    rng = np.random.RandomState(rng_seed)
    f = (np.full(shape, base, dtype=np.int16)
         + rng.randint(-noise, noise + 1, shape)).astype(np.int16)
    y0, y1, x0, x1 = obj_box
    f[y0:y1, x0:x1] = object_intensity + rng.randint(-5, 5,
                                                      (y1 - y0, x1 - x0))
    return np.clip(f, 0, 255).astype(np.uint8)


class TestFinderHappy:

    def test_finds_single_bright_object(self):
        frame = _frame_with_bright_object()
        mask = find_rat_seed_by_intensity(
            frame, intensity_percentile=92,
            min_area_px=500, morph_close_k=5)
        assert mask is not None
        assert int((mask > 0).sum()) >= 500
        # Object spans y=[50:80], x=[130:170]; centre is (65, 150)
        assert mask[65, 150] > 0

    def test_picks_largest_when_two_bright_objects(self):
        """If there are two bright objects of different sizes, the
        finder returns the LARGER one (rat is bigger than any
        bright cable highlight)."""
        frame = _frame_with_bright_object(obj_box=(50, 100, 80, 200))
        # Add a small bright object somewhere else
        frame[140:155, 220:240] = 230
        mask = find_rat_seed_by_intensity(
            frame, intensity_percentile=92,
            min_area_px=200, morph_close_k=3)
        assert mask is not None
        # The big box (the "rat") is in the mask
        assert mask[75, 140] > 0
        # The small box (the "highlight") is NOT in the mask, since
        # only the largest CC is returned
        assert mask[148, 230] == 0


class TestFinderEdgeCases:

    def test_returns_none_when_no_object(self):
        """Pure noise with no bright objects → no CC reaches
        min_area_px, so the finder returns None."""
        rng = np.random.RandomState(0)
        frame = (rng.randint(60, 100, (150, 200))).astype(np.uint8)
        mask = find_rat_seed_by_intensity(
            frame, intensity_percentile=95,
            min_area_px=2000, morph_close_k=3)
        assert mask is None

    def test_returns_none_when_objects_too_small(self):
        """With a very high min_area_px, even a real bright object
        is rejected."""
        frame = _frame_with_bright_object(obj_box=(80, 95, 120, 140))
        # The object is only ~300 px², require 5000
        mask = find_rat_seed_by_intensity(
            frame, intensity_percentile=95,
            min_area_px=5000, morph_close_k=3)
        assert mask is None

    def test_roi_mask_restricts_search(self):
        """A bright object outside the ROI must NOT be found."""
        shape = (200, 300)
        frame = _frame_with_bright_object(
            shape=shape, obj_box=(20, 50, 230, 270))   # right edge
        # ROI excludes the right side
        roi = np.zeros(shape, dtype=np.uint8)
        roi[:, :200] = 255
        mask = find_rat_seed_by_intensity(
            frame, roi_mask=roi,
            intensity_percentile=92, min_area_px=300,
            morph_close_k=3)
        assert mask is None    # rat is outside ROI

    def test_uses_intensity_inside_roi(self):
        """Same bright object — when the ROI includes it, it's
        found."""
        shape = (200, 300)
        frame = _frame_with_bright_object(
            shape=shape, obj_box=(60, 100, 100, 180))
        roi = np.zeros(shape, dtype=np.uint8)
        roi[:, :250] = 255
        mask = find_rat_seed_by_intensity(
            frame, roi_mask=roi,
            intensity_percentile=92, min_area_px=300,
            morph_close_k=3)
        assert mask is not None
        assert mask[80, 140] > 0


class TestFinderParameters:

    def test_lower_percentile_includes_more_pixels(self):
        """Dropping the percentile threshold from 95 to 80 lets in
        more pixels (the dim shoulders of a Gaussian bright spot)."""
        rng = np.random.RandomState(0)
        shape = (150, 200)
        # Wide Gaussian bright blob — pixels gradient from bright
        # centre to dim edges
        yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
        cy, cx = 75, 100
        blob = 200 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2)
                              / (2 * 30 ** 2))
        frame = np.clip(60 + blob, 0, 255).astype(np.uint8)
        mask_strict = find_rat_seed_by_intensity(
            frame, intensity_percentile=95, min_area_px=100,
            morph_close_k=3)
        mask_loose = find_rat_seed_by_intensity(
            frame, intensity_percentile=80, min_area_px=100,
            morph_close_k=3)
        assert mask_strict is not None
        assert mask_loose  is not None
        # Loose finder gathers more pixels (lower threshold)
        assert int((mask_loose > 0).sum()) > int((mask_strict > 0).sum())
