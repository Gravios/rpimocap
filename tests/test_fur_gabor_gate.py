"""
tests/test_fur_gabor_gate.py
=============================
Tests for the two new shape/texture filters added to address the
"cable wins the largest-CC pick" failure mode:

  * --fur-gabor-min:    suppress smooth wide surfaces by requiring
                         minimum Gabor texture energy per pixel
  * --max-aspect-ratio: reject elongated thin blobs (e.g., cables)
                         by their fitted-ellipse aspect ratio

CAVEAT on --fur-gabor-min: a Gabor filter's edge response extends
inward by roughly its largest wavelength (default 16 px). A smooth
surface narrower than ~2x that width will have its entire interior
flagged as high-Gabor by edge influence, and the gate does NOT
suppress it. So --fur-gabor-min works for wide smooth surfaces
(acrylic wall panels) but NOT for thin features like the tether
cable. --max-aspect-ratio is the right tool for thin features.
"""
from __future__ import annotations

import numpy as np
import cv2

from rpimocap.detection.segment import (
    BackgroundModel, ForegroundDetector)


def _empty_bg(shape=(300, 400)):
    return BackgroundModel(
        bg0=np.full(shape, 50, dtype=np.float32),
        bg1=np.full(shape, 50, dtype=np.float32),
    )


class TestFurGaborGate:
    """The Gabor gate suppresses WIDE smooth surfaces. It is not
    expected to fully eliminate narrow ones (see module docstring)."""

    def test_wide_smooth_surface_is_suppressed(self):
        """A 200x200 smooth bright blob: with the gate at 0.05, the
        interior (far from the edge influence zone) gets gated out.
        At least 50% of the pre-gate mask should be removed."""
        shape = (300, 400)
        frame = np.full(shape, 50, dtype=np.uint8)
        frame[50:250, 50:250] = 200

        det_off = ForegroundDetector(_empty_bg(shape), threshold=30,
                                       min_area_px=500, morph_k=3,
                                       fur_gabor_min=0.0)
        det_on  = ForegroundDetector(_empty_bg(shape), threshold=30,
                                       min_area_px=500, morph_k=3,
                                       fur_gabor_min=0.05)
        r_off = det_off.detect(frame, cam=0)
        r_on  = det_on.detect(frame, cam=0)

        pixels_off = int(r_off.mask.sum() // 255)
        pixels_on  = int(r_on.mask.sum()  // 255)
        assert pixels_on < 0.5 * pixels_off, (
            f"expected >50% reduction; got {pixels_off} -> {pixels_on}")

    def test_gate_does_not_remove_pure_texture(self):
        """A textured bright surface (stripes) must survive the gate."""
        shape = (300, 400)
        frame = np.full(shape, 50, dtype=np.uint8)
        yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
        stripes = (np.sin(xx * 0.8) * 100).astype(np.int16)
        mask = (yy >= 50) & (yy < 250) & (xx >= 50) & (xx < 250)
        frame[mask] = np.clip(200 + stripes[mask], 0, 255).astype(np.uint8)

        det = ForegroundDetector(_empty_bg(shape), threshold=30,
                                  min_area_px=500, morph_k=3,
                                  fur_gabor_min=0.05)
        r = det.detect(frame, cam=0)
        pixels = int(r.mask.sum() // 255)
        assert pixels > 30000, (
            f"textured surface should mostly survive; got {pixels} px")

    def test_gate_off_no_change(self):
        """fur_gabor_min=0 leaves the mask alone."""
        shape = (300, 400)
        frame = np.full(shape, 50, dtype=np.uint8)
        frame[50:250, 50:250] = 200
        det = ForegroundDetector(_empty_bg(shape), threshold=30,
                                  min_area_px=500, morph_k=3,
                                  fur_gabor_min=0.0)
        r = det.detect(frame, cam=0)
        assert r.mask.any(), "gate off should leave foreground intact"


class TestMaxAspectRatio:
    """The aspect-ratio filter rejects highly elongated blobs (cables)
    while keeping roundish ones (rat bodies). This is the primary tool
    for the cable-wins-pick failure mode."""

    def test_thin_cable_rejected(self):
        """A 200x10 thin bright strip (aspect ~20) is rejected."""
        shape = (300, 400)
        frame = np.full(shape, 50, dtype=np.uint8)
        frame[150:160, 100:300] = 200

        det = ForegroundDetector(_empty_bg(shape), threshold=30,
                                  min_area_px=500, morph_k=3,
                                  max_aspect_ratio=8.0)
        r = det.detect(frame, cam=0)
        assert r.n_blobs == 0, (
            f"thin cable (aspect ~20) should be filtered out by "
            f"max_aspect_ratio=8, got {r.n_blobs} blobs")

    def test_roundish_rat_kept(self):
        """A 60x80 roundish blob (aspect ~1.3) passes."""
        shape = (300, 400)
        frame = np.full(shape, 50, dtype=np.uint8)
        cv2.ellipse(frame, (200, 150), (40, 30), 0, 0, 360, 200, -1)

        det = ForegroundDetector(_empty_bg(shape), threshold=30,
                                  min_area_px=500, morph_k=3,
                                  max_aspect_ratio=8.0)
        r = det.detect(frame, cam=0)
        assert r.n_blobs >= 1, (
            f"roundish rat-like blob should pass aspect filter; "
            f"got {r.n_blobs} blobs")

    def test_cable_rejected_rat_kept_simultaneously(self):
        """The realistic scenario: both rat and cable present in
        the frame as separate blobs. Cable is rejected, rat is kept."""
        shape = (300, 400)
        frame = np.full(shape, 50, dtype=np.uint8)
        cv2.ellipse(frame, (100, 100), (40, 30), 0, 0, 360, 200, -1)
        frame[200:208, 200:380] = 200

        det = ForegroundDetector(_empty_bg(shape), threshold=30,
                                  min_area_px=500, morph_k=3,
                                  max_aspect_ratio=8.0)
        r = det.detect(frame, cam=0)
        assert r.n_blobs == 1, (
            f"expected 1 blob (rat), got {r.n_blobs}")

    def test_filter_off_keeps_everything(self):
        """max_aspect_ratio=None passes thin elongated blobs."""
        shape = (300, 400)
        frame = np.full(shape, 50, dtype=np.uint8)
        frame[150:160, 100:300] = 200

        det = ForegroundDetector(_empty_bg(shape), threshold=30,
                                  min_area_px=500, morph_k=3,
                                  max_aspect_ratio=None)
        r = det.detect(frame, cam=0)
        assert r.n_blobs >= 1, "filter off should keep the cable"
