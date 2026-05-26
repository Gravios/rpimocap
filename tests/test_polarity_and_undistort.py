"""
tests/test_polarity_and_undistort.py
======================================
Tests for:
  1. ForegroundDetector.polarity — bright/dark/either bg-sub.
     Bug fixed: shadow regions (darker than bg) were getting caught
     as foreground because diff = |gray - bg| treats both directions
     identically. The 'bright' mode suppresses shadows entirely.

  2. rpimocap-preview --undistort default = False.
     Bug fixed: preview was undistorting frames then projecting through
     a DLT P matrix fit to DISTORTED corner pixels, putting dots in
     the wrong place — often off-screen for fisheye lenses, which
     looked like 'preview isn't drawing dots'.
"""
from __future__ import annotations

import numpy as np
import pytest


def _bg_and_frame_with_shadow(h=80, w=80):
    """Build a bg (mid-gray) and a frame containing a BRIGHT blob +
    a DARK shadow blob. With 'either' polarity, both blobs get caught.
    With 'bright', only the bright one. With 'dark', only the shadow."""
    import cv2
    bg = np.full((h, w), 100, dtype=np.float32)
    frame = np.full((h, w, 3), 100, dtype=np.uint8)
    # Bright rat blob (brighter than bg)
    cv2.circle(frame, (25, 40), 8, (180, 180, 180), -1)
    # Dark cast shadow (darker than bg)
    cv2.circle(frame, (55, 40), 8, (40, 40, 40), -1)
    return bg, frame


class TestPolarityBgSub:

    def _det(self, polarity):
        from rpimocap.detection.segment import (
            BackgroundModel, ForegroundDetector)
        bg_arr = np.full((80, 80), 100, dtype=np.float32)
        bg     = BackgroundModel(bg_arr, bg_arr, method="median")
        return ForegroundDetector(
            bg, threshold=20, min_area_px=30,
            polarity=polarity)

    def test_polarity_either_default(self):
        """Default polarity='either' must catch both blobs."""
        det = self._det("either")
        _, frame = _bg_and_frame_with_shadow()
        r = det.detect(frame, 0)
        # Two distinct blobs in the labelmap
        assert r.n_blobs == 2

    def test_polarity_bright_drops_shadow(self):
        """polarity='bright' must catch only the bright blob — the
        cast shadow's darker-than-bg pixels are zeroed."""
        det = self._det("bright")
        _, frame = _bg_and_frame_with_shadow()
        r = det.detect(frame, 0)
        assert r.n_blobs == 1
        # The surviving blob should be on the LEFT (bright rat at x=25)
        ys, xs = np.where(r.label_map > 0)
        assert xs.mean() < 40, "surviving blob should be the bright one (left)"

    def test_polarity_dark_drops_bright(self):
        """polarity='dark' is the mirror case — only catches the shadow."""
        det = self._det("dark")
        _, frame = _bg_and_frame_with_shadow()
        r = det.detect(frame, 0)
        assert r.n_blobs == 1
        ys, xs = np.where(r.label_map > 0)
        assert xs.mean() > 40, "surviving blob should be the dark one (right)"

    def test_invalid_polarity_raises(self):
        from rpimocap.detection.segment import (
            BackgroundModel, ForegroundDetector)
        bg = BackgroundModel(np.zeros((10, 10), np.float32),
                             np.zeros((10, 10), np.float32))
        with pytest.raises(ValueError, match="polarity"):
            ForegroundDetector(bg, polarity="brigth")  # typo

    def test_ctor_default_is_either(self):
        """Default value must be 'either' for back-compat."""
        import inspect
        from rpimocap.detection.segment import ForegroundDetector
        sig = inspect.signature(ForegroundDetector.__init__)
        assert sig.parameters["polarity"].default == "either"


class TestSegmentTrackerForwardsPolarity:

    def test_tracker_ctor_has_polarity_param(self):
        import inspect
        from rpimocap.detection.tracker import SegmentTracker
        sig = inspect.signature(SegmentTracker.__init__)
        assert "polarity" in sig.parameters
        assert sig.parameters["polarity"].default == "either"


class TestPreviewUndistortFlag:

    def test_undistort_default_false(self):
        """rpimocap-preview's --undistort flag must default to False
        so dots project onto distorted frames (matching DLT fit)."""
        # Inspect the source — building a full argparse parser requires
        # the whole preview CLI to load.
        import inspect
        from rpimocap.cli import preview as pv
        src = inspect.getsource(pv.main)
        assert "--undistort" in src
        assert 'action="store_true"' in src
        assert "default=False" in src

    def test_undistort_remap_gated(self):
        """The cv2.remap calls must be inside a conditional (not
        unconditional like before)."""
        import inspect
        from rpimocap.cli import preview as pv
        src = inspect.getsource(pv.main)
        # The remap calls must be conditional on map0x being non-None
        # (i.e., guarded inside an 'if map0x is not None:' block)
        assert "if map0x is not None:" in src
