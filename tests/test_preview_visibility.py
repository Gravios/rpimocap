"""
tests/test_preview_visibility.py
==================================
Regression tests for the two visibility bugs in rpimocap-preview:

1. Scale math was backwards: dot_r = int(args.dot_radius * args.scale)
   meant a --dot-radius 6 --scale 0.5 invocation produced 1.5-px dots
   in the output instead of 6 px. The fix multiplies by 1/scale at
   draw time so the post-resize size matches user intent.

2. _draw_skeleton silently dropped keypoints whose projection fell
   outside the canvas — no log, no count. Added drawn/dropped return
   tuple + a 'WARNING: every keypoint projected OUTSIDE the frame'
   summary so projection mis-config is now loud.
"""
from __future__ import annotations

import inspect

import numpy as np
import pytest


class TestDrawSkeletonReturnTuple:

    def test_returns_canvas_drawn_dropped(self):
        from rpimocap.cli.preview import _draw_skeleton
        canvas = np.zeros((100, 100, 3), dtype=np.uint8)
        out, d, dr = _draw_skeleton(
            canvas,
            {"animal": (50, 50), "offscreen": (200, 200)},
            5, 2, 0.85)
        assert out.shape == canvas.shape
        assert d  == 1     # animal is in-frame
        assert dr == 1     # 'offscreen' is out

    def test_all_in_frame(self):
        from rpimocap.cli.preview import _draw_skeleton
        canvas = np.zeros((100, 100, 3), dtype=np.uint8)
        _, d, dr = _draw_skeleton(
            canvas, {"animal": (50, 50), "head": (60, 60)},
            5, 2, 0.85)
        assert d  == 2
        assert dr == 0

    def test_all_out_of_frame(self):
        from rpimocap.cli.preview import _draw_skeleton
        canvas = np.zeros((100, 100, 3), dtype=np.uint8)
        _, d, dr = _draw_skeleton(
            canvas, {"animal": (-5, 50), "head": (50, 200)},
            5, 2, 0.85)
        assert d  == 0
        assert dr == 2


class TestScaleMathIsInverse:
    """The dot_r / line_w computation must scale BY 1/args.scale, not
    by args.scale. We verify via source-inspection because the actual
    branch lives inside main() which needs full argparse + h5 + video
    captures to exercise."""

    def test_dot_r_uses_inverse_scale(self):
        from rpimocap.cli import preview as pv
        src = inspect.getsource(pv.main)
        # Sentinel for the corrected formula
        assert "inv_s" in src, "preview.py main() missing inv_s reciprocal"
        # And the explicit comment-about-why we want post-resize correctness
        assert "args.dot_radius * inv_s" in src
        assert "args.line_width * inv_s" in src

    def test_old_buggy_formula_absent(self):
        from rpimocap.cli import preview as pv
        src = inspect.getsource(pv.main)
        # The pre-fix expression that produced quarter-size dots:
        assert "args.dot_radius * args.scale" not in src, (
            "preview.py still uses the backwards dot scaling formula; "
            "it should be args.dot_radius * inv_s")


class TestSanityPrintExists:

    def test_sanity_check_block_exists(self):
        from rpimocap.cli import preview as pv
        src = inspect.getsource(pv.main)
        assert "Projection sanity check" in src
        assert "Skeleton draw summary" in src
        # The off-frame warning fires only when EVERYTHING dropped
        assert "every keypoint projected OUTSIDE the frame" in src
