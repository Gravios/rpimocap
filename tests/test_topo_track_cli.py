"""Light tests for the topo_track CLI (tools/topo_track.py).

The heavy detection logic is covered by tests/test_topo_detect.py; here we
only check the CLI's own wiring — the arena-corner constant, the green-channel
helper, and that the required arguments are enforced.
"""
import importlib.util
import os

import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location(
    "topo_track",
    os.path.join(os.path.dirname(__file__), "..", "tools", "topo_track.py"))
tt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tt)


class TestTopoTrackCLI:

    def test_arena_corners_shape_and_layout(self):
        assert tt._ARENA_CORNERS.shape == (8, 3)
        # floor corners first (z=0), ceiling corners last (z=388)
        assert (tt._ARENA_CORNERS[:4, 2] == 0).all()
        assert (tt._ARENA_CORNERS[4:, 2] == 388).all()
        # x in [-140, 140], y in [-215, 215]
        assert tt._ARENA_CORNERS[:, 0].min() == -140
        assert tt._ARENA_CORNERS[:, 1].max() == 215

    def test_green_channel_helper(self):
        rgb = np.zeros((8, 8, 3), np.uint8); rgb[..., 1] = 7
        assert (tt._green(rgb) == 7).all()
        gray = np.full((8, 8), 3, np.uint8)
        assert (tt._green(gray) == 3).all()          # 2-D passes through
        assert tt._green(None) is None

    def test_required_args_enforced(self):
        with pytest.raises(SystemExit):
            tt.main([])                              # missing --cam0/--cam1/...

    def test_overlay_renders_bgr(self):
        from rpimocap.detection.topo_detect import Detection, StereoResult
        H, W = 120, 160
        mask = np.zeros((H, W), np.uint8)
        mask[40:80, 60:100] = 1
        det = Detection(True, (80.0, 60.0), mask, mask, -3.0, [(80.0, 60.0)])
        g = np.full((H, W), 100, np.uint8)
        R = StereoResult(np.array([0.0, 195.0, 40.0]), True, 2.5, det, det)
        out = tt._overlay(g, g, R, 7, out_w=200)
        assert out.dtype == np.uint8 and out.ndim == 3 and out.shape[2] == 3
        assert out.shape[1] == 400               # two 200-wide panels
        # the no-consistent-pair case must render without crashing
        R2 = StereoResult(None, False, float("nan"), det, det)
        assert tt._overlay(g, g, R2, 3, out_w=200).dtype == np.uint8
