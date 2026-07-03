"""Regression test for the volume-ROI over-crop fix (patch 0076).

The volume ROI extrudes the arena floor footprint up by ``max_height_mm``
and takes the convex hull of the projected corners. At 120 mm the band
clipped a wall-hugging rat much of the time (verified against the real
DLT calibration: ~54% near-wall retention); the default is now 260 mm.

These tests use a synthetic oblique camera so they are self-contained.
The key properties are geometric and hold for any standard projection:
lifting the top corners higher can only grow the projected hull, so a
larger band retains at least as much rat-reachable floor, and a band at
the ceiling height degenerates to ``box``.
"""
import inspect

import cv2
import numpy as np

from rpimocap.detection.segment import arena_roi_corners, arena_roi_mask


ARENA = np.array([[-140, -215, 0], [140, -215, 0], [140, 215, 0], [-140, 215, 0],
                  [-140, -215, 388], [140, -215, 388], [140, 215, 388],
                  [-140, 215, 388]], dtype=float)
SHAPE = (1080, 2028)


def _synthetic_P():
    """An oblique DLT camera looking down-and-forward at the arena."""
    f, cx, cy = 900.0, 1014.0, 540.0
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
    C = np.array([0.0, -750.0, 620.0])            # camera centre (mm)
    fwd = np.array([0.0, 0.0, 194.0]) - C
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 0, 1.0]); right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    R = np.vstack([right, down, fwd])             # world -> camera
    t = -R @ C
    return K @ np.hstack([R, t.reshape(3, 1)])


def _proj(P, X):
    w = P @ np.append(X, 1.0)
    return np.array([w[0] / w[2], w[1] / w[2]])


def _column_inside(mask, P, x, y, z_top):
    for zz in np.linspace(0, z_top, 4):
        for dx in (-35, 35):
            for dy in (-35, 35):
                u, v = _proj(P, [x + dx, y + dy, zz]).astype(int)
                if not (0 <= v < SHAPE[0] and 0 <= u < SHAPE[1]
                        and mask[v, u] > 0):
                    return False
    return True


def _retention(P, mode, h, z_top=80):
    m = arena_roi_mask(P, arena_roi_corners(ARENA, mode, h), SHAPE, 20)
    gx = np.arange(-130, 131, 20); gy = np.arange(-205, 206, 20)
    return np.mean([_column_inside(m, P, x, y, z_top) for x in gx for y in gy])


class TestVolumeRoiBand:

    def test_default_is_260(self):
        assert inspect.signature(arena_roi_corners).parameters[
            "max_height_mm"].default == 260.0

    def test_corner_subset_shapes(self):
        assert arena_roi_corners(ARENA, "box").shape == (8, 3)
        assert arena_roi_corners(ARENA, "floor").shape == (4, 3)
        assert arena_roi_corners(ARENA, "volume", 260).shape == (8, 3)

    def test_volume_at_ceiling_degenerates_to_box(self):
        # floor corners lifted to the ceiling height == the box corners
        vol = arena_roi_corners(ARENA, "volume", 388)
        box = arena_roi_corners(ARENA, "box")
        assert set(map(tuple, vol.tolist())) == set(map(tuple, box.tolist()))

    def test_larger_band_grows_the_hull(self):
        P = _synthetic_P()
        def area(h):
            m = arena_roi_mask(P, arena_roi_corners(ARENA, "volume", h), SHAPE, 20)
            return int((m > 0).sum())
        a120, a260, a388 = area(120), area(260), area(388)
        assert a260 >= a120                       # bigger band, bigger ROI
        box = int((arena_roi_mask(P, arena_roi_corners(ARENA, "box"), SHAPE, 20) > 0).sum())
        assert abs(a388 - box) / box < 0.02       # ceiling band ~ box

    def test_260_retains_at_least_as_much_as_120(self):
        P = _synthetic_P()
        assert _retention(P, "volume", 260) >= _retention(P, "volume", 120)

    def test_260_recovers_reared_wall_positions_that_120_clips(self):
        # a rearing animal (z up to 230mm) near the walls
        P = _synthetic_P()
        r120 = _retention(P, "volume", 120, z_top=230)
        r260 = _retention(P, "volume", 260, z_top=230)
        assert r260 > r120
