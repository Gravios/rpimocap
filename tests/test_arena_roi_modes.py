"""Tests for arena ROI corner-subset selection (box/floor/volume)."""
import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from rpimocap.detection.segment import arena_roi_corners, arena_roi_mask


BOX = np.array([
    [-140, -215, 0], [140, -215, 0], [140, 215, 0], [-140, 215, 0],
    [-140, -215, 388], [140, -215, 388], [140, 215, 388], [-140, 215, 388],
], dtype=float)


def _make_P():
    K = np.array([[1500, 0, 1014], [0, 1500, 540], [0, 0, 1.]])
    cp = np.array([-300, -400, 700.]); la = np.array([0, 0, 194.])
    f = la - cp; f /= np.linalg.norm(f)
    up = np.array([0, 0, 1.]); r = np.cross(f, up); r /= np.linalg.norm(r)
    u = np.cross(r, f)
    R = np.vstack([r, -u, f])
    return K @ np.hstack([R, (-R @ cp).reshape(3, 1)])


class TestArenaRoiCorners:

    def test_box_is_all_eight(self):
        c = arena_roi_corners(BOX, "box")
        assert c.shape == (8, 3)
        assert np.allclose(c, BOX)

    def test_floor_is_four_bottom(self):
        c = arena_roi_corners(BOX, "floor")
        assert c.shape == (4, 3)
        assert np.allclose(c[:, 2], 0.0)          # all on the floor

    def test_volume_is_floor_plus_height_band(self):
        c = arena_roi_corners(BOX, "volume", max_height_mm=120)
        assert c.shape == (8, 3)
        assert np.allclose(c[:4, 2], 0.0)         # floor
        assert np.allclose(c[4:, 2], 120.0)       # lifted band
        # xy footprint matches the floor (no beyond-wall expansion)
        assert np.allclose(c[:4, :2], c[4:, :2])

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError):
            arena_roi_corners(BOX, "bogus")


class TestRoiAreaShrinks:

    def test_floor_and_volume_smaller_than_box(self):
        P = _make_P(); shape = (1080, 2028)
        a = {m: int((arena_roi_mask(
                P, arena_roi_corners(BOX, m), shape, 20) > 0).sum())
             for m in ("box", "floor", "volume")}
        assert a["floor"] < a["volume"] < a["box"]
        # floor should be well under half the box hull (the box hull is
        # dominated by the through-wall region)
        assert a["floor"] < 0.5 * a["box"]

    def test_beyond_wall_point_excluded_by_floor(self):
        P = _make_P(); shape = (1080, 2028)
        box_m = arena_roi_mask(P, arena_roi_corners(BOX, "box"), shape, 20)
        floor_m = arena_roi_mask(P, arena_roi_corners(BOX, "floor"),
                                 shape, 20)

        def inside(m, X):
            p = P @ np.append(X, 1.0)
            u, v = int(round(p[0] / p[2])), int(round(p[1] / p[2]))
            return (0 <= v < shape[0] and 0 <= u < shape[1]
                    and m[v, u] > 0)

        rat = np.array([0, 0, 20.])
        beyond = np.array([0, 320, 0.])           # past the +y wall
        assert inside(box_m, rat) and inside(floor_m, rat)
        assert inside(box_m, beyond)              # box wrongly includes it
        assert not inside(floor_m, beyond)        # floor excludes it
