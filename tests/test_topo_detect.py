"""Tests for the topological rat detector (rpimocap.detection.topo_detect).

Synthetic scenes: a grainy 'bedding' field (correlated noise, many small
local maxima) with a smooth bright 'rat' blob (few maxima). The detector
should find the blob as the low-grain-count region, place the centroid on
it, segment it against the grain barrier, and — in stereo — triangulate to
the 3D point and reject one below the floor.
"""
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

from rpimocap.detection.topo_detect import (
    Detection, body_blob, circle_grow_segment, detect, detect_stereo,
    grain_count_map, grain_peaks, median_bandpass)


def _grainy_with_blob(H=400, W=500, cx=250, cy=200, R=60, seed=0):
    rng = np.random.default_rng(seed)
    bg = gaussian_filter(rng.normal(0, 1, (H, W)).astype(np.float32), 0.8) * 40 + 110
    img = bg.copy()
    yy, xx = np.ogrid[:H, :W]
    disk = (xx - cx) ** 2 + (yy - cy) ** 2 <= R ** 2
    img[disk] = 185 + rng.normal(0, 2, int(disk.sum()))   # smooth bright blob
    return np.clip(img, 0, 255).astype(np.uint8), disk.astype(np.uint8)


class TestGrainFeatures:

    def test_blob_is_low_grain(self):
        img, disk = _grainy_with_blob()
        gc = grain_count_map(median_bandpass(img), patch=64)
        assert gc[disk > 0].mean() < gc[disk == 0].mean()

    def test_strict_maxima_excludes_plateau(self):
        flat = np.full((50, 50), 5.0, np.float32)
        assert grain_peaks(flat).sum() == 0          # no spurious plateau peaks

    def test_body_blob_peaks_on_the_blob(self):
        img, disk = _grainy_with_blob()
        bb = body_blob(img, sigma=40.0)
        py, px = np.unravel_index(int(np.argmax(bb)), bb.shape)
        assert disk[py, px] > 0                       # peak lands inside the blob


class TestDetect:

    def test_finds_blob(self):
        img, disk = _grainy_with_blob()
        floor = np.full(img.shape, 255, np.uint8)
        det = detect(img, floor, patch=64, blob_sigma=40, min_area=500)
        assert det.found
        assert det.separation < 0                     # rat is the low-grain region
        assert abs(det.centroid[0] - 250) < 45 and abs(det.centroid[1] - 200) < 45
        inter = int(((det.mask > 0) & (disk > 0)).sum())
        assert inter > 0.3 * int(disk.sum())          # mask overlaps the blob

    def test_returns_dataclass(self):
        img, _ = _grainy_with_blob()
        det = detect(img, np.full(img.shape, 255, np.uint8),
                     patch=64, blob_sigma=40, min_area=500)
        assert isinstance(det, Detection)
        assert det.mask.shape == img.shape

    def test_empty_floor_not_found(self):
        img, _ = _grainy_with_blob()
        det = detect(img, np.zeros(img.shape, np.uint8), patch=64)
        assert not det.found


class TestSegment:

    def test_circle_grow_covers_blob(self):
        img, disk = _grainy_with_blob()
        gc = grain_count_map(median_bandpass(img), 64)
        floor = np.full(img.shape, 255, np.uint8)
        seed = cv2.erode(disk, np.ones((15, 15), np.uint8))
        mask = circle_grow_segment(seed, gc, floor, barrier_pct=50)
        assert int(((mask > 0) & (disk > 0)).sum()) > 0.3 * int(disk.sum())

    def test_circle_grow_deterministic(self):
        img, disk = _grainy_with_blob()
        gc = grain_count_map(median_bandpass(img), 64)
        floor = np.full(img.shape, 255, np.uint8)
        seed = cv2.erode(disk, np.ones((15, 15), np.uint8))
        a = circle_grow_segment(seed, gc, floor, rng=np.random.default_rng(3))
        b = circle_grow_segment(seed, gc, floor, rng=np.random.default_rng(3))
        assert np.array_equal(a, b)


class TestStereo:

    @staticmethod
    def _P(C):
        f, cx, cy = 900.0, 300.0, 250.0
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
        fwd = np.array([0, 0, 100.0]) - C; fwd /= np.linalg.norm(fwd)
        right = np.cross(fwd, [0, 0, 1.0]); right /= np.linalg.norm(right)
        down = np.cross(fwd, right); R = np.vstack([right, down, fwd])
        return K @ np.hstack([R, (-R @ C).reshape(3, 1)])

    @staticmethod
    def _proj(P, X):
        w = P @ np.append(X, 1.0); return w[:2] / w[2]

    def _scenes(self, X3):
        P0 = self._P(np.array([-400.0, -600.0, 500.0]))
        P1 = self._P(np.array([400.0, -600.0, 500.0]))
        g0 = _grainy_with_blob(500, 600, *self._proj(P0, X3).astype(int),
                               R=55, seed=1)[0]
        g1 = _grainy_with_blob(500, 600, *self._proj(P1, X3).astype(int),
                               R=55, seed=2)[0]
        return P0, P1, g0, g1

    def test_triangulates_in_arena_point(self):
        Xtrue = np.array([0.0, 0.0, 40.0])
        P0, P1, g0, g1 = self._scenes(Xtrue)
        fl = np.full(g0.shape, 255, np.uint8)
        X, acc, d0, d1 = detect_stereo(g0, g1, fl, fl, P0, P1,
                                       patch=64, blob_sigma=40, min_area=500)
        assert d0.found and d1.found and X is not None
        assert np.linalg.norm(X[:3] - Xtrue) < 45      # near truth
        assert acc is True                              # in-arena, above floor

    def test_below_floor_rejected(self):
        # a reflection triangulates below z=0 and must be gated out
        Xrefl = np.array([0.0, 0.0, -40.0])
        P0, P1, g0, g1 = self._scenes(Xrefl)
        fl = np.full(g0.shape, 255, np.uint8)
        X, acc, d0, d1 = detect_stereo(g0, g1, fl, fl, P0, P1,
                                       patch=64, blob_sigma=40, min_area=500)
        assert (X is None) or (acc is False)
