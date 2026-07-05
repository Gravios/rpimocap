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
    Detection, StereoResult, body_blob, cable_suppressed_map,
    circle_grow_segment, combine_barriers, detect, detect_stereo,
    grain_count_map, grain_peaks, laplacian_magnitude, median_bandpass)


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


class TestLaplacianBarrier:

    def test_magnitude_low_on_blob(self):
        img, disk = _grainy_with_blob()
        lm = laplacian_magnitude(median_bandpass(img), sigma=3.0)
        assert lm[disk > 0].mean() < lm[disk == 0].mean()    # rat low, bed high

    def test_magnitude_is_sigma_robust(self):
        img, disk = _grainy_with_blob()
        mbp = median_bandpass(img)

        def sep(s):
            lm = laplacian_magnitude(mbp, s)
            return lm[disk == 0].mean() / max(lm[disk > 0].mean(), 1e-6)

        s2, s8 = sep(2.0), sep(8.0)
        assert s2 > 1.5 and s8 > 1.5               # separates at both
        assert abs(s2 - s8) / s2 < 0.4             # and is roughly flat in sigma

    def test_combine_is_and_of_lows(self):
        img, disk = _grainy_with_blob()
        mbp = median_bandpass(img)
        floor = np.full(img.shape, 255, np.uint8)
        comb = combine_barriers(
            [grain_count_map(mbp, 64), laplacian_magnitude(mbp, 3.0)], floor)
        assert comb[disk > 0].mean() < comb[disk == 0].mean()


class TestSegBarrierOptions:

    def test_all_barriers_find_and_cover_blob(self):
        img, disk = _grainy_with_blob()
        floor = np.full(img.shape, 255, np.uint8)
        for mode in ("grain", "laplacian", "both"):
            det = detect(img, floor, patch=64, blob_sigma=40, min_area=500,
                         seg_barrier=mode)
            assert det.found, mode
            inter = int(((det.mask > 0) & (disk > 0)).sum())
            assert inter > 0.3 * int(disk.sum()), mode


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
        R = detect_stereo(g0, g1, fl, fl, P0, P1,
                          patch=64, blob_sigma=40, min_area=500)
        assert isinstance(R, StereoResult)
        assert R.det0.found and R.det1.found and R.point is not None
        assert R.accepted is True                       # in-arena, consistent
        assert np.linalg.norm(R.point[:3] - Xtrue) < 45  # near truth

    def test_below_floor_rejected(self):
        # a reflection triangulates below z=0 and must be gated out
        Xrefl = np.array([0.0, 0.0, -40.0])
        P0, P1, g0, g1 = self._scenes(Xrefl)
        fl = np.full(g0.shape, 255, np.uint8)
        R = detect_stereo(g0, g1, fl, fl, P0, P1,
                          patch=64, blob_sigma=40, min_area=500)
        assert (R.point is None) or (not R.accepted)

    def test_epipolar_rejects_noncorresponding_pair(self):
        # cam0 sees a blob at A, cam1 sees one at a DIFFERENT 3D point B —
        # no epipolar-consistent pairing, so the match must be dropped.
        P0, P1, g0, _ = self._scenes(np.array([0.0, 0.0, 40.0]))
        _, _, _, g1 = self._scenes(np.array([120.0, -180.0, 60.0]))
        fl = np.full(g0.shape, 255, np.uint8)
        R = detect_stereo(g0, g1, fl, fl, P0, P1, patch=64, blob_sigma=40,
                          min_area=500, max_epipolar_px=10, max_reproj_px=10)
        assert (R.point is None) or (not R.accepted)


class TestCableSuppressionAndCandidates:

    def test_cable_suppressed_map_low_on_blob(self):
        img, disk = _grainy_with_blob()
        mix = cable_suppressed_map(img, median_bandpass(img),
                                   np.full(img.shape, 255, np.uint8),
                                   illum_sigma=81.0, barrier_sigma=16.0)
        assert mix[disk > 0].mean() < mix[disk == 0].mean()   # rat is the min

    def test_candidates_populated_and_best_first(self):
        img, _ = _grainy_with_blob()
        det = detect(img, np.full(img.shape, 255, np.uint8),
                     patch=64, blob_sigma=40, min_area=500, max_candidates=3)
        assert len(det.candidates) >= 1
        assert det.candidates[0] == det.centroid          # best candidate first

    def test_cable_suppress_still_finds_blob(self):
        img, disk = _grainy_with_blob()
        det = detect(img, np.full(img.shape, 255, np.uint8), patch=64,
                     blob_sigma=40, min_area=500, cable_suppress=True,
                     illum_sigma=81.0, cable_barrier_sigma=16.0)
        assert det.found
        assert abs(det.centroid[0] - 250) < 55 and abs(det.centroid[1] - 200) < 55
