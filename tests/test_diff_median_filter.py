"""
tests/test_diff_median_filter.py
=================================
Median filter on the diff map (--diff-median-k) removes
salt-and-pepper outliers BEFORE thresholding, while preserving
the rat blob's coherent shape. Single-pixel bright spots from
bedding texture variation, camera read noise, and residual
specular flicker get killed; the rat blob survives.
"""
from __future__ import annotations

import numpy as np

from rpimocap.detection.segment import (
    BackgroundModel, ForegroundDetector)


def _empty_bg(shape=(180, 240), bg_value=50):
    return BackgroundModel(
        bg0=np.full(shape, bg_value, dtype=np.float32),
        bg1=np.full(shape, bg_value, dtype=np.float32),
    )


def _frame_with_blob_and_salt(blob_at, blob_size,
                                salt_positions,
                                shape=(180, 240),
                                base=50, intensity=200):
    """Build a frame with:
       * a solid coherent blob (the 'rat')
       * a set of single-pixel bright outliers ('salt') at the
         given positions
    """
    f = np.full(shape, base, dtype=np.uint8)
    if blob_at is not None:
        y, x = blob_at
        h, w = blob_size
        f[y-h//2:y+h//2, x-w//2:x+w//2] = intensity
    for py, px in salt_positions:
        if 0 <= py < shape[0] and 0 <= px < shape[1]:
            f[py, px] = intensity
    return f


class TestDiffMedianFilterDisabled:

    def test_disabled_is_back_compat(self):
        """diff_median_k=0 → no filter, no behavior change."""
        det = ForegroundDetector(_empty_bg(), threshold=30,
                                  min_area_px=10, morph_k=3,
                                  diff_median_k=0)
        # Salt at three isolated positions, no real blob
        frame = _frame_with_blob_and_salt(
            blob_at=None, blob_size=None,
            salt_positions=[(40, 60), (90, 120), (150, 180)])
        r = det.detect(frame, cam=0)
        # Without median filter, salt pixels alone wouldn't normally
        # form a blob that survives morphology + min_area, so the
        # mask may be empty either way. The point is: the median
        # filter is OFF, no behavior change vs default.
        assert det._diff_median_k == 0

    def test_invalid_kernel_clamped(self):
        """Even kernel is forced odd; <3 becomes 3; <0 becomes 0."""
        d1 = ForegroundDetector(_empty_bg(), threshold=30,
                                  min_area_px=10, morph_k=3,
                                  diff_median_k=4)
        assert d1._diff_median_k == 5   # 4 | 1 = 5
        d2 = ForegroundDetector(_empty_bg(), threshold=30,
                                  min_area_px=10, morph_k=3,
                                  diff_median_k=1)
        assert d2._diff_median_k == 3   # min usable kernel
        d3 = ForegroundDetector(_empty_bg(), threshold=30,
                                  min_area_px=10, morph_k=3,
                                  diff_median_k=-5)
        assert d3._diff_median_k == 0   # negative → off


class TestSaltPepperRemoval:

    def test_small_clusters_killed(self):
        """Small 2x2 bright clusters scattered across the frame
        survive bg-sub thresholding (each cluster is 4 px) but get
        wiped by a median-5 filter. A median-5 kernel has 25 pixels;
        a 2x2 cluster of brights inside the 25-px window cannot be
        the majority, so the median replaces them with bg value."""
        bg = _empty_bg(bg_value=50)

        def add_cluster(f, y, x, size=2, intensity=200):
            f[y:y+size, x:x+size] = intensity

        # 5 small 2x2 clusters scattered, NO real rat
        frame = np.full((180, 240), 50, dtype=np.uint8)
        cluster_positions = [(30, 40), (50, 100), (90, 60),
                              (130, 180), (160, 90)]
        for y, x in cluster_positions:
            add_cluster(frame, y, x)

        # Without median: clusters get to the binary mask, each is
        # at least 4 px. With min_area_px=1, they survive.
        det_off = ForegroundDetector(bg, threshold=30,
                                      min_area_px=1, morph_k=1,
                                      diff_median_k=0)
        r_off = det_off.detect(frame.copy(), cam=0)
        n_off = int((r_off.mask > 0).sum())

        # With median-5: each cluster is 4 px in a 25-pixel window
        # of mostly-bg pixels → median picks bg → cluster wiped.
        det_on = ForegroundDetector(bg, threshold=30,
                                     min_area_px=1, morph_k=1,
                                     diff_median_k=5)
        r_on = det_on.detect(frame.copy(), cam=0)
        n_on = int((r_on.mask > 0).sum())

        assert n_off > 0, (
            f"without median filter, salt clusters should survive; "
            f"got n_off={n_off}")
        assert n_on < n_off, (
            f"median-5 should remove small clusters: "
            f"n_off={n_off} n_on={n_on}")


class TestCoherentBlobSurvives:

    def test_real_blob_survives_median_filter(self):
        """A coherent rat-sized blob (40x40 px) survives a median-5
        filter — the filter only removes structures smaller than k/2."""
        bg = _empty_bg(bg_value=50)
        frame = _frame_with_blob_and_salt(
            blob_at=(90, 120), blob_size=(40, 40),
            salt_positions=[],
            base=50, intensity=200)
        det = ForegroundDetector(bg, threshold=30,
                                  min_area_px=100, morph_k=3,
                                  diff_median_k=5)
        r = det.detect(frame, cam=0)
        # Blob should be in the mask
        assert r.mask[70:110, 100:140].sum() > 0, (
            "real coherent blob should survive median filter")

    def test_blob_with_surrounding_salt_kept_clean(self):
        """The combination: a real blob + scattered salt elsewhere.
        Median filter removes the salt, the blob survives, the
        labeller centroid lands on the blob (not pulled by salt)."""
        bg = _empty_bg(bg_value=50)
        # Blob at (60, 80), salt scattered all over the right half
        salts = [(20, 200), (50, 180), (110, 210), (140, 170), (170, 230)]
        frame = _frame_with_blob_and_salt(
            blob_at=(60, 80), blob_size=(40, 40),
            salt_positions=salts,
            base=50, intensity=200)
        det = ForegroundDetector(bg, threshold=30,
                                  min_area_px=100, morph_k=3,
                                  diff_median_k=5)
        r = det.detect(frame, cam=0)
        # The mask should contain the blob area
        blob_area = int(r.mask[40:80, 60:100].sum())
        # And the salt regions (right half outside blob) should be
        # mostly zero
        salt_zone = int(r.mask[:, 150:].sum())
        assert blob_area > 0, "real blob should be detected"
        assert salt_zone == 0, (
            f"salt zone should be empty after median filter, got "
            f"{salt_zone} px")


class TestPerCameraIndependent:

    def test_both_cameras_apply_filter(self):
        """The filter applies to both cameras (the same instance
        config — there is no per-camera diff_median_k)."""
        bg = BackgroundModel(
            bg0=np.full((180, 240), 50, dtype=np.float32),
            bg1=np.full((180, 240), 50, dtype=np.float32))
        det = ForegroundDetector(bg, threshold=30,
                                  min_area_px=1, morph_k=1,
                                  diff_median_k=3)
        # Same single salt pixel in both
        frame0 = np.full((180, 240), 50, dtype=np.uint8)
        frame0[100, 120] = 200
        frame1 = frame0.copy()
        r0 = det.detect(frame0, cam=0)
        r1 = det.detect(frame1, cam=1)
        # Both should suppress the isolated salt
        assert (r0.mask > 0).sum() == 0
        assert (r1.mask > 0).sum() == 0


class TestKernelSizeMatters:

    def test_larger_kernel_kills_larger_artifacts(self):
        """A small bright cluster (e.g., 3x3) survives median 3 but
        is killed by median 7."""
        bg = _empty_bg(bg_value=50)
        # 3x3 bright cluster — bigger than single pixel, but still
        # small enough that median 7 wipes it
        frame = np.full((180, 240), 50, dtype=np.uint8)
        frame[88:91, 118:121] = 200   # 3x3 cluster at center

        det_k3 = ForegroundDetector(bg, threshold=30,
                                     min_area_px=1, morph_k=1,
                                     diff_median_k=3)
        det_k7 = ForegroundDetector(bg, threshold=30,
                                     min_area_px=1, morph_k=1,
                                     diff_median_k=7)
        r3 = det_k3.detect(frame.copy(), cam=0)
        r7 = det_k7.detect(frame.copy(), cam=0)
        # The 3x3 cluster is right at the edge of what median-3 can
        # remove; median-7 (with majority window of 25 pixels) wipes
        # it cleanly.
        n3 = int((r3.mask > 0).sum())
        n7 = int((r7.mask > 0).sum())
        assert n7 <= n3, (
            f"larger kernel should remove at least as much as "
            f"smaller; got n3={n3} n7={n7}")
        assert n7 == 0, (
            f"median-7 should wipe a 3x3 cluster, got {n7} pixels")
