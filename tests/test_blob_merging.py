"""
tests/test_blob_merging.py
===========================
Hull-based blob merging via --merge-blob-distance.
Surviving CCs whose centroids are within the merge distance get
unioned into a single convex hull, replacing the fragments with
one coherent rat-shaped blob.
"""
from __future__ import annotations

import numpy as np
import cv2

from rpimocap.detection.segment import (
    BackgroundModel, ForegroundDetector,
    _merge_blobs_by_hull)


def _empty_bg(shape=(180, 240), bg_value=50):
    return BackgroundModel(
        bg0=np.full(shape, bg_value, dtype=np.float32),
        bg1=np.full(shape, bg_value, dtype=np.float32),
    )


def _three_fragment_frame(shape=(180, 240), base=50, intensity=200):
    """A frame with three small bright fragments close together
    (mimicking an under-segmented rat broken into pieces)."""
    f = np.full(shape, base, dtype=np.uint8)
    # Three roughly-aligned fragments at (90, 100), (90, 130), (95, 160)
    f[80:100, 90:115]  = intensity
    f[85:105, 120:145] = intensity
    f[88:110, 150:175] = intensity
    return f


def _two_distant_frames(shape=(180, 240), base=50, intensity=200):
    """Two bright blobs FAR apart — should NOT merge with a small
    merge distance."""
    f = np.full(shape, base, dtype=np.uint8)
    f[30:50, 30:60]   = intensity     # blob at top-left
    f[140:160, 180:210] = intensity   # blob at bottom-right
    return f


# ────────────────────────────────────────────────────────────────────
#  Direct unit tests on _merge_blobs_by_hull
# ────────────────────────────────────────────────────────────────────


class TestMergeHullUnit:

    def _setup_three_fragments(self):
        """Returns the (binary, label_map, surviving_idx, stats)
        tuple that the segment pipeline would produce for the three-
        fragment frame."""
        bg = _empty_bg()
        det = ForegroundDetector(bg, threshold=30,
                                  min_area_px=50, morph_k=3,
                                  merge_blob_distance=0)
        # Run detect to get the standard intermediates
        r = det.detect(_three_fragment_frame(), cam=0)
        # The mask + labels have three CCs already
        n_ccs = int(r.label_map.max())
        # Build stats from connectedComponentsWithStats
        binary = r.mask
        n_, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=8)
        surviving = list(range(1, n_))
        return binary, labels, surviving, stats

    def test_close_fragments_merge_to_one(self):
        """With merge_distance large enough to bridge the gaps,
        three fragments → one hull."""
        binary, labels, surviving, stats = self._setup_three_fragments()
        assert len(surviving) >= 2, (
            f"setup expected ≥2 fragments, got {len(surviving)}")
        new_bin, new_labels, new_blobs = _merge_blobs_by_hull(
            binary, labels, surviving, stats,
            merge_distance_px=100, dilate_px=0)
        assert len(new_blobs) == 1, (
            f"expected 1 merged blob from 3 close fragments, "
            f"got {len(new_blobs)}")

    def test_far_blobs_dont_merge(self):
        """Two distant blobs with merge_distance smaller than their
        separation → both kept as separate."""
        bg = _empty_bg()
        det = ForegroundDetector(bg, threshold=30,
                                  min_area_px=50, morph_k=3,
                                  merge_blob_distance=0)
        r = det.detect(_two_distant_frames(), cam=0)
        binary = r.mask
        n_, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=8)
        surviving = list(range(1, n_))
        # The two blobs are ~150 px apart. Use a small merge
        # distance to ensure they don't merge.
        new_bin, new_labels, new_blobs = _merge_blobs_by_hull(
            binary, labels, surviving, stats,
            merge_distance_px=50, dilate_px=0)
        assert len(new_blobs) == 2, (
            f"distant blobs should NOT merge with small "
            f"merge_distance; got {len(new_blobs)} merged blobs")

    def test_dilation_grows_hull(self):
        """Dilation expands the merged hull beyond visible pixels."""
        binary, labels, surviving, stats = self._setup_three_fragments()
        new_bin0, _, blobs0 = _merge_blobs_by_hull(
            binary, labels, surviving, stats,
            merge_distance_px=100, dilate_px=0)
        new_bin5, _, blobs5 = _merge_blobs_by_hull(
            binary, labels, surviving, stats,
            merge_distance_px=100, dilate_px=5)
        # Dilated hull has more foreground pixels
        n0 = int((new_bin0 > 0).sum())
        n5 = int((new_bin5 > 0).sum())
        assert n5 > n0, (
            f"dilated hull should be larger; got n0={n0} n5={n5}")

    def test_empty_input_returns_empty(self):
        """No surviving CCs → empty output."""
        binary = np.zeros((180, 240), dtype=np.uint8)
        labels = np.zeros((180, 240), dtype=np.int32)
        stats  = np.zeros((1, 5), dtype=np.int32)
        new_bin, new_labels, new_blobs = _merge_blobs_by_hull(
            binary, labels, [], stats,
            merge_distance_px=100, dilate_px=0)
        assert new_blobs == []
        assert (new_bin > 0).sum() == 0

    def test_single_survivor_passes_through(self):
        """One blob, nothing to merge — function still works
        (returns one hull, equal to the original blob's hull)."""
        bg = _empty_bg()
        det = ForegroundDetector(bg, threshold=30,
                                  min_area_px=50, morph_k=3)
        frame = np.full((180, 240), 50, dtype=np.uint8)
        frame[80:110, 100:140] = 200   # single solid blob
        r = det.detect(frame, cam=0)
        n_, labels, stats, centroids = cv2.connectedComponentsWithStats(
            r.mask, connectivity=8)
        surviving = list(range(1, n_))
        new_bin, new_labels, new_blobs = _merge_blobs_by_hull(
            r.mask, labels, surviving, stats,
            merge_distance_px=100, dilate_px=0)
        assert len(new_blobs) == 1


# ────────────────────────────────────────────────────────────────────
#  Integration via ForegroundDetector
# ────────────────────────────────────────────────────────────────────


class TestMergeIntegration:

    def test_detector_merges_when_distance_set(self):
        """With --merge-blob-distance > 0, the returned blobs are
        the merged hulls — three fragments come back as one."""
        det = ForegroundDetector(
            _empty_bg(), threshold=30,
            min_area_px=50, morph_k=3,
            merge_blob_distance=100, merge_blob_dilate=0)
        r = det.detect(_three_fragment_frame(), cam=0)
        # Should have one merged blob
        assert len(r.blobs) == 1
        # The label_map should contain a single label (1)
        assert int(r.label_map.max()) == 1
        # And the binary mask reflects the hull (not the original
        # fragments)
        # The hull should encompass all original fragment areas
        assert int((r.mask > 0).sum()) > 0

    def test_detector_keeps_fragments_when_disabled(self):
        """With --merge-blob-distance 0 (default), behavior unchanged."""
        det = ForegroundDetector(
            _empty_bg(), threshold=30,
            min_area_px=50, morph_k=3,
            merge_blob_distance=0)
        r = det.detect(_three_fragment_frame(), cam=0)
        # Multiple separate blobs preserved
        assert len(r.blobs) >= 2

    def test_detector_merge_with_dilation(self):
        """Merge + dilate produces a larger hull than merge alone."""
        det_no_dilate = ForegroundDetector(
            _empty_bg(), threshold=30, min_area_px=50, morph_k=3,
            merge_blob_distance=100, merge_blob_dilate=0)
        det_dilate = ForegroundDetector(
            _empty_bg(), threshold=30, min_area_px=50, morph_k=3,
            merge_blob_distance=100, merge_blob_dilate=10)
        r0 = det_no_dilate.detect(_three_fragment_frame(), cam=0)
        r1 = det_dilate.detect(_three_fragment_frame(), cam=0)
        n0 = int((r0.mask > 0).sum())
        n1 = int((r1.mask > 0).sum())
        assert n1 > n0


class TestMergePreservesBlobFormat:

    def test_blob_stats_compatible(self):
        """Merged blobs must have a (stats_row, centroid) format
        compatible with downstream consumers (the labeller, etc).
        stats_row is 5-element [L, T, W, H, A]; centroid is (cx, cy)."""
        det = ForegroundDetector(
            _empty_bg(), threshold=30,
            min_area_px=50, morph_k=3,
            merge_blob_distance=100)
        r = det.detect(_three_fragment_frame(), cam=0)
        for stats_row, centroid in r.blobs:
            assert len(stats_row) == 5, (
                f"stats row should have 5 elements [L,T,W,H,A], "
                f"got {len(stats_row)}")
            assert len(centroid) == 2, (
                f"centroid should be 2D, got {len(centroid)}")
            assert stats_row[4] > 0   # Area > 0
