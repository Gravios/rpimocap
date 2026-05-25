"""
tests/test_step_stats.py
==========================
Tests for the pipeline-step counters added to hull_centroid +
SegmentTracker. The counters answer 'which refinement steps fired
and succeeded?' — useful for tuning --cable-erosion, --gabor-refine,
--body-length etc. without staring at composites.
"""
from __future__ import annotations

import numpy as np
import pytest


def _detector():
    from rpimocap.detection.segment import (
        BackgroundModel, ForegroundDetector)
    bg = BackgroundModel(np.zeros((200, 200), np.float32),
                         np.zeros((200, 200), np.float32))
    return ForegroundDetector(bg, threshold=10, min_area_px=10)


def _result_with_blob(h=200, w=200):
    """Build a ForegroundResult containing one filled disc as the only blob."""
    import cv2
    from rpimocap.detection.segment import ForegroundResult
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (100, 100), 30, 255, -1)
    label_map = (mask > 0).astype(np.int32)
    return ForegroundResult(
        mask=mask, blobs=[], frame_gray=np.zeros((h, w), np.uint8),
        n_blobs=1, label_map=label_map, gabor_energy=None)


class TestHullCentroidStats:

    def test_stats_kwarg_back_compat(self):
        """Default behaviour unchanged: hull_centroid still returns a
        (cx, cy) tuple and works without stats."""
        det = _detector()
        r = _result_with_blob()
        out = det.hull_centroid(r, 100, 100)
        assert isinstance(out, tuple) and len(out) == 2

    def test_cable_erosion_counters(self):
        det = _detector()
        r = _result_with_blob()
        stats: dict = {}
        det.hull_centroid(r, 100, 100,
                          cable_erosion_px=5, stats=stats)
        assert stats.get("cable_erosion_attempted", 0) == 1
        assert stats.get("cable_erosion_succeeded", 0) == 1   # disc survives

    def test_cable_erosion_attempted_but_failed(self):
        """If erosion is too aggressive, succeeded should NOT increment."""
        det = _detector()
        r = _result_with_blob()
        stats: dict = {}
        # 30-radius disc + 50-px erosion → completely erased
        det.hull_centroid(r, 100, 100,
                          cable_erosion_px=50, stats=stats)
        assert stats.get("cable_erosion_attempted", 0) == 1
        assert stats.get("cable_erosion_succeeded", 0) == 0

    def test_gabor_skipped_without_gabor_energy(self):
        """No gabor_energy on the ForegroundResult → step 3b is skipped
        and counters stay at 0 even if gabor_refine=True."""
        det = _detector()
        r = _result_with_blob()
        assert r.gabor_energy is None
        stats: dict = {}
        det.hull_centroid(r, 100, 100, gabor_refine=True, stats=stats)
        assert stats.get("gabor_refine_attempted", 0) == 0

    def test_gabor_attempted_with_energy(self):
        det = _detector()
        r = _result_with_blob()
        # Synthetic Gabor energy: high on bedding, low on body
        e = np.full(r.label_map.shape, 0.8, dtype=np.float32)
        e[r.label_map > 0] = 0.1
        r.gabor_energy = e
        stats: dict = {}
        det.hull_centroid(r, 100, 100, gabor_refine=True, stats=stats)
        assert stats.get("gabor_refine_attempted", 0) == 1

    def test_anatomical_prior_attempted_with_P_and_body_dims(self):
        det = _detector()
        r = _result_with_blob()
        # Identity-ish P that doesn't degenerate
        P = np.array([[800, 0, 100, 0],
                      [0, 800, 100, 0],
                      [0, 0,   1,  500]], dtype=np.float64)
        stats: dict = {}
        det.hull_centroid(r, 100, 100,
                          P=P, body_length_mm=180.0, body_width_mm=70.0,
                          stats=stats)
        assert stats.get("anatomical_prior_attempted", 0) == 1

    def test_fallback_counters_mutually_exclusive(self):
        """Exactly one of {anatomical_prior_succeeded, fallback_ellipse,
        fallback_hull} should fire per call."""
        det = _detector()
        r = _result_with_blob()
        stats: dict = {}
        # No P → step 5 skipped → ellipse fit succeeds → fallback_ellipse
        det.hull_centroid(r, 100, 100, stats=stats)
        total = (stats.get("anatomical_prior_succeeded", 0)
                 + stats.get("fallback_ellipse", 0)
                 + stats.get("fallback_hull", 0))
        assert total == 1, f"expected exactly one return path; got {stats}"


class TestSegmentTrackerStepStats:

    def test_step_stats_property_exists_and_returns_dict(self):
        import inspect
        from rpimocap.detection.tracker import SegmentTracker
        assert isinstance(
            inspect.getattr_static(SegmentTracker, "step_stats"),
            property)

    def test_step_stats_starts_empty(self):
        """Construction-time step_stats is empty (no sequence run yet)."""
        import inspect
        # Inspect the source to confirm initialisation; full ctor needs
        # a BackgroundModel and a bunch of components, more weight than
        # we want for a counter test.
        from rpimocap.detection import tracker as t_mod
        src = inspect.getsource(t_mod.SegmentTracker.__init__)
        assert "self._step_stats: dict = {}" in src

    def test_step_stats_reset_at_track_sequence_start(self):
        """The first thing track_sequence does is wipe _step_stats."""
        import inspect
        from rpimocap.detection import tracker as t_mod
        src = inspect.getsource(t_mod.SegmentTracker.track_sequence)
        assert "self._step_stats = {}" in src
