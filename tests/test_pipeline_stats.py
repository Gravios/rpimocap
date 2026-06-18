"""
tests/test_pipeline_stats.py
============================
Per-stage detection counters on ForegroundDetector. When detection
fails on a frame, the stats summary tells us which stage rejected
the rat.
"""
from __future__ import annotations

import numpy as np

from rpimocap.detection.segment import (
    BackgroundModel, ForegroundDetector)


def _frame_with_bright_blob(shape=(200, 300),
                              bg_intensity=80,
                              blob_intensity=200,
                              blob_box=(80, 130, 100, 200)):
    f = np.full(shape, bg_intensity, dtype=np.uint8)
    y0, y1, x0, x1 = blob_box
    f[y0:y1, x0:x1] = blob_intensity
    return f


class TestPipelineStatsBasic:

    def test_fresh_counts_zero(self):
        bg = BackgroundModel(
            bg0=np.zeros((200, 300), dtype=np.float32),
            bg1=np.zeros((200, 300), dtype=np.float32))
        det = ForegroundDetector(background=bg, threshold=10)
        stats = det.get_pipeline_stats()
        assert stats[0]["frames"] == 0
        assert stats[1]["frames"] == 0
        for v in stats[0].values():
            assert v == 0

    def test_frame_count_increments_per_detect(self):
        bg = BackgroundModel(
            bg0=np.full((200, 300), 80, dtype=np.float32),
            bg1=np.full((200, 300), 80, dtype=np.float32))
        det = ForegroundDetector(
            background=bg, threshold=30, min_area_px=200, morph_k=3)
        frame = _frame_with_bright_blob()
        det.detect(frame.copy(), cam=0)
        det.detect(frame.copy(), cam=0)
        det.detect(frame.copy(), cam=1)
        stats = det.get_pipeline_stats()
        assert stats[0]["frames"] == 2
        assert stats[1]["frames"] == 1

    def test_reset_zeroes_counters(self):
        bg = BackgroundModel(
            bg0=np.full((200, 300), 80, dtype=np.float32),
            bg1=np.full((200, 300), 80, dtype=np.float32))
        det = ForegroundDetector(
            background=bg, threshold=30, min_area_px=200, morph_k=3)
        frame = _frame_with_bright_blob()
        det.detect(frame.copy(), cam=0)
        assert det.get_pipeline_stats()[0]["frames"] == 1
        det.reset_pipeline_stats()
        assert det.get_pipeline_stats()[0]["frames"] == 0


class TestPipelineStatsTracksStages:

    def test_passing_blob_advances_all_stages(self):
        """A frame with a clean, well-bounded bright blob passes
        every stage and counts toward each stage's survival count."""
        bg = BackgroundModel(
            bg0=np.full((200, 300), 80, dtype=np.float32),
            bg1=np.full((200, 300), 80, dtype=np.float32))
        det = ForegroundDetector(
            background=bg,
            threshold=30, min_area_px=200, morph_k=3,
            max_aspect_ratio=10)
        det.detect(_frame_with_bright_blob().copy(), cam=0)
        s = det.get_pipeline_stats()[0]
        assert s["frames"] == 1
        assert s["bg_sub_has_pixels"] == 1
        assert s["after_morph_open"] == 1
        assert s["after_area_solidity"] == 1
        assert s["after_aspect"] == 1
        # No texture bank → after_texture_bank counts same as aspect
        assert s["after_texture_bank"] == 1
        assert s["after_merge"] == 1
        assert s["final"] == 1
        assert s["bg_sub_blobs_total"] >= 1
        assert s["final_blobs_total"] >= 1

    def test_area_filter_rejects_tiny_blob(self):
        """A tiny bright blob below min_area_px gets dropped at the
        area filter, and the stats show the drop-off."""
        bg = BackgroundModel(
            bg0=np.full((200, 300), 80, dtype=np.float32),
            bg1=np.full((200, 300), 80, dtype=np.float32))
        # Need min_area_px high enough to reject the small blob.
        # The frame contains a 5x5 blob (area 25 pre-morph).
        det = ForegroundDetector(
            background=bg, threshold=30, min_area_px=10000, morph_k=3)
        frame = _frame_with_bright_blob(blob_box=(95, 100, 145, 150))
        det.detect(frame.copy(), cam=0)
        s = det.get_pipeline_stats()[0]
        assert s["frames"] == 1
        # bg-sub fires, blob exists pre-morph
        # but the area filter rejects it
        assert s["after_area_solidity"] == 0
        assert s["after_texture_bank"] == 0
        assert s["final"] == 0


class TestStageMaskCapture:

    def test_no_capture_by_default(self):
        bg = BackgroundModel(
            bg0=np.full((200, 300), 80, dtype=np.float32),
            bg1=np.full((200, 300), 80, dtype=np.float32))
        det = ForegroundDetector(
            background=bg, threshold=30, min_area_px=200, morph_k=3)
        det.detect(_frame_with_bright_blob().copy(), cam=0)
        # Capture is off by default → last_stage_masks should be empty
        assert det._last_stage_masks == {}

    def test_capture_populates_stages(self):
        bg = BackgroundModel(
            bg0=np.full((200, 300), 80, dtype=np.float32),
            bg1=np.full((200, 300), 80, dtype=np.float32))
        det = ForegroundDetector(
            background=bg, threshold=30, min_area_px=200, morph_k=3)
        det._capture_stage_masks = True
        det.detect(_frame_with_bright_blob().copy(), cam=0)
        det._capture_stage_masks = False
        captured = det._last_stage_masks
        assert "cam" in captured
        assert captured["cam"] == 0
        stages = captured["stages"]
        # Should have at least bg_sub, filtered, merged, final
        # (no edge refine without texture bank)
        assert "1_bg_sub" in stages
        assert "2_filtered" in stages
        assert "3_merged" in stages
        assert "6_final" in stages
        # Each captured mask must be (H, W) and uint8
        for name, m in stages.items():
            assert m is None or (m.ndim == 2 and m.dtype == np.uint8)
