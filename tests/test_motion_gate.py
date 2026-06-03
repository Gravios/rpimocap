"""
tests/test_motion_gate.py
==========================
The --motion-min gate removes bright features that pass bg-sub but
don't actually move between frames — cable mount hardware,
plexiglass reflections, specular highlights, edges. These are
physically fixed: their optical flow is zero. The rat is moving:
its optical flow is nonzero.

The gate is the same kind of pre-labelling filter as fur-gabor and
aspect-ratio, attacking the cable-wins-pick failure mode at yet
another angle.
"""
from __future__ import annotations

import numpy as np
import cv2

from rpimocap.detection.segment import (
    BackgroundModel, ForegroundDetector)


def _empty_bg(shape=(180, 240)):
    return BackgroundModel(
        bg0=np.full(shape, 50, dtype=np.float32),
        bg1=np.full(shape, 50, dtype=np.float32),
    )


def _frame_with(static_at=None, moving_at=None, shape=(180, 240)):
    """Build a frame with:
       * a STATIC bright square (same position across calls) at static_at
       * a MOVING bright square (caller passes different positions) at moving_at
    """
    f = np.full(shape, 50, dtype=np.uint8)
    if static_at is not None:
        y, x = static_at
        f[y-15:y+15, x-15:x+15] = 200
    if moving_at is not None:
        y, x = moving_at
        f[y-15:y+15, x-15:x+15] = 200
    return f


class TestMotionGateInputValidation:

    def test_rejects_invalid_motion_method(self):
        try:
            ForegroundDetector(_empty_bg(), threshold=30,
                                min_area_px=10, morph_k=3,
                                motion_min=1.0,
                                motion_method="bogus")
        except ValueError as e:
            assert "motion_method" in str(e)
        else:
            assert False, "expected ValueError on invalid motion_method"


class TestMotionGateFirstFrameNoOp:

    def test_first_frame_motion_gate_does_not_fire(self):
        """No previous frame yet -> motion gate is a no-op."""
        det = ForegroundDetector(_empty_bg(), threshold=30,
                                  min_area_px=10, morph_k=3,
                                  motion_min=1.0)
        frame = _frame_with(static_at=(90, 60))
        r = det.detect(frame, cam=0)
        # gate did NOT fire on first frame, so the static blob passes
        assert r.mask.any(), "first frame should not be gated"


class TestMotionGateRejectsStaticKeepsMoving:

    def test_static_blob_gated_out(self):
        """A blob that's at the same pixel in two consecutive frames
        is gated out (flow ≈ 0)."""
        det = ForegroundDetector(_empty_bg(), threshold=30,
                                  min_area_px=10, morph_k=3,
                                  motion_min=0.5,
                                  motion_method="framediff")
        # Frame 1: blob at (90, 60). First frame, gate doesn't fire.
        f1 = _frame_with(static_at=(90, 60))
        det.detect(f1, cam=0)
        # Frame 2: same blob, same position. Should be gated out now.
        f2 = _frame_with(static_at=(90, 60))
        r = det.detect(f2, cam=0)
        # No motion in the blob region -> binary should be empty there
        assert r.mask[75:105, 45:75].sum() == 0, (
            "static blob (zero motion between frames) should be "
            "gated out by motion-min")

    def test_moving_blob_survives(self):
        """A blob that shifts position between frames has nonzero
        framediff in BOTH the old and new locations, so it survives
        the gate."""
        det = ForegroundDetector(_empty_bg(), threshold=30,
                                  min_area_px=10, morph_k=3,
                                  motion_min=0.5,
                                  motion_method="framediff")
        f1 = _frame_with(moving_at=(90, 60))
        det.detect(f1, cam=0)
        # Blob moved by 20 px
        f2 = _frame_with(moving_at=(90, 80))
        r = det.detect(f2, cam=0)
        # Should pick up the new blob position
        assert r.mask.any(), "moving blob should survive motion gate"

    def test_static_rejected_moving_kept_in_same_frame(self):
        """The realistic scenario: a STATIC bright feature (cable
        mount hardware) and a MOVING bright feature (rat) both
        present. Static is rejected, moving survives."""
        det = ForegroundDetector(_empty_bg(), threshold=30,
                                  min_area_px=10, morph_k=3,
                                  motion_min=0.5,
                                  motion_method="framediff")
        # Frame 1: static at (50, 50), moving at (130, 90)
        f1 = _frame_with(static_at=(50, 50), moving_at=(130, 90))
        det.detect(f1, cam=0)
        # Frame 2: static at same place, moving shifted by 25 px
        f2 = _frame_with(static_at=(50, 50), moving_at=(130, 115))
        r = det.detect(f2, cam=0)
        # Static region should be empty
        static_pixels = r.mask[35:65, 35:65].sum() // 255
        # Moving region (new position) should have content
        moving_pixels = r.mask[100:130, 100:130].sum() // 255
        assert static_pixels == 0, (
            f"static blob region should be gated out; got "
            f"{static_pixels} pixels")
        assert moving_pixels > 0, (
            f"moving blob should survive; got {moving_pixels} pixels")


class TestMotionGateFlowMethod:
    """Optical-flow specifically (Farneback). On a static frame the
    flow is exactly zero in the interior; on a translating object it
    matches the actual displacement."""

    def test_flow_zero_for_static_blob(self):
        det = ForegroundDetector(_empty_bg(), threshold=30,
                                  min_area_px=10, morph_k=3,
                                  motion_min=0.5,
                                  motion_method="flow")
        f1 = _frame_with(static_at=(90, 60))
        det.detect(f1, cam=0)
        f2 = _frame_with(static_at=(90, 60))
        r = det.detect(f2, cam=0)
        # Static -> flow zero -> blob region empty after gate
        assert r.mask[75:105, 45:75].sum() == 0


class TestMotionGateDisabled:

    def test_disabled_no_op(self):
        det = ForegroundDetector(_empty_bg(), threshold=30,
                                  min_area_px=10, morph_k=3,
                                  motion_min=0.0)
        f1 = _frame_with(static_at=(90, 60))
        det.detect(f1, cam=0)
        f2 = _frame_with(static_at=(90, 60))
        r = det.detect(f2, cam=0)
        # Gate off -> static blob still detected
        assert r.mask.any()


class TestMotionGatePerCameraIndependence:
    """Each camera maintains its own prev-frame buffer."""

    def test_cam0_and_cam1_independent(self):
        det = ForegroundDetector(_empty_bg(), threshold=30,
                                  min_area_px=10, morph_k=3,
                                  motion_min=0.5,
                                  motion_method="framediff")
        # Build up cam0 prev frame
        det.detect(_frame_with(static_at=(90, 60)), cam=0)
        # First detect on cam1: gate should NOT fire (no prev cam1 frame)
        r1 = det.detect(_frame_with(static_at=(90, 60)), cam=1)
        assert r1.mask.any(), (
            "cam1's first detect should not be gated even though "
            "cam0 already has a prev frame")
        # Now cam1 has a prev frame; second detect of static gets gated
        r1b = det.detect(_frame_with(static_at=(90, 60)), cam=1)
        assert r1b.mask.sum() == 0
