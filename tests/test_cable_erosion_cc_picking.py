"""
tests/test_cable_erosion_cc_picking.py
========================================
Regression for the "cable midpoint" bug: when the rat body is
fragmented by fur texture in bg-sub, but the cable remains
contiguous, the previous code picked the *largest* connected
component after cable erosion — which was the cable, not the rat
body. The fix picks the CC closest to the labeller centroid (which
is reliably on the rat in the pre-erosion blob).

Symptom in the field: triangulated xyz lands at the cable midpoint
(Y ≈ -180, Z ≈ +180 in a typical tether-from-front-top geometry),
not on the rat at Z ≈ 0.
"""
from __future__ import annotations

import numpy as np
import cv2


def _make_rat_with_cable(h=200, w=300, rat_xy=(80, 100), cable_to=(280, 30),
                          rat_holes=False):
    """Build a synthetic blob: a roundish rat at rat_xy connected by
    a thin straight cable to point cable_to (the 'mount').

    rat_holes=True simulates fur-texture fragmentation: the rat is
    drawn as a ring with several gaps so cv2.erode breaks it into
    multiple small fragments while leaving the cable intact."""
    mask = np.zeros((h, w), dtype=np.uint8)
    rat_cx, rat_cy = rat_xy
    if rat_holes:
        # 3 disconnected arcs ≈ a roundish rat with internal gaps
        cv2.ellipse(mask, (rat_cx, rat_cy), (28, 22), 0, 0, 90,
                    255, thickness=-1)
        cv2.ellipse(mask, (rat_cx, rat_cy), (28, 22), 0, 130, 220,
                    255, thickness=-1)
        cv2.ellipse(mask, (rat_cx, rat_cy), (28, 22), 0, 250, 320,
                    255, thickness=-1)
    else:
        cv2.circle(mask, (rat_cx, rat_cy), 25, 255, -1)
    # Thin cable from the rat to the mount
    cv2.line(mask, (rat_cx, rat_cy), cable_to, 255, 4)
    # Mount blob
    cv2.circle(mask, cable_to, 6, 255, -1)
    return mask


def _make_result_from_mask(mask):
    from rpimocap.detection.segment import ForegroundResult
    label_map = (mask > 0).astype(np.int32)
    return ForegroundResult(
        mask=mask, blobs=[], frame_gray=np.zeros_like(mask),
        n_blobs=1, label_map=label_map, gabor_energy=None)


def _detector():
    from rpimocap.detection.segment import (
        BackgroundModel, ForegroundDetector)
    bg = BackgroundModel(np.zeros((200, 300), np.float32),
                         np.zeros((200, 300), np.float32))
    return ForegroundDetector(bg, threshold=10, min_area_px=10)


class TestCablePickingByProximity:

    def test_picks_rat_fragment_not_cable_when_rat_fragmented(self):
        """The original failure mode: rat is fragmented, cable is
        contiguous → old code picked cable (largest CC after erosion)
        → centroid was at the cable midpoint. New code picks the CC
        closest to the labeller centroid (which is on the rat)."""
        det = _detector()
        mask = _make_rat_with_cable(rat_holes=True)
        r = _make_result_from_mask(mask)

        # Labeller centroid is roughly on the rat body (the pre-erosion
        # mass is dominated by the rat fragments + cable, but the rat
        # is at x=80 so the combined centroid is around x ≈ 100-120)
        labeller_cx, labeller_cy = 95.0, 100.0
        rcx, rcy = det.hull_centroid(
            r, labeller_cx, labeller_cy, cable_erosion_px=6)

        # Refined centroid should be CLOSE to the labeller centroid
        # (still on the rat), NOT close to the cable midpoint
        # (≈ (180, 65)) or the mount (≈ (280, 30)).
        dist_to_rat   = np.hypot(rcx - labeller_cx, rcy - labeller_cy)
        cable_midx, cable_midy = 180, 65
        dist_to_cable = np.hypot(rcx - cable_midx, rcy - cable_midy)
        assert dist_to_rat < dist_to_cable, (
            f"refined centroid ({rcx:.1f}, {rcy:.1f}) closer to cable "
            f"midpoint ({cable_midx},{cable_midy}) than to rat "
            f"({labeller_cx},{labeller_cy})")

    def test_picks_rat_when_no_fragmentation(self):
        """Sanity: with a solid rat + cable, the refined centroid
        should still be on the rat (closest to labeller)."""
        det = _detector()
        mask = _make_rat_with_cable(rat_holes=False)
        r = _make_result_from_mask(mask)
        labeller_cx, labeller_cy = 95.0, 100.0
        rcx, rcy = det.hull_centroid(
            r, labeller_cx, labeller_cy, cable_erosion_px=6)
        # Refined should be close to the rat (within ~25 px)
        assert np.hypot(rcx - 80, rcy - 100) < 25, \
            f"refined ({rcx:.1f}, {rcy:.1f}) far from rat (80, 100)"

    def test_stats_track_proximity_pick(self):
        det = _detector()
        mask = _make_rat_with_cable(rat_holes=True)
        r = _make_result_from_mask(mask)
        stats: dict = {}
        det.hull_centroid(r, 95.0, 100.0,
                          cable_erosion_px=6, stats=stats)
        # Cable erosion still counts as attempted + succeeded
        assert stats.get("cable_erosion_attempted", 0) == 1
        assert stats.get("cable_erosion_succeeded", 0) == 1
