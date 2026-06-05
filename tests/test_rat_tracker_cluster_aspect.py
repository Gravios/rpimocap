"""
tests/test_rat_tracker_cluster_aspect.py
=========================================
Cluster aspect-ratio filter on EdgeMotionRatTracker.

The KLT corners along a tether cable form an elongated cluster
(aspect ~10-30 along the wire direction) because the cable's
specular highlights are strong Shi-Tomasi features that move
coherently with the rat. Diagnostic OVERLAYs from the user's
v12 run showed exactly this — clusters of dots along the cable
path rather than on the rat body.

This filter rejects clusters whose minAreaRect aspect exceeds
--rat-cluster-max-aspect-ratio (default 4.0). The rat's cluster
is roundish (aspect 1.5-3); the cable's is highly elongated.
"""
from __future__ import annotations

import numpy as np

from rpimocap.detection.rat_tracker import EdgeMotionRatTracker


class TestClusterAspectFilterUnit:
    """Direct unit tests on _observe — feed synthetic KLT clusters."""

    def _tracker(self, max_aspect=4.0):
        return EdgeMotionRatTracker(
            frame_shape=(240, 320),
            body_half_width_px=20,
            motion_min=0.5,
            min_cluster_points=10,
            cluster_max_aspect_ratio=max_aspect,
            dbscan_eps_xy=40.0,
            dbscan_eps_v=2.0)

    def test_thin_cable_cluster_rejected(self):
        """A line of 20 KLT points spanning ~200 px on a single
        axis (aspect ~20). Reject at threshold=4."""
        t = self._tracker(max_aspect=4.0)
        # Make a horizontal line of points around y=120, x=50..250
        pts = np.column_stack([
            np.linspace(50, 250, 20),
            np.full(20, 120.0),
        ]).astype(np.float32)
        # All moving the same way (cable swings with rat)
        flows = np.tile([2.0, 0.0], (20, 1)).astype(np.float32)
        obs = t._observe(
            points=pts, flows=flows,
            pred_cx=150.0, pred_cy=120.0, pred_r=30.0,
            frame_idx=10)
        assert obs is None, (
            "thin elongated cluster (aspect ~20) should be rejected "
            "by max_aspect_ratio=4")

    def test_roundish_rat_cluster_accepted(self):
        """A roughly round cluster (aspect ~1.5). Should pass."""
        t = self._tracker(max_aspect=4.0)
        rng = np.random.RandomState(0)
        # Cluster of 20 points in a 60x40 box around (150, 120)
        pts = np.column_stack([
            rng.uniform(120, 180, 20),
            rng.uniform(100, 140, 20),
        ]).astype(np.float32)
        flows = np.tile([2.0, 0.0], (20, 1)).astype(np.float32)
        obs = t._observe(
            points=pts, flows=flows,
            pred_cx=150.0, pred_cy=120.0, pred_r=30.0,
            frame_idx=10)
        assert obs is not None, (
            "roundish cluster (aspect ~1.5) should pass aspect filter")

    def test_both_clusters_present_picks_rat(self):
        """A rat-like cluster + a cable-like cluster — the
        cable is rejected and the rat is picked."""
        t = self._tracker(max_aspect=4.0)
        rng = np.random.RandomState(1)
        # Rat cluster (15 pts around (80, 80), roundish)
        rat_pts = np.column_stack([
            rng.uniform(70, 90, 15),
            rng.uniform(70, 90, 15),
        ])
        rat_flows = np.tile([1.0, 0.5], (15, 1))
        # Cable cluster (20 pts in a line near y=180, x=80..280)
        cable_pts = np.column_stack([
            np.linspace(80, 280, 20),
            np.full(20, 180.0),
        ])
        cable_flows = np.tile([1.5, 0.0], (20, 1))
        pts = np.vstack([rat_pts, cable_pts]).astype(np.float32)
        flows = np.vstack([rat_flows, cable_flows]).astype(np.float32)
        obs = t._observe(
            points=pts, flows=flows,
            pred_cx=80.0, pred_cy=80.0, pred_r=15.0,
            frame_idx=10)
        assert obs is not None, "rat cluster should still be found"
        # The picked cluster's mean should be near the rat (80, 80),
        # NOT on the cable (around (180, 180))
        assert abs(obs.cx - 80) < 30, (
            f"picked cluster should be near rat at x=80, got {obs.cx:.1f}")
        assert abs(obs.cy - 80) < 30, (
            f"picked cluster should be near rat at y=80, got {obs.cy:.1f}")

    def test_filter_disabled_keeps_thin_cluster(self):
        """When cluster_max_aspect_ratio=None, even a thin cluster passes."""
        t = self._tracker(max_aspect=None)
        # Dense line of 30 points spanning 150 px (5 px spacing
        # so DBSCAN with eps_xy=40 keeps them as one cluster)
        pts = np.column_stack([
            np.linspace(100, 250, 30),
            np.full(30, 120.0),
        ]).astype(np.float32)
        flows = np.tile([2.0, 0.0], (30, 1)).astype(np.float32)
        obs = t._observe(
            points=pts, flows=flows,
            pred_cx=175.0, pred_cy=120.0, pred_r=30.0,
            frame_idx=10)
        assert obs is not None, (
            "with filter disabled, even a thin cluster should pass")

    def test_high_threshold_keeps_thin_cluster(self):
        """A lenient threshold (50) keeps the cable cluster."""
        t = self._tracker(max_aspect=50.0)
        pts = np.column_stack([
            np.linspace(100, 250, 30),
            np.full(30, 120.0),
        ]).astype(np.float32)
        flows = np.tile([2.0, 0.0], (30, 1)).astype(np.float32)
        obs = t._observe(
            points=pts, flows=flows,
            pred_cx=175.0, pred_cy=120.0, pred_r=30.0,
            frame_idx=10)
        assert obs is not None, (
            "lenient max_aspect=50 should keep thin cluster (aspect ~30)")


class TestClusterAspectFilterCornerCases:

    def test_explicit_none_disables_filter(self):
        """Constructing with cluster_max_aspect_ratio=None stores
        None on the instance, the filter never fires."""
        t = EdgeMotionRatTracker(
            frame_shape=(240, 320),
            cluster_max_aspect_ratio=None)
        assert t.cluster_max_aspect_ratio is None

    def test_explicit_value_stored_as_float(self):
        t = EdgeMotionRatTracker(
            frame_shape=(240, 320),
            cluster_max_aspect_ratio=6)
        assert t.cluster_max_aspect_ratio == 6.0
        assert isinstance(t.cluster_max_aspect_ratio, float)
