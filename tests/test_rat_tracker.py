"""
tests/test_rat_tracker.py
==========================
EdgeMotionRatTracker — KLT + DBSCAN + Kalman hull.
Verifies the architecture works on synthetic frame sequences where:
  - a rat-like blob with texture moves coherently
  - a static cable-mount-like blob does NOT move
The tracker should lock onto the moving blob and produce hull masks
that contain the moving region but exclude the static one.
"""
from __future__ import annotations

import numpy as np
import cv2

from rpimocap.detection.rat_tracker import (
    EdgeMotionRatTracker, _dbscan_xy_v, _make_kalman)


# ────────────────────────────────────────────────────────────────────
#  Low-level building blocks
# ────────────────────────────────────────────────────────────────────


class TestDbscanXyV:

    def test_separates_two_clusters(self):
        """Two spatially-separate clusters of points → 2 labels."""
        # Cluster A: 10 points near (50, 50), all moving (1, 0)
        ptsA = np.random.RandomState(0).uniform(48, 52, size=(10, 2))
        flowsA = np.tile([1.0, 0.0], (10, 1))
        # Cluster B: 10 points near (200, 200), all moving (-1, 0)
        ptsB = np.random.RandomState(1).uniform(198, 202, size=(10, 2))
        flowsB = np.tile([-1.0, 0.0], (10, 1))
        pts = np.vstack([ptsA, ptsB])
        flows = np.vstack([flowsA, flowsB])

        labels = _dbscan_xy_v(pts, flows,
                                eps_xy=20, eps_v=0.5,
                                min_samples=3)
        # Should identify 2 clusters (labels 0 and 1)
        assert labels.max() == 1
        # Members of cluster A should all share a label, distinct from B
        assert len(np.unique(labels[:10])) == 1
        assert len(np.unique(labels[10:])) == 1
        assert labels[0] != labels[10]

    def test_velocity_separation(self):
        """Two clusters at the SAME spatial position but DIFFERENT
        velocities should still be separable thanks to the velocity
        dimension."""
        # Both clusters around (100, 100), but moving in opposite directions
        ptsA = np.random.RandomState(0).uniform(98, 102, size=(10, 2))
        flowsA = np.tile([5.0, 0.0], (10, 1))
        ptsB = np.random.RandomState(1).uniform(98, 102, size=(10, 2))
        flowsB = np.tile([-5.0, 0.0], (10, 1))
        pts = np.vstack([ptsA, ptsB])
        flows = np.vstack([flowsA, flowsB])
        labels = _dbscan_xy_v(pts, flows,
                                eps_xy=10, eps_v=2,
                                min_samples=3)
        # Despite spatial overlap, velocity space separates the two
        assert labels.max() >= 1, (
            "two distinct-velocity clusters at the same position "
            "should be separable")

    def test_empty_input(self):
        labels = _dbscan_xy_v(np.zeros((0, 2)), np.zeros((0, 2)))
        assert labels.shape == (0,)

    def test_all_noise(self):
        """Sparse scattered points → all -1 (noise)."""
        pts = np.array([[10, 10], [200, 10], [10, 200], [200, 200]])
        flows = np.tile([1.0, 0.0], (4, 1))
        labels = _dbscan_xy_v(pts, flows,
                                eps_xy=20, eps_v=0.5,
                                min_samples=3)
        assert (labels == -1).all()


class TestKalmanConstruction:

    def test_kalman_dimensions(self):
        kf = _make_kalman()
        assert kf.transitionMatrix.shape == (5, 5)
        assert kf.measurementMatrix.shape == (3, 5)
        assert kf.processNoiseCov.shape == (5, 5)
        assert kf.measurementNoiseCov.shape == (3, 3)


# ────────────────────────────────────────────────────────────────────
#  Full tracker
# ────────────────────────────────────────────────────────────────────


def _make_textured_blob(canvas, cx, cy, w=40, h=40,
                          base_intensity=180, rng=None):
    """Draw a textured "rat-like" blob at (cx, cy) on the canvas.
    Random noise inside the blob gives KLT corners to track."""
    if rng is None:
        rng = np.random.RandomState(42)
    y0, y1 = max(0, cy - h // 2), min(canvas.shape[0], cy + h // 2)
    x0, x1 = max(0, cx - w // 2), min(canvas.shape[1], cx + w // 2)
    h_, w_ = y1 - y0, x1 - x0
    if h_ <= 0 or w_ <= 0:
        return
    base = np.full((h_, w_), base_intensity, dtype=np.uint8)
    noise = rng.randint(-40, 40, size=(h_, w_)).astype(np.int16)
    patch = np.clip(base.astype(np.int16) + noise, 50, 250).astype(np.uint8)
    canvas[y0:y1, x0:x1] = patch


def _make_smooth_blob(canvas, cx, cy, w=40, h=40, intensity=210):
    """Draw a smooth "cable-mount-like" blob — no internal texture,
    so KLT has nothing to track inside it (gradient is flat)."""
    y0, y1 = max(0, cy - h // 2), min(canvas.shape[0], cy + h // 2)
    x0, x1 = max(0, cx - w // 2), min(canvas.shape[1], cx + w // 2)
    canvas[y0:y1, x0:x1] = intensity


class TestEdgeMotionRatTrackerLockOn:

    def test_locks_on_to_moving_textured_blob(self):
        """Build a 10-frame sequence with a textured blob moving to
        the right at 5 px/frame and a static smooth blob elsewhere.
        After several frames, the tracker should produce a hull
        observation whose center is near the textured blob."""
        H, W = 240, 320
        rng = np.random.RandomState(0)
        tracker = EdgeMotionRatTracker(
            frame_shape=(H, W),
            body_half_width_px=20,
            motion_min=1.0,
            min_cluster_points=3)
        rat_x, rat_y = 80, 120
        last_obs = None
        for i in range(15):
            canvas = np.full((H, W), 50, dtype=np.uint8)
            # Add some background texture so KLT has SOMETHING to seed
            # away from the rat too (otherwise rat dominates trivially)
            bg_noise = rng.randint(40, 60, size=(H, W)).astype(np.uint8)
            canvas[:] = bg_noise
            # Static smooth blob (cable mount surrogate) — no texture
            _make_smooth_blob(canvas, 250, 60, w=30, h=30)
            # Moving textured blob
            _make_textured_blob(canvas, rat_x, rat_y,
                                  w=40, h=40, rng=rng)
            obs = tracker.step(canvas, cam=0, frame_idx=i)
            if obs is not None and not obs.from_kalman_only:
                last_obs = obs
            rat_x += 5

        # After 15 frames the tracker should have made at least one
        # observation. Its center should be reasonably close to the
        # rat's last position (rat moved from 80 to 150 over the run,
        # but tracker lock-on usually takes 1-2 frames).
        assert last_obs is not None, (
            "tracker should have produced at least one real observation "
            "of the textured moving blob")
        # The observation center should be far from the static blob (250, 60)
        d_to_static = np.hypot(last_obs.cx - 250, last_obs.cy - 60)
        d_to_rat    = np.hypot(last_obs.cx - rat_x, last_obs.cy - rat_y)
        assert d_to_static > d_to_rat, (
            f"observation center ({last_obs.cx:.1f}, {last_obs.cy:.1f}) "
            f"should be closer to rat last position "
            f"({rat_x}, {rat_y}) than to static blob (250, 60); "
            f"got d_to_rat={d_to_rat:.1f} d_to_static={d_to_static:.1f}")


class TestEdgeMotionRatTrackerKalmanContinuity:

    def test_kalman_predicts_when_rat_freezes(self):
        """When the rat stops moving (no flow), the Kalman should
        keep emitting hull observations near the last known
        location, rather than returning None."""
        H, W = 240, 320
        tracker = EdgeMotionRatTracker(
            frame_shape=(H, W),
            body_half_width_px=20,
            motion_min=1.0,
            min_cluster_points=3)
        rng = np.random.RandomState(7)

        # Phase 1: rat moves to (120, 120) — establish lock
        rat_x, rat_y = 80, 120
        for i in range(10):
            canvas = rng.randint(40, 60, size=(H, W)).astype(np.uint8)
            _make_textured_blob(canvas, rat_x, rat_y, w=40, h=40, rng=rng)
            tracker.step(canvas, cam=0, frame_idx=i)
            rat_x += 4
        # We're now at rat_x = 120

        # Phase 2: rat freezes at (120, 120) — no motion
        last_pos = None
        for i in range(10, 20):
            canvas = rng.randint(40, 60, size=(H, W)).astype(np.uint8)
            _make_textured_blob(canvas, rat_x, rat_y, w=40, h=40, rng=rng)
            obs = tracker.step(canvas, cam=0, frame_idx=i)
            if obs is not None:
                last_pos = (obs.cx, obs.cy, obs.from_kalman_only)

        # During freeze, the tracker should still produce a hull
        # near the last known position via Kalman predict
        assert last_pos is not None
        assert abs(last_pos[0] - rat_x) < 40, (
            f"Kalman extrapolation drifted: hull cx={last_pos[0]:.1f}, "
            f"rat at x={rat_x}")


class TestEdgeMotionRatTrackerExcludesStatic:

    def test_static_blob_does_not_become_hull(self):
        """When ONLY a static blob is present (no rat), no real
        observation should be made. Critically uses a STABLE bg
        (not per-frame noise) since real cameras have stable bg
        between consecutive frames."""
        H, W = 240, 320
        tracker = EdgeMotionRatTracker(
            frame_shape=(H, W),
            body_half_width_px=20,
            motion_min=1.0,
            min_cluster_points=3)
        # Single canvas, identical every frame (the static-bg case)
        rng = np.random.RandomState(11)
        canvas_template = rng.randint(40, 60, size=(H, W)).astype(np.uint8)
        _make_smooth_blob(canvas_template, 200, 120,
                            w=40, h=40, intensity=210)
        real_observations = 0
        for i in range(15):
            canvas = canvas_template.copy()
            obs = tracker.step(canvas, cam=0, frame_idx=i)
            if obs is not None and not obs.from_kalman_only:
                real_observations += 1
        # With identical frames, KLT flow magnitudes are exactly
        # zero everywhere — no point clears motion_min — no
        # observation possible.
        assert real_observations == 0, (
            f"tracker should NOT find any 'moving' cluster when the "
            f"frames are identical; got {real_observations} obs")


class TestEdgeMotionRatTrackerCamerasIndependent:

    def test_cameras_independent_state(self):
        """Each camera maintains its own KLT/Kalman state — calling
        step() on cam 0 doesn't affect cam 1's state."""
        H, W = 240, 320
        tracker = EdgeMotionRatTracker(
            frame_shape=(H, W),
            body_half_width_px=20,
            motion_min=1.0,
            min_cluster_points=3)
        rng = np.random.RandomState(0)
        # Cam 0: feed a few frames
        for i in range(5):
            canvas = rng.randint(40, 60, size=(H, W)).astype(np.uint8)
            _make_textured_blob(canvas, 100, 120, w=40, h=40, rng=rng)
            tracker.step(canvas, cam=0, frame_idx=i)
        # Cam 1: nothing fed yet — first call should treat it as init
        canvas1 = rng.randint(40, 60, size=(H, W)).astype(np.uint8)
        _make_textured_blob(canvas1, 100, 120, w=40, h=40, rng=rng)
        obs1 = tracker.step(canvas1, cam=1, frame_idx=5)
        # First-frame init returns None
        assert obs1 is None

    def test_reset_clears_state(self):
        H, W = 100, 100
        tracker = EdgeMotionRatTracker(
            frame_shape=(H, W),
            body_half_width_px=10,
            motion_min=1.0,
            min_cluster_points=3)
        canvas = np.full((H, W), 100, dtype=np.uint8)
        tracker.step(canvas, cam=0, frame_idx=0)
        assert tracker._cams[0].has_init
        tracker.reset()
        assert not tracker._cams[0].has_init


class TestSnapshot:

    def test_snapshot_returns_dict(self):
        tracker = EdgeMotionRatTracker(frame_shape=(100, 100))
        snap = tracker.snapshot(cam=0)
        assert isinstance(snap, dict)
        assert "n_klt_points" in snap
        assert "has_init" in snap
        assert snap["has_init"] is False
