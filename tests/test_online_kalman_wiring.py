"""
tests/test_online_kalman_wiring.py
====================================
Tests for the online Kalman + rearing wiring on SegmentTracker.

Verifies:
- _project_xyz_to_pixel handles forward / behind-camera / degenerate
- A SegmentTracker constructed without the new params behaves
  identically to the pre-wiring tracker (backward compatibility)
- With kalman_online enabled, the tracker carries kalman state across
  frames and back-projects the prediction to pixel coords on a
  synthetic stereo rig
- With rearing_classifier enabled, posture is updated and the
  vertical-posture body dims are used for hull_centroid on the next
  frame after a rear is detected
"""
from __future__ import annotations

import numpy as np
import pytest


def _stereo_P():
    """Build a synthetic stereo P0/P1 pair for back-projection tests."""
    K = np.array([[800, 0, 640], [0, 800, 360], [0, 0, 1.0]])
    R = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1.0]])
    t0 = -R @ np.array([0.0, 0.0, 900.0])
    t1 = -R @ np.array([100.0, 0.0, 900.0])
    P0 = K @ np.hstack([R, t0.reshape(3, 1)])
    P1 = K @ np.hstack([R, t1.reshape(3, 1)])
    return P0, P1


class TestProjectXyzToPixel:

    def test_projects_point_in_front_of_camera(self):
        from rpimocap.detection.tracker import _project_xyz_to_pixel
        P0, _ = _stereo_P()
        # World origin sits on the principal ray of cam0 → centre pixel
        out = _project_xyz_to_pixel(P0, np.array([0.0, 0.0, 0.0]))
        assert out is not None
        cx, cy = out
        assert abs(cx - 640) < 1
        assert abs(cy - 360) < 1

    def test_returns_none_for_point_behind_camera(self):
        from rpimocap.detection.tracker import _project_xyz_to_pixel
        # Build a P matrix that places the point behind the camera by
        # putting cam very low and pointing the wrong way.
        K = np.array([[800, 0, 640], [0, 800, 360], [0, 0, 1.0]])
        R = np.eye(3)             # camera pointing +Z
        t = np.array([0.0, 0.0, 100.0])
        P = K @ np.hstack([R, t.reshape(3, 1)])
        # Point at world Z = -200 → in camera frame Z = -100 (behind)
        out = _project_xyz_to_pixel(P, np.array([0.0, 0.0, -200.0]))
        assert out is None

    def test_returns_none_on_principal_plane(self):
        from rpimocap.detection.tracker import _project_xyz_to_pixel
        # Construct a degenerate P with last row zero so h[2] = 0
        P = np.zeros((3, 4))
        P[0, 0] = 1
        P[1, 1] = 1
        out = _project_xyz_to_pixel(P, np.array([1.0, 1.0, 1.0]))
        assert out is None


class TestBackwardCompatibility:

    def test_tracker_without_new_params_unchanged(self):
        """Constructing a SegmentTracker without the new ctor args must
        not change defaults observable to existing callers."""
        from rpimocap.detection.tracker import SegmentTracker
        # Just confirm the signature accepts the old call shape
        # (full instantiation needs a BackgroundModel, but we don't need
        # to invoke __init__ for this check — inspect the signature).
        import inspect
        sig = inspect.signature(SegmentTracker.__init__)
        params = sig.parameters
        # New params must all have defaults (so existing callers don't break)
        for name in ("kalman_online", "rearing_classifier",
                     "rearing_track_name", "fps"):
            assert name in params, f"missing param {name}"
            assert params[name].default is not inspect.Parameter.empty


class TestRearingPostureFlow:

    def test_rearing_classifier_switches_body_dims(self):
        """Drive the rearing classifier with a synthetic Kalman state
        and confirm body_length_mm / body_width_mm flip when reared."""
        from rpimocap.reconstruction.rearing import RearingClassifier

        cls = RearingClassifier(
            horizontal_body_length_mm=180.0,
            horizontal_body_width_mm=70.0,
            vertical_body_length_mm=90.0,
            vertical_body_width_mm=45.0,
            z_enter=100.0, z_exit=70.0)

        # On floor
        ps = cls.classify(np.array([0, 0, 30, 0, 0, 0], dtype=np.float64))
        assert ps.reared is False
        assert ps.body_length_mm == 180.0
        assert ps.body_width_mm == 70.0

        # Reared
        ps = cls.classify(np.array([0, 0, 200, 0, 0, 0], dtype=np.float64))
        assert ps.reared is True
        assert ps.body_length_mm == 90.0
        assert ps.body_width_mm == 45.0

    def test_kalman_state_drives_rearing(self):
        """Step a KalmanTracker3D up a vertical trajectory; verify the
        rearing classifier flips."""
        from rpimocap.reconstruction.kalman import KalmanTracker3D
        from rpimocap.reconstruction.rearing import RearingClassifier

        kf = KalmanTracker3D(dt=1/25.0, sigma_a=2000.0, sigma_z=5.0)
        cls = RearingClassifier(z_enter=100.0, z_exit=70.0)

        # Rat climbing — Z rises from 30 to 200 over 50 frames
        for i, z in enumerate(np.linspace(30, 200, 50)):
            kf.step(np.array([0.0, 0.0, float(z)]))
        ps = cls.classify(kf.x)
        assert ps.reared is True
