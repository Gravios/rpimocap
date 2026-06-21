"""Tests for the static-scene geometric gate (arena volume + floor +
dense static depth)."""
import numpy as np

from rpimocap.reconstruction import arena_gate as ag
from rpimocap.model import synthetic_dataset as sd


def _make_P(cam_pos, look_at, f=1500, cx=1014, cy=540):
    cam_pos = np.asarray(cam_pos, float)
    look_at = np.asarray(look_at, float)
    fwd = look_at - cam_pos
    fwd /= np.linalg.norm(fwd)
    up = np.array([0, 0, 1.0])
    right = np.cross(fwd, up); right /= np.linalg.norm(right)
    up2 = np.cross(right, fwd)
    R = np.vstack([right, -up2, fwd])
    t = -R @ cam_pos
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
    return K @ np.hstack([R, t.reshape(3, 1)])


CAMS = {0: _make_P([-300, -400, 700], [0, 0, 194]),
        1: _make_P([300, -400, 700], [0, 0, 194])}
IMG = (2028, 1080)


class TestVolumeAndFloor:

    def test_real_point_accepted(self):
        assert ag.accept_point(np.array([20.0, -50.0, 120.0]))

    def test_floor_reflection_rejected(self):
        # a point clearly above the floor; its reflection (z<0) fails
        assert not ag.accept_point(np.array([20.0, -50.0, -120.0]))

    def test_out_of_volume_rejected(self):
        assert not ag.accept_point(np.array([400.0, 0.0, 150.0]))
        assert not ag.accept_point(np.array([0.0, 500.0, 150.0]))

    def test_floor_tolerance(self):
        # a paw just above the floor passes; well below fails
        assert ag.above_floor(np.array([0, 0, 2.0]), tol_mm=20)
        assert not ag.above_floor(np.array([0, 0, -50.0]), tol_mm=20)

    def test_ceiling_pad(self):
        # at the top of the arena, within pad
        assert ag.in_arena_volume(np.array([0, 0, 388.0]))
        assert not ag.in_arena_volume(np.array([0, 0, 600.0]))


class TestStaticDepthGate:

    def test_rat_above_floor_accepted(self):
        gate = ag.build_static_depth_gate(
            CAMS, IMG, floor_z=0.0, tol_mm=25.0, stride=8)
        assert gate.accept(np.array([20.0, -50.0, 120.0]), CAMS)

    def test_below_floor_rejected(self):
        gate = ag.build_static_depth_gate(
            CAMS, IMG, floor_z=0.0, tol_mm=25.0, stride=8)
        assert not gate.accept(np.array([20.0, -50.0, -80.0]), CAMS)

    def test_behind_camera_rejected(self):
        gate = ag.build_static_depth_gate(
            CAMS, IMG, floor_z=0.0, tol_mm=25.0, stride=8)
        # a point behind both cameras
        assert not gate.accept(np.array([0.0, -2000.0, 700.0]), CAMS)

    def test_depth_maps_built_per_camera(self):
        gate = ag.build_static_depth_gate(CAMS, IMG, stride=16)
        assert set(gate.depth_maps.keys()) == {0, 1}
        for m in gate.depth_maps.values():
            assert m.shape == (IMG[1], IMG[0])


class TestAgainstSyntheticGroundTruth:

    def test_all_real_keypoints_pass(self):
        ds = sd.generate_dataset(15, CAMS, IMG, seed=3)
        for s in ds.samples:
            for k in range(23):
                assert ag.accept_point(s.keypoints3d[k])

    def test_clearly_above_reflections_rejected(self):
        """Reflections of keypoints that are clearly above the floor
        (z > tol) are rejected. Paws within floor tolerance are
        legitimately ambiguous and excluded from this check."""
        ds = sd.generate_dataset(15, CAMS, IMG, seed=3)
        checked = 0
        for s in ds.samples:
            for k in range(23):
                X = s.keypoints3d[k]
                if X[2] <= 25.0:
                    continue                # ambiguous near-floor
                refl = X.copy()
                refl[2] = -refl[2]
                assert not ag.accept_point(refl)
                checked += 1
        assert checked > 100                # exercised plenty
