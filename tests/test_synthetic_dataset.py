"""Tests for the synthetic-pose dataset export (Phase A)."""
import numpy as np
import pytest

from rpimocap.model import rat_skeleton as rs
from rpimocap.model import synthetic_dataset as sd
from rpimocap.reconstruction.voxel import project_points_batch
from rpimocap.reconstruction.triangulate import triangulate_dlt


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


# ────────────────────────────────────────────────────────────────────
#  Body model
# ────────────────────────────────────────────────────────────────────


class TestRatBodyModel:

    def setup_method(self):
        self.body = sd.RatBodyModel.default()
        self.kp = rs.forward_kinematics(rs.RatPose(
            root_pos=np.array([0, 0, 150.0]),
            joint_angles=rs.sample_joint_angles(
                np.random.RandomState(0), fraction=0.4)))

    def test_capsule_count(self):
        caps = self.body.capsules(self.kp)
        # 22 bones + 5 default spheres
        assert len(caps) == 22 + 5

    def test_silhouette_nonempty(self):
        P = _make_P([0, -500, 500], [0, 0, 150])
        sil = self.body.silhouette(self.kp, P, IMG)
        assert sil.shape == (IMG[1], IMG[0])
        assert (sil > 0).sum() > 1000

    def test_occupancy_nonempty(self):
        grid = self.body.occupancy(self.kp, voxel_size=4.0)
        assert grid.n_occupied > 100
        # occupied fraction of the bbox is modest (a rat, not a blob)
        frac = grid.n_occupied / np.prod(grid.shape)
        assert 0.005 < frac < 0.6

    def test_occupancy_subset_of_silhouette(self):
        """The 2-D silhouette and 3-D occupancy come from the same
        capsules, so projected occupancy voxels must land inside the
        silhouette."""
        import cv2
        P = _make_P([0, -600, 400], [0, 0, 194])
        sil = self.body.silhouette(self.kp, P, IMG)
        grid = self.body.occupancy(self.kp, voxel_size=4.0)
        idx = np.argwhere(grid.occupancy)
        centers = grid.origin + (idx + 0.5) * grid.voxel_size
        px = project_points_batch(P, centers)
        sil_d = cv2.dilate(sil, np.ones((5, 5), np.uint8))
        H, W = sil.shape
        xi = np.clip(px[:, 0].astype(int), 0, W - 1)
        yi = np.clip(px[:, 1].astype(int), 0, H - 1)
        inside = (sil_d[yi, xi] > 0).mean()
        assert inside > 0.98

    def test_scale_grows_silhouette(self):
        P = _make_P([0, -500, 500], [0, 0, 150])
        small = rs.forward_kinematics(rs.RatPose(
            root_pos=np.array([0, 0, 150.0]), scale=0.8))
        big = rs.forward_kinematics(rs.RatPose(
            root_pos=np.array([0, 0, 150.0]), scale=1.2))
        a = (self.body.silhouette(small, P, IMG) > 0).sum()
        b = (self.body.silhouette(big, P, IMG) > 0).sum()
        assert b > a

    def test_radii_snapshot_roundtrips(self):
        snap = self.body.radii_snapshot()
        body2 = sd._body_from_snapshot(snap)
        assert body2.version == self.body.version
        assert body2.sphere_radii == self.body.sphere_radii
        assert body2.bone_radii == self.body.bone_radii

    def test_surface_points(self):
        rng = np.random.RandomState(1)
        pts = self.body.surface_points(self.kp, 500, rng)
        assert pts.shape[0] > 0 and pts.shape[1] == 3


# ────────────────────────────────────────────────────────────────────
#  Dataset generation
# ────────────────────────────────────────────────────────────────────


class TestGenerateDataset:

    def test_basic_generation(self):
        ds = sd.generate_dataset(20, CAMS, IMG, seed=1, pose_fraction=0.6)
        assert len(ds) == 20
        for s in ds.samples:
            assert s.keypoints3d.shape == (23, 3)
            assert set(s.keypoints2d.keys()) == {0, 1}
            assert s.keypoints2d[0].shape == (23, 2)
            assert s.visibility[0].shape == (23,)

    def test_determinism(self):
        a = sd.generate_dataset(15, CAMS, IMG, seed=3)
        b = sd.generate_dataset(15, CAMS, IMG, seed=3)
        for x, y in zip(a.samples, b.samples):
            assert np.allclose(x.keypoints3d, y.keypoints3d)

    def test_different_seed_differs(self):
        a = sd.generate_dataset(10, CAMS, IMG, seed=1)
        b = sd.generate_dataset(10, CAMS, IMG, seed=2)
        # extremely unlikely to coincide
        assert not np.allclose(a.samples[0].keypoints3d,
                               b.samples[0].keypoints3d)

    def test_workers_equivalence(self):
        a = sd.generate_dataset(24, CAMS, IMG, seed=5, n_workers=1)
        b = sd.generate_dataset(24, CAMS, IMG, seed=5, n_workers=4)
        for x, y in zip(a.samples, b.samples):
            assert np.allclose(x.keypoints3d, y.keypoints3d)
            assert np.allclose(x.keypoints2d[0], y.keypoints2d[0])

    def test_valid_samples_pass_is_valid(self):
        ds = sd.generate_dataset(20, CAMS, IMG, seed=8,
                                 require_in_arena=True)
        for s in ds.samples:
            if s.valid:
                assert rs.is_valid(s.pose, require_arena=True)


# ────────────────────────────────────────────────────────────────────
#  Ground-truth round trip
# ────────────────────────────────────────────────────────────────────


class TestRoundTrip:

    def test_keypoints_triangulate_back(self):
        ds = sd.generate_dataset(15, CAMS, IMG, seed=42)
        max_err = 0.0
        for s in ds.samples:
            for k in range(23):
                X = triangulate_dlt(
                    CAMS[0], CAMS[1],
                    tuple(s.keypoints2d[0][k]),
                    tuple(s.keypoints2d[1][k]))
                max_err = max(max_err,
                              np.linalg.norm(X[:3] - s.keypoints3d[k]))
        assert max_err < 1e-3


# ────────────────────────────────────────────────────────────────────
#  Persistence
# ────────────────────────────────────────────────────────────────────


class TestSaveLoad:

    def test_roundtrip(self, tmp_path):
        ds = sd.generate_dataset(18, CAMS, IMG, seed=11,
                                 pose_fraction=0.5)
        d = str(tmp_path / "ds")
        ds.save(d)
        import os
        assert set(os.listdir(d)) == {"manifest.npz", "meta.json"}
        ds2 = sd.SyntheticPoseDataset.load(d)
        assert len(ds2) == 18
        for a, b in zip(ds.samples, ds2.samples):
            assert np.allclose(a.keypoints3d, b.keypoints3d)
            assert np.allclose(a.keypoints2d[0], b.keypoints2d[0])
            assert np.allclose(a.keypoints2d[1], b.keypoints2d[1])
            assert np.array_equal(a.visibility[0], b.visibility[0])
            assert a.valid == b.valid
            assert np.allclose(a.pose.root_pos, b.pose.root_pos)
            assert np.allclose(a.pose.root_rot, b.pose.root_rot)
            assert np.isclose(a.pose.scale, b.pose.scale)

    def test_loaded_pose_regenerates_keypoints(self, tmp_path):
        """The stored pose params must regenerate the stored keypoints
        via FK (the compact form is the real source of truth)."""
        ds = sd.generate_dataset(10, CAMS, IMG, seed=13)
        d = str(tmp_path / "ds")
        ds.save(d)
        ds2 = sd.SyntheticPoseDataset.load(d)
        for s in ds2.samples:
            kp = rs.forward_kinematics(s.pose)
            assert np.allclose(kp, s.keypoints3d, atol=1e-6)

    def test_meta_has_provenance(self, tmp_path):
        ds = sd.generate_dataset(5, CAMS, IMG, seed=99)
        d = str(tmp_path / "ds")
        ds.save(d)
        import json
        import os
        with open(os.path.join(d, "meta.json")) as fh:
            meta = json.load(fh)
        assert meta["seed"] == 99
        assert meta["skeleton_version"] == "rat23"
        assert meta["body"]["version"] == "body-v1"
        assert "cameras" in meta and set(meta["cameras"]) == {"0", "1"}

    def test_ondemand_silhouette_from_loaded(self, tmp_path):
        ds = sd.generate_dataset(8, CAMS, IMG, seed=21)
        d = str(tmp_path / "ds")
        ds.save(d)
        ds2 = sd.SyntheticPoseDataset.load(d)
        sil = ds2.silhouette(2, 0)
        assert sil.shape == (IMG[1], IMG[0])
        assert (sil > 0).sum() > 100
