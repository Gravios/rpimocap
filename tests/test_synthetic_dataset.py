"""Tests for the synthetic-pose dataset export (Phase A)."""
import numpy as np
import pytest

from rpimocap.model import rat_skeleton as rs
from rpimocap.model import synthetic_dataset as sd
from rpimocap.reconstruction.voxel import project_points_batch
from rpimocap.reconstruction.triangulate import triangulate_dlt

try:
    import torch as _torch          # noqa: F401
    _HAVE_TORCH = True
except ImportError:
    _HAVE_TORCH = False


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


# ────────────────────────────────────────────────────────────────────
#  Phase B: native-3D targets
# ────────────────────────────────────────────────────────────────────


class TestComCenteredGrid:

    def test_grid_centered_on_centroid(self):
        kp = rs.forward_kinematics(rs.RatPose(
            root_pos=np.array([10, -20, 150.0])))
        grid = sd.com_centered_grid(kp, n_vox=64, voxel_size=4.0)
        gc = grid.origin + 0.5 * np.array(grid.shape) * grid.voxel_size
        assert np.allclose(gc, kp.mean(axis=0), atol=1e-6)

    def test_grid_contains_keypoints(self):
        kp = rs.forward_kinematics(rs.RatPose(
            root_pos=np.array([0, 0, 150.0]),
            joint_angles=rs.sample_joint_angles(
                np.random.RandomState(0), fraction=0.4)))
        grid = sd.com_centered_grid(kp, n_vox=64, voxel_size=4.0)
        lo = grid.origin
        hi = grid.origin + np.array(grid.shape) * grid.voxel_size
        assert np.all(kp >= lo) and np.all(kp <= hi)

    def test_explicit_center(self):
        kp = rs.forward_kinematics(rs.RatPose())
        c = np.array([5.0, 5.0, 100.0])
        grid = sd.com_centered_grid(kp, n_vox=32, voxel_size=4.0,
                                    center=c)
        gc = grid.origin + 0.5 * np.array(grid.shape) * grid.voxel_size
        assert np.allclose(gc, c)


class TestHeatmapVolume:

    def setup_method(self):
        self.kp = rs.forward_kinematics(rs.RatPose(
            root_pos=np.array([0, 0, 150.0]),
            joint_angles=rs.sample_joint_angles(
                np.random.RandomState(1), fraction=0.4)))
        self.grid = sd.com_centered_grid(self.kp, 64, 4.0)

    def test_shape_and_range(self):
        vol = sd.keypoint_heatmap_volume(self.kp, self.grid, sigma_mm=8.0)
        assert vol.shape == (23, 64, 64, 64)
        assert vol.dtype == np.float32
        assert vol.min() >= 0.0 and vol.max() <= 1.0 + 1e-6
        assert vol.max() > 0.99            # a peak near 1 at a keypoint

    def test_argmax_recovers_keypoints(self):
        vol = sd.keypoint_heatmap_volume(self.kp, self.grid, sigma_mm=8.0)
        rec = sd.heatmap_argmax_keypoints(vol, self.grid)
        err = np.linalg.norm(rec - self.kp, axis=1)
        # within one voxel half-diagonal (sqrt(3)/2 · 4 ≈ 3.46 mm)
        assert err.max() < 3.5

    def test_separable_matches_dense(self):
        """The separable Gaussian equals the dense |c-k|² form."""
        vol = sd.keypoint_heatmap_volume(self.kp, self.grid, sigma_mm=8.0)
        xs, ys, zs = sd._voxel_axis_centers(self.grid)
        gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
        k = self.kp[4]
        d2 = (gx - k[0]) ** 2 + (gy - k[1]) ** 2 + (gz - k[2]) ** 2
        dense = np.exp(-d2 / (2 * 8.0 ** 2))
        assert np.allclose(vol[4], dense, atol=1e-5)

    def test_larger_sigma_spreads(self):
        v1 = sd.keypoint_heatmap_volume(self.kp, self.grid, sigma_mm=4.0)
        v2 = sd.keypoint_heatmap_volume(self.kp, self.grid, sigma_mm=12.0)
        # larger sigma → more total mass above a threshold
        assert (v2[0] > 0.5).sum() > (v1[0] > 0.5).sum()


class TestVisualHull:

    def test_hull_contains_body(self):
        body = sd.RatBodyModel.default()
        kp = rs.forward_kinematics(rs.RatPose(
            root_pos=np.array([0, 0, 150.0]),
            joint_angles=rs.sample_joint_angles(
                np.random.RandomState(2), fraction=0.4)))
        grid = sd.com_centered_grid(kp, 64, 4.0)
        hull = sd.visual_hull(body, kp, CAMS, IMG, grid)
        assert hull.sum() > 100
        occ = body.occupancy(kp, grid=grid).occupancy
        contained = (occ & hull).sum() / max(occ.sum(), 1)
        assert contained > 0.95            # hull is a superset of the body


class TestDatasetPhaseB:

    def setup_method(self):
        self.ds = sd.generate_dataset(6, CAMS, IMG, seed=4,
                                      pose_fraction=0.5)

    def test_heatmap_volume_method(self):
        vol, grid = self.ds.heatmap_volume(2, n_vox=48, voxel_size=4.0)
        assert vol.shape == (23, 48, 48, 48)
        rec = sd.heatmap_argmax_keypoints(vol, grid)
        err = np.linalg.norm(rec - self.ds.samples[2].keypoints3d, axis=1)
        assert err.max() < 3.5

    def test_visual_hull_method(self):
        hull = self.ds.visual_hull(1, n_vox=48)
        assert hull.shape == (48, 48, 48)
        assert hull.sum() > 50

    def test_com_grid_method(self):
        grid = self.ds.com_grid(0, n_vox=32, voxel_size=4.0)
        assert grid.shape == (32, 32, 32)


@pytest.mark.skipif(not _HAVE_TORCH, reason="torch not installed")
class TestTorchAdapter:
    """torch is optional; these skip cleanly when it's absent."""

    def test_torch_dataset_heatmap(self):
        ds = sd.generate_dataset(4, CAMS, IMG, seed=6)
        td = ds.torch_dataset(target="heatmap", n_vox=32, voxel_size=4.0)
        item = td[0]
        assert item["target"].shape == (23, 32, 32, 32)
        assert item["keypoints3d"].shape == (23, 3)
        assert item["keypoints2d"].shape == (2, 23, 2)
        assert item["grid_origin"].shape == (3,)

    def test_torch_dataset_keypoints(self):
        ds = sd.generate_dataset(4, CAMS, IMG, seed=6)
        td = ds.torch_dataset(target="keypoints", valid_only=False)
        item = td[0]
        assert item["target"].shape == (23, 3)

    def test_torch_dataset_with_hull(self):
        ds = sd.generate_dataset(3, CAMS, IMG, seed=6)
        td = ds.torch_dataset(target="heatmap", n_vox=24,
                              include_hull=True, valid_only=False)
        item = td[0]
        assert item["hull"].shape == (24, 24, 24)


# ────────────────────────────────────────────────────────────────────
#  Phase C: self-occlusion + silhouette cache + shape-prior matrix
# ────────────────────────────────────────────────────────────────────


class TestSelfOcclusion:

    def test_far_side_keypoint_occluded_by_trunk(self):
        """From a camera at -y, the far-side shoulder (y>0, behind the
        trunk) is occluded; the near-side shoulder (y<0) is visible."""
        body = sd.RatBodyModel.default()
        kp = rs.forward_kinematics(rs.RatPose(
            root_pos=np.array([0, 0, 160.0])))
        P = _make_P([0, -700, 160], [0, 0, 160])
        occ = sd.keypoint_self_occlusion(body, kp, {0: P})[0]
        assert occ[rs.RAT23_INDEX["ShoulderL"]]        # far → occluded
        assert not occ[rs.RAT23_INDEX["ShoulderR"]]    # near → visible

    def test_occlusion_only_removes_visibility(self):
        """Generating with occlusion yields visibility ⊆ in-frame-only
        (occlusion can never make a keypoint visible)."""
        a = sd.generate_dataset(8, CAMS, IMG, seed=9,
                                compute_occlusion=False)
        b = sd.generate_dataset(8, CAMS, IMG, seed=9,
                                compute_occlusion=True)
        for sa, sb in zip(a.samples, b.samples):
            assert np.allclose(sa.keypoints3d, sb.keypoints3d)
            for c in (0, 1):
                assert np.all(sb.visibility[c] <= sa.visibility[c])

    def test_occlusion_removes_some(self):
        a = sd.generate_dataset(10, CAMS, IMG, seed=9,
                                compute_occlusion=False)
        b = sd.generate_dataset(10, CAMS, IMG, seed=9,
                                compute_occlusion=True)
        na = sum(int(s.visibility[0].sum() + s.visibility[1].sum())
                 for s in a.samples)
        nb = sum(int(s.visibility[0].sum() + s.visibility[1].sum())
                 for s in b.samples)
        assert nb < na

    def test_occlusion_deterministic_across_workers(self):
        a = sd.generate_dataset(12, CAMS, IMG, seed=9,
                                compute_occlusion=True, n_workers=1)
        b = sd.generate_dataset(12, CAMS, IMG, seed=9,
                                compute_occlusion=True, n_workers=4)
        for sa, sb in zip(a.samples, b.samples):
            assert np.array_equal(sa.visibility[0], sb.visibility[0])
            assert np.array_equal(sa.visibility[1], sb.visibility[1])


class TestSilhouetteCache:

    def test_cache_and_load(self, tmp_path):
        ds = sd.generate_dataset(6, CAMS, IMG, seed=14)
        d = str(tmp_path / "ds")
        n = ds.cache_silhouettes(d, downsample=4)
        assert n == 6 * 2
        sil = ds.load_cached_silhouette(d, 3, 0)
        assert sil.shape == (IMG[1] // 4, IMG[0] // 4)
        # cached (downsampled) ≈ on-demand downsampled
        import os
        assert os.path.exists(
            os.path.join(d, "silhouettes", "cam0", "000003.png"))

    def test_cache_valid_only(self, tmp_path):
        ds = sd.generate_dataset(8, CAMS, IMG, seed=15)
        n_valid = sum(s.valid for s in ds.samples)
        d = str(tmp_path / "ds")
        n = ds.cache_silhouettes(d, downsample=8, valid_only=True)
        assert n == n_valid * 2


class TestShapePriorMatrix:

    def test_matrix_shape(self):
        ds = sd.generate_dataset(10, CAMS, IMG, seed=16)
        mat, (h, w), idx = ds.silhouette_matrix(
            0, downsample=16, valid_only=True)
        assert mat.shape[1] == h * w
        assert mat.shape[0] == len(idx)
        assert mat.dtype == np.float32
        # binary masks → values in {0,1}
        assert set(np.unique(mat)).issubset({0.0, 1.0})

    def test_matrix_rows_match_indices(self):
        ds = sd.generate_dataset(8, CAMS, IMG, seed=17)
        mat, (h, w), idx = ds.silhouette_matrix(
            1, downsample=16, valid_only=False)
        assert len(idx) == 8                    # all included
        # a nonempty pose has nonzero silhouette mass
        assert mat.sum(axis=1).max() > 0
