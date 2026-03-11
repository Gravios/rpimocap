"""
tests/test_reconstruction.py
=============================
Unit tests for rpimocap.reconstruction (triangulate + voxel).
No camera hardware or video files required.
"""

import numpy as np
import pytest

from rpimocap.reconstruction.triangulate import (
    triangulate_dlt,
    reprojection_error,
    build_trajectory_dict,
    smooth_trajectory,
    fill_trajectory_gaps,
    trajectory_stats,
    Point3D,
)
from rpimocap.reconstruction.voxel import (
    VoxelGrid,
    build_voxel_grid,
    voxel_centers,
    project_points_batch,
    carve_frame,
    apply_carving,
    occupied_centers,
)


# --------------------------------------------------------------------------- #
#  Helpers                                                                     #
# --------------------------------------------------------------------------- #

def _simple_cameras():
    """Two cameras: cam0 at origin, cam1 shifted +100mm along X."""
    f, cx, cy = 900.0, 640.0, 360.0
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
    P0 = K @ np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]], dtype=float)
    R  = np.eye(3)
    t  = np.array([-100.0, 0.0, 0.0])   # cam1 is 100mm to the right
    P1 = K @ np.hstack([R, t.reshape(3,1)])
    return P0, P1, K


def _project(P, X3d):
    Xh = np.append(X3d, 1.0)
    p  = P @ Xh
    return p[:2] / p[2]


# --------------------------------------------------------------------------- #
#  triangulate_dlt                                                             #
# --------------------------------------------------------------------------- #

class TestTriangulateDLT:

    def test_round_trip_on_axis(self):
        P0, P1, _ = _simple_cameras()
        X_true = np.array([0.0, 0.0, 500.0])
        p0 = _project(P0, X_true)
        p1 = _project(P1, X_true)
        Xh = triangulate_dlt(P0, P1, tuple(p0), tuple(p1))
        X_rec = Xh[:3]
        np.testing.assert_allclose(X_rec, X_true, atol=0.1)

    def test_off_axis_point(self):
        P0, P1, _ = _simple_cameras()
        X_true = np.array([30.0, -20.0, 400.0])
        p0 = _project(P0, X_true)
        p1 = _project(P1, X_true)
        Xh = triangulate_dlt(P0, P1, tuple(p0), tuple(p1))
        np.testing.assert_allclose(Xh[:3], X_true, atol=0.5)

    def test_homogeneous_w_equals_one(self):
        P0, P1, _ = _simple_cameras()
        X_true = np.array([0.0, 0.0, 600.0])
        p0 = _project(P0, X_true)
        p1 = _project(P1, X_true)
        Xh = triangulate_dlt(P0, P1, tuple(p0), tuple(p1))
        assert Xh[3] == pytest.approx(1.0)

    def test_noisy_observations(self):
        """With 1px noise the 3D error should stay below a few mm at 500mm depth."""
        rng = np.random.default_rng(0)
        P0, P1, _ = _simple_cameras()
        X_true = np.array([10.0, -15.0, 500.0])
        p0 = _project(P0, X_true) + rng.standard_normal(2)
        p1 = _project(P1, X_true) + rng.standard_normal(2)
        Xh = triangulate_dlt(P0, P1, tuple(p0), tuple(p1))
        err = np.linalg.norm(Xh[:3] - X_true)
        assert err < 20.0, f"3D error {err:.2f}mm with 1px noise"


# --------------------------------------------------------------------------- #
#  reprojection_error                                                          #
# --------------------------------------------------------------------------- #

class TestReprojectionError:

    def test_zero_error_for_exact_projection(self):
        P0, _, _ = _simple_cameras()
        X_true = np.array([0.0, 0.0, 500.0, 1.0])
        p0 = _project(P0, X_true[:3])
        err = reprojection_error(P0, X_true, tuple(p0))
        assert err < 1e-6

    def test_nonzero_error_for_shifted_point(self):
        P0, _, _ = _simple_cameras()
        X_true = np.array([0.0, 0.0, 500.0, 1.0])
        p_bad = (640.0 + 5.0, 360.0)   # 5px offset
        err = reprojection_error(P0, X_true, p_bad)
        assert err > 4.0


# --------------------------------------------------------------------------- #
#  Trajectory utilities                                                        #
# --------------------------------------------------------------------------- #

def _make_frames(n=20):
    """Synthetic trajectory: nose moves in a circle."""
    frames = []
    for i in range(n):
        angle = 2 * np.pi * i / n
        x = 100 * np.cos(angle)
        y = 100 * np.sin(angle)
        z = 200.0 + i * 5.0
        pt = Point3D("nose", np.array([x, y, z]), confidence=0.9)
        frames.append([pt])
    return frames


class TestTrajectoryUtils:

    def test_build_trajectory_dict_shape(self):
        frames = _make_frames(20)
        traj = build_trajectory_dict(frames)
        assert "nose" in traj
        assert traj["nose"].shape == (20, 3)

    def test_build_trajectory_dict_values(self):
        frames = _make_frames(10)
        traj = build_trajectory_dict(frames)
        np.testing.assert_allclose(traj["nose"][0], frames[0][0].xyz)

    def test_smooth_trajectory_preserves_length(self):
        frames = _make_frames(30)
        smoothed = smooth_trajectory(frames, sigma=1.5)
        assert len(smoothed) == 30

    def test_smooth_trajectory_reduces_high_freq_noise(self):
        """Smoothing should reduce variance of a noisy trajectory."""
        rng = np.random.default_rng(42)
        frames = []
        for i in range(50):
            xyz = np.array([0.0, 0.0, 300.0]) + rng.standard_normal(3) * 20
            frames.append([Point3D("head", xyz, confidence=1.0)])
        orig_var  = np.var([f[0].xyz for f in frames])
        smooth    = smooth_trajectory(frames, sigma=2.0)
        sm_var    = np.var([f[0].xyz for f in smooth if f])
        assert sm_var < orig_var

    def test_fill_gaps_interpolates_short_gap(self):
        frames = _make_frames(20)
        # Punch a 3-frame gap at indices 5, 6, 7
        frames[5] = []
        frames[6] = []
        frames[7] = []
        filled = fill_trajectory_gaps(frames, max_gap=5)
        # Frames 5–7 should now contain interpolated values
        assert any(p.name == "nose" for p in filled[6])

    def test_fill_gaps_respects_max_gap(self):
        frames = _make_frames(30)
        # Punch a 10-frame gap (exceeds max_gap=5)
        for i in range(10, 20):
            frames[i] = []
        filled = fill_trajectory_gaps(frames, max_gap=5)
        # Middle of the gap (frame 15) should remain empty
        assert not any(p.name == "nose" for p in filled[15])

    def test_trajectory_stats_detection_rate(self):
        frames = _make_frames(20)
        frames[5] = []
        frames[10] = []
        stats = trajectory_stats(frames)
        rate = stats["nose"]["detection_rate"]
        assert rate == pytest.approx(18 / 20)


# --------------------------------------------------------------------------- #
#  VoxelGrid + carving                                                         #
# --------------------------------------------------------------------------- #

class TestVoxelGrid:

    def test_build_voxel_grid_shape(self):
        bounds = ((-100.0, 100.0), (-100.0, 100.0), (0.0, 200.0))
        grid = build_voxel_grid(bounds, voxel_size=20.0)
        assert grid.occupancy.all()
        assert grid.shape == (10, 10, 10)

    def test_voxel_centers_count(self):
        bounds = ((-50.0, 50.0), (-50.0, 50.0), (0.0, 100.0))
        grid = build_voxel_grid(bounds, voxel_size=25.0)
        centers = voxel_centers(grid)
        nx, ny, nz = grid.shape
        assert centers.shape == (nx * ny * nz, 3)

    def test_occupied_centers_count(self):
        bounds = ((-50.0, 50.0), (-50.0, 50.0), (0.0, 100.0))
        grid = build_voxel_grid(bounds, voxel_size=25.0)
        pts = occupied_centers(grid)
        assert pts.shape[0] == grid.occupancy.sum()

    def test_carve_removes_voxels(self):
        """After carving with a small mask, occupancy should strictly decrease."""
        P0, P1, K = _simple_cameras()
        bounds = ((-200.0, 200.0), (-200.0, 200.0), (0.0, 600.0))
        grid = build_voxel_grid(bounds, voxel_size=40.0)
        n_before = grid.occupancy.sum()

        h, w = 720, 1280
        # Small central foreground mask (subject in the middle)
        mask0 = np.zeros((h, w), dtype=np.uint8)
        mask1 = np.zeros((h, w), dtype=np.uint8)
        mask0[280:440, 540:740] = 255
        mask1[280:440, 540:740] = 255

        new_occ = carve_frame(grid, P0, P1, mask0, mask1, (w, h))
        carved  = apply_carving(grid, new_occ)
        assert carved.n_occupied < n_before

    def test_project_points_batch_shape(self):
        P0, _, _ = _simple_cameras()
        pts = np.random.default_rng(0).standard_normal((100, 3)) * 100
        pts[:, 2] += 500
        px = project_points_batch(P0, pts)
        assert px.shape == (100, 2)
