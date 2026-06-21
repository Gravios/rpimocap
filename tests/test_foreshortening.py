"""Tests for the lens-foreshortening / STMap module."""
import cv2
import numpy as np

from rpimocap.detection.foreshortening import (
    build_undistort_stmap, apply_stmap, normalize_stmap,
    footprint_anisotropy_plane, anisotropy_weight, _camera_center)


def _make_P(cam_pos, look_at, K, cx=1014, cy=540):
    cam_pos = np.asarray(cam_pos, float)
    look_at = np.asarray(look_at, float)
    fwd = look_at - cam_pos
    fwd /= np.linalg.norm(fwd)
    up = np.array([0, 1.0, 0]) if abs(fwd[2]) > 0.9 else np.array([0, 0, 1.0])
    right = np.cross(fwd, up); right /= np.linalg.norm(right)
    up2 = np.cross(right, fwd)
    R = np.vstack([right, -up2, fwd])
    t = -R @ cam_pos
    return K @ np.hstack([R, t.reshape(3, 1)])


K_TEST = np.array([[1500, 0, 1014], [0, 1500, 540], [0, 0, 1]], float)


class TestSTMap:

    def test_shape_and_dtype(self):
        stmap = build_undistort_stmap(
            K_TEST, np.array([-0.3, 0.1, 0, 0, 0]), (2028, 1080))
        assert stmap.shape == (1080, 2028, 2)
        assert stmap.dtype == np.float32

    def test_zero_distortion_is_identity(self):
        """With no distortion, the STMap maps each output pixel to
        itself."""
        stmap = build_undistort_stmap(
            K_TEST, np.zeros(5), (200, 100))
        yy, xx = np.mgrid[0:100, 0:200]
        assert np.allclose(stmap[..., 0], xx, atol=1e-2)
        assert np.allclose(stmap[..., 1], yy, atol=1e-2)

    def test_apply_preserves_shape(self):
        stmap = build_undistort_stmap(
            K_TEST, np.array([-0.2, 0, 0, 0, 0]), (400, 300))
        img = np.random.RandomState(0).randint(
            0, 255, (300, 400), np.uint8)
        out = apply_stmap(img, stmap)
        assert out.shape == img.shape

    def test_apply_identity_returns_same(self):
        stmap = build_undistort_stmap(K_TEST, np.zeros(5), (200, 150))
        img = np.random.RandomState(1).randint(
            0, 255, (150, 200), np.uint8)
        out = apply_stmap(img, stmap)
        # identity remap reproduces the image (modulo border interp)
        assert np.mean(np.abs(out.astype(int)
                              - img.astype(int))) < 1.0

    def test_normalize_range(self):
        stmap = build_undistort_stmap(K_TEST, np.zeros(5), (200, 100))
        uv = normalize_stmap(stmap, (200, 100))
        assert 0.0 <= uv[..., 0].min() and uv[..., 0].max() <= 1.0
        assert 0.0 <= uv[..., 1].min() and uv[..., 1].max() <= 1.0


class TestCameraCenter:

    def test_recovers_camera_position(self):
        cam = np.array([100.0, -200.0, 500.0])
        P = _make_P(cam, [0, 0, 0], K_TEST)
        C = _camera_center(P)
        assert np.allclose(C, cam, atol=1e-6)


class TestForeshortening:

    def test_face_on_is_unity(self):
        """A camera looking straight down at the floor sees the frame
        centre face-on → anisotropy ≈ 1."""
        P = _make_P([0, 0, 800], [0, 0, 0], K_TEST)
        a = footprint_anisotropy_plane(
            P, [0, 0, 0], [0, 0, 1], (2028, 1080), stride=8)
        assert a[540, 1014] < 1.3

    def test_corner_more_foreshortened_than_center(self):
        P = _make_P([0, 0, 800], [0, 0, 0], K_TEST)
        a = footprint_anisotropy_plane(
            P, [0, 0, 0], [0, 0, 1], (2028, 1080), stride=8)
        assert a[50, 50] > a[540, 1014]

    def test_oblique_view_more_foreshortened_overall(self):
        P_top = _make_P([0, 0, 800], [0, 0, 0], K_TEST)
        P_obl = _make_P([0, -600, 400], [0, 0, 0], K_TEST)
        a_top = footprint_anisotropy_plane(
            P_top, [0, 0, 0], [0, 0, 1], (2028, 1080), stride=8)
        a_obl = footprint_anisotropy_plane(
            P_obl, [0, 0, 0], [0, 0, 1], (2028, 1080), stride=8)
        assert a_obl.mean() > a_top.mean()

    def test_anisotropy_always_at_least_one(self):
        P = _make_P([0, -600, 400], [0, 0, 0], K_TEST)
        a = footprint_anisotropy_plane(
            P, [0, 0, 0], [0, 0, 1], (2028, 1080), stride=16)
        assert a.min() >= 1.0 - 1e-4

    def test_stride_upsamples_to_full_size(self):
        P = _make_P([0, 0, 800], [0, 0, 0], K_TEST)
        a = footprint_anisotropy_plane(
            P, [0, 0, 0], [0, 0, 1], (640, 480), stride=8)
        assert a.shape == (480, 640)


class TestAnisotropyWeight:

    def test_face_on_full_weight(self):
        a = np.ones((10, 10), np.float32)         # face-on
        w = anisotropy_weight(a, max_aniso=3.0)
        assert np.allclose(w, 1.0)

    def test_grazing_zero_weight(self):
        a = np.full((10, 10), 5.0, np.float32)    # beyond max_aniso
        w = anisotropy_weight(a, max_aniso=3.0)
        assert np.allclose(w, 0.0)

    def test_monotonic_decreasing(self):
        a = np.array([[1.0, 1.5, 2.0, 2.5, 3.0]], np.float32)
        w = anisotropy_weight(a, max_aniso=3.0)
        assert np.all(np.diff(w[0]) <= 0)
