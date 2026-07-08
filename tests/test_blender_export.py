"""Tests for the Blender scene exporter (rpimocap.gui.blender_export)."""
import numpy as np

from rpimocap.gui.blender_export import (build_scene_spec, decompose_dlt,
                                         dlt_to_blender_camera, skeleton_spec)


def _lookat_R(C, target=(0, 0, 100), up=(0, 1, 0)):
    C = np.asarray(C, float)
    f = np.asarray(target, float) - C
    f /= np.linalg.norm(f)
    up = np.asarray(up, float)
    r = np.cross(up, f); r /= np.linalg.norm(r)
    d = np.cross(f, r)
    return np.vstack([r, d, f])            # rows = camera axes in world (world→cam)


def _P(K, R, C):
    t = -R @ C
    return K @ np.hstack([R, t.reshape(3, 1)])


def _proj(P, X):
    w = (P @ np.hstack([X, np.ones((len(X), 1))]).T).T
    return w[:, :2] / w[:, 2:]


class TestDecompose:

    def test_reproduces_projection(self):
        K = np.array([[1200.0, 0, 960], [0, 1180, 540], [0, 0, 1]])
        C = np.array([80.0, -120, 760])
        R = _lookat_R(C)
        P = _P(K, R, C)
        K2, R2, t2, C2 = decompose_dlt(P)
        assert np.allclose(C2, C, atol=1e-6)
        X = np.random.default_rng(0).uniform([-140, -215, 0], [140, 215, 388],
                                             (30, 3))
        pc = (R2 @ X.T + t2[:, None]).T
        b = (K2 @ pc.T).T; b = b[:, :2] / b[:, 2:]
        assert np.allclose(_proj(P, X), b, atol=1e-6)
        assert (pc[:, 2] > 0).all()                    # scene in front

    def test_recovers_intrinsics(self):
        K = np.array([[1300.0, 0, 1000], [0, 1250, 520], [0, 0, 1]])
        C = np.array([40.0, 10, 800])
        K2, _, _, _ = decompose_dlt(_P(K, _lookat_R(C), C))
        assert abs(K2[0, 0] - 1300) < 1e-3 and abs(K2[1, 1] - 1250) < 1e-3
        assert abs(K2[0, 2] - 1000) < 1e-3 and abs(K2[1, 2] - 520) < 1e-3


class TestBlenderCamera:

    def test_params_sane(self):
        K = np.array([[1300.0, 0, 1000], [0, 1250, 520], [0, 0, 1]])
        C = np.array([40.0, 10, 800])
        p = dlt_to_blender_camera(_P(K, _lookat_R(C), C), 2000, 1040)
        assert p["lens"] > 0 and p["sensor_fit"] == "HORIZONTAL"
        assert np.allclose(p["location"], C, atol=1e-6)
        assert abs(p["pixel_aspect_y"] - 1300.0 / 1250.0) < 1e-6
        assert {"location", "rotation_c2w", "lens", "shift_x",
                "shift_y"} <= set(p)
        # rotation is a proper rotation
        R = np.array(p["rotation_c2w"])
        assert abs(np.linalg.det(R) - 1.0) < 1e-6


class TestSkeletonSpec:

    def test_structure(self):
        s = skeleton_spec()
        assert len(s["joints"]) == 23 and len(s["bones"]) == 22
        assert len(s["ik_chains"]) == 4
        assert np.allclose(s["rest"]["SpineM"], [0, 0, 0], atol=1e-6)
        assert "HandL" in {c["tip"] for c in s["ik_chains"]}


class TestSceneSpec:

    def test_build(self, tmp_path):
        import cv2
        K = np.array([[1300.0, 0, 1000], [0, 1250, 520], [0, 0, 1]])
        C0 = np.array([80.0, 0, 780]); C1 = np.array([-80.0, 0, 780])
        P0 = _P(K, _lookat_R(C0), C0); P1 = _P(K, _lookat_R(C1), C1)
        np.savez(str(tmp_path / "cal.npz"), dlt_P0=P0, dlt_P1=P1)
        cv2.imwrite(str(tmp_path / "c0.png"), np.zeros((1040, 2000), np.uint8))
        cv2.imwrite(str(tmp_path / "c1.png"), np.zeros((1040, 2000), np.uint8))
        spec = build_scene_spec(str(tmp_path / "cal.npz"),
                                str(tmp_path / "c0.png"),
                                str(tmp_path / "c1.png"),
                                str(tmp_path / "spec.json"))
        assert spec["resolution"] == [2000, 1040]
        assert len(spec["cameras"]) == 2
        assert len(spec["arena"]["corners"]) == 8
        assert (tmp_path / "spec.json").exists()
