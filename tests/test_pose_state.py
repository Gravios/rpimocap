"""Tests for the manual pose-fitting state (rpimocap.gui.pose_state)."""
import cv2
import numpy as np

from rpimocap.gui.pose_state import (PoseFitterState, pose_from_dict,
                                     pose_to_dict)
from rpimocap.model.body_model import render_silhouette
from rpimocap.model.rat_skeleton import RatPose, forward_kinematics
from tests.test_body_model import _P


def _capsule_render(pose, P, shp):
    return render_silhouette(forward_kinematics(pose), P, image_shape=shp)


def _make_state(tmp_path):
    shp = (500, 600)
    f0 = tmp_path / "cam0_0.png"
    f1 = tmp_path / "cam1_0.png"
    rng = np.random.default_rng(0)
    cv2.imwrite(str(f0), rng.integers(0, 255, shp, dtype=np.uint8))
    cv2.imwrite(str(f1), rng.integers(0, 255, shp, dtype=np.uint8))
    return PoseFitterState([(str(f0), str(f1))],
                           [_P((-400, -600, 500)), _P((400, -600, 500))],
                           _capsule_render, image_shape=shp)


class TestPoseSerialization:

    def test_round_trip(self):
        p = RatPose(root_pos=np.array([1.0, 2.0, 3.0]),
                    root_rot=np.array([0.1, 0.2, 0.3]), scale=1.2,
                    joint_angles={"SpineF": (0.0, 0.4, 0.0)})
        q = pose_from_dict(pose_to_dict(p))
        assert np.allclose(q.root_pos, p.root_pos)
        assert np.allclose(q.root_rot, p.root_rot)
        assert abs(q.scale - p.scale) < 1e-9
        assert np.allclose(q.joint_angles["SpineF"], p.joint_angles["SpineF"])


class TestPoseState:

    def test_overlay_shape(self, tmp_path):
        st = _make_state(tmp_path)
        ov = st.overlay(0, show_detected=False)
        assert ov.shape == (500, 600, 3) and ov.dtype == np.uint8
        assert int((ov > 0).sum()) > 0                 # model drawn

    def test_keyframe_save_load_restores_pose(self, tmp_path):
        st = _make_state(tmp_path)
        st.pose = RatPose(root_pos=np.array([5.0, 0.0, 60.0]), scale=1.1)
        st.save_current_pose()
        path = tmp_path / "kf.json"; st.write_poses(str(path))
        st2 = _make_state(tmp_path)
        st2.read_poses(str(path))
        st2.load_frame(0)                              # restores saved pose
        assert abs(st2.pose.scale - 1.1) < 1e-6
        assert abs(st2.pose.root_pos[0] - 5.0) < 1e-6

    def test_carry_pose_across_frames(self, tmp_path):
        st = _make_state(tmp_path)
        st.pose = RatPose(root_pos=np.array([7.0, 3.0, 55.0]), scale=1.05)
        st.load_frame(0, carry_pose=True)              # no keyframe saved
        assert abs(st.pose.root_pos[0] - 7.0) < 1e-6   # pose carried, not reset
