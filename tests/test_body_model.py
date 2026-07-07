"""Tests for the capsule body model (rpimocap.model.body_model)."""
import numpy as np

from rpimocap.model.body_model import (DEFAULT_RADII, render_pose_silhouette,
                                       render_silhouette, scale_radii,
                                       silhouette_iou)
from rpimocap.model.rat_skeleton import RatPose, forward_kinematics


def _P(C=(-400.0, -600.0, 500.0)):
    """A synthetic look-at DLT viewing the arena centre."""
    C = np.asarray(C, float)
    f, cx, cy = 900.0, 300.0, 250.0
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
    fwd = np.array([0, 0, 100.0]) - C; fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 0, 1.0]); right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    R = np.vstack([right, down, fwd])
    return K @ np.hstack([R, (-R @ C).reshape(3, 1)])


class TestBodyModel:

    def test_render_nonempty(self):
        kp = forward_kinematics(RatPose(root_pos=np.array([0.0, 0.0, 60.0])))
        sil = render_silhouette(kp, _P(), image_shape=(500, 600))
        assert sil.dtype == np.uint8
        assert int((sil > 0).sum()) > 500            # a visible body

    def test_iou_bounds(self):
        a = np.zeros((50, 50), np.uint8); a[10:30, 10:30] = 255
        assert silhouette_iou(a, a) == 1.0
        b = np.zeros((50, 50), np.uint8); b[35:45, 35:45] = 255
        assert silhouette_iou(a, b) == 0.0
        c = np.zeros((50, 50), np.uint8); c[20:40, 10:30] = 255   # half overlap
        assert 0.2 < silhouette_iou(a, c) < 0.4

    def test_scale_radii(self):
        big = scale_radii(DEFAULT_RADII, 2.0)
        b = ("SpineM", "SpineF")
        assert big[b][0] == 2.0 * DEFAULT_RADII[b][0]
        assert big[b][1] == 2.0 * DEFAULT_RADII[b][1]

    def test_pose_convenience_matches(self):
        pose = RatPose(root_pos=np.array([0.0, 0.0, 60.0]))
        s1 = render_pose_silhouette(pose, _P(), image_shape=(500, 600))
        s2 = render_silhouette(forward_kinematics(pose), _P(),
                               image_shape=(500, 600))
        assert np.array_equal(s1, s2)

    def test_bigger_scale_bigger_silhouette(self):
        P = _P()
        small = render_pose_silhouette(
            RatPose(root_pos=np.array([0.0, 0.0, 60.0]), scale=0.8), P,
            image_shape=(500, 600))
        big = render_pose_silhouette(
            RatPose(root_pos=np.array([0.0, 0.0, 60.0]), scale=1.4), P,
            image_shape=(500, 600))
        assert int((big > 0).sum()) > int((small > 0).sum())
