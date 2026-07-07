"""Tests for the pose fitter (rpimocap.model.fit).

Kept small: a synthetic look-at camera pair and low-resolution renders so the
Powell optimization runs quickly.
"""
import numpy as np

from rpimocap.model.body_model import render_silhouette
from rpimocap.model.fit import (TUCKED_ANGLES, _clamp_joint, curled_pose,
                                fit_pose, fit_pose_multistart,
                                fit_pose_staged, multiview_iou)
from rpimocap.model.rat_skeleton import (JOINT_LIMITS, RatPose,
                                         check_joint_angles,
                                         forward_kinematics)
from tests.test_body_model import _P


class TestFit:

    def test_self_iou_is_one(self):
        pose = RatPose(root_pos=np.array([0.0, 0.0, 60.0]),
                       root_rot=np.array([0.0, 0.0, 0.5]))
        P = _P()
        m = render_silhouette(forward_kinematics(pose), P, image_shape=(400, 480))
        assert multiview_iou(pose, [P], [m]) > 0.99

    def test_fit_improves_and_recovers(self):
        P0 = _P((-400.0, -600.0, 500.0))
        P1 = _P((400.0, -600.0, 500.0))
        target = RatPose(root_pos=np.array([0.0, 0.0, 60.0]),
                         root_rot=np.array([0.0, 0.0, 0.8]), scale=1.1)
        kp = forward_kinematics(target)
        masks = [render_silhouette(kp, P0, image_shape=(400, 480)),
                 render_silhouette(kp, P1, image_shape=(400, 480))]
        init = RatPose(root_pos=np.array([0.0, 0.0, 60.0]),
                       root_rot=np.array([0.0, 0.0, 0.4]), scale=1.0)
        iou0 = multiview_iou(init, [P0, P1], masks)
        fitted, iou = fit_pose(masks, [P0, P1], init, downscale=2, maxiter=40)
        assert iou > iou0            # fitting improved the match
        assert iou > 0.6             # and lands on a good overlap

    def test_multistart_returns_best(self):
        P0 = _P((-400.0, -600.0, 500.0))
        P1 = _P((400.0, -600.0, 500.0))
        target = RatPose(root_pos=np.array([0.0, 0.0, 60.0]),
                         root_rot=np.array([0.0, 0.0, 1.2]), scale=1.0)
        kp = forward_kinematics(target)
        masks = [render_silhouette(kp, P0, image_shape=(400, 480)),
                 render_silhouette(kp, P1, image_shape=(400, 480))]
        pose, iou = fit_pose_multistart(masks, [P0, P1],
                                        root_pos=[0.0, 0.0, 60.0], headings=2,
                                        downscale=3, maxiter=30)
        assert 0.0 <= iou <= 1.0
        assert isinstance(pose, RatPose)


class TestStagedFit:

    def test_clamp_joint_respects_limits(self):
        # ElbowL ry hinge is bounded well above -3 rad; clamping pulls it in.
        a = _clamp_joint("ElbowL", (0.0, -3.0, 0.0))
        (lo, hi) = JOINT_LIMITS["ElbowL"][1]
        assert lo <= a[1] <= hi
        assert a == _clamp_joint("ElbowL", a)          # idempotent

    def test_tucked_init_is_valid(self):
        assert check_joint_angles(curled_pose([0.0, 0.0, 60.0]).joint_angles)
        assert check_joint_angles(dict(TUCKED_ANGLES))

    def test_staged_runs_and_stays_valid(self):
        P0 = _P((-400.0, -600.0, 500.0))
        P1 = _P((400.0, -600.0, 500.0))
        target = curled_pose([0.0, 0.0, 60.0], root_rot=[0.0, 0.0, 0.6])
        kp = forward_kinematics(target)
        masks = [render_silhouette(kp, P0, image_shape=(400, 480)),
                 render_silhouette(kp, P1, image_shape=(400, 480))]
        pose, iou = fit_pose_staged(masks, [P0, P1], root_pos=[0.0, 0.0, 60.0],
                                    headings=2, stages=(("SpineF", "SpineL"),),
                                    downscale=3, maxiter=25)
        assert 0.0 <= iou <= 1.0
        assert check_joint_angles(pose.joint_angles)   # clamping kept it valid
