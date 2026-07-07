"""Tests for the physics-based pose prior (rpimocap.model.physics)."""
import numpy as np

from rpimocap.model.physics import (HIND_FEET, body_up, ground_contact_penalty,
                                     heading, penetration_penalty,
                                     physical_penalty, settle_pose,
                                     upright_penalty)
from rpimocap.model.rat_skeleton import (RAT23_INDEX, RatPose,
                                         forward_kinematics)


class TestPhysics:

    def test_upright_penalty(self):
        assert upright_penalty(RatPose()) < 1e-6              # rest is upright
        rolled = RatPose(root_rot=np.array([np.pi / 2, 0.0, 0.0]))  # on its side
        assert upright_penalty(rolled) > 0.9

    def test_heading_and_body_up(self):
        p = RatPose(root_rot=np.array([0.0, 0.0, 1.2]))
        assert abs(heading(p) - 1.2) < 1e-6
        assert body_up(RatPose())[2] > 0.99                  # +z up at rest

    def test_ground_contact_and_penetration(self):
        high = RatPose(root_pos=np.array([0.0, 0.0, 200.0]))
        assert ground_contact_penalty(high) > 0.0            # feet above floor
        assert penetration_penalty(high) == 0.0              # nothing below
        low = RatPose(root_pos=np.array([0.0, 0.0, -50.0]))
        assert penetration_penalty(low) > 0.0                # sunk below floor

    def test_settle_rights_and_grounds(self):
        bad = RatPose(root_pos=np.array([10.0, 20.0, 150.0]),
                      root_rot=np.array([1.0, 0.3, 0.8]))     # rolled + floating
        s = settle_pose(bad)
        assert upright_penalty(s) < 1e-6                      # righted
        kp = forward_kinematics(s)
        low = min(kp[RAT23_INDEX[f], 2] for f in HIND_FEET)
        assert abs(low) < 1e-6                                # hind foot on floor
        assert abs(heading(s) - heading(bad)) < 1e-6         # heading preserved

    def test_physical_penalty_prefers_grounded(self):
        bad = RatPose(root_pos=np.array([0.0, 0.0, 150.0]),
                      root_rot=np.array([1.0, 0.0, 0.5]))
        assert physical_penalty(settle_pose(bad)) < physical_penalty(bad)
