"""Tests for the rat skeletal model + synthetic pose generation."""
import numpy as np
import pytest

from rpimocap.model import rat_skeleton as rs
from rpimocap.reconstruction.triangulate import triangulate_dlt


# ────────────────────────────────────────────────────────────────────
#  Helpers
# ────────────────────────────────────────────────────────────────────


def _make_P(cam_pos, look_at, f=1500, cx=1014, cy=540):
    """A simple pinhole projection matrix looking at a target."""
    cam_pos = np.asarray(cam_pos, float)
    look_at = np.asarray(look_at, float)
    fwd = look_at - cam_pos
    fwd /= np.linalg.norm(fwd)
    up = np.array([0, 0, 1.0])
    right = np.cross(fwd, up)
    right /= np.linalg.norm(right)
    up2 = np.cross(right, fwd)
    R = np.vstack([right, -up2, fwd])
    t = -R @ cam_pos
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
    return K @ np.hstack([R, t.reshape(3, 1)])


P0_TEST = _make_P([-300, -400, 700], [0, 0, 194])
P1_TEST = _make_P([300, -400, 700], [0, 0, 194])


# ────────────────────────────────────────────────────────────────────
#  Skeleton definition
# ────────────────────────────────────────────────────────────────────


class TestSkeletonDefinition:

    def test_23_joints(self):
        assert len(rs.RAT23_JOINTS) == 23
        assert len(set(rs.RAT23_JOINTS)) == 23      # unique

    def test_regions_cover_all_joints(self):
        flat = [j for js in rs.RAT23_REGIONS.values() for j in js]
        assert set(flat) == set(rs.RAT23_JOINTS)
        assert len(flat) == 23

    def test_tree_single_root(self):
        roots = [n for n, p in rs.RAT23_PARENT.items() if p is None]
        assert roots == ["SpineM"]

    def test_tree_no_cycles_all_reachable(self):
        order = rs._topo_order()
        # _topo_order covers the EXTENDED skeleton (rat23 + the 5 tail
        # segments); the rat23 keypoint contract is asserted separately in
        # test_23_joints and by forward_kinematics returning (23, 3).
        assert len(order) == len(rs.RAT_JOINTS_EXT)
        assert set(order) == set(rs.RAT_JOINTS_EXT)
        assert set(rs.RAT23_JOINTS) <= set(order)
        pos = {n: i for i, n in enumerate(order)}
        for c, p in rs.RAT_PARENT_EXT.items():
            if p is not None:
                assert pos[p] < pos[c]              # parent first

    def test_bones_derived_from_tree(self):
        # one bone per non-root joint
        assert len(rs.RAT23_BONES) == 22
        for (p, c) in rs.RAT23_BONES:
            assert rs.RAT23_PARENT[c] == p

    def test_canonical_lengths_positive(self):
        for bone, L in rs.CANONICAL_BONE_LENGTHS.items():
            assert L > 0


# ────────────────────────────────────────────────────────────────────
#  Forward kinematics
# ────────────────────────────────────────────────────────────────────


class TestForwardKinematics:

    def test_rest_pose_shape(self):
        kp = rs.forward_kinematics(rs.RatPose())
        assert kp.shape == (23, 3)

    def test_root_at_origin_by_default(self):
        kp = rs.forward_kinematics(rs.RatPose())
        assert np.allclose(kp[rs.RAT23_INDEX["SpineM"]], [0, 0, 0])

    def test_rest_geometry_snout_forward_tail_back(self):
        kp = rs.forward_kinematics(rs.RatPose())
        sm = kp[rs.RAT23_INDEX["SpineM"]]
        assert kp[rs.RAT23_INDEX["Snout"]][0] > sm[0]
        assert kp[rs.RAT23_INDEX["TailBase"]][0] < sm[0]

    def test_root_translation_applied(self):
        pose = rs.RatPose(root_pos=np.array([50, -30, 200]))
        kp = rs.forward_kinematics(pose)
        assert np.allclose(kp[rs.RAT23_INDEX["SpineM"]], [50, -30, 200])

    def test_bone_lengths_preserved_under_articulation(self):
        rng = np.random.RandomState(0)
        for _ in range(20):
            pose = rs.sample_pose(rng, scale=1.0)
            kp = rs.forward_kinematics(pose)
            assert rs.check_bone_lengths(kp, scale=1.0)

    def test_scale_changes_body_size(self):
        small = rs.forward_kinematics(rs.RatPose(scale=0.8))
        big = rs.forward_kinematics(rs.RatPose(scale=1.2))

        def span(k):
            return np.linalg.norm(k[rs.RAT23_INDEX["Snout"]]
                                  - k[rs.RAT23_INDEX["TailBase"]])
        assert np.isclose(span(big) / span(small), 1.5, rtol=1e-6)


# ────────────────────────────────────────────────────────────────────
#  Sampling + validity
# ────────────────────────────────────────────────────────────────────


class TestSampling:

    def test_sampled_angles_in_limits(self):
        rng = np.random.RandomState(1)
        for _ in range(200):
            a = rs.sample_joint_angles(rng, fraction=1.0)
            assert rs.check_joint_angles(a)

    def test_fraction_zero_is_rest(self):
        rng = np.random.RandomState(2)
        a = rs.sample_joint_angles(rng, fraction=0.0)
        mx = max(max(abs(v) for v in t) for t in a.values())
        assert mx == 0.0

    def test_fraction_half_in_limits(self):
        rng = np.random.RandomState(3)
        for _ in range(50):
            assert rs.check_joint_angles(
                rs.sample_joint_angles(rng, fraction=0.5))

    def test_sample_pose_angles_valid(self):
        rng = np.random.RandomState(4)
        for _ in range(100):
            pose = rs.sample_pose(rng, scale=1.0)
            assert rs.check_joint_angles(pose.joint_angles)

    def test_sample_pose_root_in_arena(self):
        rng = np.random.RandomState(5)
        bounds = (-140, 140, -215, 215, 0, 388)
        for _ in range(50):
            pose = rs.sample_pose(rng, scale=1.0, arena_bounds=bounds)
            x, y, z = pose.root_pos
            assert -140 <= x <= 140
            assert -215 <= y <= 215
            assert 0 <= z <= 388


class TestValidity:

    def test_rejects_out_of_limit_angle(self):
        bad = rs.RatPose(
            joint_angles={"KneeL": (0.0, np.radians(200), 0.0)})
        assert not rs.check_joint_angles(bad.joint_angles)

    def test_accepts_in_limit_angle(self):
        ok = rs.RatPose(
            joint_angles={"KneeL": (0.0, np.radians(45), 0.0)})
        assert rs.check_joint_angles(ok.joint_angles)

    def test_arena_containment(self):
        # pose at origin, mild → inside
        rng = np.random.RandomState(0)
        pose = rs.RatPose(
            root_pos=np.array([0, 0, 194.0]),
            joint_angles=rs.sample_joint_angles(rng, fraction=0.3))
        kp = rs.forward_kinematics(pose)
        assert rs.check_arena_containment(kp)
        # pose pushed far outside → fails
        pose.root_pos = np.array([1000.0, 0, 194.0])
        kp = rs.forward_kinematics(pose)
        assert not rs.check_arena_containment(kp)

    def test_is_valid_combines_checks(self):
        rng = np.random.RandomState(6)
        pose = rs.RatPose(
            root_pos=np.array([0, 0, 194.0]),
            joint_angles=rs.sample_joint_angles(rng, fraction=0.3))
        assert rs.is_valid(pose, require_arena=True)


# ────────────────────────────────────────────────────────────────────
#  Projection + the ground-truth round trip (the whole point)
# ────────────────────────────────────────────────────────────────────


class TestProjectionRoundTrip:

    def test_project_shape(self):
        kp = rs.forward_kinematics(rs.RatPose(
            root_pos=np.array([0, 0, 194.0])))
        px = rs.project_pose(kp, P0_TEST)
        assert px.shape == (23, 2)

    def test_project_triangulate_recovers_exactly(self):
        """A known synthetic 3D pose, projected through two cameras and
        triangulated back, must recover within numerical error — this
        is the ground-truth validation the generator exists for."""
        rng = np.random.RandomState(42)
        max_err = 0.0
        for _ in range(20):
            pose = rs.sample_pose(rng, scale=1.0)
            kp3d = rs.forward_kinematics(pose)
            px0 = rs.project_pose(kp3d, P0_TEST)
            px1 = rs.project_pose(kp3d, P1_TEST)
            for i in range(23):
                X = triangulate_dlt(P0_TEST, P1_TEST,
                                    tuple(px0[i]), tuple(px1[i]))
                err = np.linalg.norm(X[:3] - kp3d[i])
                max_err = max(max_err, err)
        assert max_err < 1e-3, f"round-trip error {max_err} mm"

    def test_observable_subset(self):
        idx = rs.visible_subset(rs.OBSERVABLE_KEYPOINTS)
        assert len(idx) == len(rs.OBSERVABLE_KEYPOINTS)
        assert all(0 <= i < 23 for i in idx)
        # round-trips through names
        names = [rs.RAT23_JOINTS[i] for i in idx]
        assert names == rs.OBSERVABLE_KEYPOINTS


class TestEulerRotation:

    def test_identity_at_zero(self):
        assert np.allclose(rs.euler_to_R(0, 0, 0), np.eye(3))

    def test_rotation_is_orthonormal(self):
        R = rs.euler_to_R(0.3, -0.5, 1.1)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
        assert np.isclose(np.linalg.det(R), 1.0)


class TestTailExtension:
    """The tail exists to break the head/tail degeneracy in the RENDER.

    Without it a 180 deg flip costs ~1.3% of the appearance energy range
    (i.e. nothing); with it, ~37%.
    """

    def test_keypoint_contract_unchanged(self):
        """rat23 is a trained-model interface (synthetic_dataset emits
        (23,n,n,n) volumes) — the tail must NOT enter it."""
        assert len(rs.RAT23_JOINTS) == 23
        assert not (set(rs.TAIL_JOINTS) & set(rs.RAT23_JOINTS))
        assert rs.forward_kinematics(rs.RatPose()).shape == (23, 3)
        flat = [j for js in rs.RAT23_REGIONS.values() for j in js]
        assert set(flat) == set(rs.RAT23_JOINTS)

    def test_tail_chain_is_serial(self):
        assert rs.TAIL_PARENT["Tail1"] == "TailBase"
        for i in range(2, 6):
            assert rs.TAIL_PARENT[f"Tail{i}"] == f"Tail{i-1}"

    def test_tail_length_matches_body(self):
        """A rat's tail is about as long as its body."""
        kp = rs.forward_kinematics(rs.RatPose())
        tail = rs.forward_kinematics_tail(rs.RatPose())
        base = kp[rs.RAT23_INDEX["TailBase"]]
        snout = kp[rs.RAT23_INDEX["Snout"]]
        tail_len = np.linalg.norm(tail[-1] - base)
        body_len = np.linalg.norm(snout - base)
        assert tail_len == pytest.approx(140.0, abs=5.0)
        assert 0.75 < tail_len / body_len < 1.3

    def test_tail_scales_with_body(self):
        a = rs.forward_kinematics_tail(rs.RatPose(scale=1.0))
        b = rs.forward_kinematics_tail(rs.RatPose(scale=2.0))
        kp_a = rs.forward_kinematics(rs.RatPose(scale=1.0))
        kp_b = rs.forward_kinematics(rs.RatPose(scale=2.0))
        base_a = kp_a[rs.RAT23_INDEX["TailBase"]]
        base_b = kp_b[rs.RAT23_INDEX["TailBase"]]
        assert (np.linalg.norm(b[-1] - base_b)
                == pytest.approx(2.0 * np.linalg.norm(a[-1] - base_a), rel=1e-6))

    def test_tail_arc_totals_and_shape(self):
        """The 2-parameter arc must reproduce the requested TOTAL deflection —
        the ball-and-chain correlation, not 15 free angles."""
        a = rs.tail_arc(sweep=np.radians(120.0), lift=np.radians(-30.0))
        assert set(a) == set(rs.TAIL_JOINTS)
        assert sum(v[2] for v in a.values()) == pytest.approx(np.radians(120.0))
        assert sum(v[1] for v in a.values()) == pytest.approx(np.radians(-30.0))
        assert all(v[0] == 0.0 for v in a.values())      # no axial roll

    def test_tail_arc_taper_moves_curvature(self):
        even = rs.tail_arc(sweep=1.0, taper=1.0)
        whip = rs.tail_arc(sweep=1.0, taper=3.0)
        assert whip["Tail5"][2] > even["Tail5"][2]       # tip bends more
        assert whip["Tail1"][2] < even["Tail1"][2]       # base bends less

    def test_tail_within_its_limits(self):
        """A plausible arc must not exceed the per-segment limits."""
        a = rs.tail_arc(sweep=np.radians(100.0), lift=np.radians(-40.0))
        for name, (rx, ry, rz) in a.items():
            (lox, hix), (loy, hiy), (loz, hiz) = rs.TAIL_LIMITS[name]
            assert lox <= rx <= hix and loy <= ry <= hiy and loz <= rz <= hiz

    def test_mesh_with_tail_is_longer(self):
        from rpimocap.model.mesh_model import build_rat_mesh
        body = build_rat_mesh(voxel=5.0, with_tail=False)
        full = build_rat_mesh(voxel=5.0, with_tail=True)
        assert np.ptp(full.verts_rest[:, 0]) > 1.7 * np.ptp(body.verts_rest[:, 0])
        assert len(full.bone_parents) == len(body.bone_parents) + 5
