"""
tests/test_refraction.py
=========================
Unit tests for rpimocap.reconstruction.refraction.

The headline test is end-to-end: take a known 3D point inside a box
arena, simulate the optical path (Snell-refract the ray from each camera
through the appropriate wall, project back to a pixel through that same
refraction model), and verify that ``triangulate_refracted`` recovers
the original 3D point to sub-millimetre accuracy — while straight-ray
DLT on the same observations is biased by several mm.

No external data, no hardware.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from rpimocap.reconstruction.refraction import (
    ArenaRefractionModel,
    RefractivePlane,
    build_box_arena,
    closest_point_two_lines,
    load_arena_config,
    pixel_to_world_ray,
    refract_through_wall,
    save_arena_config,
    snell_refract,
    triangulate_refracted,
)
from rpimocap.reconstruction.triangulate import triangulate_dlt


# --------------------------------------------------------------------------- #
#  Helpers                                                                     #
# --------------------------------------------------------------------------- #

ARENA_XMIN, ARENA_XMAX = -140.0, 140.0
ARENA_YMIN, ARENA_YMAX = -215.0, 215.0
ARENA_ZMIN, ARENA_ZMAX = 0.0, 388.0


def _intrinsics():
    f, cx, cy = 900.0, 640.0, 360.0
    return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])


def _look_at(C, target=np.zeros(3), up=np.array([0.0, 0.0, 1.0])):
    """Build (R_w2c, T_w2c) for a camera at ``C`` looking at ``target``.

    Convention: x_cam = R_w2c · x_world + T_w2c.  Camera frame is
    OpenCV's: +z forward (into scene), +x right, +y down.
    """
    z = (np.asarray(target, float) - np.asarray(C, float))
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    if np.linalg.norm(x) < 1e-6:
        x = np.cross(z, np.array([0.0, 1.0, 0.0]))
    x /= np.linalg.norm(x)
    y = np.cross(z, x); y /= np.linalg.norm(y)
    R = np.stack([x, y, z])
    T = -R @ np.asarray(C, float)
    return R, T


def _stereo_outside_arena():
    """Realistic stereo pair: both cameras on the -y side of the arena.

    The cameras share the -y wall as their refractive surface and have a
    600 mm horizontal baseline, giving the rays a ~46° convergence angle
    at the arena centre.  Antiparallel-ray pathologies (both cameras
    facing each other across the arena) are avoided.
    """
    K = _intrinsics()
    target = np.array([0.0, 0.0, 200.0])

    C0 = np.array([-300.0, -700.0, 200.0])
    R0, T0 = _look_at(C0, target=target)

    C1 = np.array([+300.0, -700.0, 200.0])
    R1, T1 = _look_at(C1, target=target)

    return (K, R0, T0, C0), (K, R1, T1, C1)


def _project_pinhole(K, R_w2c, T_w2c, X):
    Xc = R_w2c @ X + T_w2c
    p = K @ Xc
    return p[:2] / p[2]


def _project_through_arena(K, R_w2c, T_w2c, C, X, arena):
    """Forward-project a 3D arena point to a pixel including refraction.

    Solves for the pixel ``uv`` whose camera ray, after refraction through
    whichever arena wall it crosses, passes through ``X``.  The residual
    is the 3-vector from ``X`` to the foot of its perpendicular onto the
    refracted ray (zero when the ray passes exactly through X).
    """
    from scipy.optimize import least_squares

    def residual(uv):
        O, d = pixel_to_world_ray(K, R_w2c, T_w2c, tuple(uv))
        plane, _ = arena.find_traversed_plane(O, d)
        if plane is None:
            # Ray missed every wall — penalise heavily
            return np.full(3, 1e6)
        B, d_in = refract_through_wall(O, d, plane)
        # Perpendicular foot from X onto line (B, d_in)
        v = X - B
        foot = B + (v @ d_in) * d_in
        return foot - X    # 3-vector; least_squares minimises ‖·‖²

    # Straight-ray pinhole gives a good starting point
    Xc = R_w2c @ X + T_w2c
    uv0 = (K @ Xc)[:2] / (K @ Xc)[2]
    res = least_squares(residual, uv0, xtol=1e-12, ftol=1e-12, max_nfev=400)
    if res.cost > 1e-10:
        raise RuntimeError(f"Forward projection failed (cost={res.cost:.3e})")
    return res.x


# =========================================================================== #
#  Snell's law                                                                 #
# =========================================================================== #

class TestSnellRefract:

    def test_no_refraction_when_indices_equal(self):
        d = np.array([0.3, -0.4, 0.86])
        d /= np.linalg.norm(d)
        n = np.array([0.0, 0.0, -1.0])  # surface normal pointing into incoming medium
        out = snell_refract(d, n, 1.0, 1.0)
        np.testing.assert_allclose(out, d, atol=1e-12)

    def test_normal_incidence_preserves_direction(self):
        # Ray hitting the surface head-on doesn't bend regardless of n
        d = np.array([0.0, 0.0, 1.0])
        n = np.array([0.0, 0.0, -1.0])
        out = snell_refract(d, n, 1.0, 1.49)
        np.testing.assert_allclose(out, d, atol=1e-12)

    def test_bends_toward_normal_into_denser(self):
        # Going air → glass: refracted angle < incident angle
        d = np.array([np.sin(np.deg2rad(45)), 0.0, np.cos(np.deg2rad(45))])
        n = np.array([0.0, 0.0, -1.0])
        out = snell_refract(d, n, 1.0, 1.49)
        # Out angle to +z axis: smaller than 45°
        cos_in = d[2]
        cos_out = out[2]
        assert cos_out > cos_in
        # Verify Snell exactly
        sin_in = np.sqrt(1 - cos_in**2)
        sin_out = np.sqrt(1 - cos_out**2)
        assert sin_in / sin_out == pytest.approx(1.49 / 1.0, rel=1e-9)

    def test_total_internal_reflection_raises(self):
        # Glass → air at well past critical angle
        d = np.array([np.sin(np.deg2rad(60)), 0.0, np.cos(np.deg2rad(60))])
        n = np.array([0.0, 0.0, -1.0])
        with pytest.raises(ValueError, match="otal internal"):
            snell_refract(d, n, 1.49, 1.0)


# =========================================================================== #
#  Parallel-slab invariant                                                     #
# =========================================================================== #

class TestRefractThroughWall:

    def test_emerging_direction_equals_incoming(self):
        plane = RefractivePlane(
            point=np.array([0.0, 100.0, 0.0]),
            normal=np.array([0.0, -1.0, 0.0]),  # outward normal, camera on -y side
            thickness=6.0, n_glass=1.49,
        )
        # Camera at -y, ray pointing +y but tilted in x
        origin = np.array([0.0, -50.0, 0.0])
        d_in = np.array([0.3, 0.95, 0.05])
        d_in /= np.linalg.norm(d_in)
        B, d_out = refract_through_wall(origin, d_in, plane)
        np.testing.assert_allclose(d_out, d_in, atol=1e-10)

    def test_inner_exit_on_inner_face(self):
        # The returned B must lie on the inner face: (B - (point - thickness*normal)) · normal == 0
        plane = RefractivePlane(
            point=np.array([0.0, 100.0, 0.0]),
            normal=np.array([0.0, -1.0, 0.0]),
            thickness=6.0, n_glass=1.49,
        )
        origin = np.array([0.0, -50.0, 0.0])
        d = np.array([0.2, 0.97, 0.0])
        d /= np.linalg.norm(d)
        B, _ = refract_through_wall(origin, d, plane)
        inner_face_pt = plane.point - plane.thickness * plane.normal
        assert (B - inner_face_pt) @ plane.normal == pytest.approx(0.0, abs=1e-9)

    def test_no_displacement_at_normal_incidence(self):
        plane = RefractivePlane(
            point=np.array([0.0, 100.0, 0.0]),
            normal=np.array([0.0, -1.0, 0.0]),
            thickness=6.0, n_glass=1.49,
        )
        origin = np.array([0.0, -50.0, 0.0])
        d = np.array([0.0, 1.0, 0.0])  # straight in
        B, d_out = refract_through_wall(origin, d, plane)
        # Inner face lies at point - thickness*normal = (0, 100 - 6*(-1), 0) = (0, 106, 0)
        assert B[0] == pytest.approx(0.0, abs=1e-9)
        assert B[1] == pytest.approx(106.0, abs=1e-9)
        np.testing.assert_allclose(d_out, d, atol=1e-12)

    def test_lateral_offset_known_geometry(self):
        # 45° incidence into 6mm slab of PMMA → known closed-form perpendicular
        # offset between the unrefracted line and the refracted emerging line:
        #     Δ = thickness · sin(θ₁ - θ₂) / cos(θ₂)
        theta_i = np.deg2rad(45.0)
        n_glass = 1.49
        theta_t = np.arcsin(np.sin(theta_i) / n_glass)
        expected_offset = 6.0 * np.sin(theta_i - theta_t) / np.cos(theta_t)

        plane = RefractivePlane(
            point=np.array([0.0, 100.0, 0.0]),
            normal=np.array([0.0, -1.0, 0.0]),
            thickness=6.0, n_glass=n_glass,
        )
        origin = np.array([0.0, 0.0, 0.0])
        d_in = np.array([np.sin(theta_i), np.cos(theta_i), 0.0])
        B, d_out = refract_through_wall(origin, d_in, plane)

        # Perpendicular distance from B to the unrefracted line through origin
        # along d_in:  ‖B - (B·d_in) d_in‖
        proj = (B @ d_in) * d_in
        perp = B - proj
        observed_offset = float(np.linalg.norm(perp))
        assert observed_offset == pytest.approx(expected_offset, rel=1e-6)


# =========================================================================== #
#  Geometry helpers                                                            #
# =========================================================================== #

class TestClosestPointTwoLines:

    def test_intersecting_lines(self):
        # Two lines meeting at (0, 0, 50)
        O0 = np.array([-10.0, 0.0, 0.0])
        d0 = np.array([10.0, 0.0, 50.0]); d0 /= np.linalg.norm(d0)
        O1 = np.array([10.0, 0.0, 0.0])
        d1 = np.array([-10.0, 0.0, 50.0]); d1 /= np.linalg.norm(d1)
        mid, gap = closest_point_two_lines(O0, d0, O1, d1)
        np.testing.assert_allclose(mid, [0.0, 0.0, 50.0], atol=1e-10)
        assert gap == pytest.approx(0.0, abs=1e-10)

    def test_skew_lines_gap(self):
        # Two parallel-but-offset-in-y lines along x-direction
        O0 = np.array([0.0, 0.0, 0.0]); d0 = np.array([1.0, 0.0, 0.0])
        O1 = np.array([0.0, 10.0, 0.0]); d1 = np.array([0.0, 0.0, 1.0])
        mid, gap = closest_point_two_lines(O0, d0, O1, d1)
        # Closest points: (0,0,0) on line 0 and (0,10,0) on line 1; gap = 10
        np.testing.assert_allclose(mid, [0.0, 5.0, 0.0], atol=1e-10)
        assert gap == pytest.approx(10.0, abs=1e-10)


class TestBoxArena:

    def test_default_four_walls(self):
        arena = build_box_arena(-140, 140, -215, 215, 0, 388)
        assert len(arena.planes) == 4
        labels = sorted(p.label for p in arena.planes)
        assert labels == ["+x", "+y", "-x", "-y"]

    def test_with_ceiling_and_floor(self):
        arena = build_box_arena(-140, 140, -215, 215, 0, 388,
                                 include_ceiling=True, include_floor=True)
        assert len(arena.planes) == 6

    def test_normals_point_outward(self):
        arena = build_box_arena(-140, 140, -215, 215, 0, 388)
        for p in arena.planes:
            # A point just inside the arena from the wall should be on the
            # negative side of the outward normal
            interior = np.array([0.0, 0.0, 194.0])
            assert (interior - p.point) @ p.normal < 0

    def test_find_traversed_plane_picks_first_hit(self):
        arena = build_box_arena(-140, 140, -215, 215, 0, 388)
        # Camera outside +y wall, ray heading toward -y interior
        O = np.array([0.0, 500.0, 194.0])
        d = np.array([0.0, -1.0, 0.0])
        plane, t = arena.find_traversed_plane(O, d)
        assert plane is not None
        assert plane.label == "+y"
        # outer face at y=215, so t = 500 - 215 = 285
        assert t == pytest.approx(285.0, abs=1e-9)

    def test_ray_missing_wall_extent_rejected(self):
        arena = build_box_arena(-10, 10, -10, 10, 0, 10)
        # Aim at +y wall but far above its z extent — should miss
        O = np.array([0.0, 500.0, 200.0])
        d = np.array([0.0, -1.0, 0.0])  # would hit +y plane but z=200 > z_max=10
        plane, _ = arena.find_traversed_plane(O, d)
        assert plane is None


# =========================================================================== #
#  Pixel ↔ world ray                                                           #
# =========================================================================== #

class TestPixelToWorldRay:

    def test_principal_axis(self):
        K = _intrinsics()
        # camera at origin, world frame = camera frame
        R = np.eye(3); T = np.zeros(3)
        # Principal point pixel should give the +z axis ray
        C, d = pixel_to_world_ray(K, R, T, (K[0, 2], K[1, 2]))
        np.testing.assert_allclose(C, [0, 0, 0], atol=1e-10)
        np.testing.assert_allclose(d, [0, 0, 1], atol=1e-10)

    def test_camera_center_recovered(self):
        K, R, T, C_true = _stereo_outside_arena()[0]
        C, _ = pixel_to_world_ray(K, R, T, (320.0, 240.0))
        np.testing.assert_allclose(C, C_true, atol=1e-9)

    def test_round_trip_with_projection(self):
        # Project a point, then verify the world-ray passes through it
        K, R, T, C = _stereo_outside_arena()[0]
        X = np.array([0.0, 0.0, 200.0])
        uv = _project_pinhole(K, R, T, X)
        O, d = pixel_to_world_ray(K, R, T, tuple(uv))
        # X - O should be parallel to d
        u = X - O; u /= np.linalg.norm(u)
        np.testing.assert_allclose(u, d, atol=1e-9)


# =========================================================================== #
#  End-to-end refractive triangulation                                         #
# =========================================================================== #

class TestRefractiveTriangulation:

    def _setup(self):
        arena = build_box_arena(ARENA_XMIN, ARENA_XMAX,
                                 ARENA_YMIN, ARENA_YMAX,
                                 ARENA_ZMIN, ARENA_ZMAX,
                                 thickness=6.0, n_glass=1.49)
        cam0, cam1 = _stereo_outside_arena()
        return arena, cam0, cam1

    def test_recovers_known_interior_point(self):
        arena, (K0, R0, T0, C0), (K1, R1, T1, C1) = self._setup()
        X_true = np.array([20.0, -10.0, 200.0])

        # Forward project through refraction (forward model with same physics)
        uv0 = _project_through_arena(K0, R0, T0, C0, X_true, arena)
        uv1 = _project_through_arena(K1, R1, T1, C1, X_true, arena)

        # Inverse: refractive triangulation
        O0, d0 = pixel_to_world_ray(K0, R0, T0, tuple(uv0))
        O1, d1 = pixel_to_world_ray(K1, R1, T1, tuple(uv1))
        X_rec, gap, n_iter = triangulate_refracted(O0, d0, O1, d1, arena)

        err = np.linalg.norm(X_rec - X_true)
        assert err < 0.05, f"Refractive recovery error {err:.4f} mm exceeds 0.05 mm tolerance"
        assert gap < 0.05, f"Line gap {gap:.4f} mm at convergence too large"
        assert n_iter <= 8

    def test_straight_dlt_is_biased_without_correction(self):
        """The whole point of Phase 2: straight-ray DLT must be visibly wrong
        on refractively-projected observations, otherwise we built nothing
        useful."""
        arena, (K0, R0, T0, C0), (K1, R1, T1, C1) = self._setup()
        X_true = np.array([20.0, -10.0, 200.0])

        uv0 = _project_through_arena(K0, R0, T0, C0, X_true, arena)
        uv1 = _project_through_arena(K1, R1, T1, C1, X_true, arena)

        # Straight DLT with the same observations
        P0 = K0 @ np.hstack([R0, T0.reshape(3, 1)])
        P1 = K1 @ np.hstack([R1, T1.reshape(3, 1)])
        Xh = triangulate_dlt(P0, P1, tuple(uv0), tuple(uv1))
        bias = float(np.linalg.norm(Xh[:3] - X_true))
        # Two-wall traversal at moderate incidence on 6 mm of PMMA
        # contributes at least a few mm of bias.
        assert bias > 1.0, (f"Straight DLT bias {bias:.3f} mm too small — "
                            "test geometry may not stress refraction")

    def test_multiple_interior_points(self):
        arena, (K0, R0, T0, C0), (K1, R1, T1, C1) = self._setup()
        rng = np.random.default_rng(0)
        # Sample 10 points well inside the arena (avoid edges)
        pts = np.column_stack([
            rng.uniform(ARENA_XMIN + 20, ARENA_XMAX - 20, 10),
            rng.uniform(ARENA_YMIN + 30, ARENA_YMAX - 30, 10),
            rng.uniform(ARENA_ZMIN + 30, ARENA_ZMAX - 30, 10),
        ])
        for X_true in pts:
            uv0 = _project_through_arena(K0, R0, T0, C0, X_true, arena)
            uv1 = _project_through_arena(K1, R1, T1, C1, X_true, arena)
            O0, d0 = pixel_to_world_ray(K0, R0, T0, tuple(uv0))
            O1, d1 = pixel_to_world_ray(K1, R1, T1, tuple(uv1))
            X_rec, _, _ = triangulate_refracted(O0, d0, O1, d1, arena)
            err = np.linalg.norm(X_rec - X_true)
            assert err < 0.1, (f"Refractive recovery error {err:.3f} mm for "
                                f"X={X_true} exceeds tolerance")

    def test_empty_arena_falls_back_to_dlt(self):
        # With no walls, the result should match straight-ray closest-of-lines
        arena = ArenaRefractionModel(planes=[])
        (K0, R0, T0, C0), (K1, R1, T1, C1) = _stereo_outside_arena()
        X_true = np.array([20.0, -10.0, 200.0])
        uv0 = _project_pinhole(K0, R0, T0, X_true)
        uv1 = _project_pinhole(K1, R1, T1, X_true)
        O0, d0 = pixel_to_world_ray(K0, R0, T0, tuple(uv0))
        O1, d1 = pixel_to_world_ray(K1, R1, T1, tuple(uv1))
        X_rec, gap, _ = triangulate_refracted(O0, d0, O1, d1, arena)
        np.testing.assert_allclose(X_rec, X_true, atol=1e-6)
        assert gap < 1e-6


# =========================================================================== #
#  Config I/O                                                                  #
# =========================================================================== #

class TestConfigIO:

    def test_round_trip(self, tmp_path):
        arena = build_box_arena(-140, 140, -215, 215, 0, 388,
                                 thickness=6.0, n_glass=1.49,
                                 include_ceiling=True)
        cfg_path = tmp_path / "arena.json"
        save_arena_config(cfg_path, arena)
        loaded = load_arena_config(cfg_path)
        assert len(loaded.planes) == len(arena.planes)
        for a, b in zip(arena.planes, loaded.planes):
            np.testing.assert_allclose(a.point, b.point)
            np.testing.assert_allclose(a.normal, b.normal)
            assert a.thickness == b.thickness
            assert a.n_glass == b.n_glass
            assert a.label == b.label

    def test_json_is_human_readable(self, tmp_path):
        arena = build_box_arena(-10, 10, -10, 10, 0, 20)
        cfg_path = tmp_path / "arena.json"
        save_arena_config(cfg_path, arena)
        data = json.loads(cfg_path.read_text())
        assert "planes" in data
        assert len(data["planes"]) == 4
        assert all("point" in p and "normal" in p for p in data["planes"])


# =========================================================================== #
#  triangulate_keypoints integration                                           #
# =========================================================================== #

class TestTriangulateKeypointsRefraction:
    """End-to-end check that the public ``triangulate_keypoints`` entrypoint
    honours the ``arena_model`` kwarg and produces refraction-corrected 3D
    points that match the underlying ``triangulate_refracted`` solver.
    """

    def test_recovers_known_point_via_public_api(self):
        from rpimocap.detection.detectors import Keypoint2D, Pose2DResult
        from rpimocap.reconstruction.triangulate import triangulate_keypoints

        arena = build_box_arena(-140, 140, -215, 215, 0, 388,
                                 thickness=6.0, n_glass=1.49)
        (K0, R0, T0, C0), (K1, R1, T1, C1) = _stereo_outside_arena()
        P0 = K0 @ np.hstack([R0, T0.reshape(3, 1)])
        P1 = K1 @ np.hstack([R1, T1.reshape(3, 1)])

        X_true = np.array([20.0, -10.0, 200.0])
        uv0 = _project_through_arena(K0, R0, T0, C0, X_true, arena)
        uv1 = _project_through_arena(K1, R1, T1, C1, X_true, arena)

        r0 = Pose2DResult(frame_idx=0, detected=True, keypoints=[
            Keypoint2D(name="nose", x=float(uv0[0]), y=float(uv0[1]), confidence=0.9)])
        r1 = Pose2DResult(frame_idx=0, detected=True, keypoints=[
            Keypoint2D(name="nose", x=float(uv1[0]), y=float(uv1[1]), confidence=0.9)])

        # Straight-ray DLT through the public API
        pts_dlt = triangulate_keypoints(P0, P1, r0, r1,
                                         min_confidence=0.0,
                                         max_reprojection_px=1e9)
        # Refraction-corrected through the public API
        pts_ref = triangulate_keypoints(P0, P1, r0, r1,
                                         min_confidence=0.0,
                                         max_reprojection_px=1e9,
                                         arena_model=arena,
                                         K0=K0, dist0=None, R0=R0, T0=T0,
                                         K1=K1, dist1=None, R1=R1, T1=T1)

        assert len(pts_dlt) == 1 and len(pts_ref) == 1
        err_dlt = float(np.linalg.norm(pts_dlt[0].xyz - X_true))
        err_ref = float(np.linalg.norm(pts_ref[0].xyz - X_true))
        # Refraction-corrected should match X_true to sub-mm; straight DLT
        # should be biased by at least an order of magnitude more.
        assert err_ref < 0.1, f"refractive error {err_ref:.4f} mm too large"
        assert err_dlt > 10 * err_ref, (
            f"DLT error {err_dlt:.4f} mm should be much worse than "
            f"refractive {err_ref:.4f} mm")

    def test_falls_back_to_dlt_without_required_params(self):
        """If arena_model is given but K/R/T are missing, the function must
        still produce a result (falling back silently to straight-ray DLT).
        """
        from rpimocap.detection.detectors import Keypoint2D, Pose2DResult
        from rpimocap.reconstruction.triangulate import triangulate_keypoints

        arena = build_box_arena(-140, 140, -215, 215, 0, 388)
        (K0, R0, T0, _), (K1, R1, T1, _) = _stereo_outside_arena()
        P0 = K0 @ np.hstack([R0, T0.reshape(3, 1)])
        P1 = K1 @ np.hstack([R1, T1.reshape(3, 1)])

        r0 = Pose2DResult(frame_idx=0, detected=True, keypoints=[
            Keypoint2D(name="nose", x=640.0, y=360.0, confidence=0.9)])
        r1 = Pose2DResult(frame_idx=0, detected=True, keypoints=[
            Keypoint2D(name="nose", x=640.0, y=360.0, confidence=0.9)])

        pts = triangulate_keypoints(P0, P1, r0, r1,
                                     min_confidence=0.0,
                                     max_reprojection_px=1e9,
                                     arena_model=arena)   # K/R/T missing
        # Should not raise; result is identical to straight-DLT path
        assert len(pts) == 1
