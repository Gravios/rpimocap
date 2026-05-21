"""
rpimocap.reconstruction.refraction
===================================
Planar refraction correction for triangulation through transparent arena walls.

Behavioural-neuroscience arenas typically house the subject inside a
transparent box made of PMMA (poly(methyl methacrylate), refractive index
n ≈ 1.49). Camera rays must traverse the acrylic before reaching the
subject, and at the air↔acrylic boundaries the rays bend according to
Snell's law. Ignoring this systematically biases triangulated positions
by 5–15 mm — comparable to the size of the subject — and is the main
remaining source of geometric error in the rpimocap pipeline once
calibration is sound.

This module models each wall as a parallel-faced slab and provides a
refractive triangulation routine that replaces straight-ray DLT when an
arena model is supplied.

Physics summary
---------------
A ray entering a slab with parallel faces refracts at the first surface
(angle θ₁ → θ₂ via n₁ sin θ₁ = n₂ sin θ₂), travels a distance
``thickness / cos θ₂`` through the slab, and refracts back at the second
surface. By the reversibility of Snell's law on parallel interfaces, the
emerging direction equals the incoming direction. The ray inside the arena
is therefore **parallel** to the original camera ray but laterally
**displaced**. The shift, computed in closed form here, captures the
geometric effect of the wall without needing to model the slab thickness
explicitly during triangulation.

The principle is applied independently to each camera; the two refracted
rays are then intersected to recover the 3D point. Because the choice of
which wall a ray crosses depends on the (unknown) target point, the
solver iterates from a straight-ray DLT initialisation. In practice
convergence is reached in 2–4 iterations.

Calibration is **not** modified. Arena corners used for alignment lie on
the outer face of the wall, so calibration rays do not traverse acrylic
and remain valid. Refraction is applied at triangulation time only.

Module API
----------
``RefractivePlane``       a single parallel-faced wall
``ArenaRefractionModel``  a collection of walls + entrance/exit logic
``build_box_arena``       convenience constructor for axis-aligned boxes
``snell_refract``         vector form of Snell's law
``refract_through_wall``  exit point and direction of a ray through a slab
``triangulate_refracted`` iterative refractive triangulation (one point)
``pixel_to_world_ray``    backproject a pixel to a world-space ray
``load_arena_config``     read a JSON arena description
``save_arena_config``     write a JSON arena description
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np


# --------------------------------------------------------------------------- #
#  Geometry primitives                                                         #
# --------------------------------------------------------------------------- #

@dataclass
class RefractivePlane:
    """A parallel-faced refractive slab (one wall of the arena).

    The slab is described by a point on its **outer** face (the side
    facing the cameras), an outward unit normal, a thickness, and the
    refractive index of the slab material.  The inner face lies a
    distance ``thickness`` opposite the normal direction.

    Optionally a planar quadrilateral can be specified by ``bounds_uv``
    (axis lengths along two in-plane axes) so that ray–plane hit tests
    reject rays passing outside the physical extent of the wall.

    Attributes
    ----------
    point : (3,) array
        Any point on the outer face.  By convention the centre of the
        wall is used so that bounds are symmetric.
    normal : (3,) array
        Outward unit normal of the outer face — points away from the
        arena interior, toward the camera side.
    thickness : float
        Wall thickness in calibration units (mm).  Default 6.0 mm
        matches common 1/4 inch PMMA sheet.
    n_glass : float
        Refractive index of the wall material.  Default 1.49 (PMMA at
        visible wavelengths).
    n_air : float
        Refractive index of the medium on both sides.  Default 1.0.
    half_extent : (2,) array or None
        Half-lengths of the wall along two in-plane axes (see
        ``in_plane_axes``).  None means the wall is treated as infinite.
        Used only to gate ray-plane intersections; physically the
        refraction model itself is independent of the wall's lateral
        extent.
    in_plane_axes : (2, 3) array or None
        Two orthonormal vectors spanning the plane (rows).  Computed
        automatically from ``normal`` if None.
    label : str
        Human-readable identifier (e.g. ``"+x"``, ``"ceiling"``).
    """
    point: np.ndarray
    normal: np.ndarray
    thickness: float = 6.0
    n_glass: float = 1.49
    n_air: float = 1.0
    half_extent: Optional[np.ndarray] = None
    in_plane_axes: Optional[np.ndarray] = None
    label: str = ""

    def __post_init__(self) -> None:
        self.point = np.asarray(self.point, dtype=np.float64).reshape(3)
        n = np.asarray(self.normal, dtype=np.float64).reshape(3)
        norm = np.linalg.norm(n)
        if norm < 1e-12:
            raise ValueError("RefractivePlane.normal must be non-zero")
        self.normal = n / norm
        if self.thickness < 0:
            raise ValueError("RefractivePlane.thickness must be non-negative")
        if self.n_glass <= 0 or self.n_air <= 0:
            raise ValueError("Refractive indices must be positive")

        if self.half_extent is not None:
            self.half_extent = np.asarray(self.half_extent, dtype=np.float64).reshape(2)
        if self.in_plane_axes is None and self.half_extent is not None:
            # Build any two orthonormal vectors in the plane
            ref = np.array([1.0, 0.0, 0.0])
            if abs(self.normal @ ref) > 0.9:
                ref = np.array([0.0, 1.0, 0.0])
            u = ref - (ref @ self.normal) * self.normal
            u /= np.linalg.norm(u)
            v = np.cross(self.normal, u)
            self.in_plane_axes = np.stack([u, v])
        elif self.in_plane_axes is not None:
            self.in_plane_axes = np.asarray(self.in_plane_axes, dtype=np.float64).reshape(2, 3)

    # -- serialisation ---------------------------------------------------- #

    def to_dict(self) -> dict:
        d: dict = {
            "point": self.point.tolist(),
            "normal": self.normal.tolist(),
            "thickness": float(self.thickness),
            "n_glass": float(self.n_glass),
            "n_air": float(self.n_air),
            "label": self.label,
        }
        if self.half_extent is not None:
            d["half_extent"] = self.half_extent.tolist()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "RefractivePlane":
        return cls(
            point=np.array(d["point"], dtype=np.float64),
            normal=np.array(d["normal"], dtype=np.float64),
            thickness=float(d.get("thickness", 6.0)),
            n_glass=float(d.get("n_glass", 1.49)),
            n_air=float(d.get("n_air", 1.0)),
            half_extent=(np.array(d["half_extent"], dtype=np.float64)
                         if "half_extent" in d else None),
            label=d.get("label", ""),
        )


@dataclass
class ArenaRefractionModel:
    """A collection of refractive walls bounding the arena interior.

    Attributes
    ----------
    planes : list[RefractivePlane]
        The walls.  Order does not matter; ``find_traversed_plane`` picks
        the first one a ray hits.
    """
    planes: list[RefractivePlane] = field(default_factory=list)

    def find_traversed_plane(
        self,
        origin: np.ndarray,
        direction: np.ndarray,
    ) -> tuple[Optional[RefractivePlane], float]:
        """Return the plane the ray crosses first (front-facing hit, t > 0).

        A "front-facing" hit means the ray is heading **into** the wall
        — i.e. the angle between ``direction`` and the outward ``normal``
        is greater than 90°.  This excludes plane intersections from
        behind a wall (which would happen for a camera placed inside the
        arena, an unsupported configuration).

        If ``half_extent`` is set on the plane, hits outside the wall's
        rectangular extent are rejected so a ray that misses the wall
        edge continues to the next plane.

        Returns
        -------
        (plane, t) where ``t`` is the parametric distance to the hit
        (``origin + t * direction`` lies on the outer face), or
        ``(None, inf)`` if no wall is hit.
        """
        best_plane: Optional[RefractivePlane] = None
        best_t = np.inf
        direction = np.asarray(direction, dtype=np.float64).reshape(3)
        origin = np.asarray(origin, dtype=np.float64).reshape(3)
        for pl in self.planes:
            denom = direction @ pl.normal
            # Want ray going *into* the wall: direction·normal < 0
            if denom > -1e-9:
                continue
            t = ((pl.point - origin) @ pl.normal) / denom
            if t <= 1e-9 or t >= best_t:
                continue
            if pl.half_extent is not None and pl.in_plane_axes is not None:
                hit = origin + t * direction
                local = pl.in_plane_axes @ (hit - pl.point)
                if (abs(local[0]) > pl.half_extent[0] + 1e-6 or
                        abs(local[1]) > pl.half_extent[1] + 1e-6):
                    continue
            best_plane = pl
            best_t = float(t)
        return best_plane, best_t

    def to_dict(self) -> dict:
        return {"planes": [p.to_dict() for p in self.planes]}

    @classmethod
    def from_dict(cls, d: dict) -> "ArenaRefractionModel":
        return cls(planes=[RefractivePlane.from_dict(p) for p in d.get("planes", [])])


# --------------------------------------------------------------------------- #
#  Box-arena convenience constructor                                           #
# --------------------------------------------------------------------------- #

def build_box_arena(
    xmin: float, xmax: float,
    ymin: float, ymax: float,
    zmin: float, zmax: float,
    *,
    thickness: float = 6.0,
    n_glass: float = 1.49,
    include_walls: tuple = ("+x", "-x", "+y", "-y"),
    include_ceiling: bool = False,
    include_floor: bool = False,
) -> ArenaRefractionModel:
    """Build a refraction model for an axis-aligned rectangular arena.

    The four vertical walls are included by default.  Ceiling and floor
    are opt-in because in most rodent setups the floor is opaque and
    cameras look down through the open top (or there is no top wall).

    The wall ``point`` is placed at the centre of each face so that
    ``half_extent`` is symmetric.

    Parameters
    ----------
    xmin, xmax, ymin, ymax, zmin, zmax : float
        Arena interior bounds in mm.  Walls are positioned at the
        respective extreme of each axis.
    thickness : float
        Wall thickness, default 6 mm (≈ 1/4 inch PMMA).
    n_glass : float
        Wall refractive index, default 1.49 (PMMA).
    include_walls : tuple of str
        Which vertical walls to include.  Use a subset of
        ``("+x", "-x", "+y", "-y")`` to drop walls (e.g. a 3-sided arena).
    include_ceiling : bool
        Add a top wall at ``z = zmax`` looking down (outward normal +z).
    include_floor : bool
        Add a bottom wall at ``z = zmin`` looking up (outward normal -z).
    """
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    cz = 0.5 * (zmin + zmax)
    hx = 0.5 * (xmax - xmin)
    hy = 0.5 * (ymax - ymin)
    hz = 0.5 * (zmax - zmin)

    planes: list[RefractivePlane] = []
    wallspecs = {
        "+x": ([xmax, cy, cz], [+1, 0, 0], [hy, hz]),
        "-x": ([xmin, cy, cz], [-1, 0, 0], [hy, hz]),
        "+y": ([cx, ymax, cz], [0, +1, 0], [hx, hz]),
        "-y": ([cx, ymin, cz], [0, -1, 0], [hx, hz]),
    }
    for name in include_walls:
        if name not in wallspecs:
            raise ValueError(f"Unknown wall {name!r}; "
                             f"expected one of {list(wallspecs)}")
        pt, n, he = wallspecs[name]
        planes.append(RefractivePlane(
            point=np.array(pt, float),
            normal=np.array(n, float),
            thickness=thickness, n_glass=n_glass,
            half_extent=np.array(he, float),
            label=name,
        ))
    if include_ceiling:
        planes.append(RefractivePlane(
            point=np.array([cx, cy, zmax], float),
            normal=np.array([0, 0, +1], float),
            thickness=thickness, n_glass=n_glass,
            half_extent=np.array([hx, hy], float),
            label="ceiling",
        ))
    if include_floor:
        planes.append(RefractivePlane(
            point=np.array([cx, cy, zmin], float),
            normal=np.array([0, 0, -1], float),
            thickness=thickness, n_glass=n_glass,
            half_extent=np.array([hx, hy], float),
            label="floor",
        ))
    return ArenaRefractionModel(planes=planes)


# --------------------------------------------------------------------------- #
#  Snell's law and ray refraction                                              #
# --------------------------------------------------------------------------- #

def snell_refract(
    direction: np.ndarray,
    normal: np.ndarray,
    n1: float, n2: float,
) -> np.ndarray:
    """Refract a unit direction across a planar interface using Snell's law.

    The interface normal must point **into the medium the ray is leaving**
    (so that ``direction · normal < 0`` for a ray approaching the
    surface).  The returned direction is a unit vector in the
    new medium.  On total internal reflection (sin²θ_t > 1) a
    ``ValueError`` is raised — for typical air ↔ PMMA, TIR only happens
    going acrylic → air at >42° incidence, which does not occur for
    rays exiting parallel-faced slabs that entered from air.

    Parameters
    ----------
    direction : (3,) unit vector
        Incoming ray direction.
    normal : (3,) unit vector
        Surface normal pointing into the medium of the incoming ray.
    n1 : float
        Refractive index of the medium the ray is leaving.
    n2 : float
        Refractive index of the medium the ray is entering.

    Returns
    -------
    (3,) unit vector in the new medium.
    """
    d = np.asarray(direction, dtype=np.float64).reshape(3)
    n = np.asarray(normal, dtype=np.float64).reshape(3)
    cos_i = -float(d @ n)
    if cos_i < 0:
        # Normal pointed the wrong way for this ray; flip it.
        n = -n
        cos_i = -float(d @ n)
    eta = n1 / n2
    sin2_t = eta * eta * max(0.0, 1.0 - cos_i * cos_i)
    if sin2_t > 1.0:
        raise ValueError(
            f"Total internal reflection at n1={n1}, n2={n2}, "
            f"cos_i={cos_i:.4f}; no transmitted ray exists.")
    cos_t = float(np.sqrt(1.0 - sin2_t))
    out = eta * d + (eta * cos_i - cos_t) * n
    return out / np.linalg.norm(out)


def refract_through_wall(
    origin: np.ndarray,
    direction: np.ndarray,
    plane: RefractivePlane,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the effective ray on the **inner** side of a parallel-faced wall.

    The ray enters from outside (camera side), refracts at the outer face,
    travels through acrylic, and refracts at the inner face.  Because the
    two faces are parallel, the emerging direction equals the entering
    direction and only a lateral offset remains.

    Parameters
    ----------
    origin : (3,) array
        Ray start point in world coordinates (camera centre).
    direction : (3,) unit vector
        Ray direction in world coordinates.
    plane : RefractivePlane
        The wall.  The outer face passes through ``plane.point`` with
        outward normal ``plane.normal``; the inner face lies at distance
        ``plane.thickness`` opposite the normal.

    Returns
    -------
    (inner_exit_point, direction_out) — the inner exit lies on the inner
    face of the slab; the direction is identical (up to floating-point
    error) to the input direction.

    Notes
    -----
    The lateral offset between the incoming ray (extrapolated straight
    through) and the emerging ray inside the arena equals

        Δ = thickness · sin(θ₁ - θ₂) / cos(θ₂)

    where θ₁, θ₂ are the incidence and acrylic angles.  This is in the
    plane of incidence and perpendicular to the outgoing direction.
    """
    d_in = np.asarray(direction, dtype=np.float64).reshape(3)
    d_in = d_in / np.linalg.norm(d_in)
    origin = np.asarray(origin, dtype=np.float64).reshape(3)

    # Outer-face hit
    denom = float(d_in @ plane.normal)
    if denom >= -1e-12:
        raise ValueError(
            "Ray does not enter the wall: direction is parallel to or "
            "away from the outer face.")
    t_outer = float(((plane.point - origin) @ plane.normal) / denom)
    if t_outer <= 0:
        raise ValueError(
            "Ray origin is inside or behind the wall outer face "
            "(t_outer ≤ 0); refract_through_wall expects the camera to "
            "lie outside the arena.")
    outer_hit = origin + t_outer * d_in

    # Refract into acrylic at outer face
    d_acryl = snell_refract(d_in, plane.normal, plane.n_air, plane.n_glass)

    # Travel through slab to inner face: distance = thickness / |d_acryl·normal|
    denom2 = float(d_acryl @ plane.normal)
    if denom2 >= -1e-12:
        # Should not happen: refracted ray must continue inward
        raise RuntimeError("Refracted ray did not advance into the slab.")
    inner_hit = outer_hit + (plane.thickness / -denom2) * d_acryl

    # Refract back to air at inner face (normal pointed into acrylic = -plane.normal)
    d_out = snell_refract(d_acryl, -plane.normal, plane.n_glass, plane.n_air)
    # Numerically, d_out should equal d_in; renormalise.
    d_out = d_out / np.linalg.norm(d_out)

    return inner_hit, d_out


# --------------------------------------------------------------------------- #
#  Ray construction and closest point of two lines                            #
# --------------------------------------------------------------------------- #

def pixel_to_world_ray(
    K: np.ndarray, R_w2c: np.ndarray, T_w2c: np.ndarray,
    pixel: tuple[float, float],
    dist: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Backproject a pixel into a world-space ray (origin, unit direction).

    Parameters
    ----------
    K : (3, 3) intrinsic matrix
    R_w2c : (3, 3) world-to-camera rotation
    T_w2c : (3,) world-to-camera translation
        The camera projection model is x = K (R_w2c X + T_w2c).
    pixel : (u, v)
        Observed pixel.
    dist : optional (5,) or (8,) distortion coefficients
        If provided, the pixel is undistorted with ``cv2.undistortPoints``
        before backprojection.  This is the right thing to do when the
        calibration captured lens distortion that has not already been
        removed by rectification.

    Returns
    -------
    (camera_center_world, ray_direction_world) — the direction is a unit
    vector pointing from the camera into the scene.
    """
    K = np.asarray(K, dtype=np.float64)
    R_w2c = np.asarray(R_w2c, dtype=np.float64)
    T_w2c = np.asarray(T_w2c, dtype=np.float64).reshape(3)

    if dist is not None:
        import cv2
        pix = np.array([[pixel]], dtype=np.float64)
        ud = cv2.undistortPoints(pix, K, np.asarray(dist, np.float64),
                                  P=K).reshape(2)
        u, v = float(ud[0]), float(ud[1])
    else:
        u, v = float(pixel[0]), float(pixel[1])

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    d_cam = np.array([(u - cx) / fx, (v - cy) / fy, 1.0], dtype=np.float64)
    d_world = R_w2c.T @ d_cam
    d_world /= np.linalg.norm(d_world)

    center_world = -R_w2c.T @ T_w2c
    return center_world, d_world


def closest_point_two_lines(
    O0: np.ndarray, d0: np.ndarray,
    O1: np.ndarray, d1: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Find the midpoint of the common perpendicular between two lines.

    Each line is given as origin + s · direction (direction unit).
    Returns the midpoint of the two closest points and the gap length
    between them.  For parallel lines, returns the midpoint of the
    closest pair of foot-points relative to ``O0`` (gap = perpendicular
    distance).

    Parameters
    ----------
    O0, O1 : (3,) origins
    d0, d1 : (3,) unit directions

    Returns
    -------
    (midpoint, gap) — midpoint is (P0 + P1) / 2 where P_i is the foot on
    line i, and gap = ||P0 - P1||.
    """
    O0 = np.asarray(O0, dtype=np.float64).reshape(3)
    O1 = np.asarray(O1, dtype=np.float64).reshape(3)
    d0 = np.asarray(d0, dtype=np.float64).reshape(3)
    d1 = np.asarray(d1, dtype=np.float64).reshape(3)
    d0 = d0 / np.linalg.norm(d0)
    d1 = d1 / np.linalg.norm(d1)

    a = float(d0 @ d1)
    denom = 1.0 - a * a
    dO = O0 - O1
    if abs(denom) < 1e-12:
        # Parallel — pick the foot of O1 on line 0 as a degenerate fallback
        s = -float(d0 @ dO)
        P0 = O0 + s * d0
        gap = float(np.linalg.norm(P0 - O1))
        return 0.5 * (P0 + O1), gap

    # min ||O0 + s d0 - O1 - t d1||²
    # ∂/∂s = 0:  s - a t = -d0·dO
    # ∂/∂t = 0:  a s - t = -d1·dO
    b0 = -float(d0 @ dO)
    b1 = -float(d1 @ dO)
    t = (a * b0 - b1) / denom
    s = a * t + b0
    P0 = O0 + s * d0
    P1 = O1 + t * d1
    gap = float(np.linalg.norm(P0 - P1))
    return 0.5 * (P0 + P1), gap


# --------------------------------------------------------------------------- #
#  Refractive triangulation                                                    #
# --------------------------------------------------------------------------- #

def triangulate_refracted(
    origin0: np.ndarray, direction0: np.ndarray,
    origin1: np.ndarray, direction1: np.ndarray,
    arena: ArenaRefractionModel,
    *,
    initial_xyz: Optional[np.ndarray] = None,
    max_iter: int = 8,
    tol: float = 1e-4,
) -> tuple[np.ndarray, float, int]:
    """Iterative refractive triangulation from two world-space rays.

    The two camera rays are bent through whichever wall each one crosses
    on its way to the current estimate of the 3D point.  The refracted
    rays are then intersected (closest-point-of-two-lines) to yield a
    new estimate.  Iterating typically converges in 2–4 steps because
    the wall traversed rarely changes after the first refraction
    correction.

    If a camera ray does not cross any wall (e.g. the estimated point is
    outside the arena), that ray is used unmodified.

    Parameters
    ----------
    origin0, direction0 : (3,) arrays
        Camera 0 ray origin (centre) and unit direction in world coords.
    origin1, direction1 : (3,) arrays
        Camera 1 ray origin and unit direction.
    arena : ArenaRefractionModel
        Walls to refract through.  If empty, the result is the
        straight-ray closest-point solution (equivalent to DLT to within
        the linear/nonlinear distinction).
    initial_xyz : (3,) array or None
        Starting estimate of the 3D point.  If None, a straight-ray
        closest-point intersection is used.
    max_iter : int
        Maximum number of refraction-and-intersect iterations.
    tol : float
        Convergence threshold on ‖ΔX‖ between successive iterations (mm).

    Returns
    -------
    (xyz, gap, n_iter) — final 3D point, line-gap at convergence (mm),
    and number of iterations performed.
    """
    O0 = np.asarray(origin0, dtype=np.float64).reshape(3)
    O1 = np.asarray(origin1, dtype=np.float64).reshape(3)
    d0 = np.asarray(direction0, dtype=np.float64).reshape(3)
    d1 = np.asarray(direction1, dtype=np.float64).reshape(3)
    d0 = d0 / np.linalg.norm(d0)
    d1 = d1 / np.linalg.norm(d1)

    # The wall a camera ray crosses is determined by the *original* ray
    # direction (the pixel observation), not by the current 3D estimate.
    # For a parallel-slab model the refracted in-arena ray is parallel to
    # the original camera ray, so the geometry is a single-shot calculation:
    # refract each ray once, then intersect the two shifted parallel lines.
    plane0, _ = arena.find_traversed_plane(O0, d0)
    if plane0 is not None:
        B0, dr0 = refract_through_wall(O0, d0, plane0)
    else:
        B0, dr0 = O0, d0

    plane1, _ = arena.find_traversed_plane(O1, d1)
    if plane1 is not None:
        B1, dr1 = refract_through_wall(O1, d1, plane1)
    else:
        B1, dr1 = O1, d1

    # First-shot intersection of the refracted rays.
    X, gap = closest_point_two_lines(B0, dr0, B1, dr1)

    # The loop below is a safety net for future non-parallel-slab models
    # (e.g. curved walls or thick lenses) where the wall choice might
    # depend on X. For the current parallel-slab walls it exits on iter 1.
    if initial_xyz is not None:
        X = np.asarray(initial_xyz, dtype=np.float64).reshape(3).copy()

    last_gap = gap
    for it in range(1, max_iter + 1):
        plane0_it, _ = arena.find_traversed_plane(O0, d0)
        plane1_it, _ = arena.find_traversed_plane(O1, d1)
        if plane0_it is not None:
            B0, dr0 = refract_through_wall(O0, d0, plane0_it)
        else:
            B0, dr0 = O0, d0
        if plane1_it is not None:
            B1, dr1 = refract_through_wall(O1, d1, plane1_it)
        else:
            B1, dr1 = O1, d1

        X_new, gap = closest_point_two_lines(B0, dr0, B1, dr1)
        delta = float(np.linalg.norm(X_new - X))
        X = X_new
        last_gap = gap
        if delta < tol:
            return X, gap, it

    return X, last_gap, max_iter


# --------------------------------------------------------------------------- #
#  Config I/O                                                                  #
# --------------------------------------------------------------------------- #

def save_arena_config(path, arena: ArenaRefractionModel) -> None:
    """Write an arena refraction model to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(arena.to_dict(), fh, indent=2)


def load_arena_config(path) -> ArenaRefractionModel:
    """Load an arena refraction model from a JSON file."""
    path = Path(path)
    with open(path) as fh:
        return ArenaRefractionModel.from_dict(json.load(fh))


__all__ = [
    "RefractivePlane",
    "ArenaRefractionModel",
    "build_box_arena",
    "snell_refract",
    "refract_through_wall",
    "pixel_to_world_ray",
    "closest_point_two_lines",
    "triangulate_refracted",
    "save_arena_config",
    "load_arena_config",
]
