"""
rpimocap.model.rat_skeleton
===========================
Anatomically-constrained rat skeleton (the DANNCE "rat23" keypoint set)
with forward kinematics, joint-angle limits, bone-length scaling, and
DLT projection — for generating *synthetic but valid* rat poses.

Why
---
Synthetic poses that obey bone-length and joint-angle constraints give us
zero-noise ground truth to:
  * validate the detect → triangulate path (project a known 3D pose,
    triangulate it back, the error is purely the pipeline's),
  * stress-test the tracker through occlusions against known truth,
  * auto-label training data before any hand annotation,
  * reject impossible *detected* poses at inference (a free accuracy gate),
  * supply the valid-pose silhouette manifold as a shape prior.

The skeleton is the rat23 standard (Dunn et al. DANNCE 2021; reused by
FreiPose, s-DANNCE, MIMIC-MJX): 23 keypoints in 4 regions. The
constraint approach (bone lengths + per-DOF joint limits + temporal
smoothness) follows "Estimation of skeletal kinematics in freely moving
rodents" (Nat. Methods 2022).

Coordinate frame: arena millimetres, matching the rest of rpimocap
(x ∈ [-140, 140], y ∈ [-215, 215], z ∈ [0, 388] for the standard arena).
The model root is SpineM with a full 6-DOF pose (3 translation + 3
rotation); every other joint adds rotational DOF within its limits, and
bone lengths are fixed per individual.

Honest scope: this is a KINEMATIC model (valid geometry), not a DYNAMIC
one (no muscles/forces). Joint-limit numbers are seeded from the OpenSim
rat-hindlimb literature and the Nat. Methods schematic and are
deliberately conservative; refine from real tracked data later. The limb
joints use simplified 1-DOF hinges in the sagittal plane plus a small
out-of-plane allowance, which is enough for plausible silhouettes and
ground-truth projection, not for precise biomechanics.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ────────────────────────────────────────────────────────────────────
#  Skeleton definition (rat23)
# ────────────────────────────────────────────────────────────────────

#: The 23 keypoint names, in a fixed canonical order (index = id).
RAT23_JOINTS = [
    # head (4)
    "Snout", "EarL", "EarR", "SpineF",
    # trunk (7)
    "SpineM", "SpineL", "TailBase",
    "ShoulderL", "ShoulderR", "HipL", "HipR",
    # forelimbs (6)
    "ElbowL", "WristL", "HandL", "ElbowR", "WristR", "HandR",
    # hindlimbs (6)
    "KneeL", "AnkleL", "FootL", "KneeR", "AnkleR", "FootR",
]
RAT23_INDEX = {name: i for i, name in enumerate(RAT23_JOINTS)}

#: Region grouping (matches the s-DANNCE / Cell 2025 description).
RAT23_REGIONS = {
    "head":      ["Snout", "EarL", "EarR", "SpineF"],
    "trunk":     ["SpineM", "SpineL", "TailBase",
                  "ShoulderL", "ShoulderR", "HipL", "HipR"],
    "forelimbs": ["ElbowL", "WristL", "HandL",
                  "ElbowR", "WristR", "HandR"],
    "hindlimbs": ["KneeL", "AnkleL", "FootL",
                  "KneeR", "AnkleR", "FootR"],
}

#: Kinematic tree as parent → child. Root (SpineM) maps to None.
RAT23_PARENT = {
    "SpineM":    None,         # root
    "SpineF":    "SpineM",
    "Snout":     "SpineF",
    "EarL":      "SpineF",
    "EarR":      "SpineF",
    "SpineL":    "SpineM",
    "TailBase":  "SpineL",
    "ShoulderL": "SpineF",
    "ElbowL":    "ShoulderL",
    "WristL":    "ElbowL",
    "HandL":     "WristL",
    "ShoulderR": "SpineF",
    "ElbowR":    "ShoulderR",
    "WristR":    "ElbowR",
    "HandR":     "WristR",
    "HipL":      "SpineL",
    "KneeL":     "HipL",
    "AnkleL":    "KneeL",
    "FootL":     "AnkleL",
    "HipR":      "SpineL",
    "KneeR":     "HipR",
    "AnkleR":    "KneeR",
    "FootR":     "AnkleR",
}

#: Bones as (parent, child) ordered pairs, derived from the tree.
RAT23_BONES = [(p, c) for c, p in RAT23_PARENT.items() if p is not None]


# ────────────────────────────────────────────────────────────────────
#  Canonical geometry: rest-pose bone vectors (mm) for a reference rat
# ────────────────────────────────────────────────────────────────────
#
# Rest pose = a neutral standing rat facing +x, with +z up. Each entry is
# the bone vector from parent to child *in the parent's local frame at
# rest*. Lengths are a plausible adult-rat ratio set (~230 mm nose-to-tail
# base body); scale with `scale` for body size. These are reference
# geometry for synthetic generation, not measured anatomy.
#
# Convention: +x = body forward (snout direction), +y = animal's LEFT,
# +z = up.

_REST_BONES_MM = {
    # head / spine midline (forward = +x)
    ("SpineM", "SpineF"):  ( 35.0,   0.0,   2.0),
    ("SpineF", "Snout"):   ( 38.0,   0.0,  -2.0),
    ("SpineF", "EarL"):    (  6.0,  11.0,  10.0),
    ("SpineF", "EarR"):    (  6.0, -11.0,  10.0),
    ("SpineM", "SpineL"):  (-38.0,   0.0,   1.0),
    ("SpineL", "TailBase"):(-30.0,   0.0,  -3.0),
    # shoulders off SpineF, hips off SpineL (left = +y, right = -y)
    ("SpineF", "ShoulderL"):( -2.0,  16.0,  -4.0),
    ("SpineF", "ShoulderR"):( -2.0, -16.0,  -4.0),
    ("SpineL", "HipL"):    (  2.0,  17.0,  -4.0),
    ("SpineL", "HipR"):    (  2.0, -17.0,  -4.0),
    # forelimbs hang down (-z) and slightly forward
    ("ShoulderL", "ElbowL"):(  3.0,   1.0, -18.0),
    ("ElbowL",    "WristL"):(  4.0,   0.0, -16.0),
    ("WristL",    "HandL"): (  6.0,   0.0,  -8.0),
    ("ShoulderR", "ElbowR"):(  3.0,  -1.0, -18.0),
    ("ElbowR",    "WristR"):(  4.0,   0.0, -16.0),
    ("WristR",    "HandR"): (  6.0,   0.0,  -8.0),
    # hindlimbs: femur down/back, tibia down/forward, foot forward
    ("HipL", "KneeL"):     ( -6.0,   2.0, -20.0),
    ("KneeL", "AnkleL"):   (  4.0,   0.0, -20.0),
    ("AnkleL", "FootL"):   ( 14.0,   0.0,  -6.0),
    ("HipR", "KneeR"):     ( -6.0,  -2.0, -20.0),
    ("KneeR", "AnkleR"):   (  4.0,   0.0, -20.0),
    ("AnkleR", "FootR"):   ( 14.0,   0.0,  -6.0),
}

#: Canonical bone lengths (mm), derived from the rest vectors.
CANONICAL_BONE_LENGTHS = {
    bone: float(np.linalg.norm(v)) for bone, v in _REST_BONES_MM.items()
}


# ────────────────────────────────────────────────────────────────────
#  Joint-angle limits
# ────────────────────────────────────────────────────────────────────
#
# Each child bone can rotate relative to its rest direction by Euler
# angles (rx, ry, rz) applied in the PARENT frame, each bounded by
# (min, max) radians. This is a simplification of true anatomical joints
# (ball/hinge/universal) into per-bone Euler ranges — sufficient to keep
# synthetic poses plausible. Hinge joints (knee/elbow) get a wide range
# on ONE axis and near-zero on the others; ball joints (hip/shoulder/
# spine/head) get moderate ranges on all three.
#
# Values are conservative defaults in DEGREES (converted below); refine
# from real data. A pose is valid iff every bone's sampled angles lie
# within its limits.

def _deg(lo, hi):
    return (np.radians(lo), np.radians(hi))


# (rx_range, ry_range, rz_range) per child bone, in degrees.
_LIMITS_DEG = {
    # spine midline — moderate ball joints (the body bends/arches)
    "SpineF":    ((-20, 20), (-25, 25), (-30, 30)),
    "SpineL":    ((-20, 20), (-25, 25), (-30, 30)),
    "TailBase":  ((-30, 30), (-30, 30), (-45, 45)),
    # head — moderate ball
    "Snout":     ((-15, 15), (-30, 40), (-45, 45)),
    "EarL":      ((-5, 5), (-5, 5), (-5, 5)),   # ears ~rigid
    "EarR":      ((-5, 5), (-5, 5), (-5, 5)),
    # shoulders/hips — attachment ball joints, small range
    "ShoulderL": ((-15, 15), (-20, 20), (-20, 20)),
    "ShoulderR": ((-15, 15), (-20, 20), (-20, 20)),
    "HipL":      ((-20, 20), (-30, 30), (-25, 25)),
    "HipR":      ((-20, 20), (-30, 30), (-25, 25)),
    # forelimb: shoulder→elbow (ball-ish), elbow hinge, wrist
    "ElbowL":    ((-10, 10), (-90, 10), (-10, 10)),   # ry hinge
    "WristL":    ((-15, 15), (-40, 40), (-15, 15)),
    "HandL":     ((-10, 10), (-20, 20), (-10, 10)),
    "ElbowR":    ((-10, 10), (-90, 10), (-10, 10)),
    "WristR":    ((-15, 15), (-40, 40), (-15, 15)),
    "HandR":     ((-10, 10), (-20, 20), (-10, 10)),
    # hindlimb: hip already above; knee hinge, ankle, foot
    "KneeL":     ((-10, 10), (-10, 90), (-10, 10)),   # ry hinge
    "AnkleL":    ((-15, 15), (-45, 45), (-15, 15)),
    "FootL":     ((-10, 10), (-20, 20), (-10, 10)),
    "KneeR":     ((-10, 10), (-10, 90), (-10, 10)),
    "AnkleR":    ((-15, 15), (-45, 45), (-15, 15)),
    "FootR":     ((-10, 10), (-20, 20), (-10, 10)),
}

#: Joint limits in radians: {child: (rx_range, ry_range, rz_range)}.
JOINT_LIMITS = {
    child: tuple(_deg(lo, hi) for (lo, hi) in ranges)
    for child, ranges in _LIMITS_DEG.items()
}


# ────────────────────────────────────────────────────────────────────
#  Rotation helpers
# ────────────────────────────────────────────────────────────────────


def _rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def _rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def _rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def euler_to_R(rx: float, ry: float, rz: float) -> np.ndarray:
    """Intrinsic XYZ Euler angles → 3×3 rotation matrix."""
    return _rot_z(rz) @ _rot_y(ry) @ _rot_x(rx)


# ────────────────────────────────────────────────────────────────────
#  Pose representation + forward kinematics
# ────────────────────────────────────────────────────────────────────


@dataclass
class RatPose:
    """A full articulated pose.

    root_pos    : (3,) world position of SpineM (mm)
    root_rot    : (3,) global body orientation Euler angles (rad)
    joint_angles: {child_name: (rx, ry, rz)} relative rotations (rad).
                  Missing entries default to zero (rest direction).
    scale       : body-size multiplier on all bone lengths.
    """
    root_pos:     np.ndarray = field(
        default_factory=lambda: np.zeros(3))
    root_rot:     np.ndarray = field(
        default_factory=lambda: np.zeros(3))
    joint_angles: dict = field(default_factory=dict)
    scale:        float = 1.0


def _bone_vectors(scale: float) -> dict:
    """Rest bone vectors scaled by body size."""
    return {b: np.asarray(v, np.float64) * scale
            for b, v in _REST_BONES_MM.items()}


def forward_kinematics(pose: RatPose) -> np.ndarray:
    """Compute all 23 keypoint world positions (mm).

    Walks the kinematic tree from the root, composing each joint's
    relative rotation with the scaled rest bone vector.

    Returns
    -------
    (23, 3) array of keypoint positions, indexed by RAT23_JOINTS order.
    """
    bones = _bone_vectors(pose.scale)
    R_root = euler_to_R(*pose.root_rot)
    # world rotation accumulated at each joint (the frame its CHILD
    # bones are expressed in) and the joint's world position.
    world_pos = {"SpineM": np.asarray(pose.root_pos, np.float64)}
    world_rot = {"SpineM": R_root}

    # Process in an order where parents come before children.
    order = _topo_order()
    for child in order:
        parent = RAT23_PARENT[child]
        if parent is None:
            continue
        bone = bones[(parent, child)]
        Rp = world_rot[parent]
        # relative joint rotation (default zero = rest direction)
        rx, ry, rz = pose.joint_angles.get(child, (0.0, 0.0, 0.0))
        Rj = euler_to_R(rx, ry, rz)
        Rc = Rp @ Rj
        world_pos[child] = world_pos[parent] + Rp @ bone
        world_rot[child] = Rc

    out = np.zeros((len(RAT23_JOINTS), 3), dtype=np.float64)
    for name, i in RAT23_INDEX.items():
        out[i] = world_pos[name]
    return out


def _topo_order():
    """Joint names ordered so every parent precedes its children."""
    order = []
    seen = set()

    def visit(n):
        p = RAT23_PARENT[n]
        if p is not None and p not in seen:
            visit(p)
        if n not in seen:
            seen.add(n)
            order.append(n)

    for n in RAT23_JOINTS:
        visit(n)
    return order


# ────────────────────────────────────────────────────────────────────
#  Sampling valid poses
# ────────────────────────────────────────────────────────────────────


def sample_joint_angles(rng: np.random.RandomState,
                        fraction: float = 1.0) -> dict:
    """Sample each joint's (rx, ry, rz) uniformly within its limits.

    fraction : interpolate the sampling range between the rest pose
               (0.0 → all angles exactly 0, the canonical rest
               direction) and the full limits (1.0). Intermediate
               values sample a sub-range that always includes 0 where 0
               is within the joint's limits, so mild poses stay close to
               rest. Useful for generating mild vs extreme poses.
    """
    f = float(np.clip(fraction, 0.0, 1.0))
    angles = {}
    for child, ranges in JOINT_LIMITS.items():
        a = []
        for (lo, hi) in ranges:
            # Shrink each bound toward 0 by (1 - f). At f=0 the range
            # collapses to [0, 0]; at f=1 it's the full [lo, hi].
            slo = lo * f
            shi = hi * f
            a.append(float(rng.uniform(slo, shi)))
        angles[child] = tuple(a)
    return angles


def sample_pose(rng: np.random.RandomState,
                scale: float = 1.0,
                arena_bounds: Optional[tuple] = None,
                fraction: float = 1.0,
                z_range: tuple = (40.0, 120.0)) -> RatPose:
    """Sample a random valid pose: random body position + heading +
    joint angles within limits.

    arena_bounds : (xmin,xmax,ymin,ymax,zmin,zmax) mm. If given, the
                   root is placed inside it; default uses the standard
                   arena. z_range bounds the root height (the rat's body
                   centre is off the floor).
    """
    if arena_bounds is None:
        arena_bounds = (-140, 140, -215, 215, 0, 388)
    xmin, xmax, ymin, ymax, zmin, zmax = arena_bounds
    # keep a margin so limbs stay inside
    margin = 40.0 * scale
    rx0 = rng.uniform(xmin + margin, xmax - margin)
    ry0 = rng.uniform(ymin + margin, ymax - margin)
    rz0 = rng.uniform(max(z_range[0], zmin + 20),
                      min(z_range[1], zmax - 20))
    heading = rng.uniform(-np.pi, np.pi)          # yaw about z
    # small pitch/roll
    pitch = rng.uniform(np.radians(-15), np.radians(15))
    roll = rng.uniform(np.radians(-15), np.radians(15))
    return RatPose(
        root_pos=np.array([rx0, ry0, rz0]),
        root_rot=np.array([roll, pitch, heading]),
        joint_angles=sample_joint_angles(rng, fraction=fraction),
        scale=scale,
    )


# ────────────────────────────────────────────────────────────────────
#  Validity checks
# ────────────────────────────────────────────────────────────────────


def bone_lengths(keypoints: np.ndarray) -> dict:
    """Measured bone lengths (mm) from a (23,3) keypoint array."""
    out = {}
    for (p, c) in RAT23_BONES:
        out[(p, c)] = float(np.linalg.norm(
            keypoints[RAT23_INDEX[c]] - keypoints[RAT23_INDEX[p]]))
    return out


def check_bone_lengths(keypoints: np.ndarray, scale: float = 1.0,
                       rel_tol: float = 0.02) -> bool:
    """True iff every measured bone length matches the canonical
    (scaled) length within rel_tol. FK preserves lengths exactly, so
    this mainly guards triangulated/observed poses."""
    meas = bone_lengths(keypoints)
    for bone, L in meas.items():
        expect = CANONICAL_BONE_LENGTHS[bone] * scale
        if expect <= 0:
            continue
        if abs(L - expect) / expect > rel_tol:
            return False
    return True


def check_arena_containment(keypoints: np.ndarray,
                            arena_bounds: Optional[tuple] = None,
                            pad_mm: float = 0.0) -> bool:
    """True iff all keypoints lie within the arena box (+ pad)."""
    if arena_bounds is None:
        arena_bounds = (-140, 140, -215, 215, 0, 388)
    xmin, xmax, ymin, ymax, zmin, zmax = arena_bounds
    k = keypoints
    return bool(
        np.all(k[:, 0] >= xmin - pad_mm) and np.all(k[:, 0] <= xmax + pad_mm)
        and np.all(k[:, 1] >= ymin - pad_mm) and np.all(k[:, 1] <= ymax + pad_mm)
        and np.all(k[:, 2] >= zmin - pad_mm) and np.all(k[:, 2] <= zmax + pad_mm))


def check_joint_angles(angles: dict, tol: float = 1e-6) -> bool:
    """True iff every joint angle lies within its limits."""
    for child, ranges in JOINT_LIMITS.items():
        a = angles.get(child, (0.0, 0.0, 0.0))
        for val, (lo, hi) in zip(a, ranges):
            if val < lo - tol or val > hi + tol:
                return False
    return True


def is_valid(pose: RatPose,
             arena_bounds: Optional[tuple] = None,
             require_arena: bool = True) -> bool:
    """Full validity: joint angles within limits, (optionally) all
    keypoints inside the arena. Bone lengths are exact by construction
    under FK so they're not re-checked here."""
    if not check_joint_angles(pose.joint_angles):
        return False
    if require_arena:
        kp = forward_kinematics(pose)
        if not check_arena_containment(kp, arena_bounds):
            return False
    return True


# ────────────────────────────────────────────────────────────────────
#  Projection to camera views (reuses the project convention)
# ────────────────────────────────────────────────────────────────────


def project_pose(keypoints: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Project (23,3) world keypoints through a (3,4) DLT matrix to
    (23,2) pixel coordinates. Mirrors reconstruction.voxel.
    project_points_batch (points behind the camera → -1e9)."""
    ones = np.ones((keypoints.shape[0], 1), dtype=np.float64)
    pts_h = np.concatenate([keypoints, ones], axis=1)        # (N,4)
    proj = (P @ pts_h.T).T                                   # (N,3)
    with np.errstate(divide="ignore", invalid="ignore"):
        px = proj[:, :2] / proj[:, 2:3]
    behind = proj[:, 2] <= 0
    px[behind] = -1e9
    return px


def visible_subset(names: list) -> np.ndarray:
    """Index array selecting a subset of keypoints by name — e.g. the
    observable set on a textureless rat (snout, ears, spine, tail base,
    paws) plus whatever landmarks a given rig adds (e.g. headstage)."""
    return np.array([RAT23_INDEX[n] for n in names], dtype=int)


#: A reasonable "observable on a textureless white rat at 2 views" set,
#: per the RatBodyFormer/FreiPose observation that only face/appendages/
#: midline are reliably detectable. The headstage (if rigidly mounted) is
#: an extra high-contrast landmark this project has and others don't —
#: add it separately as it's not a natural rat keypoint.
OBSERVABLE_KEYPOINTS = [
    "Snout", "EarL", "EarR", "SpineF", "SpineM", "SpineL", "TailBase",
    "HandL", "HandR", "FootL", "FootR",
]
