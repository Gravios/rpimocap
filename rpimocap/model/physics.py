"""
rpimocap.model.physics
======================
A lightweight **physics-based pose prior** — the consequences of gravity —
that regularizes silhouette fitting so recovered poses are physically
plausible: the rat rests **upright** (belly down) with its **hind feet on the
floor**, not floating, penetrating, or rolled onto its side.

This is deliberately *not* a rigid-body dynamics engine. For pose estimation
the useful part of "gravity" is a set of static constraints, which compose
directly with the analysis-by-synthesis objective:

* **uprightness** — the body's up-axis should point to world +z,
* **ground contact** — the lowest hind foot should touch the floor (z = 0),
* **non-penetration** — nothing below the floor,
* **support** — the trunk's centre of mass should sit over the grounded feet
  (static stability under gravity, so the animal doesn't tip).

:func:`physical_penalty` scores a pose (add it to the fit objective via the
fitter's ``physics_weight``); :func:`settle_pose` *projects* a pose to a
resting one — righting the body and dropping it until the hind feet contact
the floor, like letting the model fall under gravity.
"""
from __future__ import annotations

import numpy as np

from .rat_skeleton import RAT23_INDEX, RatPose, euler_to_R, forward_kinematics

GROUND_Z = 0.0
HIND_FEET = ("FootL", "FootR")
ALL_FEET = ("FootL", "FootR", "HandL", "HandR")
_L = 30.0                       # length scale (mm) normalizing height terms


def body_up(pose: RatPose) -> np.ndarray:
    """World direction of the body's local up (+z) axis."""
    return euler_to_R(*pose.root_rot) @ np.array([0.0, 0.0, 1.0])


def heading(pose: RatPose) -> float:
    """Yaw of the body's forward (+x) axis in the world xy-plane (rad)."""
    fwd = euler_to_R(*pose.root_rot) @ np.array([1.0, 0.0, 0.0])
    return float(np.arctan2(fwd[1], fwd[0]))


def _kp(pose, kp):
    return forward_kinematics(pose) if kp is None else kp


def upright_penalty(pose: RatPose) -> float:
    """0 when the body's up-axis points to world +z; ~1 on its side; ~2 inverted."""
    up = body_up(pose)
    return float(1.0 - up[2] / (np.linalg.norm(up) + 1e-9))


def penetration_penalty(pose: RatPose, kp=None) -> float:
    """Mean squared depth of keypoints below the floor (normalized)."""
    z = _kp(pose, kp)[:, 2]
    below = np.clip(GROUND_Z - z, 0.0, None)
    return float(np.mean((below / _L) ** 2))


def ground_contact_penalty(pose: RatPose, kp=None, feet=HIND_FEET) -> float:
    """Squared height of the lowest of ``feet`` above the floor (normalized).

    Zero once at least one of the (hind) feet touches the ground.
    """
    kp = _kp(pose, kp)
    gap = max(0.0, min(kp[RAT23_INDEX[f], 2] for f in feet) - GROUND_Z)
    return float((gap / _L) ** 2)


def support_penalty(pose: RatPose, kp=None, feet=ALL_FEET) -> float:
    """Static stability: horizontal offset of the trunk centre of mass from
    the centroid of the near-ground feet (normalized). Small when the body is
    balanced over its supports."""
    kp = _kp(pose, kp)
    com = kp[[RAT23_INDEX[j] for j in ("SpineF", "SpineM", "SpineL")]].mean(0)[:2]
    grounded = [kp[RAT23_INDEX[f]] for f in feet
                if kp[RAT23_INDEX[f], 2] < 2.0 * _L]
    if not grounded:
        return 1.0
    base = np.asarray(grounded)[:, :2].mean(0)
    return float((np.linalg.norm(com - base) / (2.0 * _L)) ** 2)


def physical_penalty(pose: RatPose, kp=None, w_upright: float = 1.0,
                     w_contact: float = 1.0, w_penetration: float = 1.0,
                     w_support: float = 0.3) -> float:
    """Weighted sum of the plausibility terms (each ~O(1) when violated)."""
    kp = _kp(pose, kp)
    return (w_upright * upright_penalty(pose)
            + w_contact * ground_contact_penalty(pose, kp)
            + w_penetration * penetration_penalty(pose, kp)
            + w_support * support_penalty(pose, kp))


def settle_pose(pose: RatPose, feet=HIND_FEET) -> RatPose:
    """Project a pose to a resting one, as if dropped under gravity.

    Rights the body (upright, heading preserved) and translates it vertically
    so the lowest of ``feet`` just contacts the floor (z = 0). Joint angles
    and scale are unchanged.
    """
    up = RatPose(root_pos=np.asarray(pose.root_pos, float).copy(),
                 root_rot=np.array([0.0, 0.0, heading(pose)]),
                 joint_angles=dict(pose.joint_angles), scale=float(pose.scale))
    kp = forward_kinematics(up)
    lowest = min(kp[RAT23_INDEX[f], 2] for f in feet)
    up.root_pos[2] += GROUND_Z - lowest
    return up
