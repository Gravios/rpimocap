"""
rpimocap.model.body_model
=========================
A **visually comparable body surface** wrapped on the rat23 skeleton
(`rat_skeleton.py`), and its projection to a camera as a silhouette.

Each bone (parent → child) is a **tapered capsule** — a truncated cone with
spherical end caps — whose two radii give the limb/trunk thickness. The union
of the capsules is a rat-shaped volume that follows any articulated pose. A
capsule is convex, so its image projection is the convex hull of its projected
surface samples; the body silhouette is the union of the per-bone hulls.

This is the *synthesis* half of analysis-by-synthesis: pose the skeleton,
render the silhouette in each calibrated camera, and compare it to the
observed rat mask (e.g. from `detection.topo_detect`). Fitting the pose to
maximize that overlap lives in `rpimocap.model.fit`.

Radii are a plausible adult-rat shape (plump trunk ~40 mm across, tapering
head/tail, thin limbs); scale them per individual with `scale_radii`, and
override per bone as needed. Reference geometry, not measured anatomy.
"""
from __future__ import annotations

import cv2
import numpy as np

from .rat_skeleton import (RAT23_BONES, RAT23_INDEX, forward_kinematics,
                           project_pose)

# Per-bone (parent_radius_mm, child_radius_mm) for a reference adult rat.
DEFAULT_RADII = {
    # trunk — a plump body
    ("SpineM", "SpineF"):    (20.0, 18.0),
    ("SpineM", "SpineL"):    (20.0, 16.0),
    ("SpineL", "TailBase"):  (13.0, 6.0),
    # head
    ("SpineF", "Snout"):     (15.0, 5.0),
    ("SpineF", "EarL"):      (5.0, 3.0),
    ("SpineF", "EarR"):      (5.0, 3.0),
    # shoulders / hips (bulk near the body)
    ("SpineF", "ShoulderL"): (11.0, 8.0),
    ("SpineF", "ShoulderR"): (11.0, 8.0),
    ("SpineL", "HipL"):      (12.0, 9.0),
    ("SpineL", "HipR"):      (12.0, 9.0),
    # forelimbs
    ("ShoulderL", "ElbowL"): (7.0, 5.0),
    ("ElbowL", "WristL"):    (5.0, 4.0),
    ("WristL", "HandL"):     (4.0, 3.0),
    ("ShoulderR", "ElbowR"): (7.0, 5.0),
    ("ElbowR", "WristR"):    (5.0, 4.0),
    ("WristR", "HandR"):     (4.0, 3.0),
    # hindlimbs
    ("HipL", "KneeL"):       (8.0, 6.0),
    ("KneeL", "AnkleL"):     (6.0, 4.0),
    ("AnkleL", "FootL"):     (4.0, 3.0),
    ("HipR", "KneeR"):       (8.0, 6.0),
    ("KneeR", "AnkleR"):     (6.0, 4.0),
    ("AnkleR", "FootR"):     (4.0, 3.0),
    # tail — tapers from a thick base to a thin tip. Only used when the mesh
    # is built with the tail (see build_rat_mesh(with_tail=True)); harmless
    # otherwise, since radii are looked up per bone.
    ("TailBase", "Tail1"):   (6.0, 5.0),
    ("Tail1", "Tail2"):      (5.0, 4.0),
    ("Tail2", "Tail3"):      (4.0, 3.2),
    ("Tail3", "Tail4"):      (3.2, 2.4),
    ("Tail4", "Tail5"):      (2.4, 1.5),
}


def scale_radii(radii: dict, factor: float) -> dict:
    """Scale every radius by ``factor`` (per-individual body thickness)."""
    return {b: (r0 * factor, r1 * factor) for b, (r0, r1) in radii.items()}


def _capsule_points(p0, p1, r0, r1, n_ring=14, n_along=3):
    """3D surface samples of a tapered capsule from p0(r0) to p1(r1).

    Rings around the axis at several positions (radius interpolated) plus the
    two axial end tips. The convex hull of the *projection* of these points is
    the capsule's image silhouette (a capsule is convex)."""
    p0 = np.asarray(p0, np.float64)
    p1 = np.asarray(p1, np.float64)
    axis = p1 - p0
    L = float(np.linalg.norm(axis))
    theta = np.linspace(0.0, 2.0 * np.pi, n_ring, endpoint=False)
    if L < 1e-6:                                   # degenerate → a sphere
        r = max(r0, r1)
        e = np.eye(3)
        pts = [p0 + r * (np.cos(theta)[:, None] * e[i] + np.sin(theta)[:, None] * e[j])
               for i, j in ((0, 1), (0, 2), (1, 2))]
        return np.vstack(pts)
    hat = axis / L
    ref = np.array([1.0, 0.0, 0.0]) if abs(hat[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(hat, ref); u /= np.linalg.norm(u)
    v = np.cross(hat, u)
    ring = np.cos(theta)[:, None] * u + np.sin(theta)[:, None] * v   # (n_ring, 3)
    pts = []
    for t in np.linspace(0.0, 1.0, n_along + 2):
        c = p0 + t * axis
        r = r0 + t * (r1 - r0)
        pts.append(c + r * ring)
    pts.append((p0 - r0 * hat)[None, :])           # capsule tips (end caps)
    pts.append((p1 + r1 * hat)[None, :])
    return np.vstack(pts)


def render_silhouette(keypoints: np.ndarray, P: np.ndarray,
                      radii: dict = DEFAULT_RADII,
                      image_shape=(1080, 2028)) -> np.ndarray:
    """Project the capsule body onto one camera and rasterize a silhouette.

    Parameters
    ----------
    keypoints : (23, 3) arena-mm joint positions (``forward_kinematics``).
    P         : (3, 4) DLT matrix (arena mm → pixel), e.g. ``dlt_P0``.
    radii     : per-bone (parent, child) radii, mm.
    image_shape : (H, W) of the target mask.

    Returns a uint8 mask (0/255).
    """
    H, W = int(image_shape[0]), int(image_shape[1])
    mask = np.zeros((H, W), np.uint8)
    for (pa, pc) in RAT23_BONES:
        p0 = keypoints[RAT23_INDEX[pa]]
        p1 = keypoints[RAT23_INDEX[pc]]
        r0, r1 = radii.get((pa, pc), (5.0, 5.0))
        px = project_pose(_capsule_points(p0, p1, r0, r1), P)
        px = px[px[:, 0] > -1e8]                   # drop behind-camera points
        if len(px) < 3:
            continue
        hull = cv2.convexHull(px.astype(np.float32))
        cv2.fillConvexPoly(mask, np.round(hull).astype(np.int32), 255)
    return mask


def render_pose_silhouette(pose, P, radii=DEFAULT_RADII, image_shape=(1080, 2028)):
    """Convenience: ``render_silhouette(forward_kinematics(pose), ...)``."""
    return render_silhouette(forward_kinematics(pose), P, radii, image_shape)


def silhouette_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection-over-union of two masks (nonzero = foreground)."""
    A = a > 0
    B = b > 0
    union = int(np.logical_or(A, B).sum())
    if union == 0:
        return 0.0
    return float(np.logical_and(A, B).sum()) / union
