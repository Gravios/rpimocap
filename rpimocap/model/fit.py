"""
rpimocap.model.fit
==================
Fit a rat **pose** to multi-view camera silhouettes by analysis-by-synthesis:
pose the capsule body (`body_model.py`), project it into each calibrated
camera, and optimize the pose so the projected silhouettes best overlap the
observed rat masks (e.g. from `detection.topo_detect`).

The objective is **mean silhouette IoU** across cameras, maximized with a
derivative-free optimizer (Powell) — IoU is not smoothly differentiable
without a differentiable renderer, and the pose is low-dimensional (6-DOF
root + scale, optionally a few joint angles). Because the body's yaw is
ambiguous from a blob, `fit_pose_multistart` tries several initial headings
and keeps the best.

Seed the root position from the triangulated body centroid
(`detect_stereo(...).point`); the heading and scale are found by the fit.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy.optimize import minimize

from .body_model import DEFAULT_RADII, render_silhouette, silhouette_iou
from .rat_skeleton import JOINT_LIMITS, RatPose, forward_kinematics


def _clamp_joint(child: str, angles) -> tuple:
    """Clamp a joint's (rx, ry, rz) to its anatomical JOINT_LIMITS."""
    lims = JOINT_LIMITS.get(child)
    if lims is None:
        return tuple(float(v) for v in angles)
    return tuple(float(np.clip(v, lo, hi)) for v, (lo, hi) in zip(angles, lims))


#: A tucked / curled limb configuration — forelimb and hindlimb hinges
#: strongly flexed so hands and feet fold up toward the body, giving the
#: compact silhouette of a resting or grooming rat. Angles are within
#: JOINT_LIMITS. Use as the joint-angle init for fitting a curled animal.
TUCKED_ANGLES = {
    "ElbowL": (0.0, -1.40, 0.0), "ElbowR": (0.0, -1.40, 0.0),   # fold forearm up
    "WristL": (0.0, -0.60, 0.0), "WristR": (0.0, -0.60, 0.0),
    "KneeL":  (0.0, 1.40, 0.0),  "KneeR":  (0.0, 1.40, 0.0),     # fold shank up
    "AnkleL": (0.0, 0.70, 0.0),  "AnkleR": (0.0, 0.70, 0.0),
}


def curled_pose(root_pos, root_rot=(0.0, 0.0, 0.0), scale: float = 1.0) -> RatPose:
    """A resting/curled starting pose: given root placement, with the limbs
    tucked (:data:`TUCKED_ANGLES`)."""
    return RatPose(root_pos=np.asarray(root_pos, float),
                   root_rot=np.asarray(root_rot, float), scale=float(scale),
                   joint_angles={k: _clamp_joint(k, v)
                                 for k, v in TUCKED_ANGLES.items()})


def _scale_P(P: np.ndarray, s: float) -> np.ndarray:
    """DLT for a 1/s-resolution image: scale the u, v rows by 1/s."""
    Q = P.astype(np.float64).copy()
    Q[0] /= s
    Q[1] /= s
    return Q


def multiview_iou(pose: RatPose, Ps, masks, radii: dict = DEFAULT_RADII,
                  render_fn=None) -> float:
    """Mean silhouette IoU of ``pose`` against ``masks`` across cameras.

    ``render_fn(pose, P, image_shape) -> mask`` overrides the default capsule
    renderer — e.g. a skinned-mesh renderer from :mod:`rpimocap.model.mesh_model`.
    """
    if render_fn is None:
        kp = forward_kinematics(pose)
        ious = [silhouette_iou(render_silhouette(kp, P, radii, m.shape), m)
                for P, m in zip(Ps, masks)]
    else:
        ious = [silhouette_iou(render_fn(pose, P, m.shape), m)
                for P, m in zip(Ps, masks)]
    return float(np.mean(ious)) if ious else 0.0


def fit_pose(observed_masks, Ps, init_pose: RatPose,
             radii: dict = DEFAULT_RADII, joints=None,
             downscale: int = 4, maxiter: int = 300, clamp: bool = True,
             render_fn=None):
    """Optimize a ``RatPose`` to match observed silhouettes across cameras.

    Parameters
    ----------
    observed_masks : list of (H, W) uint8 masks, one per camera.
    Ps             : list of (3, 4) DLT matrices, same order.
    init_pose      : starting guess (root near the triangulated centroid).
    joints         : optional list of joint names to also optimize (each
                     adds 3 Euler angles); default fits only the 6-DOF root
                     + scale.
    downscale      : render/compare at 1/downscale resolution for speed.

    Returns ``(fitted_pose, iou)``.
    """
    joints = list(joints or [])
    H, W = observed_masks[0].shape
    Wd, Hd = max(1, W // downscale), max(1, H // downscale)
    small = [(cv2.resize(m, (Wd, Hd), interpolation=cv2.INTER_AREA) > 0)
             .astype(np.uint8) * 255 for m in observed_masks]
    Qs = [_scale_P(P, downscale) for P in Ps]

    def pack(p):
        x = (list(np.asarray(p.root_pos, float))
             + list(np.asarray(p.root_rot, float))
             + [np.log(max(float(p.scale), 1e-3))])
        for jn in joints:
            x += list(p.joint_angles.get(jn, (0.0, 0.0, 0.0)))
        return np.asarray(x, float)

    def unpack(x):
        p = RatPose(root_pos=np.asarray(x[0:3], float),
                    root_rot=np.asarray(x[3:6], float),
                    scale=float(np.exp(x[6])),
                    joint_angles=dict(init_pose.joint_angles))
        k = 7
        for jn in joints:
            ang = x[k:k + 3]
            p.joint_angles[jn] = (_clamp_joint(jn, ang) if clamp
                                  else tuple(float(v) for v in ang))
            k += 3
        return p

    def loss(x):
        return 1.0 - multiview_iou(unpack(x), Qs, small, radii, render_fn)

    res = minimize(loss, pack(init_pose), method="Powell",
                   options={"maxiter": maxiter, "xtol": 1e-2, "ftol": 1e-3})
    return unpack(res.x), 1.0 - float(res.fun)


def fit_pose_multistart(observed_masks, Ps, root_pos, headings: int = 6,
                        **fit_kwargs):
    """Fit from several initial yaw headings and keep the best.

    The body's heading is ambiguous from a blob silhouette, so a single start
    lands in a local optimum; this sweeps ``headings`` initial yaws about the
    seeded ``root_pos`` (e.g. the triangulated centroid) and returns the best
    ``(pose, iou)``.
    """
    best = None
    for h in np.linspace(0.0, 2.0 * np.pi, int(headings), endpoint=False):
        init = RatPose(root_pos=np.asarray(root_pos, float),
                       root_rot=np.array([0.0, 0.0, float(h)]), scale=1.0)
        pose, iou = fit_pose(observed_masks, Ps, init, **fit_kwargs)
        if best is None or iou > best[1]:
            best = (pose, iou)
    return best


def fit_pose_staged(observed_masks, Ps, root_pos, headings: int = 6,
                    tucked: bool = True,
                    stages=(("SpineF", "SpineL"),
                            ("ElbowL", "ElbowR", "KneeL", "KneeR")),
                    radii: dict = DEFAULT_RADII, downscale: int = 4,
                    maxiter: int = 200, render_fn=None):
    """Coarse-to-fine articulated fit.

    1. Root + scale, sweeping initial headings (:func:`fit_pose_multistart`).
    2. Optionally tuck the limbs (:data:`TUCKED_ANGLES`) so a *curled* rat is
       the starting shape rather than the splayed rest pose.
    3. Progressively add joint groups from ``stages`` (accumulating), each a
       :func:`fit_pose` refinement with joint-limit clamping.

    Fitting all joints at once is high-dimensional and gets stuck; adding
    them in stages, from a good root and a tucked init, is far more reliable.
    ``stages`` defaults to the spine, then the forelimb/hindlimb hinges.

    Returns ``(pose, iou)``.
    """
    pose, iou = fit_pose_multistart(observed_masks, Ps, root_pos,
                                    headings=headings, radii=radii,
                                    downscale=downscale, maxiter=maxiter,
                                    render_fn=render_fn)
    if tucked:
        pose = RatPose(root_pos=pose.root_pos, root_rot=pose.root_rot,
                       scale=pose.scale,
                       joint_angles={k: _clamp_joint(k, v)
                                     for k, v in TUCKED_ANGLES.items()})
    joints = []
    for stage in stages:
        for j in stage:
            if j not in joints:
                joints.append(j)
        pose, iou = fit_pose(observed_masks, Ps, pose, radii=radii,
                             joints=joints, downscale=downscale,
                             maxiter=maxiter, render_fn=render_fn)
    return pose, iou
