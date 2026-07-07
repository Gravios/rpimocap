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
from .rat_skeleton import RatPose, forward_kinematics


def _scale_P(P: np.ndarray, s: float) -> np.ndarray:
    """DLT for a 1/s-resolution image: scale the u, v rows by 1/s."""
    Q = P.astype(np.float64).copy()
    Q[0] /= s
    Q[1] /= s
    return Q


def multiview_iou(pose: RatPose, Ps, masks, radii: dict = DEFAULT_RADII) -> float:
    """Mean silhouette IoU of ``pose`` against ``masks`` across cameras."""
    kp = forward_kinematics(pose)
    ious = [silhouette_iou(render_silhouette(kp, P, radii, m.shape), m)
            for P, m in zip(Ps, masks)]
    return float(np.mean(ious)) if ious else 0.0


def fit_pose(observed_masks, Ps, init_pose: RatPose,
             radii: dict = DEFAULT_RADII, joints=None,
             downscale: int = 4, maxiter: int = 300):
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
            p.joint_angles[jn] = tuple(float(v) for v in x[k:k + 3])
            k += 3
        return p

    def loss(x):
        return 1.0 - multiview_iou(unpack(x), Qs, small, radii)

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
