"""
rpimocap.gui.blender_export
==========================
Turn the rpimocap calibration + skeleton into a **Blender scene specification**
so the arena, the two calibrated cameras (with the frames as camera
backgrounds), and a rat armature with IK can be built inside Blender for manual
pose fitting.

The hard part — and the part verified here — is decomposing each DLT projection
matrix ``P`` (arena-mm → pixel) into Blender camera parameters (location,
orientation, lens, sensor shift) that reproduce the same projection, so that
looking through a camera the 3D world lines up with its background frame.

This module runs in the *pipeline* environment (needs numpy + rpimocap). It
writes a plain-JSON spec that ``tools/blender_build_scene.py`` reads inside
Blender (which only needs numpy + json), keeping Blender free of rpimocap's
dependencies.
"""
from __future__ import annotations

import json
import os

import numpy as np


def rq3(M):
    """RQ decomposition of a 3x3 matrix: ``M = R @ Q`` with R upper-triangular
    (positive diagonal) and Q orthogonal."""
    P = np.fliplr(np.eye(3))
    Mp = P @ M
    Q_, R_ = np.linalg.qr(Mp.T)
    R = P @ R_.T @ P
    Q = P @ Q_.T
    s = np.sign(np.diag(R))
    s[s == 0] = 1.0
    D = np.diag(s)
    return R @ D, D @ Q


def decompose_dlt(P):
    """Decompose a 3x4 DLT matrix into ``(K, R, t, C)``.

    ``K`` intrinsics (3x3, K[2,2]=1), ``R`` world→camera rotation, ``t`` world→
    camera translation, ``C`` camera centre in world coords. Reproduces the
    projection exactly: ``P @ [X;1] ∝ K (R X + t)``.
    """
    P = np.asarray(P, float)
    _, _, Vt = np.linalg.svd(P)
    Ch = Vt[-1]
    C = Ch[:3] / Ch[3]                       # camera centre = null(P)
    K, R = rq3(P[:, :3])
    K = K / K[2, 2]
    if np.linalg.det(R) < 0:
        R = -R
    t = -R @ C
    if (R @ np.array([0.0, 0.0, 194.0]) + t)[2] < 0:   # keep scene in front
        R, t = -R, -t
    return K, R, t, C


def dlt_to_blender_camera(P, width, height, sensor_width=36.0):
    """Blender camera parameters reproducing DLT ``P`` for a ``width`` x
    ``height`` image.

    Returns a dict with ``location``, ``rotation_c2w`` (3x3 camera-to-world,
    already flipped from CV to Blender's -Z-forward/+Y-up axes), ``lens`` (mm),
    ``sensor_width``, ``sensor_fit``, ``shift_x``/``shift_y`` (sensor shifts for
    the principal point), and pixel aspect for non-square pixels.
    """
    K, R, t, C = decompose_dlt(P)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    lens = fx * sensor_width / width
    R_c2w = R.T @ np.diag([1.0, -1.0, -1.0])    # CV cam → Blender cam axes
    # principal-point shift, in units of the sensor-fit (horizontal) dimension
    shift_x = (width / 2.0 - cx) / width
    shift_y = (cy - height / 2.0) / width
    return {
        "location": C.tolist(),
        "rotation_c2w": R_c2w.tolist(),
        "lens": float(lens),
        "sensor_width": float(sensor_width),
        "sensor_fit": "HORIZONTAL",
        "shift_x": float(shift_x),
        "shift_y": float(shift_y),
        "pixel_aspect_x": 1.0,
        "pixel_aspect_y": float(fx / fy),
        "K": K.tolist(),
    }


def skeleton_spec():
    """Rat23 rest skeleton for building the Blender armature: rest joint
    positions, bones (each named by its child joint), IK chains for the four
    limbs, and per-joint angle limits."""
    from ..model.rat_skeleton import (JOINT_LIMITS, RAT23_BONES, RAT23_JOINTS,
                                       RatPose, forward_kinematics)
    kp = forward_kinematics(RatPose())
    return {
        "joints": list(RAT23_JOINTS),
        "rest": {n: kp[i].tolist() for i, n in enumerate(RAT23_JOINTS)},
        "bones": [[p, c] for (p, c) in RAT23_BONES],
        "ik_chains": [
            {"tip": "HandL", "length": 3, "target": "IK_HandL"},
            {"tip": "HandR", "length": 3, "target": "IK_HandR"},
            {"tip": "FootL", "length": 3, "target": "IK_FootL"},
            {"tip": "FootR", "length": 3, "target": "IK_FootR"},
        ],
        "limits": {k: [list(r) for r in v] for k, v in JOINT_LIMITS.items()},
    }


ARENA_CORNERS = [[-140, -215, 0], [140, -215, 0], [140, 215, 0], [-140, 215, 0],
                 [-140, -215, 388], [140, -215, 388], [140, 215, 388],
                 [-140, 215, 388]]
ARENA_EDGES = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4],
               [0, 4], [1, 5], [2, 6], [3, 7]]


def _write_obj(verts, faces, path):
    """Write a plain OBJ (vertices + triangle faces)."""
    with open(path, "w") as fh:
        for v in verts:
            fh.write(f"v {v[0]:.5f} {v[1]:.5f} {v[2]:.5f}\n")
        for f in faces:
            fh.write(f"f {f[0] + 1} {f[1] + 1} {f[2] + 1}\n")


def build_scene_spec(calib_path, cam0_image, cam1_image, out_path,
                     obj_path=None, width=None, height=None):
    """Assemble and write the Blender scene-spec JSON.

    ``calib_path`` is the ``calib_from_corners.npz`` (dlt_P0/dlt_P1); the two
    image paths are the frames to show as camera backgrounds; ``obj_path`` is an
    optional rat OBJ that is aligned to the skeleton (via
    ``mesh_model.load_obj_mesh``) and re-written next to the spec as
    ``aligned_rat.obj`` so Blender imports it already positioned. Image
    resolution is read from the frame if not given.
    """
    cal = np.load(calib_path)
    Ps = [cal["dlt_P0"], cal["dlt_P1"]]
    if width is None or height is None:
        import cv2
        g = cv2.imread(cam0_image)
        height, width = g.shape[:2]

    aligned_obj = None
    if obj_path:
        from ..model.mesh_model import load_obj_mesh
        m = load_obj_mesh(obj_path, trim_tail=False)     # aligned to the skeleton
        aligned_obj = os.path.join(os.path.dirname(os.path.abspath(out_path)),
                                   "aligned_rat.obj")
        _write_obj(m.verts_rest, m.faces, aligned_obj)

    spec = {
        "resolution": [int(width), int(height)],
        "arena": {"corners": ARENA_CORNERS, "edges": ARENA_EDGES},
        "cameras": [
            {"name": "cam0", "image": os.path.abspath(cam0_image),
             **dlt_to_blender_camera(Ps[0], width, height)},
            {"name": "cam1", "image": os.path.abspath(cam1_image),
             **dlt_to_blender_camera(Ps[1], width, height)},
        ],
        "skeleton": skeleton_spec(),
        "obj": aligned_obj,
    }
    with open(out_path, "w") as fh:
        json.dump(spec, fh, indent=2)
    return spec
