"""
rpimocap.reconstruction.arena_gate
==================================
Static-scene geometric gate (ROADMAP Phase 1.3; the ARCore/DepthLab idea
from the parallelization/AR research): reject a detected/triangulated 3-D
point that is geometrically impossible for the rat — outside the arena
volume, or below the bedding floor (a reflection in the acrylic), or
behind the known static scene from a camera's view.

Two tiers, cheap → richer:

  1. volume + floor gate (always available): a point must lie inside the
     arena box and at/above the floor (z ≥ -tol). A reflection of the rat
     in the floor triangulates to z < 0 and is killed here for free.

  2. dense static-depth gate (optional, one-time): precompute the depth of
     the empty arena's floor plane at every pixel of each camera (the
     "dense stereo depth of the static scene" ARCore builds from a moving
     camera — we have the easier fixed-calibrated version analytically).
     A candidate is rejected if it triangulates BEHIND that surface from
     either view (depth greater than the floor depth + tol), i.e. it would
     be underneath/behind the static geometry.

Pure geometry, CPU, validated against the synthetic-pose ground truth.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


STD_ARENA = (-140.0, 140.0, -215.0, 215.0, 0.0, 388.0)


# ────────────────────────────────────────────────────────────────────
#  Tier 1: arena volume + floor
# ────────────────────────────────────────────────────────────────────


def in_arena_volume(X: np.ndarray, bounds: tuple = STD_ARENA,
                    pad_mm: float = 30.0) -> bool:
    """True iff X lies within the arena box (+ pad)."""
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    return bool(
        xmin - pad_mm <= X[0] <= xmax + pad_mm
        and ymin - pad_mm <= X[1] <= ymax + pad_mm
        and zmin - pad_mm <= X[2] <= zmax + pad_mm)


def above_floor(X: np.ndarray, floor_z: float = 0.0,
                tol_mm: float = 20.0) -> bool:
    """True iff X is at/above the bedding floor (z ≥ floor_z − tol).
    A reflection of the rat in the floor triangulates BELOW it (z < 0)
    and fails this gate."""
    return bool(X[2] >= floor_z - tol_mm)


def accept_point(X: np.ndarray, bounds: tuple = STD_ARENA,
                 floor_z: float = 0.0, pad_mm: float = 30.0,
                 floor_tol_mm: float = 20.0) -> bool:
    """Tier-1 gate: inside the arena volume AND above the floor."""
    return (in_arena_volume(X, bounds, pad_mm)
            and above_floor(X, floor_z, floor_tol_mm))


# ────────────────────────────────────────────────────────────────────
#  Tier 2: dense static-scene depth
# ────────────────────────────────────────────────────────────────────


def _camera_center(P: np.ndarray) -> np.ndarray:
    M = P[:, :3]
    return -np.linalg.solve(M, P[:, 3])


def _point_depth(P: np.ndarray, X: np.ndarray) -> float:
    """Depth of X along the camera's principal ray = the (positive)
    homogeneous w after projection, normalised by the scale of the
    third projection row — i.e. distance along the optical axis."""
    Xh = np.append(X[:3], 1.0)
    p = P @ Xh
    # P's third row r3 has ||r3[:3]|| = scale; depth = (r3·Xh)/scale
    scale = np.linalg.norm(P[2, :3]) + 1e-12
    return float(p[2] / scale)


@dataclass
class StaticDepthGate:
    """A precomputed per-camera depth map of the static arena floor
    plane, for rejecting points behind the static scene.

    Built once from the calibration + the floor plane (z = floor_z). For
    each pixel, intersect the back-projected ray with the floor plane and
    store the depth of that intersection; a candidate point is rejected
    if its depth at the pixel it projects to exceeds the floor depth +
    tol (it would lie beneath the floor surface seen there).
    """
    depth_maps: dict           # cam_id -> (H, W) float32 floor depth
    image_size: tuple          # (W, H)
    floor_z:    float = 0.0
    tol_mm:     float = 25.0

    def accept(self, X: np.ndarray, cameras: dict) -> bool:
        """True iff X is NOT behind the static floor in any camera."""
        W, H = self.image_size
        for cam_id, P in cameras.items():
            if cam_id not in self.depth_maps:
                continue
            Xh = np.append(X[:3], 1.0)
            p = P @ Xh
            if p[2] <= 1e-9:
                return False           # behind the camera
            u, v = p[0] / p[2], p[1] / p[2]
            ui, vi = int(round(u)), int(round(v))
            if not (0 <= ui < W and 0 <= vi < H):
                continue               # off-frame in this view; skip
            floor_depth = self.depth_maps[cam_id][vi, ui]
            if not np.isfinite(floor_depth):
                continue
            if _point_depth(P, X) > floor_depth + self.tol_mm:
                return False           # behind the floor here
        return True


def build_static_depth_gate(
        cameras: dict, image_size: tuple,
        floor_z: float = 0.0, tol_mm: float = 25.0,
        stride: int = 4) -> StaticDepthGate:
    """Precompute the floor-plane depth map for each camera (the one-time
    dense static-scene depth). Back-projects each pixel ray and
    intersects it with z = floor_z; stores the depth of that
    intersection (inf where the ray is parallel to / above the plane)."""
    W, H = image_size
    n = np.array([0.0, 0.0, 1.0])         # floor normal
    p0 = np.array([0.0, 0.0, floor_z])
    maps = {}
    for cam_id, P in cameras.items():
        C = _camera_center(np.asarray(P))
        Minv = np.linalg.inv(np.asarray(P)[:, :3])
        ys = np.arange(0, H, stride)
        xs = np.arange(0, W, stride)
        gx, gy = np.meshgrid(xs, ys)
        pix = np.stack([gx.ravel(), gy.ravel(),
                        np.ones(gx.size)], axis=0)        # (3, N)
        d = Minv @ pix                                    # ray dirs (3,N)
        d = d / (np.linalg.norm(d, axis=0, keepdims=True) + 1e-12)
        denom = n @ d                                     # (N,)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = ((p0 - C) @ n) / denom                    # ray param
        Xs = C[:, None] + t[None, :] * d                  # (3, N)
        # depth along optical axis
        scale = np.linalg.norm(np.asarray(P)[2, :3]) + 1e-12
        depth = (np.asarray(P)[2, :3] @ Xs
                 + np.asarray(P)[2, 3]) / scale
        depth = np.where((t > 0) & np.isfinite(t), depth, np.inf)
        small = depth.reshape(gy.shape).astype(np.float32)
        if stride > 1:
            import cv2
            small = cv2.resize(small, (W, H),
                               interpolation=cv2.INTER_NEAREST)
        maps[cam_id] = small
    return StaticDepthGate(depth_maps=maps, image_size=image_size,
                           floor_z=floor_z, tol_mm=tol_mm)
