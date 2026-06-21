"""
rpimocap.detection.foreshortening
==================================
Angle-dependent texture distortion: tools to separate the GEOMETRIC lens
warp (radial/tangential distortion — a position effect we already
calibrate) from the PHOTOMETRIC footprint effect (perspective
foreshortening — the same physical texture sampled through an elongated
elliptical footprint when a surface tilts away from the sensor normal).

Why this matters for the texture detector
-----------------------------------------
The Gabor descriptor samples texture with a fixed isotropic kernel at
every pixel. But the physical fur each pixel sees is sampled through an
elliptical footprint whose elongation grows with the angle between the
surface normal and the viewing ray (≈ 1/cos θ). So the SAME fur yields
different texture statistics near the frame edge / on the rat's flanks
than near the center / on its back — a systematic bias, separate from
radial distortion.

Film/VFX separates these: rectify the geometric warp into a clean
"undistorted" space (the matchmove undistort→track→redistort pipeline,
baked into a software-agnostic STMap), and model the footprint effect
per-pixel from known geometry (the graphics EWA / anisotropic-footprint
idea). This module provides:

  * build_undistort_stmap()   — bake K/dist into a per-pixel (u,v) remap
                                field (an "STMap"); apply with cv2.remap.
  * apply_stmap()             — resample an image through an STMap.
  * footprint_ellipse_plane() — per-pixel footprint elongation + tilt for
                                a known planar surface (arena floor/wall)
                                seen by a calibrated camera.
  * anisotropy_map()          — the scalar elongation (1/cos θ) map, for
                                weighting / trusting the descriptor less
                                where the surface is steeply foreshortened.

Scope note: this is the geometric machinery + the static-scene
foreshortening map. Adapting the Gabor kernel itself (true EWA) is a
later escalation; building the background model in the same space already
cancels much of the per-pixel bias.
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


# ────────────────────────────────────────────────────────────────────
#  STMap: bake lens distortion into a per-pixel remap field
# ────────────────────────────────────────────────────────────────────


def build_undistort_stmap(
        K:          np.ndarray,
        dist:       np.ndarray,
        image_size: tuple[int, int],
        new_K:      Optional[np.ndarray] = None,
        ) -> np.ndarray:
    """Build an undistortion STMap: for every pixel in the UNDISTORTED
    output, the (x, y) source coordinate to read from the DISTORTED
    input.

    This is the VFX "STMap" — a software-agnostic per-pixel deformation
    field. Bake it once from the calibration, then undistort any frame
    with a single cv2.remap (cheaper than re-evaluating the distortion
    model per frame, and it composes with other warps).

    Parameters
    ----------
    K          : (3,3) camera intrinsics.
    dist       : (k,) OpenCV distortion coefficients.
    image_size : (W, H).
    new_K      : intrinsics for the undistorted output. Defaults to K
                 (same focal/center), so the undistorted image is in the
                 same pixel scale.

    Returns
    -------
    (H, W, 2) float32 STMap; [..., 0] = source x, [..., 1] = source y.
    Pass to apply_stmap (or split into map1/map2 for cv2.remap).
    """
    W, H = image_size
    if new_K is None:
        new_K = K
    map1, map2 = cv2.initUndistortRectifyMap(
        K, dist, None, new_K, (W, H), cv2.CV_32FC1)
    return np.stack([map1, map2], axis=-1).astype(np.float32)


def apply_stmap(img: np.ndarray, stmap: np.ndarray,
                interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
    """Resample `img` through an STMap (the (H,W,2) source-coordinate
    field from build_undistort_stmap). Output pixel p reads input at
    stmap[p]."""
    return cv2.remap(img, stmap[..., 0], stmap[..., 1], interpolation)


def normalize_stmap(stmap: np.ndarray,
                    image_size: tuple[int, int]) -> np.ndarray:
    """Convert a pixel-coordinate STMap to the normalized [0,1] (u,v)
    convention some compositors (Nuke) use, for interchange. y is
    flipped (image origin top-left → uv origin bottom-left)."""
    W, H = image_size
    out = np.empty_like(stmap)
    out[..., 0] = stmap[..., 0] / max(W - 1, 1)
    out[..., 1] = 1.0 - stmap[..., 1] / max(H - 1, 1)
    return out


# ────────────────────────────────────────────────────────────────────
#  Foreshortening: the per-pixel footprint ellipse for a known plane
# ────────────────────────────────────────────────────────────────────


def _camera_center(P: np.ndarray) -> np.ndarray:
    """Camera center C from a (3,4) projection matrix P = K[R|t]:
    C = -R^{-1} t (the null space of P)."""
    M = P[:, :3]
    p4 = P[:, 3]
    C = -np.linalg.solve(M, p4)
    return C


def footprint_anisotropy_plane(
        P:           np.ndarray,
        plane_point: np.ndarray,
        plane_normal: np.ndarray,
        image_size:  tuple[int, int],
        stride:      int = 1,
        ) -> np.ndarray:
    """Per-pixel foreshortening (footprint elongation) for a planar
    surface seen by a calibrated camera.

    For each image pixel, back-project the ray, intersect it with the
    plane, and compute the angle θ between the viewing ray and the plane
    normal. The elliptical pixel footprint on the surface elongates by
    ≈ 1/cos θ along the tilt direction — large where the surface is seen
    edge-on (grazing), 1.0 where seen face-on (normal).

    This is the scalar anisotropy map: high values flag regions where the
    texture is steeply foreshortened and the descriptor should be trusted
    less (or where an anisotropic kernel would differ most from the
    isotropic one).

    Parameters
    ----------
    P            : (3,4) projection matrix for this camera.
    plane_point  : (3,) a point on the surface plane (e.g. arena floor
                   centre, world mm).
    plane_normal : (3,) the plane's unit normal (e.g. [0,0,1] for floor).
    image_size   : (W, H).
    stride       : compute every `stride` pixels and upsample (speed).

    Returns
    -------
    (H, W) float32 map of 1/cos θ (clamped); 1.0 = face-on, large =
    grazing. NaN-safe: rays parallel to / behind the plane → large value.
    """
    W, H = image_size
    n = np.asarray(plane_normal, np.float64)
    n = n / (np.linalg.norm(n) + 1e-12)
    p0 = np.asarray(plane_point, np.float64)
    C = _camera_center(P)
    Minv = np.linalg.inv(P[:, :3])

    ys = np.arange(0, H, stride)
    xs = np.arange(0, W, stride)
    gx, gy = np.meshgrid(xs, ys)
    pix = np.stack([gx.ravel(), gy.ravel(),
                    np.ones(gx.size)], axis=0)        # (3, N)
    # ray directions in world frame: d ∝ Minv @ [x,y,1]
    d = Minv @ pix                                    # (3, N)
    d = d / (np.linalg.norm(d, axis=0, keepdims=True) + 1e-12)
    # angle between ray and plane normal → cos θ = |d·n|
    cos_t = np.abs(n @ d)                              # (N,)
    cos_t = np.clip(cos_t, 1e-3, 1.0)
    aniso = (1.0 / cos_t).astype(np.float32)
    small = aniso.reshape(gy.shape)
    if stride > 1:
        small = cv2.resize(small, (W, H),
                           interpolation=cv2.INTER_LINEAR)
    return small


def anisotropy_weight(aniso: np.ndarray,
                      max_aniso: float = 3.0) -> np.ndarray:
    """Turn a 1/cosθ anisotropy map into a [0,1] confidence weight:
    1 where face-on (aniso≈1), decaying toward 0 as foreshortening
    grows past max_aniso. Use to down-weight texture-distance where the
    surface is steeply foreshortened."""
    a = np.clip(aniso, 1.0, None)
    w = (max_aniso - a) / (max_aniso - 1.0)
    return np.clip(w, 0.0, 1.0).astype(np.float32)
