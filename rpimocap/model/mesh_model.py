"""
rpimocap.model.mesh_model
=========================
A smooth, skinned rat **surface mesh** wrapped on the rat23 skeleton — a
better-looking body than the tapered-capsule union of ``body_model``, with
proper **joint deformation** (linear blend skinning).

Construction (grounded in rat side-profile references — a rounded body that
tapers to a pointed snout, small ears, thin tucked limbs, a thick-to-thin
tail):

1. **Implicit surface.** Each bone contributes a tapered-capsule signed
   distance; a smooth-minimum blends them (metaball style) so the trunk,
   head, and limbs merge into one organic surface with no capsule seams.
   Marching cubes extracts a watertight triangle mesh in the rest pose.
2. **Skinning.** Each vertex is bound to its nearest bones by distance, with
   an inverse-square falloff, so vertices near a joint blend the two bones'
   frames — giving smooth bending instead of a rigid crease.
3. **Posing.** ``skin_mesh`` linear-blend-skins the rest mesh to any
   ``RatPose``; ``render_mesh_silhouette`` projects and rasterizes it per
   camera for the same analysis-by-synthesis fitting as the capsule model.

Build the mesh once (marching cubes + weights is the expensive part); skin and
render per pose are fast.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .body_model import DEFAULT_RADII
from .rat_skeleton import (RAT23_BONES, RAT23_INDEX, RatPose,
                           forward_kinematics, forward_kinematics_transforms,
                           project_pose)


def _capsule_sdf(pts, p0, p1, r0, r1):
    """Approx signed distance to a tapered capsule (negative inside)."""
    d = p1 - p0
    L2 = float(d @ d)
    if L2 < 1e-9:
        return np.linalg.norm(pts - p0, axis=1) - max(r0, r1)
    t = np.clip((pts - p0) @ d / L2, 0.0, 1.0)
    proj = p0[None, :] + t[:, None] * d[None, :]
    r = r0 + t * (r1 - r0)
    return np.linalg.norm(pts - proj, axis=1) - r


def _seg_dist(pts, p0, p1):
    """Distance from points to a segment (for skin weights)."""
    d = p1 - p0
    L2 = float(d @ d)
    if L2 < 1e-9:
        return np.linalg.norm(pts - p0, axis=1)
    t = np.clip((pts - p0) @ d / L2, 0.0, 1.0)
    proj = p0[None, :] + t[:, None] * d[None, :]
    return np.linalg.norm(pts - proj, axis=1)


@dataclass
class RatMesh:
    verts_rest: np.ndarray      # (V, 3) rest-pose world vertices (mm)
    faces: np.ndarray           # (F, 3) triangle vertex indices
    weights: np.ndarray         # (V, B) skinning weights over bones
    bone_parents: list          # parent-joint name per bone (len B)
    _rest_R: dict               # rest world rotation per joint
    _rest_t: dict               # rest world position per joint


def build_rat_mesh(radii: dict = DEFAULT_RADII, voxel: float = 2.5,
                   margin: float = 28.0, smooth_k: float = 0.18,
                   weight_bones: int = 4) -> RatMesh:
    """Build the skinned rat mesh from the rat23 rest skeleton.

    ``voxel`` is the marching-cubes grid spacing (mm); ``smooth_k`` sets the
    metaball blend (larger = sharper joints); ``radii`` are the per-bone
    ``(parent, child)`` thicknesses (shared with the capsule model).
    """
    from skimage.measure import marching_cubes

    kp = forward_kinematics(RatPose())
    segs = [(kp[RAT23_INDEX[p]], kp[RAT23_INDEX[c]], radii.get((p, c), (5.0, 5.0)))
            for (p, c) in RAT23_BONES]
    lo = kp.min(0) - margin
    hi = kp.max(0) + margin
    xs = np.arange(lo[0], hi[0] + voxel, voxel)
    ys = np.arange(lo[1], hi[1] + voxel, voxel)
    zs = np.arange(lo[2], hi[2] + voxel, voxel)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)

    # smooth-minimum of per-bone SDFs (metaball union)
    sdfs = np.stack([_capsule_sdf(pts, p0, p1, r0, r1)
                     for (p0, p1, (r0, r1)) in segs])          # (B, M)
    field = (-np.log(np.exp(-smooth_k * sdfs).sum(0)) / smooth_k).reshape(X.shape)
    verts_vox, faces, _, _ = marching_cubes(field, level=0.0)
    verts = lo[None, :] + verts_vox * voxel                     # → world mm

    # skin weights: nearest bones by distance to segment, inverse-square
    D = np.stack([_seg_dist(verts, p0, p1) for (p0, p1, _) in segs], axis=1)  # (V,B)
    k = min(int(weight_bones), D.shape[1])
    order = np.argsort(D, axis=1)[:, :k]
    dk = np.take_along_axis(D, order, axis=1)
    wk = 1.0 / (dk ** 2 + 1.0)
    wk /= wk.sum(1, keepdims=True)
    W = np.zeros_like(D)
    np.put_along_axis(W, order, wk, axis=1)

    bone_parents = [p for (p, c) in RAT23_BONES]   # bone rides its parent frame
    rest_R, rest_t = forward_kinematics_transforms(RatPose())
    return RatMesh(verts.astype(np.float64), faces.astype(np.int32),
                   W.astype(np.float64), bone_parents, rest_R, rest_t)


def skin_mesh(mesh: RatMesh, pose: RatPose) -> np.ndarray:
    """Linear-blend-skin the rest mesh to ``pose`` → (V, 3) posed vertices."""
    Rp, tp = forward_kinematics_transforms(pose)
    V = mesh.verts_rest
    s = float(pose.scale)
    out = np.zeros_like(V)
    for b, parent in enumerate(mesh.bone_parents):
        w = mesh.weights[:, b]
        if not w.any():
            continue
        Rr = mesh._rest_R[parent]
        tr = mesh._rest_t[parent]
        M = Rp[parent] @ Rr.T                         # rest→posed rotation
        vb = (s * (V - tr)) @ M.T + tp[parent]
        out += w[:, None] * vb
    return out


def render_mesh_silhouette(verts_posed: np.ndarray, faces: np.ndarray,
                           P: np.ndarray, image_shape=(1080, 2028)) -> np.ndarray:
    """Project posed vertices and rasterize all triangles → silhouette mask."""
    px = project_pose(verts_posed, P)
    H, W = int(image_shape[0]), int(image_shape[1])
    mask = np.zeros((H, W), np.uint8)
    ok = (px[:, 0] > -1e8)[faces].all(axis=1)
    if not ok.any():
        return mask
    tris = np.round(px[faces[ok]]).astype(np.int32)   # (F', 3, 2), vectorized
    cv2.fillPoly(mask, tris, 255)
    return mask


def render_mesh_pose_silhouette(mesh: RatMesh, pose: RatPose, P: np.ndarray,
                                image_shape=(1080, 2028)) -> np.ndarray:
    """Convenience: skin ``mesh`` to ``pose`` then render its silhouette."""
    return render_mesh_silhouette(skin_mesh(mesh, pose), mesh.faces, P,
                                  image_shape)


def _parse_obj(path):
    """Minimal OBJ reader → (verts (V,3), tri faces (F,3)). Triangulates
    polygons by fanning; handles 1-based and negative face indices."""
    verts, faces = [], []
    with open(path, "r", errors="ignore") as fh:
        for line in fh:
            if line.startswith("v "):
                verts.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                idx = [int(p.split("/")[0]) for p in line.split()[1:]]
                idx = [i - 1 if i > 0 else len(verts) + i for i in idx]
                for k in range(1, len(idx) - 1):
                    faces.append([idx[0], idx[k], idx[k + 1]])
    return np.asarray(verts, np.float64), np.asarray(faces, np.int32)


def load_obj_mesh(obj_path: str, radii: dict = DEFAULT_RADII,
                  weight_bones: int = 4, forward_axis: int = 2,
                  up_axis: int = 1, forward_sign: float = 1.0,
                  scale_mult: float = 1.2, nose_dx: float = 5.0,
                  feet_dz: float = 5.0, trim_tail: bool = True,
                  tail_margin: float = 5.0, decimate=None) -> RatMesh:
    """Load an external artist OBJ rat mesh, align it to the rat23 rest
    skeleton, and bind it (linear-blend skinning) for the pipeline.

    The mesh is rotated into the rat convention (+x forward, +y left, +z up),
    scaled so its height matches the skeleton (times ``scale_mult``), and
    translated so the nose aligns with the snout and the feet with the floor.
    Vertices are then bound to the nearest bones (inverse-square falloff), so
    the returned :class:`RatMesh` skins and renders exactly like the built-in
    one — pass it to the fitter via ``render_mesh_pose_silhouette``.

    Defaults suit a Y-up, +Z-forward model (head at +Z), such as the sample
    artist asset. ``forward_axis``/``up_axis``/``forward_sign`` and the
    alignment offsets can be adjusted for a differently-oriented model.
    ``trim_tail`` drops the thin tail behind the tail base (the detector masks
    exclude it); ``decimate`` (fraction of faces to remove, e.g. 0.85; needs
    ``fast-simplification``) reduces a heavy mesh so fitting stays fast.

    Note: the trunk and hindquarters bind well; a sculpted forelimb pose that
    differs from the skeleton's straight-down rest binds only approximately,
    so use root/scale (+ spine) fitting rather than relying on forelimb
    articulation.
    """
    V, F = _parse_obj(obj_path)
    if decimate is not None:                          # optional face reduction
        import fast_simplification              # (needs `pip install fast-simplification`)
        V, F = fast_simplification.simplify(
            V.astype(np.float32), F.astype(np.int32), target_reduction=float(decimate))
        V = V.astype(np.float64)
        F = F.astype(np.int32)
    kp = forward_kinematics(RatPose())
    side_axis = 3 - forward_axis - up_axis
    Vr = np.empty_like(V)
    Vr[:, 0] = forward_sign * V[:, forward_axis]     # +x forward
    Vr[:, 1] = V[:, side_axis]                        # +y left
    Vr[:, 2] = V[:, up_axis]                          # +z up
    scale = (np.ptp(kp[:, 2]) / np.ptp(Vr[:, 2])) * scale_mult
    Vr *= scale
    Vr[:, 2] += kp[:, 2].min() - Vr[:, 2].min() + feet_dz          # feet to floor
    Vr[:, 1] += kp[:, 1].mean() - Vr[:, 1].mean()                  # centre width
    Vr[:, 0] += kp[:, 0].max() - np.percentile(Vr[:, 0], 98) + nose_dx  # nose→snout

    if trim_tail:                                     # drop the thin tail behind
        tb_x = kp[RAT23_INDEX["TailBase"], 0]         # the tail base — the
        fc = Vr[F].mean(axis=1)[:, 0]                 # detector masks exclude it
        F = F[fc >= tb_x - tail_margin]

    segs = [(kp[RAT23_INDEX[p]], kp[RAT23_INDEX[c]]) for (p, c) in RAT23_BONES]
    D = np.stack([_seg_dist(Vr, p0, p1) for (p0, p1) in segs], axis=1)
    k = min(int(weight_bones), D.shape[1])
    order = np.argsort(D, axis=1)[:, :k]
    dk = np.take_along_axis(D, order, axis=1)
    wk = 1.0 / (dk ** 2 + 1.0)
    wk /= wk.sum(1, keepdims=True)
    W = np.zeros_like(D)
    np.put_along_axis(W, order, wk, axis=1)
    rest_R, rest_t = forward_kinematics_transforms(RatPose())
    return RatMesh(Vr, F.astype(np.int32), W.astype(np.float64),
                   [p for (p, c) in RAT23_BONES], rest_R, rest_t)
