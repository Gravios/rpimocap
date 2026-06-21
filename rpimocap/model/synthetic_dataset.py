"""
rpimocap.model.synthetic_dataset
================================
Phase A of the synthetic-pose export (see SYNTHETIC_POSE_EXPORT_DESIGN.md).

Turns the kinematic rat skeleton (rat_skeleton.py) into a labeled-data
source shared by three consumers: validation (detect→triangulate against
exact ground truth), the shape prior (valid-pose silhouette manifold),
and native-3D training (DANNCE-style 3-D heatmap regression).

Core design decision: STORE the compact parametric pose + keypoint data;
RASTERIZE the heavy artifacts (silhouettes, occupancy volumes) on demand
from a deterministic body model. A 23-keypoint heatmap volume is ~24 MB;
storing them for 10^5 poses would be terabytes, so we store the ~few-
hundred-float pose instead and regenerate.

This module (Phase A) provides:
  * RatBodyModel  — soft tissue around the skeleton as a union of tapered
                    capsules (+ bulk spheres); the SAME radii drive both
                    the 2-D silhouette and the 3-D occupancy, so they
                    agree by construction.
  * SyntheticPoseSample / SyntheticPoseDataset — the data model + manifest
                    save/load.
  * generate_dataset — sample N valid poses, project to each camera,
                    record visibility; deterministic per-pose seeding so
                    generation parallelizes across cores with no change to
                    the output.

Heatmap volumes + torch adapters are Phase B; silhouette cache +
self-occlusion + the multiprocessing pool tuning are Phase C.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from rpimocap.model import rat_skeleton as rs
from rpimocap.reconstruction.voxel import VoxelGrid


# Stable joint ordering for serialization (matches JOINT_LIMITS order).
JOINT_ORDER = list(rs.JOINT_LIMITS.keys())

STD_ARENA = (-140.0, 140.0, -215.0, 215.0, 0.0, 388.0)


# ────────────────────────────────────────────────────────────────────
#  Body model: soft tissue as tapered capsules + bulk spheres
# ────────────────────────────────────────────────────────────────────


@dataclass
class Capsule:
    """A tapered capsule (truncated cone with spherical caps) between two
    keypoints. r_a/r_b are the radii at a/b (mm). a==b → a sphere."""
    a:   np.ndarray
    b:   np.ndarray
    r_a: float
    r_b: float


# Default per-bone radii (mm) at (parent, child). Tapering gives the tail
# its point and the limbs their thinning. Seed values — refine from the
# texture/edge sampler's measured body width later.
_DEFAULT_BONE_RADII = {
    ("SpineM", "SpineF"):   (22.0, 18.0),   # trunk, fat
    ("SpineM", "SpineL"):   (22.0, 18.0),
    ("SpineL", "TailBase"): (12.0, 8.0),    # tail base, tapering
    ("SpineF", "Snout"):    (14.0, 8.0),    # head → nose
    ("SpineF", "EarL"):     (8.0, 5.0),
    ("SpineF", "EarR"):     (8.0, 5.0),
    ("SpineF", "ShoulderL"):(14.0, 10.0),
    ("SpineF", "ShoulderR"):(14.0, 10.0),
    ("ShoulderL", "ElbowL"):(8.0, 7.0),     # upper arm
    ("ElbowL", "WristL"):   (6.0, 5.0),     # forearm
    ("WristL", "HandL"):    (5.0, 4.0),
    ("ShoulderR", "ElbowR"):(8.0, 7.0),
    ("ElbowR", "WristR"):   (6.0, 5.0),
    ("WristR", "HandR"):    (5.0, 4.0),
    ("SpineL", "HipL"):     (15.0, 11.0),
    ("HipL", "KneeL"):      (10.0, 8.0),    # thigh
    ("KneeL", "AnkleL"):    (7.0, 5.0),     # shank
    ("AnkleL", "FootL"):    (5.0, 4.0),
    ("SpineL", "HipR"):     (15.0, 11.0),
    ("HipR", "KneeR"):      (10.0, 8.0),
    ("KneeR", "AnkleR"):    (7.0, 5.0),
    ("AnkleR", "FootR"):    (5.0, 4.0),
}

# Extra bulk spheres at trunk joints (the body is a fat ellipsoid around
# the spine, not a thin tube).
_DEFAULT_SPHERE_RADII = {
    "SpineM": 24.0, "SpineF": 18.0, "SpineL": 20.0,
    "Snout": 9.0, "TailBase": 8.0,
}


@dataclass
class RatBodyModel:
    bone_radii:   dict = field(
        default_factory=lambda: dict(_DEFAULT_BONE_RADII))
    sphere_radii: dict = field(
        default_factory=lambda: dict(_DEFAULT_SPHERE_RADII))
    version:      str = "body-v1"

    @classmethod
    def default(cls) -> "RatBodyModel":
        return cls()

    # ── geometry ────────────────────────────────────────────────────

    def capsules(self, kpts3d: np.ndarray) -> list[Capsule]:
        """Build the capsule + sphere list for a posed skeleton."""
        caps: list[Capsule] = []
        for (p, c), (ra, rb) in self.bone_radii.items():
            a = kpts3d[rs.RAT23_INDEX[p]]
            b = kpts3d[rs.RAT23_INDEX[c]]
            caps.append(Capsule(a.astype(np.float64),
                                b.astype(np.float64), ra, rb))
        for name, r in self.sphere_radii.items():
            v = kpts3d[rs.RAT23_INDEX[name]].astype(np.float64)
            caps.append(Capsule(v, v, r, r))      # sphere
        return caps

    # ── 2-D silhouette ──────────────────────────────────────────────

    def silhouette(self, kpts3d: np.ndarray, P: np.ndarray,
                   image_size: tuple[int, int]) -> np.ndarray:
        """Rasterize the projected body silhouette (uint8 0/255) into an
        (H, W) mask for camera P. Each capsule projects to a filled
        tapered-capsule polygon; their union is the silhouette."""
        W, H = image_size
        mask = np.zeros((H, W), np.uint8)
        C = _camera_center(P)
        for cap in self.capsules(kpts3d):
            _draw_capsule_2d(mask, cap, P, C)
        return mask

    # ── 3-D occupancy ───────────────────────────────────────────────

    def occupancy(self, kpts3d: np.ndarray,
                  grid: Optional[VoxelGrid] = None,
                  voxel_size: float = 4.0,
                  pad_mm: float = 15.0) -> VoxelGrid:
        """Rasterize the body into a 3-D occupancy VoxelGrid (visual
        hull). If grid is None, a grid is built around the pose with the
        given voxel_size and padding."""
        if grid is None:
            grid = _grid_around(kpts3d, voxel_size, pad_mm,
                                self._max_radius())
        occ = np.zeros(grid.shape, dtype=bool)
        nx, ny, nz = grid.shape
        vs = grid.voxel_size
        org = grid.origin
        for cap in self.capsules(kpts3d):
            _fill_capsule_3d(occ, cap, org, vs, (nx, ny, nz))
        return VoxelGrid(origin=grid.origin, voxel_size=grid.voxel_size,
                         shape=grid.shape, occupancy=occ)

    def surface_points(self, kpts3d: np.ndarray, n: int,
                       rng: np.random.RandomState) -> np.ndarray:
        """Sample ~n points on the body surface (for point-based shape
        work). Samples capsules proportional to their lateral area."""
        caps = self.capsules(kpts3d)
        # weight by approximate lateral area
        areas = []
        for cap in caps:
            L = float(np.linalg.norm(cap.b - cap.a))
            rm = 0.5 * (cap.r_a + cap.r_b)
            areas.append(max(2 * np.pi * rm * L, np.pi * rm * rm))
        areas = np.array(areas)
        probs = areas / areas.sum()
        counts = rng.multinomial(n, probs)
        pts = []
        for cap, k in zip(caps, counts):
            if k == 0:
                continue
            pts.append(_sample_capsule_surface(cap, k, rng))
        return np.concatenate(pts, axis=0) if pts else np.zeros((0, 3))

    # ── helpers ─────────────────────────────────────────────────────

    def _max_radius(self) -> float:
        rmax = max((max(r) for r in self.bone_radii.values()), default=0)
        return max(rmax, max(self.sphere_radii.values(), default=0))

    def radii_snapshot(self) -> dict:
        """JSON-serializable snapshot for the dataset manifest header."""
        return {
            "version": self.version,
            "bone_radii": {f"{p}->{c}": list(v)
                           for (p, c), v in self.bone_radii.items()},
            "sphere_radii": dict(self.sphere_radii),
        }


# ────────────────────────────────────────────────────────────────────
#  Rasterization primitives
# ────────────────────────────────────────────────────────────────────


def _camera_center(P: np.ndarray) -> np.ndarray:
    """Camera center C = -M^{-1} p4 from P = [M | p4]."""
    M = P[:, :3]
    return -np.linalg.solve(M, P[:, 3])


def _project(P: np.ndarray, X: np.ndarray) -> Optional[np.ndarray]:
    """Project a single world point; None if behind the camera."""
    Xh = np.append(X, 1.0)
    p = P @ Xh
    if p[2] <= 1e-9:
        return None
    return p[:2] / p[2]


def _projected_radius(P: np.ndarray, C: np.ndarray,
                      X: np.ndarray, r: float) -> float:
    """Projected pixel radius of a sphere of world-radius r at X, via a
    perpendicular offset — robust for any projection matrix (no focal
    assumption)."""
    if r <= 0:
        return 0.0
    d = X - C
    nd = np.linalg.norm(d)
    if nd < 1e-9:
        return 0.0
    d = d / nd
    up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(d, up)) > 0.99:
        up = np.array([1.0, 0.0, 0.0])
    u = np.cross(d, up)
    u = u / (np.linalg.norm(u) + 1e-12)
    p0 = _project(P, X)
    p1 = _project(P, X + r * u)
    if p0 is None or p1 is None:
        return 0.0
    return float(np.linalg.norm(p1 - p0))


def _draw_capsule_2d(mask: np.ndarray, cap: Capsule,
                     P: np.ndarray, C: np.ndarray) -> None:
    """Fill the projected tapered capsule into mask."""
    pa = _project(P, cap.a)
    pb = _project(P, cap.b)
    if pa is None and pb is None:
        return
    ra_px = _projected_radius(P, C, cap.a, cap.r_a)
    rb_px = _projected_radius(P, C, cap.b, cap.r_b)
    # endpoint disks
    if pa is not None and ra_px >= 0.5:
        cv2.circle(mask, (int(round(pa[0])), int(round(pa[1]))),
                   int(round(ra_px)), 255, -1)
    if pb is not None and rb_px >= 0.5:
        cv2.circle(mask, (int(round(pb[0])), int(round(pb[1]))),
                   int(round(rb_px)), 255, -1)
    # connecting trapezoid (outer tangents) when both endpoints valid
    if pa is not None and pb is not None:
        axis = pb - pa
        L = np.linalg.norm(axis)
        if L > 1e-6:
            n = np.array([-axis[1], axis[0]]) / L     # perpendicular
            quad = np.array([
                pa + ra_px * n, pb + rb_px * n,
                pb - rb_px * n, pa - ra_px * n,
            ], dtype=np.int32)
            cv2.fillConvexPoly(mask, quad, 255)


def _fill_capsule_3d(occ: np.ndarray, cap: Capsule,
                     origin: np.ndarray, vs: float,
                     dims: tuple) -> None:
    """OR a capsule's occupancy into `occ`, restricted to its bbox."""
    nx, ny, nz = dims
    a, b = cap.a, cap.b
    rmax = max(cap.r_a, cap.r_b)
    lo = np.minimum(a, b) - rmax
    hi = np.maximum(a, b) + rmax
    # voxel index bbox (inclusive), clipped to grid
    i0 = np.floor((lo - origin) / vs).astype(int)
    i1 = np.ceil((hi - origin) / vs).astype(int)
    i0 = np.maximum(i0, 0)
    i1 = np.minimum(i1, np.array([nx, ny, nz]))
    if np.any(i1 <= i0):
        return
    xs = origin[0] + (np.arange(i0[0], i1[0]) + 0.5) * vs
    ys = origin[1] + (np.arange(i0[1], i1[1]) + 0.5) * vs
    zs = origin[2] + (np.arange(i0[2], i1[2]) + 0.5) * vs
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    c = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
    ab = b - a
    L2 = float(ab @ ab)
    if L2 < 1e-9:                       # sphere
        d = np.linalg.norm(c - a, axis=1)
        inside = d <= cap.r_a
    else:
        t = np.clip(((c - a) @ ab) / L2, 0.0, 1.0)
        closest = a + t[:, None] * ab
        r_at = cap.r_a + t * (cap.r_b - cap.r_a)
        inside = np.linalg.norm(c - closest, axis=1) <= r_at
    sub = inside.reshape(gx.shape)
    occ[i0[0]:i1[0], i0[1]:i1[1], i0[2]:i1[2]] |= sub


def _sample_capsule_surface(cap: Capsule, k: int,
                            rng: np.random.RandomState) -> np.ndarray:
    """Sample k points on a capsule's lateral surface (approx)."""
    ab = cap.b - cap.a
    L = np.linalg.norm(ab)
    if L < 1e-9:                        # sphere
        v = rng.normal(size=(k, 3))
        v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
        return cap.a + cap.r_a * v
    axis = ab / L
    # an orthonormal basis perpendicular to axis
    tmp = np.array([0.0, 0.0, 1.0])
    if abs(axis @ tmp) > 0.99:
        tmp = np.array([1.0, 0.0, 0.0])
    e1 = np.cross(axis, tmp); e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis, e1)
    t = rng.uniform(0, 1, size=k)
    theta = rng.uniform(0, 2 * np.pi, size=k)
    r_at = cap.r_a + t * (cap.r_b - cap.r_a)
    pts = (cap.a[None, :] + t[:, None] * ab[None, :]
           + (r_at[:, None] * (np.cos(theta)[:, None] * e1[None, :]
                               + np.sin(theta)[:, None] * e2[None, :])))
    return pts


def _grid_around(kpts3d: np.ndarray, vs: float, pad: float,
                 rmax: float) -> VoxelGrid:
    """Build an empty VoxelGrid bounding the posed body."""
    lo = kpts3d.min(axis=0) - (pad + rmax)
    hi = kpts3d.max(axis=0) + (pad + rmax)
    dims = np.ceil((hi - lo) / vs).astype(int)
    dims = np.maximum(dims, 1)
    return VoxelGrid(origin=lo.astype(np.float64), voxel_size=float(vs),
                     shape=tuple(int(d) for d in dims),
                     occupancy=np.zeros(tuple(int(d) for d in dims),
                                        dtype=bool))


# ────────────────────────────────────────────────────────────────────
#  Data model
# ────────────────────────────────────────────────────────────────────


@dataclass
class SyntheticPoseSample:
    """One labeled synthetic pose. The `pose` is the compact source of
    truth; keypoints3d/2d are cached derivations. Heavy artifacts
    (silhouette/occupancy/heatmap) are NOT stored — regenerate from
    `pose` + a RatBodyModel via the dataset's on-demand rasterizers."""
    pose:        rs.RatPose
    keypoints3d: np.ndarray                 # (23,3)
    keypoints2d: dict                       # cam_id -> (23,2)
    visibility:  dict                       # cam_id -> (23,) bool
    valid:       bool
    body_version: str = "body-v1"


def _pose_rng(seed: int, i: int) -> np.random.RandomState:
    """Deterministic per-pose RNG so generation is reproducible AND
    order-independent — generating pose i gives the same result no
    matter how many worker processes split the range."""
    state = np.random.SeedSequence([int(seed), int(i)]).generate_state(1)[0]
    return np.random.RandomState(int(state))


def _in_frame(px: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    """Per-keypoint in-frame visibility (and in front of the camera —
    project_pose marks behind-camera points with -1e9)."""
    W, H = image_size
    return ((px[:, 0] >= 0) & (px[:, 0] < W)
            & (px[:, 1] >= 0) & (px[:, 1] < H)).astype(bool)


def make_sample(i: int, cameras: dict, image_size: tuple[int, int],
                seed: int, pose_fraction: float, scale_range: tuple,
                arena_bounds: tuple, require_in_arena: bool,
                body_version: str,
                max_tries: int = 50) -> SyntheticPoseSample:
    """Generate one labeled sample (pure function of i + params)."""
    rng = _pose_rng(seed, i)
    scale = float(rng.uniform(*scale_range))
    pose = None
    for _ in range(max_tries):
        cand = rs.sample_pose(rng, scale=scale, arena_bounds=arena_bounds,
                              fraction=pose_fraction)
        kp = rs.forward_kinematics(cand)
        if (not require_in_arena
                or rs.check_arena_containment(kp, arena_bounds)):
            pose = cand
            break
    if pose is None:
        pose = cand
    kp = rs.forward_kinematics(pose)
    valid = rs.is_valid(pose, arena_bounds=arena_bounds,
                        require_arena=require_in_arena)
    kpts2d, vis = {}, {}
    for cam_id, P in cameras.items():
        px = rs.project_pose(kp, P)
        kpts2d[cam_id] = px
        vis[cam_id] = _in_frame(px, image_size)
    return SyntheticPoseSample(
        pose=pose, keypoints3d=kp, keypoints2d=kpts2d,
        visibility=vis, valid=valid, body_version=body_version)


# ────────────────────────────────────────────────────────────────────
#  Dataset
# ────────────────────────────────────────────────────────────────────


class SyntheticPoseDataset:
    """A collection of labeled synthetic poses + a reproducibility
    manifest. Stores compact data; rasterizes heavy artifacts on
    demand."""

    def __init__(self, samples, meta, body: Optional[RatBodyModel] = None):
        self.samples = list(samples)
        self.meta = dict(meta)
        self.body = body or RatBodyModel.default()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]

    # ── on-demand rasterizers ───────────────────────────────────────

    def silhouette(self, i: int, cam_id: int) -> np.ndarray:
        s = self.samples[i]
        P = np.asarray(self.meta["cameras"][str(cam_id)])
        return self.body.silhouette(
            s.keypoints3d, P, tuple(self.meta["image_size"]))

    def occupancy(self, i: int, voxel_size: float = 4.0) -> VoxelGrid:
        return self.body.occupancy(
            self.samples[i].keypoints3d, voxel_size=voxel_size)

    # ── persistence ─────────────────────────────────────────────────

    def save(self, dataset_dir: str) -> None:
        os.makedirs(dataset_dir, exist_ok=True)
        N = len(self.samples)
        J = len(JOINT_ORDER)
        cam_ids = list(self.meta["cameras"].keys())
        Ncam = len(cam_ids)

        root_pos = np.zeros((N, 3), np.float64)
        root_rot = np.zeros((N, 3), np.float64)
        scale = np.zeros((N,), np.float64)
        joint_angles = np.zeros((N, J, 3), np.float64)
        keypoints3d = np.zeros((N, 23, 3), np.float64)
        keypoints2d = np.zeros((N, Ncam, 23, 2), np.float64)
        visibility = np.zeros((N, Ncam, 23), bool)
        valid = np.zeros((N,), bool)

        for n, s in enumerate(self.samples):
            root_pos[n] = s.pose.root_pos
            root_rot[n] = s.pose.root_rot
            scale[n] = s.pose.scale
            for j, name in enumerate(JOINT_ORDER):
                joint_angles[n, j] = s.pose.joint_angles.get(
                    name, (0.0, 0.0, 0.0))
            keypoints3d[n] = s.keypoints3d
            for ci, cid in enumerate(cam_ids):
                keypoints2d[n, ci] = s.keypoints2d[int(cid)]
                visibility[n, ci] = s.visibility[int(cid)]
            valid[n] = s.valid

        np.savez_compressed(
            os.path.join(dataset_dir, "manifest.npz"),
            root_pos=root_pos, root_rot=root_rot, scale=scale,
            joint_angles=joint_angles, keypoints3d=keypoints3d,
            keypoints2d=keypoints2d, visibility=visibility, valid=valid,
            cam_ids=np.array([int(c) for c in cam_ids]))
        with open(os.path.join(dataset_dir, "meta.json"), "w") as fh:
            json.dump(self.meta, fh, indent=2)

    @classmethod
    def load(cls, dataset_dir: str) -> "SyntheticPoseDataset":
        with open(os.path.join(dataset_dir, "meta.json")) as fh:
            meta = json.load(fh)
        m = np.load(os.path.join(dataset_dir, "manifest.npz"))
        cam_ids = [int(c) for c in m["cam_ids"]]
        N = m["root_pos"].shape[0]
        samples = []
        for n in range(N):
            ja = {name: tuple(m["joint_angles"][n, j])
                  for j, name in enumerate(JOINT_ORDER)}
            pose = rs.RatPose(
                root_pos=m["root_pos"][n], root_rot=m["root_rot"][n],
                joint_angles=ja, scale=float(m["scale"][n]))
            kpts2d = {cam_ids[ci]: m["keypoints2d"][n, ci]
                      for ci in range(len(cam_ids))}
            vis = {cam_ids[ci]: m["visibility"][n, ci]
                   for ci in range(len(cam_ids))}
            samples.append(SyntheticPoseSample(
                pose=pose, keypoints3d=m["keypoints3d"][n],
                keypoints2d=kpts2d, visibility=vis,
                valid=bool(m["valid"][n]),
                body_version=meta.get("body", {}).get(
                    "version", "body-v1")))
        body = _body_from_snapshot(meta.get("body"))
        return cls(samples, meta, body=body)


def _body_from_snapshot(snap: Optional[dict]) -> RatBodyModel:
    if not snap:
        return RatBodyModel.default()
    bone = {}
    for k, v in snap.get("bone_radii", {}).items():
        p, c = k.split("->")
        bone[(p, c)] = tuple(v)
    return RatBodyModel(
        bone_radii=bone or dict(_DEFAULT_BONE_RADII),
        sphere_radii={k: float(v) for k, v in
                      snap.get("sphere_radii", {}).items()}
        or dict(_DEFAULT_SPHERE_RADII),
        version=snap.get("version", "body-v1"))


def generate_dataset(
        n_poses: int,
        cameras: dict,
        image_size: tuple[int, int],
        *,
        body: Optional[RatBodyModel] = None,
        seed: int = 0,
        pose_fraction: float = 0.7,
        scale_range: tuple = (0.85, 1.15),
        arena_bounds: tuple = STD_ARENA,
        require_in_arena: bool = True,
        n_workers: int = 1,
        ) -> SyntheticPoseDataset:
    """Generate a synthetic-pose dataset.

    cameras    : {cam_id(int): P (3,4) np.ndarray}.
    n_workers  : >1 uses a multiprocessing pool. Per-pose deterministic
                 seeding makes the output identical regardless of
                 n_workers.
    """
    body = body or RatBodyModel.default()
    args = dict(cameras=cameras, image_size=tuple(image_size), seed=seed,
                pose_fraction=pose_fraction, scale_range=scale_range,
                arena_bounds=arena_bounds, require_in_arena=require_in_arena,
                body_version=body.version)

    if n_workers and n_workers > 1:
        from multiprocessing import Pool
        from functools import partial
        worker = partial(_worker_make, args=args)
        with Pool(n_workers) as pool:
            samples = pool.map(worker, range(n_poses),
                               chunksize=max(1, n_poses // (n_workers * 4)))
    else:
        samples = [make_sample(i, **args) for i in range(n_poses)]

    meta = {
        "seed": seed, "n_poses": n_poses,
        "pose_fraction": pose_fraction,
        "scale_range": list(scale_range),
        "arena_bounds": list(arena_bounds),
        "require_in_arena": require_in_arena,
        "skeleton_version": "rat23",
        "joint_order": JOINT_ORDER,
        "image_size": list(image_size),
        "cameras": {str(cid): np.asarray(P).tolist()
                    for cid, P in cameras.items()},
        "body": body.radii_snapshot(),
    }
    return SyntheticPoseDataset(samples, meta, body=body)


def _worker_make(i, args):
    return make_sample(i, **args)
