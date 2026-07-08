"""
rpimocap.gui.pose_state
=======================
Framework-agnostic state for interactive pose fitting — everything the Qt GUI
(``tools/pose_gui.py``) needs, with no Qt dependency so it is unit-testable.

A :class:`PoseFitterState` holds the stereo frame list, the calibration, a body
model renderer, and the current editable :class:`RatPose`. It renders model
overlays for display, runs the detector for target silhouettes, saves/loads
per-frame keyframe poses, and auto-fits — either freely from the detection or,
crucially, **bounded around the current pose** so a hand-set keyframe seeds a
restricted search on neighbouring frames.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import cv2
import numpy as np

from ..detection.topo_detect import build_floor_mask, detect_stereo
from ..model.fit import fit_pose_local, fit_pose_multistart
from ..model.rat_skeleton import RatPose

ARENA_CORNERS = np.array([[-140, -215, 0], [140, -215, 0], [140, 215, 0],
                          [-140, 215, 0], [-140, -215, 388], [140, -215, 388],
                          [140, 215, 388], [-140, 215, 388]], float)

_MODEL_GREEN = np.array([0, 150, 70])
_MASK_ORANGE = (255, 180, 0)


def pose_to_dict(p: RatPose) -> dict:
    return {"root_pos": [float(x) for x in p.root_pos],
            "root_rot": [float(x) for x in p.root_rot],
            "scale": float(p.scale),
            "joint_angles": {k: [float(x) for x in v]
                             for k, v in p.joint_angles.items()}}


def pose_from_dict(d: dict) -> RatPose:
    return RatPose(root_pos=np.asarray(d["root_pos"], float),
                   root_rot=np.asarray(d["root_rot"], float),
                   scale=float(d.get("scale", 1.0)),
                   joint_angles={k: tuple(float(x) for x in v)
                                 for k, v in d.get("joint_angles", {}).items()})


@dataclass
class PoseFitterState:
    """Editable pose-fitting session over a list of stereo frame pairs.

    Parameters
    ----------
    frames     : list of (cam0_path, cam1_path).
    Ps         : [dlt_P0, dlt_P1].
    render_fn  : ``render_fn(pose, P, image_shape) -> uint8 mask`` — any body
                 model (capsule, procedural mesh, artist mesh).
    image_shape: (H, W) of the frames.
    """
    frames: list
    Ps: list
    render_fn: object
    image_shape: tuple = (1080, 2028)
    idx: int = 0
    pose: RatPose = field(default_factory=lambda: RatPose(
        root_pos=np.array([0.0, 0.0, 60.0])))
    saved: dict = field(default_factory=dict)

    def __post_init__(self):
        self._floor = [build_floor_mask(P, ARENA_CORNERS, self.image_shape,
                                        mode="floor") for P in self.Ps]
        self._imgs = None
        self._det_cache = {}
        self.load_frame(self.idx)

    # ---- frames --------------------------------------------------------
    @staticmethod
    def _read(path):
        g = cv2.imread(path)
        if g is None:
            raise FileNotFoundError(path)
        return (g[:, :, 1] if g.ndim == 3 else g)   # green channel (NIR)

    def frame_name(self) -> str:
        return os.path.basename(self.frames[self.idx][0])

    def load_frame(self, idx: int, carry_pose: bool = True):
        """Load frame ``idx``. If a keyframe was saved for it, restore that
        pose; otherwise keep the current pose (``carry_pose``) as a warm start."""
        self.idx = int(np.clip(idx, 0, len(self.frames) - 1))
        c0, c1 = self.frames[self.idx]
        self._imgs = [self._read(c0), self._read(c1)]
        name = self.frame_name()
        if name in self.saved:
            self.pose = pose_from_dict(self.saved[name])
        # else: keep self.pose (carry_pose) as the starting point
        return self

    # ---- detection (target silhouettes) --------------------------------
    def detection(self, **detect_kw):
        """Detector masks + triangulated seed for the current frame (cached)."""
        key = self.idx
        if key not in self._det_cache:
            g0, g1 = self._imgs
            R = detect_stereo(g0, g1, self._floor[0], self._floor[1],
                              self.Ps[0], self.Ps[1], **detect_kw)
            self._det_cache[key] = R
        return self._det_cache[key]

    # ---- rendering -----------------------------------------------------
    def overlay(self, cam: int, show_detected: bool = True) -> np.ndarray:
        """RGB overlay for ``cam``: frame + model silhouette (green fill +
        outline), and optionally the detector mask outline (orange)."""
        g = self._imgs[cam].astype(np.float32)
        lo, hi = np.percentile(g, 1), np.percentile(g, 99)
        base = np.clip((g - lo) / (hi - lo + 1e-6), 0, 1)
        rgb = cv2.cvtColor((base * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        if show_detected:
            try:
                dets = (self.detection().det0, self.detection().det1)
                dm = dets[cam].mask
                dc, _ = cv2.findContours(dm, cv2.RETR_EXTERNAL,
                                         cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(rgb, dc, -1, _MASK_ORANGE, 3)
            except Exception:
                pass
        sil = self.render_fn(self.pose, self.Ps[cam], self.image_shape)
        rgb[sil > 0] = (0.55 * rgb[sil > 0] + _MODEL_GREEN).astype(np.uint8)
        mc, _ = cv2.findContours(sil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(rgb, mc, -1, (0, 255, 120), 2)
        return rgb

    def current_iou(self) -> float:
        from ..model.body_model import silhouette_iou
        R = self.detection()
        masks = [R.det0.mask, R.det1.mask]
        ious = [silhouette_iou(self.render_fn(self.pose, P, self.image_shape), m)
                for P, m in zip(self.Ps, masks)]
        return float(np.mean(ious))

    # ---- keyframe poses ------------------------------------------------
    def save_current_pose(self):
        self.saved[self.frame_name()] = pose_to_dict(self.pose)

    def write_poses(self, path: str):
        with open(path, "w") as fh:
            json.dump(self.saved, fh, indent=2)

    def read_poses(self, path: str):
        with open(path) as fh:
            self.saved = json.load(fh)
        if self.frame_name() in self.saved:
            self.pose = pose_from_dict(self.saved[self.frame_name()])

    # ---- fitting -------------------------------------------------------
    def fit_from_detection(self, headings: int = 4, physics_weight: float = 2.0,
                           **kw):
        """Free fit for the current frame, seeded from the triangulated
        centroid — use to initialise a keyframe."""
        R = self.detection()
        masks = [R.det0.mask, R.det1.mask]
        seed = np.array([R.point[0], R.point[1], max(R.point[2], 45.0)])
        self.pose, iou = fit_pose_multistart(
            masks, self.Ps, seed, headings=headings, render_fn=self.render_fn,
            physics_weight=physics_weight, **kw)
        return iou

    def fit_local(self, pos_tol: float = 25.0, ang_tol: float = 0.35,
                  scale_tol: float = 0.15, joints=None, physics_weight: float = 2.0,
                  **kw):
        """Bounded fit around the current pose — the neighbouring-frame
        refinement. Restricts the search to the vicinity of the (hand-set or
        carried) pose."""
        R = self.detection()
        masks = [R.det0.mask, R.det1.mask]
        self.pose, iou = fit_pose_local(
            masks, self.Ps, self.pose, pos_tol=pos_tol, ang_tol=ang_tol,
            scale_tol=scale_tol, joints=joints, render_fn=self.render_fn,
            physics_weight=physics_weight, **kw)
        return iou
