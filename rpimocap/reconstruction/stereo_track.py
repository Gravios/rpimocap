"""
rpimocap.reconstruction.stereo_track
====================================
Stereo-gated detection selection (closes the loop on ROADMAP Phase 1).

A single camera cannot tell the rat on the arena floor from an external
floor patch seen THROUGH the transparent acrylic wall — both land at the
same pixel, so a 2-D ROI can't separate them. Only triangulation
resolves it: the through-the-glass patch triangulates to a point OUTSIDE
the arena volume (beyond the wall / below the floor), so the static-scene
arena gate (arena_gate.accept_point, patch 0057) rejects it; and a blob
with no epipolar-consistent partner in the other view is dropped by the
two-view coupling (epipolar.match_stereo_candidates, patch 0056).

This module wires those two gates into per-frame detection: given the
candidate blob centroids in each camera for one synchronized frame, it
returns the cam0↔cam1 pairing that is epipolar-consistent AND triangulates
to an in-arena 3-D point with low reprojection error — i.e. the rat, not
the artifact. Feed the returned 2-D centroids to the per-camera trackers
and they lock onto the rat instead of the wall patch.

Pure geometry on top of 0056/0057; validated against the synthetic-pose
ground truth (a real in-arena point + an injected out-of-arena distractor
that survives both 2-D ROIs).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from rpimocap.reconstruction.epipolar import (
    match_stereo_candidates, StereoMatch)


STD_ARENA = (-140.0, 140.0, -215.0, 215.0, 0.0, 388.0)


@dataclass
class GatedDetection:
    """The accepted stereo detection for one frame."""
    point: np.ndarray            # (3,) triangulated world point (mm)
    uv0:   tuple                 # cam0 pixel centroid of the accepted blob
    uv1:   tuple                 # cam1 pixel centroid
    i0:    int                   # index into the cam0 candidate list
    i1:    int
    reproj_err: float


def gated_stereo_detection(
        cand0: Sequence[tuple], cand1: Sequence[tuple],
        P0: np.ndarray, P1: np.ndarray, *,
        arena_bounds: tuple = STD_ARENA,
        arena_pad_mm: float = 30.0,
        max_epipolar_px: float = 8.0,
        max_reproj_px: float = 8.0,
        F: Optional[np.ndarray] = None,
        ) -> Optional[GatedDetection]:
    """Pick the best in-arena, epipolar-consistent cam0↔cam1 pairing from
    this frame's candidate centroids, or None if none qualifies (e.g.
    only the through-the-glass patch is present → rejected).

    cand0, cand1 : lists of (x, y) blob centroids per camera.
    The arena gate is applied inside match_stereo_candidates via
    require_in_arena, so out-of-volume pairings (the external floor
    patch) never get selected.
    """
    if not cand0 or not cand1:
        return None
    matches = match_stereo_candidates(
        cand0, cand1, P0, P1, F=F,
        max_epipolar_px=max_epipolar_px, max_reproj_px=max_reproj_px,
        arena_bounds=arena_bounds, arena_pad_mm=arena_pad_mm,
        require_in_arena=True)
    if not matches:
        return None
    m: StereoMatch = matches[0]            # lowest-cost, in-arena
    return GatedDetection(
        point=m.point, uv0=tuple(cand0[m.i0]), uv1=tuple(cand1[m.i1]),
        i0=m.i0, i1=m.i1, reproj_err=m.reproj_err)


def gate_trajectory(
        cand0_by_frame: dict, cand1_by_frame: dict,
        P0: np.ndarray, P1: np.ndarray, **kwargs) -> dict:
    """Apply gated_stereo_detection across many frames.

    cand{0,1}_by_frame : {frame_index: [(x,y), ...]} candidate centroids.
    Returns {frame_index: GatedDetection} for frames with an accepted
    in-arena stereo detection (frames where only artifacts were present
    are absent from the result — they're correctly rejected).
    """
    frames = sorted(set(cand0_by_frame) & set(cand1_by_frame))
    out = {}
    F = kwargs.pop("F", None)
    for fi in frames:
        det = gated_stereo_detection(
            cand0_by_frame[fi], cand1_by_frame[fi], P0, P1, F=F, **kwargs)
        if det is not None:
            out[fi] = det
    return out
