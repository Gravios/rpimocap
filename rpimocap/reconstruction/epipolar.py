"""
rpimocap.reconstruction.epipolar
================================
Tightly-coupled two-view selection (ROADMAP Phase 1.2; the VINS/FreiPose
lesson from the parallelization/AR research): instead of segmenting each
camera independently and hoping the centroids correspond, resolve the
correspondence ACROSS views using the known stereo geometry, and reject
candidates that can't form a plausible 3-D point.

The failure this fixes: a single-view detector picks a false blob in
cam1 (a door reflection, a rail glint). Loosely-coupled triangulation of
"the cam0 centroid" with "the cam1 centroid" is then poisoned. Tightly-
coupled selection asks: of all cam0×cam1 blob pairings, which one
triangulates to a point that (a) lies on both epipolar lines, (b) sits
inside the arena, and (c) reprojects with low error? A false cam1 blob
fails all three, so it's rejected before it can corrupt the 3-D estimate.

Pure geometry — no learned components, fully validated against the
synthetic-pose ground truth (project a known 3-D point through both
cameras, recover the correct pairing among distractors).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from rpimocap.reconstruction.triangulate import (
    triangulate_dlt, reprojection_error)


STD_ARENA = (-140.0, 140.0, -215.0, 215.0, 0.0, 388.0)


@dataclass
class StereoMatch:
    """A chosen cam0↔cam1 correspondence and its triangulation."""
    i0:    int                  # index into the cam0 candidate list
    i1:    int                  # index into the cam1 candidate list
    point: np.ndarray           # (3,) triangulated world point (mm)
    reproj_err: float           # max per-view reprojection error (px)
    in_arena: bool
    cost:  float                # combined score (lower = better)


def fundamental_from_projections(P0: np.ndarray,
                                 P1: np.ndarray) -> np.ndarray:
    """Fundamental matrix F such that x1ᵀ F x0 = 0, from the two 3×4
    projection matrices. Built via the camera-0 center and the
    epipole in view 1 (Hartley & Zisserman 9.2)."""
    # camera-0 center: null space of P0
    _, _, Vt = np.linalg.svd(P0)
    C0 = Vt[-1]
    C0 = C0 / C0[3]
    e1 = P1 @ C0                         # epipole in image 1
    P0_pinv = np.linalg.pinv(P0)
    F = _skew(e1) @ P1 @ P0_pinv
    return F


def _skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]], dtype=np.float64)


def epipolar_distance(F: np.ndarray, pt0: tuple, pt1: tuple) -> float:
    """Symmetric epipolar distance (px): the distance from pt1 to the
    epipolar line F·x0, plus from pt0 to Fᵀ·x1, averaged. Zero when the
    two points are a perfect stereo correspondence."""
    x0 = np.array([pt0[0], pt0[1], 1.0])
    x1 = np.array([pt1[0], pt1[1], 1.0])
    l1 = F @ x0                          # epipolar line in image 1
    l0 = F.T @ x1                        # epipolar line in image 0
    d1 = abs(x1 @ l1) / (np.hypot(l1[0], l1[1]) + 1e-12)
    d0 = abs(x0 @ l0) / (np.hypot(l0[0], l0[1]) + 1e-12)
    return float(0.5 * (d0 + d1))


def _in_arena(X: np.ndarray, bounds: tuple, pad: float) -> bool:
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    return bool(
        xmin - pad <= X[0] <= xmax + pad
        and ymin - pad <= X[1] <= ymax + pad
        and zmin - pad <= X[2] <= zmax + pad)


def match_stereo_candidates(
        cand0: Sequence[tuple],
        cand1: Sequence[tuple],
        P0: np.ndarray, P1: np.ndarray,
        *,
        F: Optional[np.ndarray] = None,
        max_epipolar_px: float = 8.0,
        max_reproj_px: float = 6.0,
        arena_bounds: Optional[tuple] = STD_ARENA,
        arena_pad_mm: float = 30.0,
        require_in_arena: bool = True,
        epipolar_weight: float = 1.0,
        reproj_weight: float = 1.0,
        ) -> list[StereoMatch]:
    """Find the best cam0↔cam1 correspondences among candidate blobs.

    cand0, cand1 : lists of (x, y) pixel centroids in each camera.
    Returns a list of accepted StereoMatch, greedily best-first, each
    cam0 and cam1 candidate used at most once (one rat → one match, but
    the API supports several for multi-animal). A pair is accepted only
    if it passes the epipolar gate, (optionally) lands in the arena, and
    reprojects within tolerance.

    The combined cost is epipolar_weight·epipolar_px +
    reproj_weight·reproj_px, so the chosen pairing is the geometrically
    most consistent one — this is the tight coupling: a false single-view
    blob has no consistent partner and is dropped.
    """
    if F is None:
        F = fundamental_from_projections(P0, P1)

    cands: list[StereoMatch] = []
    for i0, p0 in enumerate(cand0):
        for i1, p1 in enumerate(cand1):
            ep = epipolar_distance(F, p0, p1)
            if ep > max_epipolar_px:
                continue
            X = triangulate_dlt(P0, P1, p0, p1)[:3]
            ina = (_in_arena(X, arena_bounds, arena_pad_mm)
                   if arena_bounds is not None else True)
            if require_in_arena and not ina:
                continue
            e0 = reprojection_error(P0, X, p0)
            e1 = reprojection_error(P1, X, p1)
            re = max(e0, e1)
            if re > max_reproj_px:
                continue
            cost = epipolar_weight * ep + reproj_weight * re
            cands.append(StereoMatch(
                i0=i0, i1=i1, point=X, reproj_err=re,
                in_arena=ina, cost=cost))

    # greedy: accept lowest-cost matches, each index used once
    cands.sort(key=lambda m: m.cost)
    used0, used1, out = set(), set(), []
    for m in cands:
        if m.i0 in used0 or m.i1 in used1:
            continue
        used0.add(m.i0)
        used1.add(m.i1)
        out.append(m)
    return out


def best_stereo_point(
        cand0: Sequence[tuple], cand1: Sequence[tuple],
        P0: np.ndarray, P1: np.ndarray, **kwargs
        ) -> Optional[StereoMatch]:
    """Single-rat convenience: return the single best consistent match,
    or None if no candidate pairing is geometrically plausible (e.g. the
    cam1 detection is a false positive with no cam0 partner)."""
    matches = match_stereo_candidates(cand0, cand1, P0, P1, **kwargs)
    return matches[0] if matches else None
