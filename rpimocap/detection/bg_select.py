"""
rpimocap.detection.bg_select
============================
Motion-aware selection of background-model frames.

The persistent texture model is a per-pixel temporal median (+ MAD) over
a set of sampled frames. The median rejects the rat ONLY if, at each
pixel, the rat is present in a minority of those frames. Sampling at a
fixed stride breaks that assumption when the rat sits still: a dwell
period contributes many frames with the rat in the SAME spot, so the
median there absorbs the rat — a "persistence hole" that suppresses the
rat in the distance map exactly where it spends its time.

Fix (the user's idea): pick background frames from moments when the rat
is MOVING. A moving rat occupies a different position in each sampled
frame, so at every pixel it is a minority and the median stays clean. We
estimate per-frame motion from the inter-frame difference (the "movement
field") inside the arena ROI, then choose one high-motion frame per
temporal bin — high motion (rat in transit, varied position) AND spread
across the session (varied gross location, robust to any single active
bout).

Pure NumPy/OpenCV, streamed; validated on synthetic moving-blob video.
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


def _read_gray(cap, idx, green_channel):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    if frame.ndim == 3:
        return (frame[:, :, 1] if green_channel
                else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    return frame


def frame_motion_series(
        cap, frame_indices, *, green_channel: bool = True,
        roi_mask: Optional[np.ndarray] = None,
        motion_downsample: int = 4) -> np.ndarray:
    """Per-frame motion score = mean |Δ| to the previous sampled frame,
    inside the ROI, on a downsampled gray image (motion needs no detail).
    motion[0] = 0 (no predecessor). Frames that fail to read score 0."""
    n = len(frame_indices)
    motion = np.zeros(n, dtype=np.float64)
    roi_ds = None
    if roi_mask is not None and motion_downsample > 1:
        roi_ds = cv2.resize(
            roi_mask, (roi_mask.shape[1] // motion_downsample,
                       roi_mask.shape[0] // motion_downsample),
            interpolation=cv2.INTER_NEAREST) > 0
    elif roi_mask is not None:
        roi_ds = roi_mask > 0

    prev = None
    for i, fi in enumerate(frame_indices):
        g = _read_gray(cap, fi, green_channel)
        if g is None:
            prev = None
            continue
        if motion_downsample > 1:
            g = cv2.resize(
                g, (g.shape[1] // motion_downsample,
                    g.shape[0] // motion_downsample),
                interpolation=cv2.INTER_AREA)
        g = g.astype(np.float32)
        if prev is not None and prev.shape == g.shape:
            d = np.abs(g - prev)
            motion[i] = float(d[roi_ds].mean() if roi_ds is not None
                              else d.mean())
        prev = g
    return motion


def select_active_frames(
        cap, n_frames: int, start: int, end: int, *,
        green_channel: bool = True,
        roi_mask: Optional[np.ndarray] = None,
        oversample: int = 4, motion_downsample: int = 4,
        min_motion_percentile: float = 40.0,
        ) -> tuple:
    """Choose n_frames background frames biased to active rat movement.

    Scans n_frames*oversample candidates spread over [start, end), scores
    each by motion, then selects so the chosen frames are caught
    mid-movement AND spread across the session — so the rat is in varied
    positions and the per-pixel median rejects it.

    Selection: discard candidates below `min_motion_percentile` of the
    motion distribution (dead-still frames where the rat is dwelling —
    these are what contaminate the median), then take one highest-motion
    candidate per temporal bin over the SURVIVING frames. If a bin has no
    surviving candidate (a long still bout), it contributes nothing and
    its slot is filled from the global high-motion pool, so the model is
    never padded with dwell frames just to hit a temporal quota.

    Returns (selected_indices, candidate_indices, motion_scores).
    """
    end = max(end, start + 1)
    n_cand = min(max(n_frames * oversample, n_frames), end - start)
    cand = np.unique(
        np.linspace(start, end - 1, n_cand).astype(int))
    motion = frame_motion_series(
        cap, cand, green_channel=green_channel, roi_mask=roi_mask,
        motion_downsample=motion_downsample)

    # motion[0] is always 0 (no predecessor) — exclude from the
    # threshold estimate so it doesn't drag the percentile down.
    scored = motion[1:] if len(motion) > 1 else motion
    thresh = (np.percentile(scored, min_motion_percentile)
              if scored.size else 0.0)
    active = motion >= max(thresh, 1e-9)
    if active.sum() < min(n_frames, 3):
        active = np.ones(len(cand), bool)      # too still: fall back

    # one max-motion ACTIVE frame per temporal bin
    selected = []
    bins = np.array_split(np.arange(len(cand)), min(n_frames, len(cand)))
    for b in bins:
        ab = b[active[b]]
        if len(ab):
            selected.append(int(cand[ab[int(np.argmax(motion[ab]))]]))
    # backfill empty-bin slots from the global active pool (highest
    # motion first) so we never substitute dwell frames for a quota
    if len(selected) < n_frames:
        pool = [int(cand[i]) for i in np.argsort(-motion)
                if active[i] and int(cand[i]) not in selected]
        selected.extend(pool[:n_frames - len(selected)])
    return sorted(set(selected)), cand, motion
