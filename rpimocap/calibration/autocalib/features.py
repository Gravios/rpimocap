"""
autocalib_features.py — Cross-view feature correspondence for self-calibration
==============================================================================
Extracts matched feature point pairs from synchronised stereo video frames.

For self-calibration via the essential matrix constraint we need:
  - Many (≥100) well-distributed correspondences per frame pair
  - Frame pairs sampled across the full recording to cover varied subject poses
  - Robust fundamental matrix estimates (RANSAC, high inlier rate)

The output is a list of FundamentalEstimate objects, one per sampled frame,
each containing the estimated F matrix, inlier correspondences, and quality
metadata used by the Kruppa stage.

Design notes
------------
- SIFT is used over ORB because the scene (fur/skin texture in constrained
  arenas) typically has low-contrast blobs rather than sharp corners.
  ORB tends to produce many false matches in such conditions.
- Ratio test threshold 0.72 is intentionally slightly tighter than Lowe's
  canonical 0.75 to reduce the RANSAC burden.
- Frame sampling is uniform by default; dense sampling wastes compute because
  nearby frames produce nearly identical F estimates (low parallax variance).
  The recommended sampling period is ~1–2 seconds of footage.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np


# --------------------------------------------------------------------------- #
#  Data class                                                                  #
# --------------------------------------------------------------------------- #

@dataclass
class FundamentalEstimate:
    """A single robust fundamental matrix estimate from one stereo frame pair.

        Attributes
        ----------
        frame_idx     : source frame index in the video stream
        F             : (3, 3) fundamental matrix (rank-2, unit Frobenius norm)
        pts0, pts1    : (N, 2) RANSAC inlier pixel coordinates
        n_inliers     : number of RANSAC inliers
        inlier_ratio  : inliers / total matches after ratio test
        mean_sampson  : mean symmetric Sampson distance of inliers in pixels
        """
    frame_idx: int
    F: np.ndarray              # (3, 3) fundamental matrix
    pts0: np.ndarray           # (N, 2) inlier points in camera 0
    pts1: np.ndarray           # (N, 2) inlier points in camera 1
    n_inliers: int
    inlier_ratio: float
    mean_sampson: float        # mean Sampson distance of inliers (quality proxy)

    @property
    def quality(self) -> float:
        """Composite quality score ∈ [0,1]. Higher = better."""
        inlier_score = min(self.n_inliers / 300.0, 1.0)
        ratio_score  = self.inlier_ratio
        # Sampson distance: lower is better; 0.5px is excellent
        samp_score   = math.exp(-self.mean_sampson / 0.5)
        return 0.35 * inlier_score + 0.35 * ratio_score + 0.30 * samp_score


def sampson_distance(F: np.ndarray,
                     pts0: np.ndarray,
                     pts1: np.ndarray) -> np.ndarray:
    """
    Compute the symmetric Sampson distance for each point correspondence.
    Both pts arrays are (N, 2) pixel coordinates.
    """
    ones = np.ones((len(pts0), 1))
    p0 = np.hstack([pts0, ones])   # (N, 3)
    p1 = np.hstack([pts1, ones])

    Fp0  = (F   @ p0.T).T          # (N, 3)
    Ftp1 = (F.T @ p1.T).T          # (N, 3)

    num = (p1 * Fp0).sum(axis=1) ** 2
    den = Fp0[:, 0]**2 + Fp0[:, 1]**2 + Ftp1[:, 0]**2 + Ftp1[:, 1]**2
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(den > 1e-12, num / den, 0.0)


# --------------------------------------------------------------------------- #
#  Feature detector                                                            #
# --------------------------------------------------------------------------- #

class CrossViewMatcher:
    """
    Detects SIFT features in paired frames and returns robust F estimates.

    Parameters
    ----------
    n_features      : Maximum SIFT keypoints per frame
    ratio_thresh    : Lowe ratio test threshold
    ransac_thresh   : RANSAC inlier distance in pixels (Sampson)
    ransac_conf     : RANSAC confidence
    min_inliers     : Discard estimates with fewer inliers
    min_inlier_ratio: Discard estimates with lower inlier ratio
    grid_mask       : Divide frame into grid cells and cap features per cell
                      (improves spatial distribution)
    """

    def __init__(self,
                 n_features: int = 4000,
                 ratio_thresh: float = 0.72,
                 ransac_thresh: float = 1.5,
                 ransac_conf: float = 0.9995,
                 min_inliers: int = 80,
                 min_inlier_ratio: float = 0.25,
                 grid_rows: int = 4,
                 grid_cols: int = 4):

        self._sift = cv2.SIFT_create(
            nfeatures=n_features,
            contrastThreshold=0.025,
            edgeThreshold=12,
        )
        self._matcher = cv2.BFMatcher(cv2.NORM_L2)
        self._ratio_thresh = ratio_thresh
        self._ransac_thresh = ransac_thresh
        self._ransac_conf = ransac_conf
        self._min_inliers = min_inliers
        self._min_inlier_ratio = min_inlier_ratio
        self._grid_rows = grid_rows
        self._grid_cols = grid_cols

    # ── Core matching ──────────────────────────────────────────────────────

    def match_frame_pair(self,
                         frame0: np.ndarray,
                         frame1: np.ndarray,
                         frame_idx: int = 0
                         ) -> Optional[FundamentalEstimate]:
        """
        Match a single synchronised frame pair and estimate F.
        Returns None if quality thresholds are not met.
        """
        g0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
        g1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

        kp0, des0 = self._sift.detectAndCompute(g0, None)
        kp1, des1 = self._sift.detectAndCompute(g1, None)

        if des0 is None or des1 is None or len(des0) < 20 or len(des1) < 20:
            return None

        # Ratio test
        matches = self._matcher.knnMatch(des0, des1, k=2)
        good = [m for m, n in matches if m.distance < self._ratio_thresh * n.distance]
        if len(good) < self._min_inliers:
            return None

        pts0 = np.float32([kp0[m.queryIdx].pt for m in good])
        pts1 = np.float32([kp1[m.trainIdx].pt for m in good])

        # Distribute: cap features per grid cell for better coverage
        pts0, pts1 = self._grid_filter(pts0, pts1, frame0.shape[:2])

        if len(pts0) < self._min_inliers:
            return None

        # RANSAC F estimation
        F, mask = cv2.findFundamentalMat(
            pts0, pts1,
            method=cv2.FM_RANSAC,
            ransacReprojThreshold=self._ransac_thresh,
            confidence=self._ransac_conf,
        )
        if F is None or F.shape != (3, 3):
            return None

        # Enforce rank-2
        U, S, Vt = np.linalg.svd(F)
        S[2] = 0.0
        F = (U * S) @ Vt
        F = F / (np.linalg.norm(F) + 1e-12)

        mask = mask.ravel().astype(bool)
        n_in = mask.sum()
        ratio = n_in / len(pts0)

        if n_in < self._min_inliers or ratio < self._min_inlier_ratio:
            return None

        ip0, ip1 = pts0[mask], pts1[mask]
        samp = sampson_distance(F, ip0, ip1)
        mean_s = float(samp.mean())

        return FundamentalEstimate(
            frame_idx=frame_idx,
            F=F,
            pts0=ip0,
            pts1=ip1,
            n_inliers=n_in,
            inlier_ratio=ratio,
            mean_sampson=mean_s,
        )

    def _grid_filter(self,
                     pts0: np.ndarray,
                     pts1: np.ndarray,
                     frame_shape: tuple,
                     cap_per_cell: int = 30
                     ) -> tuple[np.ndarray, np.ndarray]:
        """Keep at most cap_per_cell correspondences per grid cell."""
        h, w = frame_shape
        cell_h = h / self._grid_rows
        cell_w = w / self._grid_cols
        keep = []
        cell_count: dict = {}
        for i, (p0, p1) in enumerate(zip(pts0, pts1)):
            r = int(p0[1] / cell_h)
            c = int(p0[0] / cell_w)
            key = (min(r, self._grid_rows - 1), min(c, self._grid_cols - 1))
            if cell_count.get(key, 0) < cap_per_cell:
                keep.append(i)
                cell_count[key] = cell_count.get(key, 0) + 1
        if not keep:
            return pts0, pts1
        idx = np.array(keep)
        return pts0[idx], pts1[idx]


# --------------------------------------------------------------------------- #
#  Video sampler                                                               #
# --------------------------------------------------------------------------- #

def sample_frame_pairs(
    cap0: cv2.VideoCapture,
    cap1: cv2.VideoCapture,
    matcher: CrossViewMatcher,
    sample_interval_frames: int = 60,
    max_estimates: int = 120,
    start_frame: int = 0,
    warmup_frames: int = 60,
    verbose: bool = True,
) -> list[FundamentalEstimate]:
    """
    Sample synchronised frame pairs at regular intervals and accumulate
    FundamentalEstimate objects.

    Parameters
    ----------
    sample_interval_frames : process every Nth frame pair
    max_estimates          : stop after collecting this many valid estimates
    warmup_frames          : skip the first N frames (subject entry / lighting settle)
    verbose                : print progress

    Returns
    -------
    list of FundamentalEstimate, sorted by quality descending
    """
    estimates: list[FundamentalEstimate] = []
    total = int(cap0.get(cv2.CAP_PROP_FRAME_COUNT))

    if start_frame + warmup_frames > 0:
        cap0.set(cv2.CAP_PROP_POS_FRAMES, start_frame + warmup_frames)
        cap1.set(cv2.CAP_PROP_POS_FRAMES, start_frame + warmup_frames)

    frame_idx = start_frame + warmup_frames
    step_idx = 0
    n_tried = 0
    n_ok = 0

    if verbose:
        print(f"  Sampling every {sample_interval_frames} frames  "
              f"(warmup {warmup_frames}, max {max_estimates} estimates)")

    while True:
        if len(estimates) >= max_estimates:
            break

        ret0, f0 = cap0.read()
        ret1, f1 = cap1.read()
        if not ret0 or not ret1:
            break

        if step_idx % sample_interval_frames == 0:
            n_tried += 1
            est = matcher.match_frame_pair(f0, f1, frame_idx)
            if est is not None:
                estimates.append(est)
                n_ok += 1
                if verbose and n_ok % 10 == 0:
                    print(f"    {n_ok:3d} estimates  "
                          f"frame {frame_idx}  "
                          f"inliers={est.n_inliers}  "
                          f"ratio={est.inlier_ratio:.2f}  "
                          f"sampson={est.mean_sampson:.3f}px")

        frame_idx += 1
        step_idx += 1

    if verbose:
        print(f"  Result: {n_ok}/{n_tried} frame pairs accepted  "
              f"({len(estimates)} estimates)")

    estimates.sort(key=lambda e: e.quality, reverse=True)
    return estimates


# --------------------------------------------------------------------------- #
#  Quality filtering                                                           #
# --------------------------------------------------------------------------- #

def filter_estimates(
    estimates: list[FundamentalEstimate],
    min_quality: float = 0.35,
    top_n: Optional[int] = None,
) -> list[FundamentalEstimate]:
    """
    Filter and optionally subsample estimates by quality score.

    A diverse selection (spread over the recording) is better for calibration
    than many near-identical high-quality estimates from the same region.
    """
    filtered = [e for e in estimates if e.quality >= min_quality]

    if top_n and len(filtered) > top_n:
        # Greedy temporal diversity: keep highest quality, then add others
        # that are temporally spaced at least (total_range / top_n) apart.
        if filtered:
            kept = [filtered[0]]
            frames = [filtered[0].frame_idx]
            for e in filtered[1:]:
                if all(abs(e.frame_idx - f) > 30 for f in frames):
                    kept.append(e)
                    frames.append(e.frame_idx)
                    if len(kept) >= top_n:
                        break
            filtered = kept

    return filtered
