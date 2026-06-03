"""
rpimocap.detection.rat_tracker
==============================
Edge-motion-based rat ROI tracker.

Designed to solve the aperture-problem failure of dense optical
flow on the rat's smooth fur interior: Farneback flow on the rat
body reports near-zero magnitude in interior pixels because the
gradient is locally flat, even though the rat is clearly moving.
Patch 0021's motion gate (motion_min on dense flow magnitude)
therefore wipes out the rat's interior, leaving only edge pixels.
The remaining edge-skeleton fails the labeller's min-area filter
and the cable becomes the sole surviving candidate.

This module's approach
----------------------
1. Track sparse Shi-Tomasi corners via KLT (Lucas-Kanade pyramidal
   optical flow). KLT operates only at high-gradient points where
   flow is mathematically well-conditioned, sidestepping the
   aperture problem entirely.

2. Cluster surviving points by spatial proximity AND velocity
   similarity (5-D DBSCAN over (x, y, vx, vy)). The rat is a
   group of corners moving coherently together. The cable wire is
   typically a SEPARATE cluster (its motion lags the rat and is
   constrained to swing around the mount point). Mount hardware,
   plexiglass reflections, and static highlights have no surviving
   points because their gradients are static.

3. Pick the cluster whose centroid is closest to the Kalman
   prediction of the rat position. This handles the case where
   multiple coherent-motion clusters exist (rat + cable wire) by
   following the rat through occlusions and dropouts via state
   continuity rather than per-frame intensity.

4. Convex hull of the chosen cluster, dilated by body_half_width
   px, gives the rat's spatial extent. Pass this hull mask as an
   ROI to the existing segment pipeline.

5. Kalman filter on (cx, cy, r, vx, vy) updates from observations
   when motion is detected. When no motion (rat frozen, brief
   occlusion), the predict step keeps the hull in place at the
   last known location with growing position uncertainty — so the
   rest of the pipeline never sees an empty ROI just because the
   rat stopped moving for a few frames.

Per-camera state. Stateless across recordings (instantiate once
per session).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import cv2


# ────────────────────────────────────────────────────────────────────
#  Kalman filter for hull state
# ────────────────────────────────────────────────────────────────────


def _make_kalman(dt: float = 1.0,
                  process_noise: float = 5.0,
                  meas_noise: float = 3.0) -> cv2.KalmanFilter:
    """Build a 5-state, 3-measurement Kalman filter.

    State vector  x  = [cx, cy, r, vx, vy]
    Measurement   z  = [cx, cy, r]

    Constant-velocity model for position, random-walk for radius.
    The σ for process and measurement noise are scalars that tune
    how responsive the filter is — process_noise large = trusts new
    observations more; process_noise small = smoother but laggier.
    """
    kf = cv2.KalmanFilter(5, 3, 0, cv2.CV_32F)
    kf.transitionMatrix = np.array([
        [1, 0, 0, dt, 0],
        [0, 1, 0, 0, dt],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1],
    ], dtype=np.float32)
    kf.measurementMatrix = np.array([
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
    ], dtype=np.float32)
    kf.processNoiseCov     = np.eye(5, dtype=np.float32) * (process_noise ** 2)
    kf.measurementNoiseCov = np.eye(3, dtype=np.float32) * (meas_noise ** 2)
    kf.errorCovPost        = np.eye(5, dtype=np.float32) * 1000.0
    kf.statePost           = np.zeros((5, 1), dtype=np.float32)
    return kf


# ────────────────────────────────────────────────────────────────────
#  KLT / clustering helpers
# ────────────────────────────────────────────────────────────────────


def _dbscan_xy_v(points: np.ndarray,
                  flows: np.ndarray,
                  eps_xy: float = 30.0,
                  eps_v:  float = 1.5,
                  min_samples: int = 5) -> np.ndarray:
    """Cluster (x, y, vx, vy) points by combined spatial + velocity
    proximity. Returns cluster labels (-1 = noise).

    Spatial distance uses eps_xy (pixels); velocity distance uses
    eps_v (px/frame). We collapse to a single eps by rescaling the
    velocity axis: a vector that's 1 eps_v from another in velocity
    space is the same as being 1 eps_xy from it in image space.

    Simple O(N²) implementation — fine for the few hundred KLT
    points we expect to have. A full scikit-learn DBSCAN would be
    overkill for a dependency.
    """
    if len(points) == 0:
        return np.empty(0, dtype=np.int32)

    # Normalize velocity to be in "pixel-equivalent" units
    scale_v = eps_xy / max(eps_v, 1e-6)
    feats = np.column_stack([points, flows * scale_v])  # (N, 4)

    N = len(feats)
    labels = -np.ones(N, dtype=np.int32)
    visited = np.zeros(N, dtype=bool)
    next_label = 0

    # Pairwise distance matrix (small N, OK to materialize)
    diff = feats[:, None, :] - feats[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=-1))
    neighborhood = dist < eps_xy   # bool matrix

    for i in range(N):
        if visited[i]:
            continue
        visited[i] = True
        neighbors = list(np.where(neighborhood[i])[0])
        if len(neighbors) < min_samples:
            continue   # noise (label stays -1)
        labels[i] = next_label
        # Expand cluster (iterative BFS)
        stack = list(neighbors)
        while stack:
            j = stack.pop()
            if not visited[j]:
                visited[j] = True
                j_neigh = list(np.where(neighborhood[j])[0])
                if len(j_neigh) >= min_samples:
                    stack.extend(j_neigh)
            if labels[j] == -1:
                labels[j] = next_label
        next_label += 1

    return labels


# ────────────────────────────────────────────────────────────────────
#  Per-camera tracker state
# ────────────────────────────────────────────────────────────────────


@dataclass
class CamTrackerState:
    klt_points:    Optional[np.ndarray] = None     # (N, 1, 2) float32
    last_gray:     Optional[np.ndarray] = None     # (H, W) uint8
    kalman:        Optional[cv2.KalmanFilter] = None
    last_obs_idx:  int = -10000                    # frame idx of last good obs
    has_init:      bool = False
    frames_since_refresh: int = 0


@dataclass
class HullObservation:
    """Result of one detect_hull() call. None when no usable
    cluster was found AND Kalman has not yet locked on."""
    cx:        float
    cy:        float
    radius:    float
    n_points:  int
    points:    np.ndarray      # (M, 2) — the cluster's KLT points
    hull:      np.ndarray      # (K, 2) — convex hull vertices
    mask:      np.ndarray      # (H, W) uint8, 255 inside dilated hull
    from_kalman_only: bool = False   # True when no observation, predicted


# ────────────────────────────────────────────────────────────────────
#  Main class
# ────────────────────────────────────────────────────────────────────


class EdgeMotionRatTracker:
    """Edge-motion + Kalman hull tracker.

    Usage
    -----
        tracker = EdgeMotionRatTracker(frame_shape=(H, W))
        for frame_idx, (gray0, gray1) in enumerate(frames):
            obs0 = tracker.step(gray0, cam=0, frame_idx=frame_idx)
            obs1 = tracker.step(gray1, cam=1, frame_idx=frame_idx)
            if obs0 is not None:
                # obs0.mask is an HxW uint8 image, 255 inside the
                # estimated rat hull. Pass to ForegroundDetector as
                # an ROI to AND with the bg-sub binary.
                ...

    Parameters
    ----------
    frame_shape : (H, W) of the input frames
    body_half_width_px : how much to dilate the convex hull to get
        a full-body mask. Estimate from the rat's expected pixel
        size — typically 40-80 px for a rat viewed from above.
    motion_min : minimum KLT flow magnitude (px/frame) to consider
        a point "moving". 0.5 is conservative; raise if there are
        many camera-jitter false positives.
    min_cluster_points : a cluster needs at least this many points
        to be a candidate. Below this, the rat is considered
        unobserved this frame and Kalman extrapolates.
    max_klt_points : seed Shi-Tomasi to find this many corners.
    refresh_every : refresh corner seeds every N frames regardless
        of survival (keeps the point set fresh as the rat changes
        appearance).
    process_noise, meas_noise : Kalman tuning. Defaults work for
        25 fps rat-scale motion.
    seed_roi_radius_px : when seeding the FIRST frame's corner set,
        restrict to a disc of this radius around the frame center.
        Helps avoid seeding the cable mount before the tracker has
        locked on. Set to None to seed the whole frame.
    """

    def __init__(self,
                 frame_shape:           tuple[int, int],
                 body_half_width_px:    int   = 60,
                 motion_min:            float = 0.5,
                 min_cluster_points:    int   = 5,
                 max_klt_points:        int   = 300,
                 refresh_every:         int   = 30,
                 process_noise:         float = 5.0,
                 meas_noise:            float = 3.0,
                 dbscan_eps_xy:         float = 40.0,
                 dbscan_eps_v:          float = 2.0,
                 seed_roi_radius_px:    Optional[int] = None):
        self.H, self.W = frame_shape
        self.body_half_width_px = int(body_half_width_px)
        self.motion_min         = float(motion_min)
        self.min_cluster_points = int(min_cluster_points)
        self.max_klt_points     = int(max_klt_points)
        self.refresh_every      = int(refresh_every)
        self.process_noise      = float(process_noise)
        self.meas_noise         = float(meas_noise)
        self.dbscan_eps_xy      = float(dbscan_eps_xy)
        self.dbscan_eps_v       = float(dbscan_eps_v)
        self.seed_roi_radius_px = seed_roi_radius_px

        self._cams: dict[int, CamTrackerState] = {
            0: CamTrackerState(), 1: CamTrackerState()
        }

    # ----------------------------------------------------------------
    #  Public API
    # ----------------------------------------------------------------

    def step(self,
             gray:      np.ndarray,
             cam:       int,
             frame_idx: int) -> Optional[HullObservation]:
        """Process one frame from one camera.

        Returns a HullObservation when a usable hull is available
        (either from current-frame observation or from Kalman
        prediction during a brief dropout). Returns None when the
        tracker has not yet locked on AND no plausible cluster is
        found in the current frame (typically only the very first
        few frames).
        """
        st = self._cams[cam]

        # ── First frame for this camera: seed corners, stash gray, exit
        if not st.has_init:
            self._seed_corners(st, gray)
            st.last_gray = gray.copy()
            st.kalman    = _make_kalman(dt=1.0,
                                          process_noise=self.process_noise,
                                          meas_noise=self.meas_noise)
            st.has_init  = True
            return None

        # ── KLT: propagate the current point set to the new frame
        new_pts, flows = self._klt_propagate(st, gray)

        # ── Kalman predict (always, before maybe-updating)
        pred = st.kalman.predict()
        pred_cx, pred_cy, pred_r = pred[0, 0], pred[1, 0], pred[2, 0]

        obs: Optional[HullObservation] = None
        if new_pts is not None and len(new_pts) >= self.min_cluster_points:
            obs = self._observe(
                new_pts, flows,
                pred_cx, pred_cy, pred_r, frame_idx)

        # ── Refresh corner seeds if depleted OR scheduled
        st.frames_since_refresh += 1
        too_few = (new_pts is None
                   or len(new_pts) < self.max_klt_points // 3)
        if too_few or st.frames_since_refresh >= self.refresh_every:
            self._reseed_corners_near(st, gray, pred_cx, pred_cy, pred_r)
            st.frames_since_refresh = 0
        else:
            st.klt_points = new_pts.reshape(-1, 1, 2).astype(np.float32)

        st.last_gray = gray.copy()

        if obs is not None:
            # Real observation → update Kalman
            meas = np.array([[obs.cx], [obs.cy], [obs.radius]],
                            dtype=np.float32)
            st.kalman.correct(meas)
            st.last_obs_idx = frame_idx
            return obs

        # ── No observation. Fall back on Kalman prediction if we've
        # ever locked on (last_obs_idx is set). Generate a hull mask
        # from (pred_cx, pred_cy, pred_r + body_half_width).
        if st.last_obs_idx >= 0:
            return self._hull_from_kalman(pred_cx, pred_cy, pred_r)

        return None

    def reset(self, cam: Optional[int] = None):
        """Forget all state. Useful between sessions."""
        if cam is None:
            self._cams = {0: CamTrackerState(), 1: CamTrackerState()}
        else:
            self._cams[cam] = CamTrackerState()

    # ----------------------------------------------------------------
    #  Internal helpers
    # ----------------------------------------------------------------

    def _seed_corners(self, st: CamTrackerState, gray: np.ndarray):
        """Initial Shi-Tomasi seeding. Optionally restricted to a
        center-disc ROI to bias against locking on cable mount
        hardware at frame 0."""
        roi_mask = None
        if self.seed_roi_radius_px is not None:
            cy, cx = self.H // 2, self.W // 2
            roi_mask = np.zeros((self.H, self.W), dtype=np.uint8)
            cv2.circle(roi_mask, (cx, cy),
                        int(self.seed_roi_radius_px), 255, -1)
        pts = cv2.goodFeaturesToTrack(
            gray, maxCorners=self.max_klt_points,
            qualityLevel=0.01, minDistance=8,
            mask=roi_mask, blockSize=7)
        st.klt_points = pts   # (N, 1, 2) or None
        st.frames_since_refresh = 0

    def _reseed_corners_near(self, st: CamTrackerState,
                              gray: np.ndarray,
                              cx: float, cy: float, r: float):
        """Refresh corners, restricted to a disc around the
        Kalman-predicted rat position. Prevents the corner pool
        from drifting onto the cable mount or other static
        artifacts that the bg-sub mask never identified as the rat."""
        # If Kalman not yet locked on, fall back to full-frame or
        # initial ROI.
        if not (0 <= cx <= self.W and 0 <= cy <= self.H):
            self._seed_corners(st, gray)
            return
        radius = max(60, int(r + self.body_half_width_px * 1.5))
        mask = np.zeros((self.H, self.W), dtype=np.uint8)
        cv2.circle(mask, (int(cx), int(cy)), radius, 255, -1)
        new_pts = cv2.goodFeaturesToTrack(
            gray, maxCorners=self.max_klt_points,
            qualityLevel=0.01, minDistance=8,
            mask=mask, blockSize=7)
        if new_pts is None or len(new_pts) == 0:
            # Region is featureless; keep whatever we had
            return
        # Merge with surviving points (deduplicate by proximity)
        if st.klt_points is not None and len(st.klt_points) > 0:
            existing = st.klt_points.reshape(-1, 2)
            new_flat = new_pts.reshape(-1, 2)
            keep = []
            for p in new_flat:
                d = np.min(np.linalg.norm(existing - p, axis=1))
                if d > 8:
                    keep.append(p)
            if keep:
                merged = np.vstack([existing, np.array(keep)])
                merged = merged[:self.max_klt_points]
                st.klt_points = merged.reshape(-1, 1, 2).astype(np.float32)
            else:
                st.klt_points = existing.reshape(-1, 1, 2).astype(np.float32)
        else:
            st.klt_points = new_pts

    def _klt_propagate(self, st: CamTrackerState,
                        gray: np.ndarray
                        ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Run KLT to propagate st.klt_points from last_gray to gray.
        Returns (surviving_points_xy, flow_vectors) — both (N, 2) — or
        (None, None) if propagation failed."""
        if st.klt_points is None or len(st.klt_points) == 0:
            return None, None
        if st.last_gray is None or st.last_gray.shape != gray.shape:
            return None, None
        new_pts, status, err = cv2.calcOpticalFlowPyrLK(
            st.last_gray, gray, st.klt_points, None,
            winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                      30, 0.01))
        if new_pts is None:
            return None, None
        ok = (status.flatten() == 1)
        # Also reject points that left the frame
        new_flat = new_pts.reshape(-1, 2)
        old_flat = st.klt_points.reshape(-1, 2)
        in_frame = (
            (new_flat[:, 0] >= 0) & (new_flat[:, 0] < self.W)
            & (new_flat[:, 1] >= 0) & (new_flat[:, 1] < self.H)
        )
        keep = ok & in_frame
        if not keep.any():
            return None, None
        survived_new = new_flat[keep]
        survived_old = old_flat[keep]
        flows = survived_new - survived_old   # per-point (vx, vy)
        return survived_new.astype(np.float32), flows.astype(np.float32)

    def _observe(self,
                  points: np.ndarray, flows: np.ndarray,
                  pred_cx: float, pred_cy: float, pred_r: float,
                  frame_idx: int) -> Optional[HullObservation]:
        """Build an observation from this frame's KLT points + flows.

        1. Keep only moving points (|v| > motion_min).
        2. Cluster by spatial + velocity proximity.
        3. Pick the cluster nearest the Kalman prediction.
        4. Convex hull + dilation = rat ROI mask.
        """
        speeds = np.linalg.norm(flows, axis=1)
        moving = speeds > self.motion_min
        if moving.sum() < self.min_cluster_points:
            return None
        m_points = points[moving]
        m_flows  = flows[moving]

        labels = _dbscan_xy_v(m_points, m_flows,
                                eps_xy=self.dbscan_eps_xy,
                                eps_v=self.dbscan_eps_v,
                                min_samples=self.min_cluster_points)
        if labels.max() < 0:
            return None  # all noise

        # Pick the cluster whose mean position is nearest pred (or,
        # if Kalman hasn't locked on yet, the LARGEST cluster)
        kalman_locked = (self._cams[0].last_obs_idx >= 0
                          or self._cams[1].last_obs_idx >= 0)
        best_label = -1
        best_score = float("inf")
        for lbl in range(labels.max() + 1):
            mask = labels == lbl
            if mask.sum() < self.min_cluster_points:
                continue
            cx_ = float(m_points[mask, 0].mean())
            cy_ = float(m_points[mask, 1].mean())
            if kalman_locked:
                score = np.hypot(cx_ - pred_cx, cy_ - pred_cy)
            else:
                # Prefer biggest cluster on first observation
                score = -mask.sum()
            if score < best_score:
                best_score = score
                best_label = lbl
        if best_label < 0:
            return None

        cluster_pts = m_points[labels == best_label]
        cx_ = float(cluster_pts[:, 0].mean())
        cy_ = float(cluster_pts[:, 1].mean())
        # Radius as the 95th percentile of point distance from
        # cluster center — robust to a few outliers.
        d = np.hypot(cluster_pts[:, 0] - cx_, cluster_pts[:, 1] - cy_)
        radius = float(np.percentile(d, 95)) if len(d) > 0 else 30.0

        # Build hull mask
        hull = cv2.convexHull(
            cluster_pts.astype(np.int32).reshape(-1, 1, 2))
        mask = np.zeros((self.H, self.W), dtype=np.uint8)
        cv2.fillConvexPoly(mask, hull, 255)
        # Dilate to convert edge-skeleton hull to full-body mask
        dilate_k = self.body_half_width_px
        if dilate_k > 0:
            kern = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * dilate_k + 1, 2 * dilate_k + 1))
            mask = cv2.dilate(mask, kern, iterations=1)

        return HullObservation(
            cx=cx_, cy=cy_, radius=radius,
            n_points=int(len(cluster_pts)),
            points=cluster_pts,
            hull=hull.reshape(-1, 2),
            mask=mask,
            from_kalman_only=False)

    def _hull_from_kalman(self, cx: float, cy: float, r: float
                          ) -> Optional[HullObservation]:
        """Fallback: when no current-frame observation, paint a
        disc at the Kalman-predicted location. Conservative radius."""
        if not (0 <= cx <= self.W and 0 <= cy <= self.H):
            return None
        radius = max(30.0, float(r))
        mask = np.zeros((self.H, self.W), dtype=np.uint8)
        cv2.circle(mask, (int(cx), int(cy)),
                    int(radius + self.body_half_width_px),
                    255, -1)
        return HullObservation(
            cx=cx, cy=cy, radius=radius,
            n_points=0,
            points=np.zeros((0, 2), dtype=np.float32),
            hull=np.zeros((0, 2), dtype=np.float32),
            mask=mask,
            from_kalman_only=True)

    # ----------------------------------------------------------------
    #  Diagnostics
    # ----------------------------------------------------------------

    def snapshot(self, cam: int) -> dict:
        """Return a small dict describing current state, for logging
        or per-frame diagnostics."""
        st = self._cams[cam]
        n_points = (0 if st.klt_points is None
                    else int(len(st.klt_points)))
        if st.kalman is not None:
            sp = st.kalman.statePost.flatten()
            kalman_state = dict(cx=float(sp[0]), cy=float(sp[1]),
                                r=float(sp[2]),
                                vx=float(sp[3]), vy=float(sp[4]))
        else:
            kalman_state = None
        return dict(n_klt_points=n_points,
                    last_obs_idx=st.last_obs_idx,
                    has_init=st.has_init,
                    kalman_state=kalman_state)
