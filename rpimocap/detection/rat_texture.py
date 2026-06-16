"""
rpimocap.detection.rat_texture
==============================
Texture-based rat identification using a multi-scale, multi-
orientation Gabor feature bank fit by a multivariate Gaussian.
The model is built incrementally from random-frame bootstrap
samples, then refined online during tracking from confident
detections. Each refit that produces a significant drift in the
model mean increments a monotonic version_id, so the caller can
record which version processed each frame and re-process outdated
frames later.

Rationale
---------
Intensity-based bg-subtraction can't distinguish a bright cable
specular reflection from a bright rat — both are bright pixels.
Texture features can: rat fur produces moderate Gabor response
across orientations (roughly isotropic), the cable wire produces
high response in ONE orientation (anisotropic), bedding produces
strong response at fine scales but weak at coarse, and acrylic
glints produce very low Gabor response (smooth).

In 12-D Gabor-feature space (4 orientations × 3 scales) the
rat's distribution is geometrically separable from the
non-rat-but-bright clusters that have been costing us detections.

The bank doesn't need annotated data. It bootstraps from random
frames by computing features over whatever bg-sub blobs are
present, accepting that bootstrap samples are contaminated by
non-rat detections. The single-Gaussian fit lands somewhere
between rat and non-rat clusters; the rat-dominant majority pulls
the mean toward rat values. As tracking progresses, online
updates from blobs that score above threshold (closer to the
current mean) further pull the model toward the rat cluster — a
form of unsupervised consolidation.

Algorithm (per-frame, after bootstrap)
--------------------------------------
1. Standard bg-sub produces candidate blobs (CCs after morph).
2. For each blob, compute average Gabor feature vector inside the
   blob mask. Bounding-box crop for speed.
3. Score blob via Gaussian PDF (Mahalanobis distance → softmax).
4. Reject blobs with score below threshold.
5. For kept blobs, add the feature vector to the online update
   buffer. When the buffer fills, refit the Gaussian using a
   numerically-stable combine-statistics update (Chan et al.).
6. If the refit's mean differs from the previous by > drift
   threshold, increment version_id.

The model is symmetric in cameras — one bank is shared between
cam0 and cam1. The rat's surface texture is geometry-invariant.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import cv2


# ────────────────────────────────────────────────────────────────────
#  Defaults
# ────────────────────────────────────────────────────────────────────

DEFAULT_ORIENTATIONS = (0.0, np.pi / 4.0, np.pi / 2.0, 3.0 * np.pi / 4.0)
DEFAULT_SCALES = (7, 11, 17)


# ────────────────────────────────────────────────────────────────────
#  Gabor kernel bank
# ────────────────────────────────────────────────────────────────────


def build_gabor_kernels(orientations: Sequence[float],
                         scales: Sequence[int]
                         ) -> list[np.ndarray]:
    """Build a bank of Gabor kernels at every (scale, orientation)
    pair. Each kernel is L1-normalized to make responses
    intensity-comparable across scales.

    Layout of the returned list: outer loop over scale, inner loop
    over orientation. Total length = len(scales) * len(orientations).
    The same layout is mirrored in feature vectors: features[
    s * n_orient + o] is the response at scale s, orientation o.
    """
    kernels: list[np.ndarray] = []
    for ksize in scales:
        sigma = ksize / 4.0
        lam   = ksize / 2.0
        for theta in orientations:
            k = cv2.getGaborKernel(
                (ksize, ksize),
                sigma=sigma, theta=theta,
                lambd=lam, gamma=0.5, psi=0.0,
                ktype=cv2.CV_32F)
            # L1-normalize so |response| is comparable across scales
            norm = np.abs(k).sum()
            if norm > 0:
                k = k / norm
            kernels.append(k)
    return kernels


# ────────────────────────────────────────────────────────────────────
#  RatTextureBank
# ────────────────────────────────────────────────────────────────────


@dataclass
class _OnlineState:
    """Buffer of pending samples + counters for drift logging."""
    buffer:        list = field(default_factory=list)
    n_total_seen:  int  = 0
    n_total_kept:  int  = 0
    last_refit_n:  int  = 0


class RatTextureBank:
    """Texture-based rat ID using Gabor features + Gaussian model
    with online updates and version_id tracking.

    Lifecycle
    ---------
        bank = RatTextureBank()
        # Phase 1 — bootstrap (one-shot)
        feats = []
        for frame, blob_mask in random_bootstrap_samples:
            feats.append(bank.features_in_blob(gray, blob_mask))
        bank.bootstrap(feats)                  # version_id → 1

        # Phase 2 — tracking with online updates
        for frame, blobs in tracking_loop:
            for mask in blobs:
                feats = bank.features_in_blob(gray, mask)
                score = bank.score(feats)
                if score >= min_score:
                    bank.add_sample(feats)     # buffers, may refit
                else:
                    drop_blob()
            log_frame_version(frame_idx, bank.version_id)

        bank.save('texture_bank.npz')
    """

    def __init__(self,
                 orientations: Sequence[float] = DEFAULT_ORIENTATIONS,
                 scales:       Sequence[int]  = DEFAULT_SCALES,
                 update_every:  int   = 100,
                 drift_threshold: float = 0.05,
                 reg_eps:       float = 1e-3,
                 max_history_samples: int = 10000):
        self.orientations  = tuple(float(o) for o in orientations)
        self.scales        = tuple(int(s) for s in scales)
        self.feature_dim   = len(self.orientations) * len(self.scales)
        self._kernels      = build_gabor_kernels(
                                self.orientations, self.scales)

        self.update_every  = int(update_every)
        self.drift_threshold = float(drift_threshold)
        self.reg_eps       = float(reg_eps)
        self.max_history_samples = int(max_history_samples)

        # Trained-model state
        self.version_id    = 0
        self.n_samples     = 0
        self.mean: Optional[np.ndarray]    = None  # (D,)
        self.cov:  Optional[np.ndarray]    = None  # (D, D)
        self._inv_cov: Optional[np.ndarray] = None

        # Online accumulation
        self._online = _OnlineState()

    # ----------------------------------------------------------------
    #  Status
    # ----------------------------------------------------------------

    @property
    def is_ready(self) -> bool:
        """True iff the bank has been bootstrapped and can score."""
        return (self.mean is not None
                and self._inv_cov is not None
                and self.version_id >= 1)

    def stats(self) -> dict:
        """Snapshot for logging / step_stats."""
        return dict(version_id=self.version_id,
                    n_samples=self.n_samples,
                    buffered=len(self._online.buffer),
                    seen=self._online.n_total_seen,
                    kept=self._online.n_total_kept,
                    last_refit_n=self._online.last_refit_n)

    # ----------------------------------------------------------------
    #  Feature extraction
    # ----------------------------------------------------------------

    def features_in_blob(self,
                          gray:  np.ndarray,
                          mask:  np.ndarray,
                          pad:   int = 10
                          ) -> np.ndarray:
        """Compute mean |Gabor response| inside the blob.

        gray  : (H, W) grayscale frame (uint8 or float32)
        mask  : (H, W) blob mask, nonzero = inside

        Returns the D-dimensional feature vector. If the blob is
        empty or shape-mismatched, returns a zero vector — caller
        is responsible for skipping that blob.
        """
        if mask.shape != gray.shape:
            return np.zeros(self.feature_dim, dtype=np.float32)
        ys, xs = np.where(mask > 0)
        if len(ys) < 5:
            return np.zeros(self.feature_dim, dtype=np.float32)

        # Crop to bounding box + padding for kernel context
        y0 = max(0, int(ys.min()) - pad)
        y1 = min(gray.shape[0], int(ys.max()) + pad + 1)
        x0 = max(0, int(xs.min()) - pad)
        x1 = min(gray.shape[1], int(xs.max()) + pad + 1)

        crop = gray[y0:y1, x0:x1].astype(np.float32)
        local_mask = mask[y0:y1, x0:x1] > 0
        if not local_mask.any():
            return np.zeros(self.feature_dim, dtype=np.float32)

        feats = np.empty(self.feature_dim, dtype=np.float32)
        for i, kern in enumerate(self._kernels):
            resp = cv2.filter2D(crop, cv2.CV_32F, kern)
            # |response| inside the blob
            feats[i] = float(np.abs(resp[local_mask]).mean())
        return feats

    # ----------------------------------------------------------------
    #  Scoring
    # ----------------------------------------------------------------

    def mahalanobis_squared(self, features: np.ndarray) -> float:
        """d² = (x - μ)ᵀ Σ⁻¹ (x - μ). Returns ∞ if not ready."""
        if not self.is_ready:
            return float("inf")
        d = features - self.mean
        return float(d @ self._inv_cov @ d)

    def score(self, features: np.ndarray) -> float:
        """Texture similarity in [0, 1].

        Score is exp(-d² / (2 D)) where D is the feature dim, so a
        sample exactly on the mean scores 1.0 and a sample at the
        characteristic Mahalanobis scale (√D σ-deviations) scores
        ~0.6. Threshold around 0.3 in practice.

        Before bootstrap (is_ready=False) returns 1.0 — the gate
        is a no-op until trained.
        """
        if not self.is_ready:
            return 1.0
        d2 = self.mahalanobis_squared(features)
        if not np.isfinite(d2):
            return 0.0
        return float(np.exp(-d2 / (2.0 * self.feature_dim)))

    # ----------------------------------------------------------------
    #  Bootstrap
    # ----------------------------------------------------------------

    def bootstrap(self, feature_vectors: Sequence[np.ndarray]):
        """Initial fit from a batch of feature vectors. Sets
        version_id = 1."""
        if len(feature_vectors) < 5:
            raise ValueError(
                f"Need at least 5 bootstrap samples, got "
                f"{len(feature_vectors)}")
        X = np.stack([np.asarray(v, dtype=np.float32)
                       for v in feature_vectors])
        if X.shape[1] != self.feature_dim:
            raise ValueError(
                f"Feature dim mismatch: expected {self.feature_dim}, "
                f"got {X.shape[1]}")
        self.mean = X.mean(axis=0).astype(np.float32)
        diff = X - self.mean
        self.cov = ((diff.T @ diff) / max(1, len(X) - 1)).astype(np.float32)
        self.cov += np.eye(self.feature_dim, dtype=np.float32) * self.reg_eps
        self._inv_cov = np.linalg.inv(self.cov).astype(np.float32)
        self.n_samples = int(len(X))
        self.version_id = 1
        # Clear any pending online state from a previous lifecycle
        self._online = _OnlineState()
        self._online.last_refit_n = self.n_samples

    # ----------------------------------------------------------------
    #  Online updates
    # ----------------------------------------------------------------

    def add_sample(self, features: np.ndarray) -> bool:
        """Buffer a sample for the next refit. Returns True if a
        refit was triggered THIS call (caller can use to log a
        version transition immediately)."""
        f = np.asarray(features, dtype=np.float32)
        if f.shape != (self.feature_dim,):
            return False
        self._online.buffer.append(f.copy())
        self._online.n_total_seen += 1
        self._online.n_total_kept += 1
        if len(self._online.buffer) >= self.update_every:
            return self._refit_from_buffer()
        return False

    def _refit_from_buffer(self) -> bool:
        """Merge buffered samples into current model. Uses the
        Chan / Welford-style parallel-statistics combine, which is
        numerically stable for online use without storing the full
        history. Returns True if version_id incremented."""
        if not self._online.buffer:
            return False
        new_X = np.stack(self._online.buffer)
        n_new = len(new_X)
        new_mean = new_X.mean(axis=0)
        new_cov  = ((new_X - new_mean).T @ (new_X - new_mean)
                    / max(1, n_new - 1))

        if self.is_ready:
            old_mean = self.mean.copy()
            n_old = self.n_samples
            n_total = n_old + n_new
            # Combined mean
            combined_mean = (n_old * self.mean + n_new * new_mean) / n_total
            # Combined covariance via Chan et al. parallel formula
            d_old = self.mean - combined_mean
            d_new = new_mean - combined_mean
            combined_cov = (
                (n_old - 1) * self.cov + (n_new - 1) * new_cov
                + n_old * np.outer(d_old, d_old)
                + n_new * np.outer(d_new, d_new)
            ) / max(1, n_total - 1)
            self.mean = combined_mean.astype(np.float32)
            self.cov = combined_cov.astype(np.float32)
            self.cov += np.eye(self.feature_dim,
                                dtype=np.float32) * self.reg_eps
            self._inv_cov = np.linalg.inv(self.cov).astype(np.float32)
            # Cap n_samples to prevent infinite memory in the
            # weighted-average — after a lot of samples, new ones
            # have negligible effect otherwise.
            self.n_samples = min(n_total, self.max_history_samples)
            # Drift check
            old_norm = float(np.linalg.norm(old_mean))
            rel_change = float(
                np.linalg.norm(self.mean - old_mean)
                / max(old_norm, 1e-6))
            drifted = rel_change > self.drift_threshold
        else:
            # First fit
            self.bootstrap(list(new_X))
            drifted = False    # version went from 0 → 1, not a "drift"

        self._online.last_refit_n = n_new
        self._online.buffer.clear()
        if drifted:
            self.version_id += 1
        return drifted

    def flush_pending(self) -> bool:
        """Force a refit even if the buffer isn't full. For use at
        end of session before save."""
        return self._refit_from_buffer()

    def refine_blob_mask(self,
                          gray:           np.ndarray,
                          hull_mask:      np.ndarray,
                          expand_px:      int   = 30,
                          score_threshold: float = 0.15,
                          smooth_window:  int   = 7
                          ) -> np.ndarray:
        """Grow an initial blob mask outward, including pixels
        whose local Gabor texture matches the bank, stopping where
        texture diverges. The result is a mask that snaps to the
        actual texture boundary of the rat rather than its convex
        hull.

        Parameters
        ----------
        gray            : (H, W) uint8 grayscale frame
        hull_mask       : (H, W) uint8 starting mask (e.g. the
                          merged convex hull). Pixels with value > 0
                          are seeds — they're always kept.
        expand_px       : maximum number of pixels to expand outward
                          from the hull. Defines the search band.
        score_threshold : per-pixel texture score required to
                          include. Lower than the blob-level
                          threshold because per-pixel feature
                          vectors are noisier. Typical 0.10-0.25.
        smooth_window   : odd box-filter size applied to each Gabor
                          response before scoring. Brings per-pixel
                          features closer to the blob-averaged
                          training distribution.

        Returns
        -------
        (H, W) uint8 mask, ≥ hull_mask in extent. The mask is the
        connected component(s) of the texture-pass region that
        overlap the original hull (geodesic constraint — disconnected
        islands of rat-texture noise are not included).

        Bank must be ready (is_ready True). If not, returns hull_mask
        unchanged.
        """
        if not self.is_ready:
            return hull_mask.copy()
        if hull_mask.shape != gray.shape:
            return hull_mask.copy()
        if int((hull_mask > 0).sum()) == 0:
            return hull_mask.copy()

        # Build search ROI: hull dilated by expand_px
        kern = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * expand_px + 1, 2 * expand_px + 1))
        roi = cv2.dilate(hull_mask.astype(np.uint8), kern)

        # Crop to bounding box for speed
        ys, xs = np.where(roi > 0)
        if len(ys) == 0:
            return hull_mask.copy()
        margin = max(self.scales) + 5  # account for largest Gabor kernel
        H, W = gray.shape
        y0 = max(0, int(ys.min()) - margin)
        y1 = min(H, int(ys.max()) + margin + 1)
        x0 = max(0, int(xs.min()) - margin)
        x1 = min(W, int(xs.max()) + margin + 1)

        crop_gray = gray[y0:y1, x0:x1].astype(np.float32)
        crop_hull = (hull_mask[y0:y1, x0:x1] > 0)
        crop_roi  = (roi[y0:y1, x0:x1] > 0)
        ch, cw = crop_gray.shape

        # Compute Gabor responses over the whole crop — one filter2D
        # per kernel. Box-filter each response with smooth_window to
        # bring per-pixel features closer to blob-averaged statistics.
        smooth_k = int(max(1, smooth_window) | 1)   # odd
        responses = np.empty(
            (self.feature_dim, ch, cw), dtype=np.float32)
        for i, kern_ in enumerate(self._kernels):
            r = np.abs(cv2.filter2D(crop_gray, cv2.CV_32F, kern_))
            if smooth_k > 1:
                r = cv2.boxFilter(r, cv2.CV_32F, (smooth_k, smooth_k))
            responses[i] = r

        # Score every pixel in the ROI band (between hull and dilated)
        candidate_mask = crop_roi & ~crop_hull
        if not candidate_mask.any():
            return hull_mask.copy()

        # Vectorized Mahalanobis: d² = diff @ inv_cov @ diff per pixel
        feats = responses[:, candidate_mask].T   # (N, D)
        diffs = feats - self.mean[None, :]       # (N, D)
        d2 = np.einsum("ni,ij,nj->n", diffs, self._inv_cov, diffs)
        # Clip d2 to avoid overflow in exp() — anything beyond ~30 is
        # already effectively zero score.
        d2 = np.clip(d2, 0, 200)
        scores = np.exp(-d2 / (2.0 * self.feature_dim))
        keep = scores >= score_threshold

        # Build refined mask in crop space
        refined = crop_hull.copy()
        cy_idx, cx_idx = np.where(candidate_mask)
        refined[cy_idx[keep], cx_idx[keep]] = True

        # Geodesic constraint: keep only CCs that touch the hull.
        # Disconnected islands of rat-texture noise far from the
        # rat are excluded.
        refined_u8 = refined.astype(np.uint8) * 255
        n_cc, cc_labels = cv2.connectedComponents(refined_u8)
        if n_cc <= 1:
            # No components survived
            full = np.zeros_like(hull_mask)
            full[y0:y1, x0:x1] = refined_u8
            return full
        # Find labels that overlap with the original hull
        hull_labels = set(int(l) for l in cc_labels[crop_hull].tolist()
                            if l > 0)
        if not hull_labels:
            full = np.zeros_like(hull_mask)
            full[y0:y1, x0:x1] = (crop_hull.astype(np.uint8) * 255)
            return full
        keep_mask = np.isin(cc_labels, list(hull_labels))
        final_crop = (keep_mask.astype(np.uint8) * 255)

        # Place back in full-image frame
        full = np.zeros_like(hull_mask)
        full[y0:y1, x0:x1] = final_crop
        return full

    # ----------------------------------------------------------------
    #  Persistence
    # ----------------------------------------------------------------

    def save(self, path: str | Path):
        """Save the trained bank to .npz. Must be is_ready."""
        if not self.is_ready:
            raise RuntimeError("cannot save an untrained bank")
        np.savez(
            str(path),
            version_id=np.array([self.version_id], dtype=np.int64),
            n_samples=np.array([self.n_samples], dtype=np.int64),
            mean=self.mean,
            cov=self.cov,
            orientations=np.array(self.orientations, dtype=np.float64),
            scales=np.array(self.scales, dtype=np.int64),
            update_every=np.array([self.update_every], dtype=np.int64),
            drift_threshold=np.array([self.drift_threshold],
                                       dtype=np.float64),
        )

    @classmethod
    def load(cls, path: str | Path) -> "RatTextureBank":
        """Load a trained bank. The Gabor kernels are rebuilt from
        stored orientations + scales — they're not stored to keep
        the file small and to allow OpenCV-version-independence."""
        data = np.load(str(path))
        bank = cls(
            orientations=tuple(float(o) for o in data["orientations"]),
            scales=tuple(int(s) for s in data["scales"]),
            update_every=int(data["update_every"][0])
                if "update_every" in data.files else 100,
            drift_threshold=float(data["drift_threshold"][0])
                if "drift_threshold" in data.files else 0.05,
        )
        bank.version_id = int(data["version_id"][0])
        bank.n_samples  = int(data["n_samples"][0])
        bank.mean = data["mean"].astype(np.float32)
        bank.cov  = data["cov"].astype(np.float32)
        bank._inv_cov = np.linalg.inv(bank.cov).astype(np.float32)
        return bank


# ────────────────────────────────────────────────────────────────────
#  Bootstrap helper
# ────────────────────────────────────────────────────────────────────


def bootstrap_from_random_frames(
        bank:           RatTextureBank,
        sample_features: list[np.ndarray],
        min_samples:    int = 20) -> RatTextureBank:
    """Convenience wrapper. Given a list of feature vectors
    collected externally (e.g., by running detection on random
    frames in the calling pipeline), bootstrap the bank if there
    are enough samples. Raises if too few."""
    if len(sample_features) < min_samples:
        raise RuntimeError(
            f"too few bootstrap samples: got {len(sample_features)}, "
            f"need at least {min_samples}. Try sampling more frames "
            f"or relaxing detection thresholds during bootstrap.")
    bank.bootstrap(sample_features)
    return bank
