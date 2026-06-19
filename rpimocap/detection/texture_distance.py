"""
rpimocap.detection.texture_distance
====================================
Texture-change foreground detection (diagnostic / experimental).

The production detector is intensity-bg-sub-first, texture-check-
second: the texture bank only ever sees blobs that bg-sub already
produced. When bg-sub fragments (forcing tiny --min-area) or an
artifact defeats the Mahalanobis gate (the cam1 specular-reflection
failure), the texture stage never runs.

This module inverts that ordering for evaluation. It makes *texture
difference from the background* the primary signal:

  1. Build a per-pixel background texture model (mean + std of the
     rotation-invariant Gabor descriptor) from background frames.
  2. For a current frame, compute the same dense descriptor and
     measure, per pixel, the Mahalanobis-style distance to the
     background model.
  3. The distance map lights up wherever texture has *changed* —
     i.e. where fur has replaced bedding — and stays dark on static
     features (acrylic reflections, frame rails) because those have
     the same texture in background and current frame.

Why this is structurally better for the failure modes seen:
  * Static bright artifacts are already in the background, so their
    texture-distance is ~0 — rejected without any artifact mask.
  * Texture distance is intensity-invariant, so IR flicker /
    exposure drift don't trigger it.
  * Working at patch granularity + spatial smoothing yields one
    coherent region instead of a fragmented constellation, so the
    aggressive --min-area / merge machinery isn't needed.

This is deliberately a standalone diagnostic: it does NOT touch the
ForegroundDetector pipeline. Run texture_distance_map on the
existing diagnostic frames and compare the output to the current
stage1_bg_sub.png to decide whether the full graph-cut
(MRF) segmentation is worth building.

Honest caveats:
  * Two cameras give 2D silhouettes; the *volume* is a loose visual
    hull, but the silhouette centroid triangulates to the 3D
    position you need.
  * Bedding disturbance genuinely changes texture; a region the rat
    kicked can false-positive after it leaves. The background model
    needs to adapt (same problem class as --bg-adapt-alpha).
  * Wobbling specular artifacts are only rejected cleanly if the
    background model captures texture *variance*, not just mean —
    hence the per-pixel std stored here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import cv2
import numpy as np


# ────────────────────────────────────────────────────────────────────
#  Dense per-pixel rotation-invariant Gabor descriptor
# ────────────────────────────────────────────────────────────────────


def dense_gabor_descriptor(
        gray:        np.ndarray,
        kernels:     Sequence[np.ndarray],
        n_orient:    int,
        n_scales:    int,
        smooth_k:    int = 7,
        rotation_invariant: bool = True,
        ) -> np.ndarray:
    """Compute a dense per-pixel Gabor texture descriptor.

    This is the same descriptor the RatTextureBank uses (max / mean /
    std pooled across orientations per scale, for rotation
    invariance), but produced densely for every pixel rather than
    averaged inside a blob mask. The artifact-mask builder already
    computed this inline; this is the factored-out, reusable version.

    Parameters
    ----------
    gray      : (H, W) grayscale frame (uint8 or float)
    kernels   : flat list of Gabor kernels, ordered
                scale-major / orientation-minor (i.e. index
                = scale_idx * n_orient + orient_idx), exactly as
                RatTextureBank._kernels is built.
    n_orient  : number of orientations
    n_scales  : number of scales
    smooth_k  : box-filter window applied to each |response| (px).
                Smoothing makes the descriptor less noisy and gives
                each pixel a small spatial-context window. 0/1
                disables.
    rotation_invariant
              : when True, pool the n_orient responses per scale into
                (max, mean, std) → 3*n_scales features. When False,
                keep all n_orient*n_scales raw responses.

    Returns
    -------
    (D, H, W) float32 descriptor stack, where
        D = 3 * n_scales        (rotation_invariant=True)
        D = n_orient * n_scales  (rotation_invariant=False)
    """
    gray_f = gray.astype(np.float32)
    H, W = gray_f.shape

    if rotation_invariant:
        D = 3 * n_scales
        out = np.empty((D, H, W), dtype=np.float32)
        for s_idx in range(n_scales):
            resp_o = np.empty((n_orient, H, W), dtype=np.float32)
            for o_idx in range(n_orient):
                kern = kernels[s_idx * n_orient + o_idx]
                r = np.abs(cv2.filter2D(gray_f, cv2.CV_32F, kern))
                if smooth_k > 1:
                    r = cv2.boxFilter(r, cv2.CV_32F, (smooth_k, smooth_k))
                resp_o[o_idx] = r
            out[s_idx * 3 + 0] = resp_o.max(axis=0)
            out[s_idx * 3 + 1] = resp_o.mean(axis=0)
            out[s_idx * 3 + 2] = resp_o.std(axis=0)
    else:
        D = n_orient * n_scales
        out = np.empty((D, H, W), dtype=np.float32)
        for i, kern in enumerate(kernels):
            r = np.abs(cv2.filter2D(gray_f, cv2.CV_32F, kern))
            if smooth_k > 1:
                r = cv2.boxFilter(r, cv2.CV_32F, (smooth_k, smooth_k))
            out[i] = r
    return out


# ────────────────────────────────────────────────────────────────────
#  Per-pixel background texture model (mean + std of descriptor)
# ────────────────────────────────────────────────────────────────────


@dataclass
class BackgroundTextureModel:
    """Per-pixel mean + std of the dense Gabor descriptor, learned
    from background frames.

    The distance of a current frame's descriptor to this model
    (normalized by per-pixel std) is the texture-change signal. The
    std term is what lets wobbling specular artifacts be rejected:
    a reflection that flickers has a large background std, so the
    same flicker in the current frame produces a small normalized
    distance.

    Build it by calling .accumulate(descriptor) for each background
    frame's dense descriptor, then .finalize(). Or use the
    convenience builder build_background_texture_model().
    """
    mean:  Optional[np.ndarray] = None     # (D, H, W)
    std:   Optional[np.ndarray] = None     # (D, H, W)
    n:     int = 0
    # Welford running accumulators (internal)
    _M2:   Optional[np.ndarray] = field(default=None, repr=False)
    _mean: Optional[np.ndarray] = field(default=None, repr=False)

    def accumulate(self, descriptor: np.ndarray) -> None:
        """Add one frame's dense descriptor (D, H, W) via Welford's
        online algorithm, accumulating per-pixel mean and variance."""
        if self._mean is None:
            self._mean = descriptor.astype(np.float64).copy()
            self._M2   = np.zeros_like(self._mean)
            self.n     = 1
            return
        self.n += 1
        delta  = descriptor - self._mean
        self._mean += delta / self.n
        delta2 = descriptor - self._mean
        self._M2 += delta * delta2

    def finalize(self, std_floor: float = 1e-3) -> "BackgroundTextureModel":
        """Compute final mean and std arrays from the accumulators.
        std_floor prevents divide-by-zero where a descriptor channel
        was perfectly constant across background frames."""
        if self._mean is None or self.n < 1:
            raise RuntimeError(
                "no frames accumulated; call accumulate() first")
        self.mean = self._mean.astype(np.float32)
        if self.n < 2:
            # Single frame — no variance estimate; use a flat std
            self.std = np.full_like(self.mean, std_floor)
        else:
            var = self._M2 / (self.n - 1)
            self.std = np.sqrt(var).astype(np.float32)
            self.std = np.maximum(self.std, std_floor)
        return self

    # ── persistence ────────────────────────────────────────────────

    def save(self, path: str) -> None:
        if self.mean is None or self.std is None:
            raise RuntimeError("model not finalized; call finalize()")
        np.savez_compressed(path, mean=self.mean, std=self.std,
                            n=np.int64(self.n))

    @classmethod
    def load(cls, path: str) -> "BackgroundTextureModel":
        d = np.load(path)
        m = cls(mean=d["mean"], std=d["std"], n=int(d["n"]))
        return m


def build_background_texture_model(
        bg_frames:   Sequence[np.ndarray],
        kernels:     Sequence[np.ndarray],
        n_orient:    int,
        n_scales:    int,
        smooth_k:    int = 7,
        rotation_invariant: bool = True,
        std_floor:   float = 1e-3,
        ) -> BackgroundTextureModel:
    """Convenience builder: accumulate a texture model over a list of
    background frames."""
    model = BackgroundTextureModel()
    for f in bg_frames:
        desc = dense_gabor_descriptor(
            f, kernels, n_orient, n_scales,
            smooth_k=smooth_k,
            rotation_invariant=rotation_invariant)
        model.accumulate(desc)
    return model.finalize(std_floor=std_floor)


# ────────────────────────────────────────────────────────────────────
#  Texture-distance map (the diagnostic signal)
# ────────────────────────────────────────────────────────────────────


def texture_distance_map(
        gray:        np.ndarray,
        model:       BackgroundTextureModel,
        kernels:     Sequence[np.ndarray],
        n_orient:    int,
        n_scales:    int,
        smooth_k:    int = 7,
        rotation_invariant: bool = True,
        roi_mask:    Optional[np.ndarray] = None,
        post_smooth_k: int = 15,
        ) -> np.ndarray:
    """Compute a per-pixel texture-distance map between a current
    frame and the background texture model.

    For each pixel, the distance is the RMS over descriptor channels
    of the per-channel z-score:

        dist[y,x] = sqrt( mean_d( ((desc[d,y,x] - mean[d,y,x])
                                    / std[d,y,x])^2 ) )

    This is a diagonal-covariance Mahalanobis distance — it uses the
    per-pixel, per-channel std but assumes channels are independent
    (cheap and adequate for a diagnostic; the full covariance is the
    obvious upgrade if this proves out).

    Parameters
    ----------
    gray         : (H, W) current grayscale frame
    model        : finalized BackgroundTextureModel
    kernels/n_orient/n_scales/smooth_k/rotation_invariant
                 : must match what built the model
    roi_mask     : optional (H, W); pixels outside are set to 0
    post_smooth_k: box-filter window applied to the final distance
                   map (px). This is the crude stand-in for the
                   smoothness term of a full MRF — it consolidates
                   the per-pixel signal into coherent regions.
                   0/1 disables.

    Returns
    -------
    (H, W) float32 distance map. Higher = more texture change from
    background. Static features (same texture in bg and current)
    are near 0.
    """
    if model.mean is None or model.std is None:
        raise RuntimeError("model not finalized; call finalize()")
    desc = dense_gabor_descriptor(
        gray, kernels, n_orient, n_scales,
        smooth_k=smooth_k,
        rotation_invariant=rotation_invariant)
    # Per-channel z-score, then RMS across channels
    z = (desc - model.mean) / model.std
    dist = np.sqrt(np.mean(z * z, axis=0)).astype(np.float32)
    if roi_mask is not None:
        dist = dist * (roi_mask > 0).astype(np.float32)
    if post_smooth_k and post_smooth_k > 1:
        dist = cv2.boxFilter(dist, cv2.CV_32F,
                            (post_smooth_k, post_smooth_k))
    return dist


def threshold_distance_map(
        dist:        np.ndarray,
        method:      str = "otsu",
        abs_thresh:  float = 3.0,
        percentile:  float = 95.0,
        min_area_px: int = 1000,
        morph_close_k: int = 7,
        ) -> tuple[np.ndarray, float]:
    """Threshold a texture-distance map into a binary foreground
    mask, keep connected components above min_area_px.

    method:
      "otsu"       — Otsu's threshold on the distance histogram
      "absolute"   — fixed abs_thresh (z-score units)
      "percentile" — the given percentile of in-ROI distances

    Returns (mask_uint8, threshold_used). This crude threshold +
    CC-filter is the cheap stand-in for the graph-cut; if the
    resulting mask cleanly isolates the rat we know the full MRF is
    worth building.
    """
    nz = dist[dist > 0]
    if nz.size == 0:
        return np.zeros_like(dist, dtype=np.uint8), 0.0

    if method == "otsu":
        # Otsu needs uint8; scale the distance to 0-255
        dmax = float(nz.max())
        scaled = np.clip(dist / (dmax + 1e-6) * 255.0,
                        0, 255).astype(np.uint8)
        thr_u8, _ = cv2.threshold(
            scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thr = float(thr_u8) / 255.0 * dmax
    elif method == "percentile":
        thr = float(np.percentile(nz, percentile))
    else:  # absolute
        thr = float(abs_thresh)

    mask = (dist >= thr).astype(np.uint8) * 255
    if morph_close_k and morph_close_k > 0:
        kern = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (morph_close_k, morph_close_k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kern)
    # Keep CCs above min_area
    n_cc, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    out = np.zeros_like(mask)
    for i in range(1, n_cc):
        if stats[i, cv2.CC_STAT_AREA] >= min_area_px:
            out[labels == i] = 255
    return out, thr


def colorize_distance_map(dist: np.ndarray,
                            vmax: Optional[float] = None) -> np.ndarray:
    """Render a distance map as a BGR heatmap for diagnostics
    (blue=low/no-change, red=high/large-texture-change)."""
    if vmax is None:
        nz = dist[dist > 0]
        vmax = float(np.percentile(nz, 99)) if nz.size else 1.0
    scaled = np.clip(dist / (vmax + 1e-6) * 255.0, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(scaled, cv2.COLORMAP_JET)
