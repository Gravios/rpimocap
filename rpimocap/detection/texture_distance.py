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
#  Static shadow / illumination model (flat-field normalization)
# ────────────────────────────────────────────────────────────────────


def build_illumination_field(
        frames:      Sequence[np.ndarray],
        blur_sigma:  float = 0.0,
        roi_mask:    Optional[np.ndarray] = None,
        ) -> np.ndarray:
    """Estimate the static per-pixel illumination ('shadow') field
    from a stack of frames.

    The arena has a fixed IR illumination falloff — brighter under
    the emitter, dimmer at the edges. The same surface texture
    therefore produces a stronger Gabor response where it's well-lit
    and a weaker one in shadow, so the illumination gradient is baked
    into the texture descriptor and gives the distance map uneven
    spatial sensitivity.

    With the rat present in every frame but only ever in one place,
    the per-pixel temporal MEDIAN of intensity is the static scene —
    the rat is rejected as a minority outlier (same logic as the
    persistent texture model). That median IS the illumination field
    (plus static structure).

    Parameters
    ----------
    frames     : grayscale frames spread across the session.
    blur_sigma : if > 0, Gaussian-blur the median field to isolate
                 only the LOW-FREQUENCY illumination falloff (the
                 'shadow model' proper), leaving sharp static
                 structures (rails, reflections) out of the field so
                 they aren't divided away. If 0, return the raw
                 median field (full flat-field — removes illumination
                 AND static structure).
    roi_mask   : optional; outside-ROI pixels are filled with the
                 in-ROI mean so the divide is well-defined there.

    Returns
    -------
    (H, W) float32 illumination field, strictly positive (floored).
    """
    if len(frames) < 3:
        raise RuntimeError(
            "need at least 3 frames for an illumination field")
    stack = np.stack([f.astype(np.float32) for f in frames], axis=0)
    field = np.median(stack, axis=0).astype(np.float32)
    if blur_sigma and blur_sigma > 0:
        # Kernel size ~6 sigma, odd
        k = int(2 * round(3 * blur_sigma) + 1)
        field = cv2.GaussianBlur(field, (k, k), blur_sigma)
    if roi_mask is not None:
        m = roi_mask > 0
        if m.any():
            field[~m] = float(field[m].mean())
    # Floor to keep the divide well-defined
    field = np.maximum(field, 1.0)
    return field


def apply_illumination_correction(
        gray:            np.ndarray,
        illumination:    np.ndarray,
        target_level:    Optional[float] = None,
        ) -> np.ndarray:
    """Flat-field correct a frame by the illumination field.

        corrected = gray / illumination * target_level

    This flattens the static shadow gradient so the same texture
    yields the same descriptor everywhere in the arena. The rat —
    which is NOT part of the static illumination field — keeps its
    contrast against the now-uniform background.

    Parameters
    ----------
    gray         : (H, W) frame to correct (uint8 or float)
    illumination : (H, W) field from build_illumination_field
    target_level : the post-correction reference brightness. Defaults
                   to the mean of the illumination field (so overall
                   brightness is preserved). Pixels where the frame
                   matches the field map to ~target_level.

    Returns
    -------
    (H, W) uint8 illumination-corrected frame, clipped to [0, 255].
    """
    g = gray.astype(np.float32)
    field = np.maximum(illumination.astype(np.float32), 1.0)
    if target_level is None:
        target_level = float(field.mean())
    corrected = g / field * target_level
    return np.clip(corrected, 0, 255).astype(np.uint8)


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
    background frames using Welford running mean/std.

    NOTE: the running mean is contaminated by the rat — wherever the
    rat sits during the sampled frames, that pixel's mean is pulled
    toward fur texture. For a session where the rat is present in
    every frame, prefer build_persistent_texture_model() which uses
    a per-pixel temporal MEDIAN (robust to the moving rat, since any
    given pixel shows background in the majority of frames)."""
    model = BackgroundTextureModel()
    for f in bg_frames:
        desc = dense_gabor_descriptor(
            f, kernels, n_orient, n_scales,
            smooth_k=smooth_k,
            rotation_invariant=rotation_invariant)
        model.accumulate(desc)
    return model.finalize(std_floor=std_floor)


def build_persistent_texture_model(
        bg_frames:   Sequence[np.ndarray],
        kernels:     Sequence[np.ndarray],
        n_orient:    int,
        n_scales:    int,
        smooth_k:    int = 7,
        rotation_invariant: bool = True,
        std_floor:   float = 1e-3,
        ) -> tuple["BackgroundTextureModel", np.ndarray]:
    """Build a background texture model robust to the moving rat,
    plus a persistence (temporal-stability) map.

    The key insight for a session where the rat is in every frame:
    any single pixel is BACKGROUND in the majority of frames (the rat
    is only ever in one place at a time). So the per-pixel temporal
    MEDIAN of the descriptor is the persistent background texture —
    the rat, being a transient minority at any given pixel, is
    rejected as an outlier. This is far better than a running mean,
    which the rat contaminates.

    The per-pixel temporal MAD (median absolute deviation) gives a
    robust spread estimate, used both as the distance-normalizer
    (like std) and to build the persistence map.

    Persistence map:
        Pixels whose descriptor is STABLE over the session (low
        temporal MAD) are persistent structure — bedding, frame
        rails, static reflections. Pixels with HIGH temporal MAD are
        where things move (rat paths, cable, disturbed bedding). The
        map is normalized to [0, 1] where 1 = perfectly persistent.

    Frames should be sampled across the WHOLE session (large stride),
    not a front window, so that lighting/bedding drift is captured in
    the median rather than read as foreground later.

    Parameters
    ----------
    bg_frames : list of grayscale frames spread across the session.
                More frames = better median; 40-80 is reasonable.
    (others)  : as dense_gabor_descriptor.

    Returns
    -------
    (model, persistence_map)
      model           : BackgroundTextureModel with mean=median,
                        std=MAD-derived spread
      persistence_map : (H, W) float32 in [0, 1]; 1 = most persistent
    """
    if len(bg_frames) < 3:
        raise RuntimeError(
            "need at least 3 frames for a median texture model")
    # Stack all descriptors: (T, D, H, W). Memory note — for a
    # 2028×1080 frame with D=9 and T=60 this is ~1.2 GB float32, so
    # downstream callers may want to subsample spatially or cap T.
    descs = []
    for f in bg_frames:
        descs.append(dense_gabor_descriptor(
            f, kernels, n_orient, n_scales,
            smooth_k=smooth_k, rotation_invariant=rotation_invariant))
    stack = np.stack(descs, axis=0)            # (T, D, H, W)

    # Per-pixel, per-channel temporal median = persistent background
    median = np.median(stack, axis=0).astype(np.float32)   # (D, H, W)
    # Median absolute deviation → robust spread
    mad = np.median(np.abs(stack - median[None]), axis=0)
    # Scale MAD to be std-comparable (for a normal dist, std≈1.4826*MAD)
    spread = (1.4826 * mad).astype(np.float32)
    spread = np.maximum(spread, std_floor)

    model = BackgroundTextureModel(
        mean=median, std=spread, n=len(bg_frames))

    # Persistence map: low channel-summed spread → high persistence.
    # Use the mean spread across channels as the instability measure.
    instability = spread.mean(axis=0)          # (H, W)
    # Normalize: map [p5, p95] of instability to [1, 0]
    lo = float(np.percentile(instability, 5))
    hi = float(np.percentile(instability, 95))
    norm = np.clip((instability - lo) / (hi - lo + 1e-6), 0, 1)
    persistence_map = (1.0 - norm).astype(np.float32)
    return model, persistence_map


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
        persistence_map: Optional[np.ndarray] = None,
        persistence_power: float = 1.0,
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
    persistence_map : optional (H, W) in [0,1] from
                   build_persistent_texture_model. Pixels with high
                   persistence (stable background) are damped by
                   (1 - persistence)^persistence_power, suppressing
                   static structure that the median model captures
                   imperfectly. This is what prevents the global
                   wash-out seen when bedding/lighting drift.
    persistence_power : exponent on the damping factor. >1 makes the
                   suppression of persistent pixels more aggressive.
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
    # Persistence gating. Multiply the distance by (1 - persistence)^p
    # so that pixels which are PERSISTENT background (high
    # persistence) are suppressed even if their instantaneous z-score
    # is high. This kills the global wash-out: a frame rail or
    # reflection that the median model captures imperfectly still has
    # high persistence, so its residual distance is damped. Transient
    # pixels (the rat) have low persistence → full distance kept.
    if persistence_map is not None:
        damp = np.clip(1.0 - persistence_map, 0.0, 1.0)
        if persistence_power != 1.0:
            damp = damp ** float(persistence_power)
        dist = dist * damp.astype(np.float32)
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
        max_aspect_ratio: float = 0.0,
        min_fill_ratio:   float = 0.0,
        ) -> tuple[np.ndarray, float]:
    """Threshold a texture-distance map into a binary foreground
    mask, keep connected components above min_area_px.

    method:
      "otsu"       — Otsu's threshold on the distance histogram
      "absolute"   — fixed abs_thresh (z-score units)
      "percentile" — the given percentile of in-ROI distances

    Shape filters (reject the cable, keep the rat):
      max_aspect_ratio : if > 0, reject CCs whose minAreaRect
                        long/short ratio exceeds this. The cable is a
                        thin line (aspect 10-30); the rat body is
                        compact (aspect 1.5-3). The texture heatmap
                        shows the cable as a high-distance STREAK and
                        the rat as a high-distance BLOB, so this
                        separates them cleanly.
      min_fill_ratio   : if > 0, reject CCs whose area / bbox-area is
                        below this. A line barely fills its bounding
                        box (~0.1-0.2); a blob fills much more (~0.5+).
                        Complements the aspect filter for diagonal
                        cables whose minAreaRect is misleading.

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
    # Keep CCs above min_area, optionally applying shape filters
    n_cc, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    out = np.zeros_like(mask)
    for i in range(1, n_cc):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area_px:
            continue
        # Shape filters to reject the cable
        if max_aspect_ratio > 0 or min_fill_ratio > 0:
            ys, xs = np.where(labels == i)
            if len(xs) >= 5:
                pts = np.column_stack([xs, ys]).astype(np.float32)
                (_, _), (w, h), _ = cv2.minAreaRect(pts)
                short = min(w, h)
                long_ = max(w, h)
                if max_aspect_ratio > 0 and short > 0:
                    if (long_ / short) > max_aspect_ratio:
                        continue
                if min_fill_ratio > 0:
                    bbox_area = (stats[i, cv2.CC_STAT_WIDTH]
                                 * stats[i, cv2.CC_STAT_HEIGHT])
                    if bbox_area > 0 and (area / bbox_area) < min_fill_ratio:
                        continue
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
