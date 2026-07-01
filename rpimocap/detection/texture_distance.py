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


def _second_layer_kernels(scales, n_orient):
    """Build the 2nd-layer Gabor bank (cached by config). Reuses the
    same getGaborKernel construction as layer 1."""
    key = (tuple(scales), int(n_orient))
    k = _SECOND_LAYER_CACHE.get(key)
    if k is None:
        from rpimocap.detection.rat_texture import build_gabor_kernels
        k = build_gabor_kernels(
            [i * np.pi / n_orient for i in range(n_orient)], list(scales))
        _SECOND_LAYER_CACHE[key] = k
    return k


_SECOND_LAYER_CACHE: dict = {}
_SECOND_LAYER_SCALES = (9, 17)
_SECOND_LAYER_ORIENTS = 6


def dense_gabor_descriptor(
        gray:        np.ndarray,
        kernels:     Sequence[np.ndarray],
        n_orient:    int,
        n_scales:    int,
        smooth_k:    int = 7,
        rotation_invariant: bool = True,
        log_transform: bool = False,
        second_layer: bool = True,
        second_layer_scales: Sequence[int] = _SECOND_LAYER_SCALES,
        second_layer_orient: int = _SECOND_LAYER_ORIENTS,
        second_smooth_k: int = 9,
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
    second_layer
              : when True (default) AND rotation_invariant, append a
                SECOND Gabor layer computed on the rectified,
                rotation-pooled layer-1 energy maps — the scattering
                transform |G2 * |G1 * x||. This captures the spatial
                ARRANGEMENT of layer-1 texture energy (2nd-order
                statistics), which separates fur from structured
                background clutter (metal frame, equipment, clothing,
                cable) far better than the 1st layer alone — measured
                +13-106% per-pixel d-prime on real footage across both
                cameras. Adds 2 pooled channels per
                (layer1_scale x second_layer_scale). Ignored when
                rotation_invariant=False.
    second_layer_scales, second_layer_orient, second_smooth_k
              : the 2nd-layer bank config (default (9,17) at 6
                orientations, 9px post-smoothing — the swept optimum).

    Returns
    -------
    (D, H, W) float32 descriptor stack. With rotation_invariant=True:
        D = 3*n_scales                                   (second_layer=False)
        D = 3*n_scales + 2*n_scales*len(second_layer_scales)
                                                         (second_layer=True)
    With rotation_invariant=False: D = n_orient*n_scales (second_layer ignored).
    """
    gray_f = gray.astype(np.float32)
    H, W = gray_f.shape

    if rotation_invariant:
        D = 3 * n_scales
        out = np.empty((D, H, W), dtype=np.float32)
        mean_maps = []                       # rotation-pooled energy / scale
        for s_idx in range(n_scales):
            resp_o = np.empty((n_orient, H, W), dtype=np.float32)
            for o_idx in range(n_orient):
                kern = kernels[s_idx * n_orient + o_idx]
                r = np.abs(cv2.filter2D(gray_f, cv2.CV_32F, kern))
                if smooth_k > 1:
                    r = cv2.boxFilter(r, cv2.CV_32F, (smooth_k, smooth_k))
                resp_o[o_idx] = r
            m = resp_o.mean(axis=0)
            out[s_idx * 3 + 0] = resp_o.max(axis=0)
            out[s_idx * 3 + 1] = m
            out[s_idx * 3 + 2] = resp_o.std(axis=0)
            mean_maps.append(m)

        if second_layer:
            k2 = _second_layer_kernels(second_layer_scales,
                                       second_layer_orient)
            n_o2 = second_layer_orient
            n_s2 = len(second_layer_scales)
            l2 = np.empty((2 * n_scales * n_s2, H, W), dtype=np.float32)
            j = 0
            for base in mean_maps:                # per layer-1 scale
                for s2 in range(n_s2):
                    ro = np.empty((n_o2, H, W), dtype=np.float32)
                    for o2 in range(n_o2):
                        r = np.abs(cv2.filter2D(
                            base, cv2.CV_32F, k2[s2 * n_o2 + o2]))
                        if second_smooth_k > 1:
                            r = cv2.boxFilter(
                                r, cv2.CV_32F,
                                (second_smooth_k, second_smooth_k))
                        ro[o2] = r
                    l2[j] = ro.max(axis=0); j += 1
                    l2[j] = ro.std(axis=0); j += 1
            out = np.concatenate([out, l2], axis=0)
    else:
        D = n_orient * n_scales
        out = np.empty((D, H, W), dtype=np.float32)
        for i, kern in enumerate(kernels):
            r = np.abs(cv2.filter2D(gray_f, cv2.CV_32F, kern))
            if smooth_k > 1:
                r = cv2.boxFilter(r, cv2.CV_32F, (smooth_k, smooth_k))
            out[i] = r
    if log_transform:
        # The pooled background descriptor is exponential-like (CoV ~ 1,
        # tails 8-16x the mean over 1e9 samples). log1p is the variance-
        # stabilizing transform for that family: it compresses the heavy
        # background tail toward the bulk, so the downstream z-score /
        # Mahalanobis (a Gaussian assumption) is far better founded and
        # the tail produces fewer false positives. Must be applied
        # consistently to BOTH the bg-model build and the per-frame
        # descriptor (the model mean/std then live in log space).
        out = np.log1p(out)
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
#  Dynamic shadow model (slow EMA, rat-masked)
# ────────────────────────────────────────────────────────────────────


class DynamicShadowModel:
    """Per-pixel illumination field that adapts over time.

    The static illumination field (build_illumination_field) is a
    single median over the whole session — it can't track two things
    that change during a long recording:
      1. SLOW DRIFT: IR emitter output and ambient level wander over
         tens of minutes.
      2. The rat's CAST SHADOW: a moving dim region next to the rat
         that a static field treats as background, so it shows up as
         a faint texture-change halo.

    This model holds a running illumination estimate and nudges it
    each frame with a slow exponential moving average (EMA):

        field ← (1 - α) · field  +  α · frame      (per pixel)

    Crucially the update is MASKED: pixels currently believed to be
    rat (passed in via update_mask) are NOT folded into the field, so
    the bright rat never poisons the illumination estimate. Pixels
    around the rat (its cast shadow) ARE allowed to update, so the
    field tracks the moving shadow and stops flagging it.

    α is small (e.g. 0.02) so the field follows slow drift but
    ignores the fast-moving rat. This is the texture-domain analogue
    of the bg-sub --bg-adapt-alpha.
    """

    def __init__(self,
                 initial_field: np.ndarray,
                 alpha:         float = 0.02,
                 blur_sigma:    float = 51.0,
                 floor:         float = 1.0):
        """
        Parameters
        ----------
        initial_field : (H, W) seed illumination, typically the static
                        median field from build_illumination_field.
        alpha         : EMA rate. Larger = adapts faster (tracks drift
                        but risks absorbing a slow/still rat). 0.01-
                        0.05 is a reasonable range for slow IR drift.
        blur_sigma    : if > 0, the per-frame update is low-pass
                        filtered before blending, so only the smooth
                        illumination component adapts (sharp structure
                        isn't pulled into the field). Matches the
                        static field's blur convention.
        floor         : minimum field value (keeps the divide safe).
        """
        self.field = np.maximum(
            initial_field.astype(np.float32), floor)
        self.alpha = float(alpha)
        self.blur_sigma = float(blur_sigma)
        self.floor = float(floor)
        self.n_updates = 0

    def update(self,
               gray:        np.ndarray,
               update_mask: Optional[np.ndarray] = None) -> None:
        """Fold one frame into the running illumination field.

        update_mask : optional (H, W); pixels where it is TRUE (>0)
                      are EXCLUDED from the update (they're the rat).
                      Everything else — including the rat's cast
                      shadow — updates normally.
        """
        g = gray.astype(np.float32)
        if self.blur_sigma and self.blur_sigma > 0:
            k = int(2 * round(3 * self.blur_sigma) + 1)
            g = cv2.GaussianBlur(g, (k, k), self.blur_sigma)
        # Per-pixel blended update
        blended = (1.0 - self.alpha) * self.field + self.alpha * g
        if update_mask is not None:
            keep = (update_mask > 0)        # rat pixels: keep old field
            blended = np.where(keep, self.field, blended)
        self.field = np.maximum(blended, self.floor)
        self.n_updates += 1

    def correct(self,
                gray:         np.ndarray,
                target_level: Optional[float] = None) -> np.ndarray:
        """Flat-field correct a frame by the CURRENT dynamic field."""
        return apply_illumination_correction(
            gray, self.field, target_level=target_level)

    def get_field(self) -> np.ndarray:
        return self.field.copy()


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


def _detect_rat_mask_intensity(
        gray:        np.ndarray,
        percentile:  float = 96.0,
        min_area_px: int = 1500,
        dilate_px:   int = 25,
        roi_mask:    Optional[np.ndarray] = None,
        ) -> np.ndarray:
    """Quick per-frame rat mask via intensity: the rat is the
    brightest compact blob under IR. Threshold the top-percentile
    intensities, take the largest connected component, dilate it
    generously so the rat's soft edges and immediate cast shadow are
    covered. Returns a (H, W) uint8 mask (255 = rat).

    Used to EXCLUDE rat pixels from the persistence/MAD computation
    so the rat's frequented spots don't get falsely marked as
    low-persistence (transient) background. Generous dilation is
    intentional here — over-masking the rat is safe (those frames
    just don't contribute to that pixel's spread), under-masking
    leaves rat residue in the persistence map.

    roi_mask : optional (H, W) arena mask. When provided, the
               percentile threshold AND the blob search are restricted
               to inside the arena, so bright things OUTSIDE the arena
               (experimenter's hands, room behind the acrylic, door
               reflections) can't be mistaken for the rat."""
    if roi_mask is not None:
        inside = gray[roi_mask > 0]
        if inside.size == 0:
            return np.zeros_like(gray, dtype=np.uint8)
        thr = float(np.percentile(inside, percentile))
        m = ((gray >= thr) & (roi_mask > 0)).astype(np.uint8) * 255
    else:
        thr = float(np.percentile(gray, percentile))
        m = (gray >= thr).astype(np.uint8) * 255
    n_cc, labels, stats, _ = cv2.connectedComponentsWithStats(m)
    if n_cc < 2:
        return np.zeros_like(gray, dtype=np.uint8)
    sizes = stats[1:, cv2.CC_STAT_AREA]
    if int(sizes.max()) < min_area_px:
        return np.zeros_like(gray, dtype=np.uint8)
    largest = int(sizes.argmax()) + 1
    out = (labels == largest).astype(np.uint8) * 255
    if dilate_px > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
        out = cv2.dilate(out, k)
    if roi_mask is not None:
        # Keep the dilated mask inside the arena
        out = cv2.bitwise_and(out, (roi_mask > 0).astype(np.uint8) * 255)
    return out


def build_persistent_texture_model(
        bg_frames:   Sequence[np.ndarray],
        kernels:     Sequence[np.ndarray],
        n_orient:    int,
        n_scales:    int,
        smooth_k:    int = 7,
        rotation_invariant: bool = True,
        std_floor:   float = 1e-3,
        mask_rat:    bool = False,
        rat_percentile: float = 96.0,
        rat_min_area_px: int = 1500,
        rat_dilate_px:   int = 25,
        roi_mask:    Optional[np.ndarray] = None,
        log_transform: bool = False,
        second_layer: bool = True,
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

    RAT MASKING (mask_rat=True)
        The median is robust to the rat, but the MAD is NOT: when the
        rat passes through a pixel, that frame's descriptor is a large
        deviation from the median, inflating the MAD and LOWERING the
        persistence there. The result is that the rat's frequented
        spots get marked low-persistence (transient) even though the
        underlying background is static — and since the distance map
        multiplies by persistence, the rat's own favorite locations
        get SUPPRESSED.

        With mask_rat=True, the rat is detected per bg-frame (brightest
        compact blob) and those pixels are EXCLUDED from that pixel's
        spread computation. Since the rat moves, every pixel still has
        plenty of background frames to estimate spread from. The
        persistence map then reflects the STATIC SCENE only, so the
        rat is no longer suppressed where it tends to dwell.

    Frames should be sampled across the WHOLE session (large stride),
    not a front window, so that lighting/bedding drift is captured in
    the median rather than read as foreground later.

    Parameters
    ----------
    bg_frames : list of grayscale frames spread across the session.
                More frames = better median; 40-80 is reasonable.
    mask_rat  : if True, exclude per-frame rat pixels from the spread
                / persistence computation (see above).
    rat_percentile / rat_min_area_px / rat_dilate_px
              : parameters for the per-frame intensity rat detector.
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
    rat_masks = []
    for f in bg_frames:
        descs.append(dense_gabor_descriptor(
            f, kernels, n_orient, n_scales,
            smooth_k=smooth_k, rotation_invariant=rotation_invariant,
            log_transform=log_transform, second_layer=second_layer))
        if mask_rat:
            rat_masks.append(_detect_rat_mask_intensity(
                f, percentile=rat_percentile,
                min_area_px=rat_min_area_px,
                dilate_px=rat_dilate_px,
                roi_mask=roi_mask) > 0)
    stack = np.stack(descs, axis=0)            # (T, D, H, W)

    if mask_rat:
        # Mask the rat in BOTH the median and the spread. If the rat
        # dwells at a pixel for ≥50% of frames, an unmasked median
        # sits between rat and background instead of on background,
        # which then inflates the deviation of even the background
        # frames. Masking the median fixes this — the persistent
        # background texture is estimated from background frames only.
        rat_stack = np.stack(rat_masks, axis=0)            # (T, H, W)
        rat_bcast = np.broadcast_to(
            rat_stack[:, None, :, :], stack.shape)
        masked_stack = np.ma.array(stack, mask=rat_bcast)
        median_m = np.ma.median(masked_stack, axis=0)
        # Fallback for fully-masked pixels (rat almost always there)
        median_full = np.median(stack, axis=0)
        median = np.where(np.ma.getmaskarray(median_m),
                          median_full,
                          median_m.filled(0)).astype(np.float32)
        abs_dev = np.abs(stack - median[None])             # (T,D,H,W)
        masked_dev = np.ma.array(abs_dev, mask=rat_bcast)
        mad_masked = np.ma.median(masked_dev, axis=0).filled(np.nan)
        mad_full = np.median(abs_dev, axis=0)
        mad = np.where(np.isnan(mad_masked), mad_full, mad_masked)
    else:
        # Per-pixel, per-channel temporal median = persistent bg
        median = np.median(stack, axis=0).astype(np.float32)  # (D,H,W)
        mad = np.median(np.abs(stack - median[None]), axis=0)

    # Scale MAD to be std-comparable (for a normal dist, std≈1.4826*MAD)
    spread = (1.4826 * mad).astype(np.float32)
    spread = np.maximum(spread, std_floor)

    model = BackgroundTextureModel(
        mean=median, std=spread, n=len(bg_frames))

    # Persistence map: low channel-summed spread → high persistence.
    # Use the mean spread across channels as the instability measure.
    instability = spread.mean(axis=0)          # (H, W)
    # Normalize against the in-ROI instability distribution so that
    # outside-arena pixels (hands, room, reflections) don't skew the
    # p5/p95 scaling of the arena interior.
    if roi_mask is not None:
        inside_vals = instability[roi_mask > 0]
        if inside_vals.size == 0:
            inside_vals = instability.ravel()
    else:
        inside_vals = instability.ravel()
    lo = float(np.percentile(inside_vals, 5))
    hi = float(np.percentile(inside_vals, 95))
    norm = np.clip((instability - lo) / (hi - lo + 1e-6), 0, 1)
    persistence_map = (1.0 - norm).astype(np.float32)
    # Zero persistence outside the arena — nothing out there should
    # ever contribute to (or be measured by) the distance map.
    if roi_mask is not None:
        persistence_map = persistence_map * (roi_mask > 0).astype(
            np.float32)
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
        log_transform: bool = False,
        second_layer: bool = True,
        roi_mask:    Optional[np.ndarray] = None,
        persistence_map: Optional[np.ndarray] = None,
        persistence_power: float = 1.0,
        anisotropy_weight: Optional[np.ndarray] = None,
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
    anisotropy_weight : optional (H, W) in [0,1] from
                   foreshortening.anisotropy_weight (1 = surface seen
                   face-on, 0 = steeply foreshortened / grazing). The
                   distance is multiplied by this, so texture in
                   steeply-foreshortened regions — where the same fur
                   reads differently because the pixel footprint is
                   elongated (perspective foreshortening, NOT radial
                   lens distortion) — is trusted less. This directly
                   targets the frame-edge / flank weakness where the
                   descriptor is least reliable. None disables.
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
        rotation_invariant=rotation_invariant,
        log_transform=log_transform, second_layer=second_layer)
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
    # Foreshortening gating. Multiply by a [0,1] confidence that is 1
    # where the surface is seen face-on and decays toward 0 where it is
    # steeply foreshortened (grazing), so the descriptor is trusted less
    # exactly where the elongated pixel footprint makes the same physical
    # texture read differently. Parallel to persistence: a multiplicative
    # per-pixel confidence on the distance.
    if anisotropy_weight is not None:
        w = np.clip(anisotropy_weight, 0.0, 1.0).astype(np.float32)
        dist = dist * w
    if roi_mask is not None:
        dist = dist * (roi_mask > 0).astype(np.float32)
    if post_smooth_k and post_smooth_k > 1:
        dist = cv2.boxFilter(dist, cv2.CV_32F,
                            (post_smooth_k, post_smooth_k))
    return dist


def suppress_thin_structures(
        mask:          np.ndarray,
        min_width_px:  int = 25,
        restore_radius_px: Optional[int] = None,
        ) -> np.ndarray:
    """Remove thin structures (the tether cable) from a binary mask
    while preserving the compact rat body.

    The aspect-ratio / fill-ratio filters in threshold_distance_map
    operate on WHOLE connected components, so they fail when the
    cable physically connects to the rat (it attaches to the
    headstage): rat+cable is one component with a moderate aspect
    ratio that passes the filter. This function instead works WITHIN
    a component by width.

    Mechanism — morphological opening with a disk:
      1. ERODE by r = min_width_px // 2. Anything thinner than
         2r (the cable, a few px wide) is completely eaten away;
         the thick rat body shrinks but survives as a 'core'.
      2. DILATE the surviving core back by restore_radius_px (default
         a bit more than r) to recover the rat body's original
         extent.
      3. AND the restored core with the original mask, so the result
         is the rat-shaped subset of the input — the cable, having no
         surviving core, is gone.

    Because the rat body is much wider than the cable, the opening
    severs the cable at the headstage and discards it while keeping
    the rat intact.

    Parameters
    ----------
    mask              : (H, W) binary uint8 mask.
    min_width_px      : structures narrower than this are removed.
                        Set to roughly the cable width × 3, well below
                        the rat body width. For a ~6-10px cable and a
                        ~280px rat, 25-40 is a good range.
    restore_radius_px : dilation radius to regrow the eroded core.
                        Defaults to min_width_px//2 + 2. Larger
                        recovers more of the rat edge but risks
                        re-attaching a stub of cable.

    Returns
    -------
    (H, W) uint8 mask with thin structures removed.
    """
    r = max(1, int(min_width_px) // 2)
    if restore_radius_px is None:
        restore_radius_px = r + 2
    m = (mask > 0).astype(np.uint8)
    if int(m.sum()) == 0:
        return mask.copy()
    erode_k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    core = cv2.erode(m, erode_k)
    if int(core.sum()) == 0:
        # Nothing survived erosion (e.g. the whole mask was thin) —
        # return empty rather than the cable.
        return np.zeros_like(mask)
    rr = int(restore_radius_px)
    dil_k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * rr + 1, 2 * rr + 1))
    restored = cv2.dilate(core, dil_k)
    out = cv2.bitwise_and(m, restored) * 255
    return out.astype(np.uint8)


def threshold_distance_map(
        dist:        np.ndarray,
        method:      str = "otsu",
        abs_thresh:  float = 3.0,
        percentile:  float = 95.0,
        min_area_px: int = 1000,
        morph_close_k: int = 7,
        max_aspect_ratio: float = 0.0,
        min_fill_ratio:   float = 0.0,
        suppress_thin_width: int = 0,
        ) -> tuple[np.ndarray, float]:
    """Threshold a texture-distance map into a binary foreground
    mask, keep connected components above min_area_px.

    method:
      "otsu"       — Otsu's threshold on the distance histogram
      "absolute"   — fixed abs_thresh (z-score units)
      "percentile" — the given percentile of in-ROI distances

    Cable suppression (runs BEFORE component filtering):
      suppress_thin_width : if > 0, remove structures narrower than
                        this many px via morphological opening
                        (suppress_thin_structures). This severs the
                        cable from the rat WITHIN a merged component,
                        which the whole-component aspect/fill filters
                        below cannot do. Set to ~3x the cable width,
                        well under the rat-body width (25-40 typical).

    Shape filters (reject isolated cable fragments):
      max_aspect_ratio : if > 0, reject CCs whose minAreaRect
                        long/short ratio exceeds this. The cable is a
                        thin line (aspect 10-30); the rat body is
                        compact (aspect 1.5-3).
      min_fill_ratio   : if > 0, reject CCs whose area / bbox-area is
                        below this. A line barely fills its bounding
                        box (~0.1-0.2); a blob fills much more (~0.5+).

    Returns (mask_uint8, threshold_used).
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
    # Cable suppression BEFORE component analysis, so the cable is
    # severed from the rat before CCs are counted/filtered.
    if suppress_thin_width and suppress_thin_width > 0:
        mask = suppress_thin_structures(
            mask, min_width_px=suppress_thin_width)
    # Keep CCs above min_area, optionally applying shape filters
    n_cc, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    out = np.zeros_like(mask)
    for i in range(1, n_cc):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area_px:
            continue
        # Shape filters to reject isolated cable fragments
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


# ────────────────────────────────────────────────────────────────────
#  Graph-cut (MRF) segmentation of the texture-distance map
# ────────────────────────────────────────────────────────────────────


def crop_box_from_prediction(
        pred:        Optional[tuple],
        image_shape: tuple,
        pad_px:      int = 120,
        min_size:    int = 200,
        ) -> Optional[tuple]:
    """Build a (x0, y0, x1, y1) crop box around a Kalman prediction
    (cx, cy, r) for the predicted-ROI graph cut.

    The box is centered on the predicted blob, sized to the predicted
    radius plus `pad_px` of slack (motion + blob-size uncertainty), and
    clamped to the image and to a minimum size. Returns None if the
    prediction is None (no track yet) so the caller falls back to a
    full-frame cut.

    pad_px should comfortably exceed the rat's per-frame displacement
    plus the radius uncertainty — too tight and a fast move escapes the
    box; the cost of a generous pad is only a slightly larger graph.
    """
    if pred is None:
        return None
    H, W = image_shape[:2]
    cx, cy, r = pred
    half = max(float(r) + pad_px, min_size / 2.0)
    x0 = int(max(0, np.floor(cx - half)))
    y0 = int(max(0, np.floor(cy - half)))
    x1 = int(min(W, np.ceil(cx + half)))
    y1 = int(min(H, np.ceil(cy + half)))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    return (x0, y0, x1, y1)


def graphcut_segment_distance(
        dist:           np.ndarray,
        gray:           Optional[np.ndarray] = None,
        roi_mask:       Optional[np.ndarray] = None,
        fg_thresh:      float = 4.0,
        data_scale:     float = 1.0,
        smooth_weight:  float = 2.0,
        edge_sigma:     float = 10.0,
        min_area_px:    int = 1000,
        suppress_thin_width: int = 0,
        crop_box:       Optional[tuple] = None,
        ) -> tuple[np.ndarray, dict]:
    """Segment the texture-distance map into a foreground mask by
    minimizing a binary MRF energy with a graph cut (Boykov-Kolmogorov
    max-flow), instead of the threshold + morphology + CC heuristic.

    This is the first principled step toward the variational
    formulation (Cremers et al.): the energy has a DATA term (how
    well each pixel matches foreground vs background, from the texture
    distance) plus a contrast-sensitive SMOOTHNESS term (a penalty for
    neighboring pixels taking different labels, *reduced* where the
    image has a strong edge). The min-cut is the globally-optimal
    binary labeling of that energy — so the silhouette is coherent by
    construction rather than via post-hoc morphology.

    Energy minimized (binary labels l_p ∈ {bg=0, fg=1}):

        E(l) = Σ_p D_p(l_p)  +  Σ_{p~q} V_pq · [l_p ≠ l_q]

    Data term D_p — derived from the texture distance d_p. We map the
    distance through a logistic centered at fg_thresh to a foreground
    probability, then use negative log-likelihoods as the unary costs:

        P_fg(p) = sigmoid(data_scale · (d_p − fg_thresh))
        D_p(fg) = −log P_fg(p)      (cheap to call fg when d_p large)
        D_p(bg) = −log(1 − P_fg(p)) (cheap to call bg when d_p small)

    So fg_thresh plays the role of the old absolute threshold (the
    distance at which fg/bg are equally likely), but the cut can
    override it locally to keep the region coherent.

    Smoothness term V_pq — contrast-sensitive Potts (Boykov-Jolly):

        V_pq = smooth_weight · exp(−(I_p − I_q)² / (2·edge_sigma²))

    High where neighbors have similar intensity (penalize splitting a
    smooth region), low across a real intensity edge (cheap to place
    the boundary there). If `gray` is None, V_pq = smooth_weight
    everywhere (plain Potts; boundary driven only by the data term).

    Parameters
    ----------
    dist          : (H, W) texture-distance map.
    gray          : (H, W) intensity image for the contrast-sensitive
                    smoothness term. Use the illumination-corrected
                    frame for consistency with the distance map.
    roi_mask      : (H, W) arena mask. Pixels outside are forced to
                    background (their fg data cost is set to +inf) so
                    nothing outside the arena can be labeled rat.
    fg_thresh     : distance at which fg/bg are equally likely. Same
                    role as the old --abs-thresh.
    data_scale    : logistic steepness. Larger = the data term behaves
                    more like a hard threshold; smaller = softer, lets
                    the smoothness term do more work.
    smooth_weight : global weight on the smoothness term. Larger =
                    smoother/rounder silhouette, fewer islands. This is
                    the main regularization knob.
    edge_sigma    : intensity scale (in gray levels) for the
                    contrast-sensitive term. Edges with |I_p−I_q| >>
                    edge_sigma are treated as cheap boundaries.
    min_area_px   : drop final components below this area.
    suppress_thin_width : if > 0, apply suppress_thin_structures to the
                    cut result (cable removal still helps even with the
                    smoothness term).
    crop_box      : optional (x0, y0, x1, y1) pixel box. When given, the
                    max-flow is solved ONLY over this sub-window (e.g. a
                    dilated box around the Kalman-predicted rat) instead
                    of the whole frame, and the result is placed back
                    into a full-size mask (everything outside the box is
                    background). This is the cheapest large perf win —
                    a ~400×400 band is ~30k nodes vs ~2.2M for a full
                    2028×1080 frame — with no change to the cut inside
                    the box. Use crop_box_from_prediction() to build it
                    from the tracker. None = full frame.

    Returns
    -------
    (mask_uint8, info) where info has the energy and pixel counts.
    """
    try:
        import maxflow
    except ImportError as e:           # pragma: no cover
        raise ImportError(
            "graphcut_segment_distance requires PyMaxflow "
            "(`pip install PyMaxflow`).") from e

    H, W = dist.shape
    d = dist.astype(np.float32)

    # ── Optional predicted-ROI crop ────────────────────────────────
    # Solve the max-flow only over a sub-window (e.g. a dilated box
    # around the Kalman-predicted rat). We slice the data here and paste
    # the resulting cut back into a full-size mask after the solve, so
    # the output shape and all downstream post-processing are unchanged.
    full_HW = (H, W)
    cb = None
    roi_full = roi_mask          # keep the full-frame roi for the post clamp
    if crop_box is not None:
        x0, y0, x1, y1 = crop_box
        x0 = int(max(0, min(x0, W - 1)))
        y0 = int(max(0, min(y0, H - 1)))
        x1 = int(max(x0 + 1, min(x1, W)))
        y1 = int(max(y0 + 1, min(y1, H)))
        cb = (x0, y0, x1, y1)
        d = d[y0:y1, x0:x1]
        if gray is not None:
            gray = gray[y0:y1, x0:x1]
        if roi_mask is not None:
            roi_mask = roi_mask[y0:y1, x0:x1]
        H, W = d.shape

    # ── Data term ──────────────────────────────────────────────────
    # Foreground probability via logistic centered at fg_thresh.
    z = data_scale * (d - fg_thresh)
    z = np.clip(z, -50.0, 50.0)        # avoid overflow in exp
    p_fg = 1.0 / (1.0 + np.exp(-z))
    eps = 1e-6
    p_fg = np.clip(p_fg, eps, 1.0 - eps)
    # Unary costs (negative log-likelihood)
    cost_fg = -np.log(p_fg)            # cost of labeling pixel FG
    cost_bg = -np.log(1.0 - p_fg)      # cost of labeling pixel BG

    # Force outside-ROI pixels to background: make FG infinitely
    # expensive there.
    if roi_mask is not None:
        outside = (roi_mask == 0)
        cost_fg = cost_fg.copy()
        cost_fg[outside] = 1e9

    # ── Build the graph ────────────────────────────────────────────
    g = maxflow.Graph[float]()
    nodeids = g.add_grid_nodes((H, W))

    # Smoothness term — contrast-sensitive 4-connected Potts.
    if gray is not None:
        gi = gray.astype(np.float32)
        # Horizontal edges (p=(y,x), q=(y,x+1))
        dh = gi[:, 1:] - gi[:, :-1]
        wh = smooth_weight * np.exp(-(dh * dh)
                                    / (2.0 * edge_sigma * edge_sigma))
        # Vertical edges (p=(y,x), q=(y+1,x))
        dv = gi[1:, :] - gi[:-1, :]
        wv = smooth_weight * np.exp(-(dv * dv)
                                    / (2.0 * edge_sigma * edge_sigma))
    else:
        wh = np.full((H, W - 1), smooth_weight, np.float32)
        wv = np.full((H - 1, W), smooth_weight, np.float32)

    # Add pairwise edges. maxflow's add_grid_edges with a structure
    # would apply a uniform weight; we need per-edge weights, so add
    # them explicitly but vectorized per direction.
    # Horizontal
    src = nodeids[:, :-1].ravel()
    dst = nodeids[:, 1:].ravel()
    g.add_edges(src, dst, wh.ravel(), wh.ravel())
    # Vertical
    src = nodeids[:-1, :].ravel()
    dst = nodeids[1:, :].ravel()
    g.add_edges(src, dst, wv.ravel(), wv.ravel())

    # Terminal (data) edges: add_grid_tedges(nodeids, E_source, E_sink)
    # In maxflow's convention, the SOURCE side is label 1 (fg) after
    # get_grid_segments returns True for sink-connected = bg. We set:
    #   capacity to SOURCE = cost of assigning to SINK label, etc.
    # Use the standard mapping: tedge(node, cap_source=cost_bg,
    # cap_sink=cost_fg) so that cutting the cheaper terminal keeps the
    # cheaper label. get_grid_segments returns True where the node is
    # on the SINK side. We define fg = NOT sink.
    g.add_grid_tedges(nodeids, cost_bg, cost_fg)

    flow = g.maxflow()
    sgm = g.get_grid_segments(nodeids)     # True = sink side
    # By our tedge convention, sink side = background.
    mask = (~sgm).astype(np.uint8) * 255

    # Paste the cropped cut back into a full-size mask (outside the
    # crop box stays background). Post-processing then runs full-frame.
    if cb is not None:
        x0, y0, x1, y1 = cb
        full = np.zeros(full_HW, np.uint8)
        full[y0:y1, x0:x1] = mask
        mask = full
        roi_mask = roi_full       # restore full-frame roi for the clamp

    # ── Post: ROI clamp, cable suppression, area filter ────────────
    if roi_mask is not None:
        mask = cv2.bitwise_and(mask, (roi_mask > 0).astype(np.uint8) * 255)
    if suppress_thin_width and suppress_thin_width > 0:
        mask = suppress_thin_structures(
            mask, min_width_px=suppress_thin_width)
    n_cc, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    out = np.zeros_like(mask)
    kept = 0
    for i in range(1, n_cc):
        if stats[i, cv2.CC_STAT_AREA] >= min_area_px:
            out[labels == i] = 255
            kept += 1

    info = {
        "flow": float(flow),
        "fg_px": int((out > 0).sum()),
        "n_components": int(kept),
        "fg_thresh": float(fg_thresh),
        "smooth_weight": float(smooth_weight),
    }
    return out, info


def colorize_distance_map(dist: np.ndarray,
                            vmax: Optional[float] = None) -> np.ndarray:
    """Render a distance map as a BGR heatmap for diagnostics
    (blue=low/no-change, red=high/large-texture-change)."""
    if vmax is None:
        nz = dist[dist > 0]
        vmax = float(np.percentile(nz, 99)) if nz.size else 1.0
    scaled = np.clip(dist / (vmax + 1e-6) * 255.0, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(scaled, cv2.COLORMAP_JET)


# ────────────────────────────────────────────────────────────────────
#  Kalman tracker for texture-distance blobs
# ────────────────────────────────────────────────────────────────────


class TextureBlobTracker:
    """Kalman filter over the texture-distance rat blob, giving
    temporal coherence to the per-frame segmentation.

    Each frame the texture-distance threshold produces zero or more
    candidate blobs. Segmenting independently means a frame where the
    rat fragments, or where a stray reflection briefly wins, produces
    a bad detection with no memory. This tracker fixes that:

      * PREDICT the rat's next position/size from its motion.
      * GATE incoming candidates — reject any whose centroid is more
        than `gate_px` from the prediction (a reflection flashing in
        the corner is ignored because it's far from where the rat is).
      * Among in-gate candidates, pick the best (largest area, or
        closest to prediction) and CORRECT the filter with it.
      * COAST through dropouts — if no candidate passes the gate,
        output the prediction and increment a miss counter, up to
        `max_coast` frames before declaring the track lost.

    State vector  x = [cx, cy, r, vx, vy]   (matches the project's
    EdgeMotionRatTracker convention) where r is an effective radius
    = sqrt(area / pi). Constant-velocity for position, random-walk
    for radius.
    """

    def __init__(self,
                 dt:            float = 1.0,
                 process_noise: float = 8.0,
                 meas_noise:    float = 6.0,
                 gate_px:       float = 120.0,
                 max_coast:     int   = 10,
                 select:        str   = "compactness",
                 fill_power:    float = 1.0,
                 min_fill:      float = 0.0):
        """
        Parameters
        ----------
        dt            : frame timestep (1.0 = per-frame units).
        process_noise : Kalman process σ (larger = trusts new
                        detections more, less smoothing).
        meas_noise    : Kalman measurement σ.
        gate_px       : max centroid distance from prediction for a
                        candidate to be accepted. The single most
                        important parameter — too tight loses the rat
                        on fast moves, too loose admits reflections.
        max_coast     : consecutive missed frames the track survives
                        on prediction alone before being marked lost.
        select        : how to rank in-gate candidates —
                        'area'        : largest blob (legacy; a large
                                        sparse cable+headstage composite
                                        out-competes the compact rat).
                        'nearest'     : closest to prediction.
                        'compactness' : (default) rank by
                                        area * fill_ratio**fill_power,
                                        where fill_ratio = area / bbox.
                                        A thin cable (even fused with the
                                        headstage by morphological close)
                                        has low fill and is demoted below
                                        the compact rat, which fills its
                                        bounding box. The rat's shape, not
                                        just its size, decides.
        fill_power    : exponent on fill_ratio for 'compactness'. Higher
                        = harsher penalty on elongated/sparse blobs.
        min_fill      : reject candidates whose fill_ratio is below this
                        outright (0 = keep all). A hard floor for very
                        sparse cable blobs.
        """
        self.dt            = float(dt)
        self.process_noise = float(process_noise)
        self.meas_noise    = float(meas_noise)
        self.gate_px       = float(gate_px)
        self.max_coast     = int(max_coast)
        self.select        = select
        self.fill_power    = float(fill_power)
        self.min_fill      = float(min_fill)
        self._kf           = None      # built lazily on first detection
        self._initialized  = False
        self.coast_count   = 0
        self.lost          = True
        self.n_updates     = 0
        self.n_coasts      = 0
        self.n_gated_out   = 0

    # ── internal ───────────────────────────────────────────────────

    def _build_kf(self, cx, cy, r):
        kf = cv2.KalmanFilter(5, 3, 0, cv2.CV_32F)
        dt = self.dt
        kf.transitionMatrix = np.array([
            [1, 0, 0, dt, 0],
            [0, 1, 0, 0, dt],
            [0, 0, 1, 0,  0],
            [0, 0, 0, 1,  0],
            [0, 0, 0, 0,  1],
        ], dtype=np.float32)
        kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
        ], dtype=np.float32)
        kf.processNoiseCov = (np.eye(5, dtype=np.float32)
                              * (self.process_noise ** 2))
        kf.measurementNoiseCov = (np.eye(3, dtype=np.float32)
                                  * (self.meas_noise ** 2))
        kf.errorCovPost = np.eye(5, dtype=np.float32) * 1000.0
        kf.statePost = np.array(
            [[cx], [cy], [r], [0], [0]], dtype=np.float32)
        self._kf = kf
        self._initialized = True
        self.lost = False
        self.coast_count = 0

    @staticmethod
    def _blob_measurement(stats_row):
        """(cx, cy, r) from a connectedComponentsWithStats row."""
        x = stats_row[cv2.CC_STAT_LEFT]
        y = stats_row[cv2.CC_STAT_TOP]
        w = stats_row[cv2.CC_STAT_WIDTH]
        h = stats_row[cv2.CC_STAT_HEIGHT]
        a = stats_row[cv2.CC_STAT_AREA]
        cx = x + w / 2.0
        cy = y + h / 2.0
        r  = float(np.sqrt(a / np.pi))
        return cx, cy, r

    @staticmethod
    def _fill_ratio(stats_row):
        """area / bounding-box-area — 1.0 = fills its box (compact rat),
        low = sparse/elongated (thin cable, even fused with headstage)."""
        w = stats_row[cv2.CC_STAT_WIDTH]
        h = stats_row[cv2.CC_STAT_HEIGHT]
        a = stats_row[cv2.CC_STAT_AREA]
        return a / float(max(w * h, 1))

    def _select_score(self, stats_row):
        """Compactness-weighted area: area * fill_ratio**fill_power.
        Demotes large-but-sparse cable/headstage composites below the
        compact rat."""
        a = float(stats_row[cv2.CC_STAT_AREA])
        return a * (self._fill_ratio(stats_row) ** self.fill_power)

    def predict(self):
        """Advance the filter one step; return (cx, cy, r) prediction
        or None if uninitialized."""
        if not self._initialized:
            return None
        p = self._kf.predict()
        return float(p[0, 0]), float(p[1, 0]), float(p[2, 0])

    def peek_prediction(self):
        """Predicted (cx, cy, r) WITHOUT advancing the filter.

        `predict()` (and `update()`, which calls it) mutate the Kalman
        state, so calling `predict()` separately to build a predicted-ROI
        crop before `update()` would double-advance the motion model.
        This reads the one-step prediction non-destructively as
        A · statePost, leaving the filter untouched, so the caller can
        build a crop box and then call `update()` normally. Returns None
        if uninitialized."""
        if not self._initialized:
            return None
        x = self._kf.transitionMatrix @ self._kf.statePost
        return float(x[0, 0]), float(x[1, 0]), float(x[2, 0])

    # ── public step ────────────────────────────────────────────────

    def update(self, mask: np.ndarray):
        """Process one frame's foreground mask.

        Returns a dict:
          {state: (cx,cy,r) or None,
           measured: bool,        # True if a real detection was used
           coasting: bool,        # True if output is prediction-only
           lost: bool,
           n_candidates: int}
        """
        n_cc, labels, stats, _ = cv2.connectedComponentsWithStats(
            (mask > 0).astype(np.uint8))
        cands = [stats[i] for i in range(1, n_cc)]
        n_cand = len(cands)

        pred = self.predict()

        # No track yet — initialize from the best-scoring candidate
        if not self._initialized:
            usable = [s for s in cands
                      if self._fill_ratio(s) >= self.min_fill]
            if not usable:
                return dict(state=None, measured=False,
                            coasting=False, lost=True,
                            n_candidates=0)
            if self.select == "area":
                first = max(usable, key=lambda s: s[cv2.CC_STAT_AREA])
            else:
                first = max(usable, key=self._select_score)
            cx, cy, r = self._blob_measurement(first)
            self._build_kf(cx, cy, r)
            self.n_updates += 1
            return dict(state=(cx, cy, r), measured=True,
                        coasting=False, lost=False,
                        n_candidates=n_cand)

        # Gate candidates against the prediction
        px, py, _ = pred
        in_gate = []
        for s in cands:
            if self._fill_ratio(s) < self.min_fill:
                continue
            cx, cy, r = self._blob_measurement(s)
            d = np.hypot(cx - px, cy - py)
            if d <= self.gate_px:
                in_gate.append((s, cx, cy, r, d))
            else:
                self.n_gated_out += 1

        if not in_gate:
            # Coast on prediction
            self.coast_count += 1
            self.n_coasts += 1
            if self.coast_count > self.max_coast:
                self.lost = True
                self._initialized = False
                return dict(state=None, measured=False,
                            coasting=False, lost=True,
                            n_candidates=n_cand)
            return dict(state=pred, measured=False,
                        coasting=True, lost=False,
                        n_candidates=n_cand)

        # Select the best in-gate candidate
        if self.select == "nearest":
            best = min(in_gate, key=lambda t: t[4])
        elif self.select == "area":
            best = max(in_gate, key=lambda t: t[0][cv2.CC_STAT_AREA])
        else:  # compactness (default): area weighted by fill_ratio
            best = max(in_gate, key=lambda t: self._select_score(t[0]))
        _, cx, cy, r, _ = best
        meas = np.array([[cx], [cy], [r]], dtype=np.float32)
        self._kf.correct(meas)
        self.coast_count = 0
        self.lost = False
        self.n_updates += 1
        st = self._kf.statePost
        return dict(state=(float(st[0, 0]), float(st[1, 0]),
                           float(st[2, 0])),
                    measured=True, coasting=False, lost=False,
                    n_candidates=n_cand)
