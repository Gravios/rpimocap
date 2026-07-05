"""Topological rat detector — median-bandpass grain density.

Motivation
----------
Every intensity- or energy-based feature we tried (raw brightness, Gabor
energy at any scale, LoG amplitude, median high-pass) plateaus at ~1-2 sigma
of rat/bedding separation, because the white fur and the bedding are similar
in brightness and even in local contrast — and on the low-contrast camera the
rat is measured as *less* anomalous than bedding. The discriminating fact is
topological rather than photometric: bedding is a field of many countable
grains, whereas the rat's fur is internally smooth. Measuring grain DENSITY
(not amplitude) in the median-bandpass space gives -4 to -7 sigma separation
and, crucially, holds on both cameras.

Validated on real 021722 frames (both views, 10/10 over the dwell sequence).

Pipeline — everything lives in the median-bandpass space
--------------------------------------------------------
    median bandpass = medianBlur(k_small) - medianBlur(k_large)
        bedding -> grainy (high amplitude); smooth fur -> ~0.
    grain-count map = box-sum of STRICT local maxima (patch ~112 px)
        LOW over the rat, HIGH over bedding.            [localize + seed]
    body-scale -LoG = Mexican hat (sigma ~80 px)
        a filled, centred positive peak on the rat body. [centroid]
    circle grow     = grow circles from the seed blob outward against the
        grain-count barrier; their union is the mask.    [segment]

The body-scale centroid is used (not the seed-blob centroid) because the
-LoG peak sits on the body centre rather than drifting toward the smooth
cable, which trims the localization offset. The centroid then feeds the
existing reconstruction stack (``triangulate_dlt`` + ``accept_point``), which
rejects the floor reflection — it triangulates below z = 0 — and any
through-wall artifact.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import cv2
import numpy as np
from scipy.ndimage import gaussian_laplace, maximum_filter, minimum_filter


# ────────────────────────────────────────────────────────────────────
#  Median-bandpass grain features
# ────────────────────────────────────────────────────────────────────
def _as_uint8(gray: np.ndarray) -> np.ndarray:
    if gray.dtype == np.uint8:
        return gray
    g = gray.astype(np.float32)
    return cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def median_bandpass(gray: np.ndarray, small_k: int = 3,
                    large_k: int = 21) -> np.ndarray:
    """Median bandpass: robust, edge-preserving, straddling the grain scale.

    The small/large median difference keeps the bedding grain (a field of
    ~3-8 px features) while cancelling on smooth fur (both medians agree).
    Median (not Gaussian) is used because it is robust to the bright rat and
    to speckle. Requires uint8 for OpenCV's large-kernel median; floats are
    min/max normalized first.
    """
    g8 = _as_uint8(gray)
    small = cv2.medianBlur(g8, int(small_k)).astype(np.float32)
    large = cv2.medianBlur(g8, int(large_k)).astype(np.float32)
    return small - large


def grain_peaks(mbp: np.ndarray, peak_frac: float = 0.5) -> np.ndarray:
    """Binary map of STRICT local maxima of the median bandpass above a
    fraction of its std.

    The strictness (``mbp > minimum_filter``) excludes flat plateaus: a naive
    ``mbp == maximum_filter`` flags every pixel of a smooth region (the rat)
    as a maximum, which inverts the signal.
    """
    mx = maximum_filter(mbp, size=3)
    mn = minimum_filter(mbp, size=3)
    return ((mbp == mx) & (mbp > mn)
            & (mbp > mbp.std() * float(peak_frac))).astype(np.float32)


def grain_count_map(mbp: np.ndarray, patch: int = 112,
                    peak_frac: float = 0.5) -> np.ndarray:
    """Per-pixel count of grain peaks in a ``patch``x``patch`` window.

    Computed as a single box filter of the strict-maxima map — O(N) for any
    patch size (no per-patch labeling, no per-threshold pass). LOW over the
    smooth rat, HIGH over bedding. A larger patch tightens the bedding
    distribution (bigger separation) at the cost of localization resolution;
    ~112 px is the detection sweet spot for a ~250 px rat.
    """
    peaks = grain_peaks(mbp, peak_frac)
    return cv2.boxFilter(peaks, cv2.CV_32F, (int(patch), int(patch)),
                         normalize=False)


def body_blob(gray: np.ndarray, sigma: float = 80.0) -> np.ndarray:
    """Body-scale Mexican hat (scale-normalized -LoG).

    A bright, body-sized blob (the rat) produces a positive peak at its
    centre. Used to place the centroid on the body rather than on the seed
    blob, whose centroid drifts toward the smooth cable.
    """
    g = gray.astype(np.float32)
    return -(float(sigma) ** 2) * gaussian_laplace(g, float(sigma))


def laplacian_magnitude(mbp: np.ndarray, sigma: float = 3.0) -> np.ndarray:
    """Sigma-robust texture-energy barrier: ``Gaussian(|Laplacian(mbp)|)``.

    The Laplacian of the median bandpass turns each bedding grain into a
    +/- zero-crossing pair; its magnitude is a non-negative energy — HIGH
    over bedding, LOW over the smooth rat. Because it is rectified, Gaussian
    smoothing preserves the local mean, so the rat/bedding separation is
    nearly independent of sigma (measured ~6x for cam0, ~3.7x for cam1, flat
    from sigma 0.5 to 20) — unlike the *signed* Laplacian field, whose
    separation collapses as the grain oscillations cancel. sigma ~ 2-4 keeps
    the map contiguous while denoising, and the choice is forgiving.

    An alternative to the grain-count map as the segmentation barrier: same
    "high = bedding, low = rat" polarity, keyed on grain energy rather than
    grain count. Pair the two with :func:`combine_barriers` for robustness.
    """
    lap = cv2.Laplacian(mbp.astype(np.float32), cv2.CV_32F, ksize=3)
    return cv2.GaussianBlur(np.abs(lap), (0, 0), float(sigma))


def combine_barriers(maps, floor_mask) -> np.ndarray:
    """Combine several "high = bedding" barrier maps into one.

    Each map is z-scored within the floor ROI (so grain-count and Laplacian
    energy are comparable), then the elementwise maximum is taken. Because a
    pixel is 'inside the rat' only where the combined barrier is LOW, and
    max(a, b) < t iff a < t AND b < t, thresholding the combined map keeps
    only pixels that are smooth by *every* measure — the double barrier that
    closes the low-texture leak channels a single measure can miss.
    """
    floor = (floor_mask > 0)
    z = []
    for m in maps:
        mf = m[floor]
        z.append((m - float(mf.mean())) / (float(mf.std()) + 1e-6))
    return np.maximum.reduce(z).astype(np.float32)


def cable_suppressed_map(gray: np.ndarray, mbp: np.ndarray,
                         floor_mask: np.ndarray, illum_sigma: float = 201.0,
                         barrier_sigma: float = 32.0) -> np.ndarray:
    """Cable-suppressed rat map — rat = LOW, cable folded into bedding.

    The ``|Laplacian|`` texture barrier cannot separate the rat from the
    cable (both are smooth), but *intensity* can: the rat is bright, the
    cable is dark. This mixes the inverted, illumination-flattened,
    floor-normalized intensity 50/50 with the smoothed ``|Laplacian|``
    barrier. The rat is low in BOTH (smooth *and* bright); the cable is low
    in the barrier but HIGH in the inverted intensity (it's dark), so it
    averages up into the bedding cluster and stops dragging the centroid.

    The pieces are the ones that were swept on real frames: a broad
    illumination map (sigma 201, wider than the rat so it isn't muted),
    normalization to the *floor* range (so the bright rails/glove don't set
    the scale), and a sigma-``barrier_sigma`` (default 32) ``|Laplacian|``
    barrier — the value that maximized the rat gap while keeping the best
    centroid.

    Returns a float map in ~[0, 1]; the rat is the distinct minimum.
    """
    g = gray.astype(np.float32)
    floor = (floor_mask > 0)
    illum = cv2.GaussianBlur(g, (0, 0), float(illum_sigma))
    flat = g / np.maximum(illum, 1.0)
    lo, hi = np.percentile(flat[floor], 2), np.percentile(flat[floor], 98)
    inv = 1.0 - np.clip((flat - lo) / (hi - lo + 1e-6), 0, 1)
    lap = np.abs(cv2.Laplacian(mbp.astype(np.float32), cv2.CV_32F, ksize=3))
    laps = cv2.GaussianBlur(lap, (0, 0), float(barrier_sigma))
    a, b = np.percentile(laps, 1), np.percentile(laps, 99)
    bar = np.clip((laps - a) / (b - a + 1e-6), 0, 1)
    return (0.5 * inv + 0.5 * bar).astype(np.float32)


# ────────────────────────────────────────────────────────────────────
#  Segmentation — grow circles from the seed against the grain barrier
# ────────────────────────────────────────────────────────────────────
def circle_grow_segment(seed_blob: np.ndarray, barrier: np.ndarray,
                        floor_mask: np.ndarray, barrier_pct: float = 45.0,
                        n_seeds: int = 120, min_radius: int = 3,
                        open_k: int = 7, close_k: int = 21,
                        rng: Optional[np.random.Generator] = None
                        ) -> np.ndarray:
    """Grow circles from ``seed_blob`` outward against a texture ``barrier``.

    ``barrier`` is any "high = bedding, low = rat" map — the grain-count map
    (:func:`grain_count_map`), the Laplacian energy
    (:func:`laplacian_magnitude`), or their combination
    (:func:`combine_barriers`). 'Inside the rat' means the (smoothed) barrier
    is below the ``barrier_pct`` percentile of the in-floor distribution — as
    smooth as the rat. Circles seeded across the blob grow to the distance to
    the nearest barrier (a distance transform of the smooth region); their
    union is the silhouette. Because the barrier is *texture*, not brightness,
    this holds on the low-contrast camera where a brightness fill leaks.

    ``barrier_pct`` is the one knob for the harder view: raise it to let the
    circles reach a little further before the texture wall stops them.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    floor = (floor_mask > 0)
    gcs = cv2.GaussianBlur(barrier, (0, 0), 9)
    thr = float(np.percentile(gcs[floor], barrier_pct))
    free = ((gcs < thr) & floor).astype(np.uint8)
    if open_k:
        free = cv2.morphologyEx(free, cv2.MORPH_OPEN,
                                np.ones((open_k, open_k), np.uint8))
    dt = cv2.distanceTransform(free, cv2.DIST_L2, 5)
    ys, xs = np.where(seed_blob > 0)
    if len(xs) == 0:
        return np.zeros_like(free)
    pick = rng.choice(len(xs), min(int(n_seeds), len(xs)), replace=False)
    union = np.zeros_like(free)
    for k in pick:
        cx, cy = int(xs[k]), int(ys[k])
        rad = int(dt[cy, cx])
        if rad >= min_radius:
            cv2.circle(union, (cx, cy), rad, 1, -1)
    if close_k:
        union = cv2.morphologyEx(union, cv2.MORPH_CLOSE,
                                 np.ones((close_k, close_k), np.uint8))
    return union


# ────────────────────────────────────────────────────────────────────
#  Single-view detection
# ────────────────────────────────────────────────────────────────────
@dataclass
class Detection:
    found: bool
    centroid: Optional[Tuple[float, float]]   # (x, y) px — best candidate
    mask: np.ndarray                          # segmented silhouette (uint8)
    seed_blob: np.ndarray                     # low-grain-count localization
    separation: float                         # rat vs bedding grain sigma (<0)
    candidates: list = field(default_factory=list)   # top-K (x,y), best first
                                              # — fed to the stereo epipolar match


def detect(gray: np.ndarray, floor_mask: np.ndarray, patch: int = 112,
           blob_sigma: float = 80.0, detect_pct: float = 90.0,
           min_area: int = 1500, barrier_pct: float = 45.0,
           peak_frac: float = 0.5, seg_barrier: str = "grain",
           barrier_sigma: float = 3.0, cable_suppress: bool = False,
           illum_sigma: float = 201.0, cable_barrier_sigma: float = 32.0,
           max_candidates: int = 3,
           rng: Optional[np.random.Generator] = None) -> Detection:
    """Detect the rat in one camera view.

    Localization uses the grain-count map (the robust localizer). The
    segmentation barrier is selectable via ``seg_barrier`` ('grain' default,
    'laplacian', or 'both').

    Centroid: the body-scale -LoG peak by default. With ``cable_suppress``
    the centroid is taken from the low region of the cable-suppressed map
    (:func:`cable_suppressed_map`) instead, which folds the thin cable into
    bedding so it no longer drags the centroid (measured ~64-67 px vs ~70-78
    px on real frames).

    Returns a :class:`Detection` whose ``candidates`` holds the top
    ``max_candidates`` low-grain blobs (best first) as (x, y) centroids —
    these feed the stereo epipolar match, so the correct rat can be chosen
    across views even when it isn't the largest blob in one of them.
    """
    floor = (floor_mask > 0).astype(np.uint8)
    empty = np.zeros_like(floor)
    none = Detection(False, None, empty, empty, 0.0, [])
    mbp = median_bandpass(gray)
    gc = grain_count_map(mbp, patch, peak_frac)

    # 1) localize: low-grain-count blobs within the floor ROI
    inv = gc.max() - gc
    inv[floor == 0] = 0
    invs = cv2.GaussianBlur(inv, (0, 0), 21)
    infloor = invs[floor > 0]
    if infloor.size == 0:
        return none
    m = ((invs > np.percentile(infloor, detect_pct)) & (floor > 0)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((13, 13), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((41, 41), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    comps = [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= min_area]
    if not comps:
        return none
    comps.sort(key=lambda i: -st[i, cv2.CC_STAT_AREA])
    comps = comps[:max(1, int(max_candidates))]

    # centroid helper: cable-suppressed low region if requested, else -LoG peak
    mix = (cable_suppressed_map(gray, mbp, floor, illum_sigma,
                                cable_barrier_sigma)
           if cable_suppress else None)
    bb = body_blob(gray, blob_sigma)

    def _centroid(i):
        blob_i = (lab == i)
        region = cv2.dilate(blob_i.astype(np.uint8), np.ones((31, 31), np.uint8))
        if mix is not None:
            within = np.where(region > 0, mix, 1.0)
            thr = np.percentile(within[region > 0], 35)
            low = (within < thr) & (region > 0)
            if int(low.sum()) > 50:
                ys, xs = np.where(low)
                return (float(xs.mean()), float(ys.mean()))
        masked = np.where(region > 0, bb, -np.inf)
        py, px = np.unravel_index(int(np.argmax(masked)), masked.shape)
        return (float(px), float(py))

    candidates = [_centroid(i) for i in comps]
    seed_blob = (lab == comps[0]).astype(np.uint8)
    centroid = candidates[0]

    # separation (grain count: rat region vs in-floor bedding), in sigma
    bed = gc[(floor > 0) & (seed_blob == 0)]
    ratv = float(gc[seed_blob > 0].mean())
    sep = (ratv - float(bed.mean())) / (float(bed.std()) + 1e-6)

    # segment the primary blob against the texture barrier
    if seg_barrier == "laplacian":
        barrier = laplacian_magnitude(mbp, barrier_sigma)
    elif seg_barrier == "both":
        barrier = combine_barriers(
            [gc, laplacian_magnitude(mbp, barrier_sigma)], floor)
    else:  # "grain"
        barrier = gc
    mask = circle_grow_segment(seed_blob, barrier, floor, barrier_pct, rng=rng)

    return Detection(True, centroid, mask, seed_blob, float(sep), candidates)


# ────────────────────────────────────────────────────────────────────
#  Stereo detection → 3D (uses the existing reconstruction stack)
# ────────────────────────────────────────────────────────────────────
def build_floor_mask(dlt_P: np.ndarray, arena_corners: np.ndarray,
                     image_shape: tuple, mode: str = "volume",
                     max_height_mm: float = 260.0, pad_px: int = 20
                     ) -> np.ndarray:
    """Convenience: the projected arena ROI mask for one view (defaults to the
    corrected volume band from patch 0076)."""
    from rpimocap.detection.segment import (arena_roi_corners,
                                            arena_roi_mask)
    corners = arena_roi_corners(arena_corners, mode, max_height_mm)
    return arena_roi_mask(dlt_P, corners, image_shape, pad_px)


@dataclass
class StereoResult:
    point: Optional[np.ndarray]     # (3,) arena-mm 3D point, or None
    accepted: bool                  # epipolar-consistent + in-arena
    reproj_err: float               # max per-view reprojection error (px)
    det0: Detection
    det1: Detection


def detect_stereo(gray0: np.ndarray, gray1: np.ndarray,
                  floor0: np.ndarray, floor1: np.ndarray,
                  dlt_P0: np.ndarray, dlt_P1: np.ndarray,
                  max_epipolar_px: float = 60.0, max_reproj_px: float = 60.0,
                  rng: Optional[np.random.Generator] = None,
                  **detect_kw) -> StereoResult:
    """Epipolar-consistent stereo detection → 3D.

    Detects candidate rats in both views, then uses the reconstruction
    epipolar matcher (:func:`best_stereo_point`) to choose the cam0-cam1
    pairing that is geometrically consistent — lies on each other's epipolar
    line (within ``max_epipolar_px``), triangulates inside the arena, and
    reprojects within ``max_reproj_px``. This is the tight coupling that was
    missing: instead of blindly triangulating each view's single best blob
    (which fails when they don't correspond), the correct rat is matched
    across views and single-view false positives — including the floor
    reflection, which has no consistent partner — are dropped.

    Tolerances default loose (60 px) because the topology centroids are the
    rat's approximate centre (~70 px), not a sub-pixel keypoint; tighten
    them once a sharper centroid (e.g. ``cable_suppress=True``) is in use.

    Returns a :class:`StereoResult`.
    """
    from rpimocap.reconstruction.arena_gate import STD_ARENA
    from rpimocap.reconstruction.epipolar import best_stereo_point

    d0 = detect(gray0, floor0, rng=rng, **detect_kw)
    d1 = detect(gray1, floor1, rng=rng, **detect_kw)
    if not (d0.found and d1.found):
        return StereoResult(None, False, float("nan"), d0, d1)

    match = best_stereo_point(
        d0.candidates, d1.candidates, dlt_P0, dlt_P1,
        max_epipolar_px=max_epipolar_px, max_reproj_px=max_reproj_px,
        arena_bounds=STD_ARENA, require_in_arena=True)
    if match is None:
        return StereoResult(None, False, float("nan"), d0, d1)
    return StereoResult(np.asarray(match.point), True,
                        float(match.reproj_err), d0, d1)
