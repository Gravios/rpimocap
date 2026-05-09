"""
rpimocap.detection.segment
===========================
Background-subtraction-based animal segmentation with optional SAM/SAM2
integration and epipolar-constrained stereo region matching.

Architecture
------------
BackgroundModel
    Estimates per-pixel background by taking the median (or mode) over
    a sample of frames.  Robust to the animal being present in some of
    those frames as long as it does not dominate a single pixel position.

ForegroundDetector
    Subtracts the background, thresholds, applies morphological cleanup,
    and returns binary foreground masks together with connected-component
    blobs.

BodyRegion
    One detected body region: centroid (px), mask (bool H×W), label,
    bounding box, orientation, and a confidence score.

GeometricLabeller
    Assigns anatomical labels to regions using blob shape only.
    No ML required.  Works by:
      1. Fitting an ellipse to the foreground blob → spine axis
      2. Skeletonising the blob → finding endpoints (nose / tail)
      3. Detecting high-variance sub-blobs near the anterior end → ears
      4. Dividing the blob into head / neck / back / rump / tail zones
         along the principal axis

SAMLabeller  (optional — requires segment-anything or sam2)
    Prompts SAM with the foreground centroid → body part sub-masks.
    Falls back to GeometricLabeller automatically if SAM is not installed.

EpipolarMatcher
    Matches BodyRegion lists from cam0 and cam1 using the epipolar
    constraint derived from the calibrated stereo rig.  For each cam0
    centroid the epipolar line in cam1 is computed; the cam1 region
    whose centroid is closest to that line (within a pixel threshold)
    is accepted as the match.

Usage
-----
    from rpimocap.detection.segment import (
        BackgroundModel, ForegroundDetector,
        GeometricLabeller, EpipolarMatcher)

    bg = BackgroundModel.from_captures(cap0, cap1, n_frames=200)
    det = ForegroundDetector(bg)
    lbl = GeometricLabeller()
    matcher = EpipolarMatcher.from_calibration(cal)

    regions0 = lbl.label(det.detect(frame0))
    regions1 = lbl.label(det.detect(frame1))
    matches  = matcher.match(regions0, regions1)
    xyz_dict = matcher.triangulate(matches)   # {label: np.ndarray (3,)}
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# Body part labels in anterior-to-posterior order
PART_ORDER = ["nose", "left_ear", "right_ear", "head",
              "neck", "back", "rump", "tail_base", "tail_tip"]


# --------------------------------------------------------------------------- #
#  Data classes                                                                #
# --------------------------------------------------------------------------- #

@dataclass
class BodyRegion:
    """One detected body region in a single camera view.

    Attributes
    ----------
    label       : anatomical label (e.g. ``"nose"``, ``"back"``)
    cx, cy      : centroid in video pixel coordinates
    mask        : (H, W) boolean array aligned to the full frame size,
                  or None if only the centroid was detected
    bbox        : (x, y, w, h) bounding box in pixels
    area_px     : region area in pixels
    confidence  : float in [0, 1]; 1.0 for geometric labels, SAM score otherwise
    orientation : major-axis angle in degrees (0 = horizontal right)
    """

    label:       str
    cx:          float
    cy:          float
    mask:        Optional[np.ndarray]  = field(default=None, repr=False)
    bbox:        tuple                 = (0, 0, 0, 0)
    area_px:     float                 = 0.0
    confidence:  float                 = 1.0
    orientation: float                 = 0.0

    def as_array(self) -> np.ndarray:
        return np.array([self.cx, self.cy], dtype=np.float64)


@dataclass
class ForegroundResult:
    """Output of ForegroundDetector for one frame.

    Attributes
    ----------
    mask        : (H, W) uint8 binary foreground mask (255 = foreground)
    blobs       : list of per-blob (stats, centroid) from connectedComponentsWithStats
    frame_gray  : (H, W) uint8 grayscale frame used for detection
    n_blobs     : number of blobs above the minimum area threshold
    """

    mask:       np.ndarray
    blobs:      list
    frame_gray: np.ndarray
    n_blobs:    int


# --------------------------------------------------------------------------- #
#  Background model                                                            #
# --------------------------------------------------------------------------- #

class BackgroundModel:
    """Per-pixel background image estimated from a sample of frames.

    Parameters
    ----------
    bg0, bg1 : (H, W) float32 arrays — background for cam0 and cam1
    method   : ``"median"`` or ``"mean"``
    """

    def __init__(self, bg0: np.ndarray, bg1: np.ndarray,
                 method: str = "median"):
        self.bg0    = bg0.astype(np.float32)
        self.bg1    = bg1.astype(np.float32)
        self.method = method

    # ------------------------------------------------------------------ #

    @classmethod
    def from_captures(
        cls,
        cap0,
        cap1,
        n_frames:      int   = 200,
        method:        str   = "median",
        start_frame:   int   = 0,
        bayer_pattern: str   = "RGGB",
        verbose:       bool  = True,
    ) -> "BackgroundModel":
        """Build a background model by sampling ``n_frames`` from both captures.

        Parameters
        ----------
        cap0, cap1    : cv2.VideoCapture-compatible objects (including TiffCapture)
        n_frames      : number of frames to sample (evenly spaced)
        method        : ``"median"`` (robust to animal presence) or ``"mean"``
        start_frame   : first frame index to consider
        bayer_pattern : passed to _to_gray for Bayer demosaic consistency
        verbose       : print progress
        """
        total = int(min(cap0.get(cv2.CAP_PROP_FRAME_COUNT),
                        cap1.get(cv2.CAP_PROP_FRAME_COUNT)))
        n_frames = min(n_frames, total - start_frame)
        step = max(1, (total - start_frame) // n_frames)
        indices = list(range(start_frame,
                             min(start_frame + step * n_frames, total),
                             step))[:n_frames]

        if verbose:
            print(f"  Building background model from {len(indices)} frames...")

        stack0, stack1 = [], []
        for i, idx in enumerate(indices):
            for cap, stack in [(cap0, stack0), (cap1, stack1)]:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    stack.append(cls._to_gray(frame))
            if verbose and (i + 1) % 50 == 0:
                print(f"    {i + 1}/{len(indices)}")

        if not stack0:
            raise RuntimeError("No frames could be read for background model")

        fn = np.median if method == "median" else np.mean
        bg0 = fn(np.stack(stack0, axis=0), axis=0).astype(np.float32)
        bg1 = fn(np.stack(stack1, axis=0), axis=0).astype(np.float32)

        if verbose:
            print(f"  Background model built  ({method}, {len(stack0)} frames)")

        return cls(bg0, bg1, method)

    @classmethod
    def from_multiple_captures(
        cls,
        caps0:         list,
        caps1:         list,
        n_frames_each: int  = 100,
        method:        str  = "median",
        start_frame:   int  = 0,
        verbose:       bool = True,
    ) -> "BackgroundModel":
        """Build a background model from multiple stereo video pairs.

        Each pair contributes ``n_frames_each`` evenly-spaced frames.
        The final background is the median (or mean) across ALL sampled
        frames from ALL sessions combined — the animal appears in
        different positions across sessions, so the median converges to
        the true background far more reliably than single-session sampling.

        Parameters
        ----------
        caps0, caps1   : lists of VideoCapture-compatible objects, one
                         per session (must be same length)
        n_frames_each  : frames to sample from each session pair
        method         : ``"median"`` or ``"mean"``
        start_frame    : first frame index to consider in each session
        verbose        : print progress
        """
        assert len(caps0) == len(caps1), "caps0 and caps1 must have same length"

        stack0, stack1 = [], []
        for si, (cap0, cap1) in enumerate(zip(caps0, caps1)):
            total = int(min(cap0.get(cv2.CAP_PROP_FRAME_COUNT),
                            cap1.get(cv2.CAP_PROP_FRAME_COUNT)))
            n = min(n_frames_each, total - start_frame)
            step = max(1, (total - start_frame) // n)
            indices = list(range(start_frame,
                                  min(start_frame + step * n, total),
                                  step))[:n]
            if verbose:
                print(f"  Session {si+1}/{len(caps0)}: "
                      f"sampling {len(indices)} frames ...")
            for idx in indices:
                for cap, stack in [(cap0, stack0), (cap1, stack1)]:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    if ret:
                        stack.append(cls._to_gray(frame))

        if not stack0:
            raise RuntimeError("No frames could be read for background model")

        fn = np.median if method == "median" else np.mean
        bg0 = fn(np.stack(stack0, axis=0), axis=0).astype(np.float32)
        bg1 = fn(np.stack(stack1, axis=0), axis=0).astype(np.float32)

        total_frames = len(stack0)
        n_sessions   = len(caps0)
        if verbose:
            print(f"  Background model built  "
                  f"({method}, {total_frames} frames, {n_sessions} sessions)")
        return cls(bg0, bg1, method)

    @classmethod
    def from_npz(cls, path: str | Path) -> "BackgroundModel":
        """Load a saved background model."""
        d = np.load(path)
        return cls(d["bg0"], d["bg1"], str(d["method"]))

    def save(self, path: str | Path) -> None:
        """Save background arrays to a .npz file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, bg0=self.bg0, bg1=self.bg1,
                            method=self.method)

    @staticmethod
    def _to_gray(frame: np.ndarray) -> np.ndarray:
        """Convert a BGR frame to uint8 grayscale."""
        if frame.ndim == 2:
            return frame.astype(np.uint8)
        return cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_BGR2GRAY)


# --------------------------------------------------------------------------- #
#  Foreground detector                                                         #
# --------------------------------------------------------------------------- #

class ForegroundDetector:
    """Subtract background and return binary foreground masks.

    Preprocessing pipeline (applied before background subtraction):

    1. Channel extraction  — green channel only (``use_green_channel=True``)
       Under NIR illumination the Bayer green pixels carry ~2x the signal
       of red/blue.  Extracting the green channel gives a cleaner
       single-channel image with better SNR.

    2. CLAHE               — adaptive histogram equalisation (``clahe=True``)
       Enhances local contrast so fur texture is distinguishable from
       bedding even when the animal is dark against a dark background.
       Applied per-tile so bright arena walls don't suppress the animal.

    3. Bilateral filter    — edge-preserving smooth (``bilateral=True``)
       Reduces sensor noise while keeping the fur/bedding boundary sharp.
       Better than Gaussian for this task.

    Parameters
    ----------
    background        : BackgroundModel
    threshold         : absolute pixel difference threshold (0–255)
    min_area_px       : discard blobs smaller than this (noise filter)
    morph_k           : morphological kernel size for opening/closing
    blur_k            : Gaussian blur kernel (0 = skip; overridden by bilateral)
    clahe             : apply CLAHE before background subtraction
    clahe_clip        : CLAHE clip limit (higher = more contrast, more noise)
    clahe_tile        : CLAHE tile grid size (smaller = more local)
    use_green_channel : extract green Bayer channel instead of luminance
    bilateral         : apply bilateral filter instead of Gaussian blur
    bilateral_d       : bilateral filter diameter (neighbourhood size)
    bilateral_sigma   : bilateral sigma for colour and spatial
    """

    def __init__(
        self,
        background:        BackgroundModel,
        threshold:         float = 25.0,
        min_area_px:       int   = 500,
        morph_k:           int   = 7,
        blur_k:            int   = 5,
        clahe:             bool  = False,
        clahe_clip:        float = 2.0,
        clahe_tile:        int   = 8,
        use_green_channel: bool  = False,
        bilateral:         bool  = False,
        bilateral_d:       int   = 9,
        bilateral_sigma:   float = 50.0,
    ):
        self.bg                = background
        self.threshold         = threshold
        self.min_area_px       = min_area_px
        self.use_green_channel = use_green_channel
        self.bilateral         = bilateral
        self.bilateral_d       = bilateral_d
        self.bilateral_sigma   = bilateral_sigma
        self._kernel           = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (morph_k, morph_k))
        self._blur_k = blur_k | 1

        # CLAHE instance (shared across frames for efficiency)
        self._clahe = (cv2.createCLAHE(
                           clipLimit=clahe_clip,
                           tileGridSize=(clahe_tile, clahe_tile))
                       if clahe else None)

    # ------------------------------------------------------------------ #

    def _to_enhanced_gray(self, frame: np.ndarray) -> np.ndarray:
        """Convert frame to enhanced grayscale applying the configured pipeline.

        Steps:
          1. Channel extraction (green or luminance)
          2. CLAHE (if enabled)
          3. Bilateral or Gaussian smooth (if enabled)

        Returns uint8 (H, W) array.
        """
        # Step 1 — channel extraction
        if self.use_green_channel and frame.ndim == 3 and frame.shape[2] == 3:
            # BGR: index 1 is green.  Under NIR the green channel has
            # the most signal — use it directly.
            gray = frame[:, :, 1].copy()
        else:
            gray = BackgroundModel._to_gray(frame)

        # Step 2 — CLAHE
        if self._clahe is not None:
            gray = self._clahe.apply(gray)

        # Step 3 — smoothing
        if self.bilateral:
            gray = cv2.bilateralFilter(
                gray.astype(np.uint8),
                self.bilateral_d,
                self.bilateral_sigma,
                self.bilateral_sigma)
        elif self._blur_k > 1:
            gray = cv2.GaussianBlur(
                gray.astype(np.uint8),
                (self._blur_k, self._blur_k), 0)

        return gray.astype(np.uint8)

    # ------------------------------------------------------------------ #

    def detect(self, frame: np.ndarray, cam: int = 0) -> ForegroundResult:
        """Detect foreground regions in one frame.

        Parameters
        ----------
        frame : BGR uint8 frame
        cam   : 0 or 1 — selects which background model to use

        Returns
        -------
        ForegroundResult
        """
        # ── Current frame: full enhancement pipeline ─────────────────────
        gray = self._to_enhanced_gray(frame).astype(np.float32)

        # ── Background: extract same channel but NO CLAHE ─────────────
        # CLAHE must NOT be applied to the background — it normalises
        # the histogram of both images to the same distribution, which
        # collapses the difference signal and makes the animal invisible.
        # Instead: extract the same channel (green or luminance) from
        # the background so the subtraction is in the same colour space,
        # but leave the histogram untouched.
        bg_raw = self.bg.bg0 if cam == 0 else self.bg.bg1

        if self.use_green_channel and frame.ndim == 3:
            # Background was built from BGR→gray luminance.
            # Re-extract green channel from a synthetic gray BGR for
            # consistency: since bg_raw is already gray, all channels
            # are equal so green channel == the gray value.
            bg = bg_raw.copy()
        else:
            bg = bg_raw.copy()

        # Resize if needed
        if bg.shape != gray.shape:
            bg = cv2.resize(bg.astype(np.float32),
                            (gray.shape[1], gray.shape[0]),
                            interpolation=cv2.INTER_LINEAR)
        else:
            bg = bg.astype(np.float32)

        # ── Diff and threshold ────────────────────────────────────────
        diff   = np.abs(gray - bg)
        binary = (diff > self.threshold).astype(np.uint8) * 255
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  self._kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, self._kernel)

        n, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=8)

        blobs = []
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] >= self.min_area_px:
                blobs.append((stats[i], centroids[i]))

        return ForegroundResult(
            mask=binary,
            blobs=blobs,
            frame_gray=gray.astype(np.uint8),
            n_blobs=len(blobs))

    def largest_blob_mask(self, result: ForegroundResult) -> Optional[np.ndarray]:
        """Return a boolean mask for the largest detected blob, or None."""
        if not result.blobs:
            return None
        best = max(result.blobs, key=lambda b: b[0][cv2.CC_STAT_AREA])
        x, y, w, h = (best[0][cv2.CC_STAT_LEFT],
                      best[0][cv2.CC_STAT_TOP],
                      best[0][cv2.CC_STAT_WIDTH],
                      best[0][cv2.CC_STAT_HEIGHT])
        mask = np.zeros(result.mask.shape, dtype=bool)
        roi  = result.mask[y:y+h, x:x+w] > 0
        mask[y:y+h, x:x+w] = roi
        return mask


# --------------------------------------------------------------------------- #
#  Geometric body-part labeller (no ML)                                       #
# --------------------------------------------------------------------------- #

class GeometricLabeller:
    """Assign anatomical labels using blob shape analysis only.

    Algorithm
    ---------
    1. PCA on foreground pixels → principal axis (spine direction)
    2. Skeletonise blob → find two endpoints (nose, tail_tip)
    3. Assign nose to the more pointed (lower width) end
    4. Divide blob into zones along the spine axis:
       [0–20%] head, [20–30%] neck, [30–65%] back,
       [65–80%] rump, [80–100%] tail
    5. Detect ears: small high-variance blobs in the head zone

    Parameters
    ----------
    ear_variance_percentile : texture variance threshold for ear detection
    min_ear_area_px         : minimum blob area to be considered an ear
    """

    ZONES = [
        ("nose",      0.00, 0.08),
        ("head",      0.08, 0.22),
        ("neck",      0.22, 0.35),
        ("back",      0.35, 0.65),
        ("rump",      0.65, 0.80),
        ("tail_base", 0.80, 0.90),
        ("tail_tip",  0.90, 1.00),
    ]

    def __init__(
        self,
        ear_variance_percentile: float = 85.0,
        min_ear_area_px:         int   = 40,
    ):
        self._ear_var_pct    = ear_variance_percentile
        self._min_ear_area   = min_ear_area_px

    # ------------------------------------------------------------------ #

    def label(
        self,
        fg: ForegroundResult,
        frame_gray: Optional[np.ndarray] = None,
    ) -> list[BodyRegion]:
        """Return labelled BodyRegions from a ForegroundResult.

        Parameters
        ----------
        fg         : output of ForegroundDetector.detect()
        frame_gray : (H, W) uint8 — used for ear texture detection.
                     Falls back to fg.frame_gray if None.
        """
        if not fg.blobs:
            return []

        if frame_gray is None:
            frame_gray = fg.frame_gray

        # Use largest blob as the animal
        best_stats, best_centroid = max(fg.blobs,
                                         key=lambda b: b[0][cv2.CC_STAT_AREA])
        x0, y0 = best_stats[cv2.CC_STAT_LEFT], best_stats[cv2.CC_STAT_TOP]
        w0, h0 = best_stats[cv2.CC_STAT_WIDTH], best_stats[cv2.CC_STAT_HEIGHT]

        blob_mask = np.zeros(fg.mask.shape, dtype=np.uint8)
        blob_mask[y0:y0+h0, x0:x0+w0] = (
            fg.mask[y0:y0+h0, x0:x0+w0] > 0).astype(np.uint8)

        # PCA on foreground pixels → spine axis
        ys, xs = np.where(blob_mask > 0)
        pts = np.stack([xs, ys], axis=1).astype(np.float32)
        if len(pts) < 20:
            return []

        mean = pts.mean(axis=0)
        cov  = np.cov(pts.T)
        evals, evecs = np.linalg.eigh(cov)
        principal = evecs[:, np.argmax(evals)]  # major axis direction
        angle_deg = math.degrees(math.atan2(principal[1], principal[0]))

        # Project all points onto the principal axis
        proj = (pts - mean) @ principal      # signed scalar projection
        proj_min, proj_max = proj.min(), proj.max()
        span = proj_max - proj_min
        if span < 1:
            return []

        # Determine nose end: the end with smaller mean width
        # (nose is pointed, tail end can also be pointed — use local width)
        head_mask  = proj <= proj_min + 0.25 * span
        tail_mask  = proj >= proj_max - 0.25 * span
        head_pts   = pts[head_mask]
        tail_pts   = pts[tail_mask]

        def _transverse_width(region_pts, axis):
            """RMS distance from the axis in the perpendicular direction."""
            perp = np.array([-axis[1], axis[0]])
            return np.abs((region_pts - mean) @ perp).std() + 1e-6

        head_w = _transverse_width(head_pts, principal) if len(head_pts) > 5 else 1e6
        tail_w = _transverse_width(tail_pts, principal) if len(tail_pts) > 5 else 1e6

        # Nose = narrower end; flip principal axis if tail is narrower
        if tail_w < head_w:
            principal = -principal
            proj      = -proj
            proj_min, proj_max = proj.min(), proj.max()
            span = proj_max - proj_min

        # Normalise projection to [0, 1] along spine
        t = (proj - proj_min) / span    # 0 = nose, 1 = tail_tip

        regions: list[BodyRegion] = []

        # Zone centroids
        for label, t0, t1 in self.ZONES:
            in_zone = (t >= t0) & (t <= t1)
            if in_zone.sum() < 4:
                continue
            zone_pts = pts[in_zone]
            cx, cy   = zone_pts.mean(axis=0)
            zone_mask = np.zeros_like(blob_mask, dtype=bool)
            zone_mask[zone_pts[:,1].astype(int),
                      zone_pts[:,0].astype(int)] = True
            regions.append(BodyRegion(
                label=label, cx=float(cx), cy=float(cy),
                mask=zone_mask,
                area_px=float(in_zone.sum()),
                orientation=angle_deg,
                confidence=0.7))

        # Ear detection: high local variance in the head zone
        ear_regions = self._detect_ears(
            frame_gray, blob_mask, pts, t, mean, principal, angle_deg)
        regions.extend(ear_regions)

        return regions

    def _detect_ears(
        self,
        gray:      np.ndarray,
        blob_mask: np.ndarray,
        pts:       np.ndarray,
        t:         np.ndarray,
        mean:      np.ndarray,
        principal: np.ndarray,
        angle_deg: float,
    ) -> list[BodyRegion]:
        """Detect ears as high-variance sub-regions in the head zone."""
        # Restrict to head zone (t ∈ [0.0, 0.25])
        head_zone = (t >= 0.0) & (t <= 0.25)
        if head_zone.sum() < self._min_ear_area:
            return []

        head_pts = pts[head_zone].astype(int)
        head_gray = np.zeros_like(gray)
        head_gray[head_pts[:,1], head_pts[:,0]] = gray[head_pts[:,1],
                                                        head_pts[:,0]]

        # Local variance via Laplacian magnitude
        lap  = cv2.Laplacian(head_gray, cv2.CV_64F)
        var  = np.abs(lap)

        # Threshold on high-variance pixels within the head blob
        thresh = np.percentile(var[head_pts[:,1], head_pts[:,0]],
                                self._ear_var_pct)
        ear_mask = ((var > thresh) & (head_gray > 0)).astype(np.uint8)

        # Morphological cleanup
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        ear_mask = cv2.morphologyEx(ear_mask, cv2.MORPH_OPEN, k)

        n, labels, stats, centroids = cv2.connectedComponentsWithStats(
            ear_mask, connectivity=8)

        # Perpendicular to spine (left/right axis)
        perp = np.array([-principal[1], principal[0]])

        ear_blobs = []
        for i in range(1, n):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < self._min_ear_area:
                continue
            cx, cy = centroids[i]
            # Lateral sign: positive = left, negative = right
            lateral = float(np.dot(np.array([cx, cy]) - mean, perp))
            ear_blobs.append((area, cx, cy, lateral, i, labels, stats))

        if not ear_blobs:
            return []

        # Sort by lateral position
        ear_blobs.sort(key=lambda b: b[3], reverse=True)

        ears = []
        for j, blob in enumerate(ear_blobs[:2]):
            area, cx, cy, lat, li, lab, st = blob
            side  = "left" if lat >= 0 else "right"
            label = f"{side}_ear"
            bmask = (lab == li)
            ears.append(BodyRegion(
                label=label, cx=float(cx), cy=float(cy),
                mask=bmask, area_px=float(area),
                orientation=angle_deg, confidence=0.5))

        return ears


# --------------------------------------------------------------------------- #
#  SAM-based labeller (optional)                                              #
# --------------------------------------------------------------------------- #

class SAMLabeller:
    """Prompt SAM/SAM2 with foreground centroid and sub-region clicks.

    Falls back to GeometricLabeller if segment_anything is not installed.

    Parameters
    ----------
    checkpoint  : path to SAM weights (.pth file)
    model_type  : ``"vit_h"``, ``"vit_l"``, ``"vit_b"`` (SAM1) or
                  ``"sam2"`` (auto-selects SAM2 if available)
    device      : ``"cuda"`` or ``"cpu"``
    """

    def __init__(
        self,
        checkpoint:  str | Path,
        model_type:  str = "vit_h",
        device:      str = "cuda",
    ):
        self._ckpt       = str(checkpoint)
        self._model_type = model_type
        self._device     = device
        self._predictor  = None
        self._fallback   = GeometricLabeller()
        self._available  = self._try_load()

    def _try_load(self) -> bool:
        try:
            if self._model_type == "sam2":
                from sam2.build_sam import build_sam2
                from sam2.sam2_image_predictor import SAM2ImagePredictor
                model = build_sam2(self._ckpt, device=self._device)
                self._predictor = SAM2ImagePredictor(model)
            else:
                from segment_anything import sam_model_registry, SamPredictor
                sam = sam_model_registry[self._model_type](
                    checkpoint=self._ckpt)
                sam.to(device=self._device)
                self._predictor = SamPredictor(sam)
            return True
        except (ImportError, FileNotFoundError, Exception):
            return False

    @property
    def available(self) -> bool:
        return self._available

    def label(
        self,
        fg:         ForegroundResult,
        frame_bgr:  np.ndarray,
        frame_gray: Optional[np.ndarray] = None,
    ) -> list[BodyRegion]:
        """Label body regions using SAM, falling back to geometric if unavailable."""
        if not self._available or self._predictor is None:
            return self._fallback.label(fg, frame_gray)

        if not fg.blobs:
            return []

        try:
            return self._label_with_sam(fg, frame_bgr)
        except Exception as e:
            print(f"  SAM labelling failed ({e}), falling back to geometric")
            return self._fallback.label(fg, frame_gray)

    def _label_with_sam(
        self,
        fg:        ForegroundResult,
        frame_bgr: np.ndarray,
    ) -> list[BodyRegion]:
        """Internal: run SAM on the frame with foreground-guided prompts."""
        self._predictor.set_image(
            cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

        # Build prompt grid within the foreground mask
        ys, xs = np.where(fg.mask > 0)
        if len(xs) < 20:
            return []

        # Sample ~8 points from the foreground as positive prompts
        step     = max(1, len(xs) // 8)
        px       = xs[::step][:8].reshape(-1, 1)
        py       = ys[::step][:8].reshape(-1, 1)
        pts_in   = np.hstack([px, py])
        labels_in = np.ones(len(pts_in), dtype=int)

        masks, scores, _ = self._predictor.predict(
            point_coords=pts_in,
            point_labels=labels_in,
            multimask_output=True,
        )

        regions = []
        for mask, score in zip(masks, scores):
            mask_u8 = mask.astype(np.uint8)
            ys_m, xs_m = np.where(mask_u8 > 0)
            if len(xs_m) < 10:
                continue
            cx, cy = xs_m.mean(), ys_m.mean()
            x, y, w, h = (xs_m.min(), ys_m.min(),
                          xs_m.max()-xs_m.min(), ys_m.max()-ys_m.min())
            regions.append(BodyRegion(
                label="body",        # geometric pass assigns fine labels
                cx=float(cx), cy=float(cy),
                mask=mask.astype(bool),
                bbox=(int(x), int(y), int(w), int(h)),
                area_px=float(len(xs_m)),
                confidence=float(score)))

        # Assign anatomical labels geometrically on top of SAM masks
        return self._assign_labels(regions, fg)

    def _assign_labels(
        self,
        regions: list[BodyRegion],
        fg:      ForegroundResult,
    ) -> list[BodyRegion]:
        """Sort SAM masks by position along spine and assign zone labels."""
        if not regions:
            return regions

        # Global spine axis from all foreground pixels
        ys, xs = np.where(fg.mask > 0)
        pts    = np.stack([xs, ys], axis=1).astype(np.float32)
        mean   = pts.mean(axis=0)
        cov    = np.cov(pts.T)
        _, evecs = np.linalg.eigh(cov)
        axis   = evecs[:, 1]

        # Project each region centroid onto the spine
        for r in regions:
            t = float(np.dot(np.array([r.cx, r.cy]) - mean, axis))
            r._spine_t = t   # type: ignore[attr-defined]

        regions.sort(key=lambda r: r._spine_t)  # type: ignore[attr-defined]

        # Map sorted regions to zone labels
        zone_labels = ["nose", "head", "neck", "back", "rump",
                       "tail_base", "tail_tip"]
        for i, r in enumerate(regions):
            if i < len(zone_labels):
                r.label = zone_labels[i]

        return regions


# --------------------------------------------------------------------------- #
#  Epipolar stereo matcher                                                     #
# --------------------------------------------------------------------------- #

class EpipolarMatcher:
    """Match BodyRegion lists between two cameras using epipolar geometry.

    Computes the fundamental matrix F from the calibrated stereo rig.
    For each cam0 region centroid the epipolar line in cam1 is found;
    the cam1 region closest to that line (within ``max_epipolar_px``)
    is accepted as the stereo match.

    Parameters
    ----------
    P0, P1             : (3, 4) projection matrices
    K0, K1             : (3, 3) intrinsic matrices
    dist0, dist1       : (1, 5) distortion vectors
    R, T               : stereo rotation and translation
    max_epipolar_px    : maximum distance from epipolar line to accept a match
    undistort_pts      : if True, undistort centroids before triangulation
    """

    def __init__(
        self,
        P0:              np.ndarray,
        P1:              np.ndarray,
        K0:              np.ndarray,
        K1:              np.ndarray,
        dist0:           np.ndarray,
        dist1:           np.ndarray,
        R:               np.ndarray,
        T:               np.ndarray,
        max_epipolar_px: float = 8.0,
        undistort_pts:   bool  = True,
    ):
        self.P0 = P0; self.P1 = P1
        self.K0 = K0; self.K1 = K1
        self.dist0 = dist0; self.dist1 = dist1
        self.R = R; self.T = T
        self.max_epipolar_px = max_epipolar_px
        self.undistort_pts   = undistort_pts
        self.F = self._compute_F(K0, K1, R, T)

    @classmethod
    def from_calibration(
        cls,
        cal:             np.ndarray | dict,
        max_epipolar_px: float = 8.0,
    ) -> "EpipolarMatcher":
        """Construct from a loaded calibration .npz.

        Parameters
        ----------
        cal : result of ``np.load("calibration.npz")`` or equivalent dict
        """
        if hasattr(cal, "__getitem__"):
            K0    = cal["K0"]
            K1    = cal["K1"]
            dist0 = np.ravel(cal.get("dist0", np.zeros(5)))
            dist1 = np.ravel(cal.get("dist1", np.zeros(5)))
            R     = cal["R"]
            T     = cal["T"].ravel()
            # Prefer DLT projection matrices (from rpimocap-calibrate-from-corners)
            # which work directly in arena frame without R/T decomposition issues
            if "dlt_P0" in cal.files:
                P0 = cal["dlt_P0"]
                P1 = cal["dlt_P1"]
            else:
                P0 = cal.get("P0", K0 @ np.hstack([np.eye(3), np.zeros((3,1))]))
                P1 = cal.get("P1", K1 @ np.hstack([R, T.reshape(3,1)]))
        else:
            raise TypeError("cal must be a dict-like (npz or dict)")
        return cls(P0=P0, P1=P1, K0=K0, K1=K1,
                   dist0=dist0, dist1=dist1, R=R, T=T,
                   max_epipolar_px=max_epipolar_px)

    @staticmethod
    def _compute_F(K0, K1, R, T) -> np.ndarray:
        """Compute the fundamental matrix F from stereo calibration.

        F = K1^{-T} · [t]× · R · K0^{-1}

        where [t]× is the skew-symmetric cross-product matrix of T.
        """
        t = T.ravel()
        tx = np.array([[ 0,    -t[2],  t[1]],
                       [ t[2],  0,    -t[0]],
                       [-t[1],  t[0],  0   ]], dtype=np.float64)
        E  = tx @ R.astype(np.float64)
        F  = np.linalg.inv(K1).T @ E @ np.linalg.inv(K0)
        return F / (np.abs(F).max() + 1e-12)

    def _epipolar_line(self, x: float, y: float) -> np.ndarray:
        """Epipolar line in cam1 for point (x, y) in cam0. Returns (a, b, c)."""
        return self.F @ np.array([x, y, 1.0])

    def _point_to_line_dist(self, line: np.ndarray,
                             x: float, y: float) -> float:
        """Signed distance from point (x, y) to line ax+by+c=0."""
        a, b, c = line
        denom = math.sqrt(a*a + b*b)
        if denom < 1e-10:
            return float("inf")
        return abs(a*x + b*y + c) / denom

    def _undistort(self, pts: np.ndarray, cam: int) -> np.ndarray:
        """Undistort a (N, 2) array of pixel coordinates."""
        K    = self.K0 if cam == 0 else self.K1
        dist = self.dist0 if cam == 0 else self.dist1
        if np.all(np.abs(dist) < 1e-8):
            return pts
        pts_in = pts.reshape(-1, 1, 2).astype(np.float32)
        pts_out = cv2.undistortPoints(pts_in, K,
                                       dist.reshape(1, -1),
                                       P=K)
        return pts_out.reshape(-1, 2).astype(np.float64)

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def match(
        self,
        regions0: list[BodyRegion],
        regions1: list[BodyRegion],
    ) -> list[tuple[BodyRegion, BodyRegion]]:
        """Return matched (cam0_region, cam1_region) pairs.

        Matching is label-first: if both cameras have a region with the
        same anatomical label, they are matched directly.  Unmatched
        regions fall back to epipolar-nearest matching.
        """
        if not regions0 or not regions1:
            return []

        used1: set[int] = set()
        matched: list[tuple[BodyRegion, BodyRegion]] = []

        # Pass 1: label-based matching
        by_label1 = {r.label: (i, r) for i, r in enumerate(regions1)}
        remaining0 = []
        for r0 in regions0:
            if r0.label in by_label1:
                i1, r1 = by_label1[r0.label]
                matched.append((r0, r1))
                used1.add(i1)
            else:
                remaining0.append(r0)

        # Pass 2: epipolar nearest for unmatched
        for r0 in remaining0:
            cx0, cy0 = r0.cx, r0.cy
            line     = self._epipolar_line(cx0, cy0)
            best_d, best_i, best_r1 = float("inf"), -1, None
            for i1, r1 in enumerate(regions1):
                if i1 in used1:
                    continue
                d = self._point_to_line_dist(line, r1.cx, r1.cy)
                if d < best_d:
                    best_d, best_i, best_r1 = d, i1, r1
            if best_r1 is not None and best_d <= self.max_epipolar_px:
                matched.append((r0, best_r1))
                used1.add(best_i)

        return matched

    def triangulate(
        self,
        matches: list[tuple[BodyRegion, BodyRegion]],
    ) -> dict[str, np.ndarray]:
        """Triangulate matched region pairs → {label: xyz_mm}.

        Returns
        -------
        dict mapping label string to (3,) world coordinate array (mm)
        """
        from rpimocap.reconstruction.triangulate import triangulate_dlt
        result: dict[str, np.ndarray] = {}
        for r0, r1 in matches:
            pts0 = np.array([[r0.cx, r0.cy]])
            pts1 = np.array([[r1.cx, r1.cy]])
            if self.undistort_pts:
                pts0 = self._undistort(pts0, 0)
                pts1 = self._undistort(pts1, 1)
            X = triangulate_dlt(self.P0, self.P1,
                                  (pts0[0,0], pts0[0,1]),
                                  (pts1[0,0], pts1[0,1]))
            # Always use cam0 label as canonical name.
            # Epipolar-matched pairs may have different labels when the
            # geometric labeller determines spine orientation differently
            # between cameras (common when the animal faces across the
            # baseline).  The cam0 label is used as ground truth.
            label = r0.label
            result[label] = X[:3]
        return result

    def reprojection_error(
        self,
        xyz:     np.ndarray,
        r0:      BodyRegion,
        r1:      BodyRegion,
    ) -> tuple[float, float]:
        """Compute reprojection error for a triangulated point.

        Returns
        -------
        (err0_px, err1_px) — RMS pixel error in each camera
        """
        from rpimocap.reconstruction.triangulate import reprojection_error
        return reprojection_error(xyz, self.P0, self.P1,
                                   np.array([r0.cx, r0.cy]),
                                   np.array([r1.cx, r1.cy]))


# =========================================================================== #
#  Diagnostic image writer                                                     #
# =========================================================================== #

def save_diagnostics(
    cap0,
    cap1,
    detector:   "ForegroundDetector",
    labeller:   "GeometricLabeller",
    out_dir:    str | Path = Path("/tmp/rpimocap_diag"),
    n_frames:   int = 6,
    frame_indices: list[int] | None = None,
    cam_labels: tuple[str, str] = ("cam0", "cam1"),
) -> None:
    """Write a set of diagnostic images to out_dir for visual inspection.

    Images written
    --------------
    background/
        bg_cam0.png, bg_cam1.png         — raw background model
        bg_cam0_enhanced.png             — background after contrast pipeline
    frames/
        frame_{idx}_{cam}_raw.png        — demosaiced frame (BGR)
        frame_{idx}_{cam}_enhanced.png   — after green/CLAHE/bilateral
        frame_{idx}_{cam}_diff.png       — |enhanced - bg| difference map
        frame_{idx}_{cam}_mask.png       — binary foreground mask
        frame_{idx}_{cam}_overlay.png    — mask + body part labels on raw frame
        frame_{idx}_{cam}_composite.png  — 4-up grid: raw/enhanced/diff/overlay

    Parameters
    ----------
    cap0, cap1      : VideoCapture-compatible objects (rewound after use)
    detector        : ForegroundDetector (with contrast pipeline configured)
    labeller        : GeometricLabeller
    out_dir         : destination directory (created if needed)
    n_frames        : number of evenly-spaced frames to sample
    frame_indices   : explicit list of frame indices (overrides n_frames)
    cam_labels      : display labels for each camera
    """
    out_dir = Path(out_dir)
    (out_dir / "background").mkdir(parents=True, exist_ok=True)
    (out_dir / "frames").mkdir(parents=True, exist_ok=True)

    # ── Background images ─────────────────────────────────────────────────
    for bg_arr, cam_lbl in [(detector.bg.bg0, cam_labels[0]),
                             (detector.bg.bg1, cam_labels[1])]:
        # Raw background
        bg_u8 = np.clip(bg_arr, 0, 255).astype(np.uint8)
        cv2.imwrite(str(out_dir / "background" / f"bg_{cam_lbl}.png"),
                    cv2.cvtColor(bg_u8, cv2.COLOR_GRAY2BGR))
        # Enhanced background (same pipeline as frames)
        bg_bgr = cv2.cvtColor(bg_u8, cv2.COLOR_GRAY2BGR)
        bg_enh = detector._to_enhanced_gray(bg_bgr)
        cv2.imwrite(str(out_dir / "background" / f"bg_{cam_lbl}_enhanced.png"),
                    cv2.cvtColor(bg_enh, cv2.COLOR_GRAY2BGR))
    print(f"  [diag] background images → {out_dir / 'background'}")

    # ── Frame samples ─────────────────────────────────────────────────────
    total = int(min(cap0.get(cv2.CAP_PROP_FRAME_COUNT),
                    cap1.get(cv2.CAP_PROP_FRAME_COUNT)))
    if frame_indices is None:
        step   = max(1, total // (n_frames + 1))
        frame_indices = [step * (i + 1) for i in range(n_frames)]
    frame_indices = [min(i, total - 1) for i in frame_indices]

    _PART_COLOURS = {
        "nose":      (0,   80, 255),
        "head":      (0,  160, 255),
        "left_ear":  (255, 200, 0),
        "right_ear": (255, 200, 0),
        "neck":      (0,  255, 160),
        "back":      (0,  255,  80),
        "rump":      (255, 120,  0),
        "tail_base": (200,  0, 200),
        "tail_tip":  (255,  0, 200),
    }

    for idx in frame_indices:
        for cam_id, (cap, cam_lbl) in enumerate(
                zip([cap0, cap1], cam_labels)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue

            prefix = out_dir / "frames" / f"frame_{idx:06d}_{cam_lbl}"

            # 1. Raw frame
            cv2.imwrite(str(prefix) + "_raw.png", frame)

            # 2. Enhanced gray (as BGR for saving)
            enhanced = detector._to_enhanced_gray(frame)
            enh_bgr  = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
            cv2.imwrite(str(prefix) + "_enhanced.png", enh_bgr)

            # 3. Diff map (colorised — blue=low, red=high)
            bg_arr = detector.bg.bg0 if cam_id == 0 else detector.bg.bg1
            bg_arr = bg_arr.copy()
            if bg_arr.shape != enhanced.shape:
                bg_arr = cv2.resize(bg_arr,
                                    (enhanced.shape[1], enhanced.shape[0]),
                                    interpolation=cv2.INTER_LINEAR)
            # Do NOT apply CLAHE to background (same fix as in detect())
            diff = np.abs(enhanced.astype(np.float32) - bg_arr.astype(np.float32))
            diff_norm = np.clip(diff / max(diff.max(), 1.0) * 255,
                                0, 255).astype(np.uint8)
            diff_colour = cv2.applyColorMap(diff_norm, cv2.COLORMAP_JET)
            cv2.imwrite(str(prefix) + "_diff.png", diff_colour)

            # 4. Binary mask
            fg = detector.detect(frame, cam=cam_id)
            mask_bgr = cv2.cvtColor(fg.mask, cv2.COLOR_GRAY2BGR)
            # Tint blobs in green, non-blobs in dark
            cv2.imwrite(str(prefix) + "_mask.png", mask_bgr)

            # 5. Overlay: raw frame + mask outline + body part dots + labels
            overlay = frame.copy()
            contours, _ = cv2.findContours(
                fg.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)

            regions = labeller.label(fg)
            for r in regions:
                col = _PART_COLOURS.get(r.label, (200, 200, 200))
                cx, cy = int(r.cx), int(r.cy)
                cv2.circle(overlay, (cx, cy), 6, col, -1)
                cv2.circle(overlay, (cx, cy), 7, (0, 0, 0), 1)
                cv2.putText(overlay, r.label, (cx + 8, cy - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                            col, 1, cv2.LINE_AA)
            cv2.imwrite(str(prefix) + "_overlay.png", overlay)

            # 6. 4-up composite: raw | enhanced | diff | overlay
            H, W = frame.shape[:2]
            th   = max(1, min(H, W) // 20)   # text height reference
            row1 = np.hstack([frame, enh_bgr])
            row2 = np.hstack([diff_colour, overlay])
            composite = np.vstack([row1, row2])
            # Labels
            for text, (tx, ty) in [
                ("RAW",      (4,  16)),
                ("ENHANCED", (W + 4, 16)),
                ("DIFF",     (4,  H + 16)),
                ("OVERLAY",  (W + 4, H + 16)),
            ]:
                cv2.putText(composite, text, (tx, ty),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imwrite(str(prefix) + "_composite.png", composite)

    n_written = len(frame_indices) * 2 * 6
    print(f"  [diag] {len(frame_indices)} frames × 2 cameras "
          f"× 6 images = {n_written} images")
    print(f"  [diag] frames → {out_dir / 'frames'}")
