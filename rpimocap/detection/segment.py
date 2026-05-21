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

# Centroid-only mode: just the largest blob centre, labelled "animal"
CENTROID_ONLY = False  # flipped per-run by --centroid-only flag


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
    label_map   : (H, W) int32 connected-component label image.
                  Pixel value == blob index into ``blobs`` + 1
                  (0 = background).  Used by hull_centroid() to extract
                  the pixels of a specific blob after epipolar selection.
    """

    mask:       np.ndarray
    blobs:      list
    frame_gray: np.ndarray
    n_blobs:    int
    label_map:  Optional[np.ndarray] = field(default=None, repr=False)


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
        if len(caps0) != len(caps1):
            raise ValueError(
                f"caps0 and caps1 must have same length "
                f"(got {len(caps0)} and {len(caps1)})")

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

    # ------------------------------------------------------------------ #
    #  Temporal adaptation                                                #
    # ------------------------------------------------------------------ #

    def update(
        self,
        frame0:        np.ndarray,
        frame1:        np.ndarray,
        mask0:         "np.ndarray | None" = None,
        mask1:         "np.ndarray | None" = None,
        alpha:         float = 0.995,
        bayer_pattern: str = "RGGB",
    ) -> None:
        """Adapt the background model toward a new stereo frame pair.

        Performs an exponential moving average update **only at pixels
        where the foreground mask is zero**, so disturbances that the
        animal made earlier in the recording (bedding kicked aside,
        objects nudged) gradually become part of the new background
        while the animal itself is excluded.

        Parameters
        ----------
        frame0, frame1 : raw camera frames (BGR or grayscale, uint8).
        mask0, mask1   : optional (H, W) boolean foreground masks. True
                         means "animal pixel — do not update here". When
                         None, every pixel is updated (i.e. uniform EMA).
        alpha          : EMA weight on the existing background, in [0, 1].
                         Higher = slower adaptation. At 25 fps:
                             alpha=0.995 → ~8 s memory (200 frames)
                             alpha=0.999 → ~40 s memory
                             alpha=0.99  → ~4 s memory
        bayer_pattern  : retained for API compatibility; ignored.

        Notes
        -----
        Call this once per frame (or every Nth frame) inside the
        tracking loop. The foreground mask should be the epipolar-
        validated animal mask, not the raw bg-subtract output, to avoid
        baking noise into the background.
        """
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha must be in [0, 1]; got {alpha}")

        g0 = self._to_gray(frame0).astype(np.float32)
        g1 = self._to_gray(frame1).astype(np.float32)
        if g0.shape != self.bg0.shape or g1.shape != self.bg1.shape:
            raise ValueError(
                f"frame shapes {g0.shape}/{g1.shape} differ from "
                f"background {self.bg0.shape}/{self.bg1.shape}")

        if mask0 is None:
            self.bg0 = alpha * self.bg0 + (1.0 - alpha) * g0
        else:
            m0 = ~np.asarray(mask0, dtype=bool)   # update where NOT animal
            self.bg0[m0] = alpha * self.bg0[m0] + (1.0 - alpha) * g0[m0]

        if mask1 is None:
            self.bg1 = alpha * self.bg1 + (1.0 - alpha) * g1
        else:
            m1 = ~np.asarray(mask1, dtype=bool)
            self.bg1[m1] = alpha * self.bg1[m1] + (1.0 - alpha) * g1[m1]

    # ------------------------------------------------------------------ #

    @staticmethod
    def _to_gray(frame: np.ndarray) -> np.ndarray:
        """Convert a BGR frame to uint8 grayscale."""
        if frame.ndim == 2:
            return frame.astype(np.uint8)
        return cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_BGR2GRAY)


# --------------------------------------------------------------------------- #
#  Foreground detector                                                         #
# --------------------------------------------------------------------------- #

def arena_roi_mask(P: np.ndarray,
                   arena_pts: np.ndarray,
                   image_shape: tuple,
                   pad_px: int = 20) -> np.ndarray:
    """Compute a filled convex-hull mask of the arena for one camera.

    Projects the 8 known 3D arena corners through the DLT projection matrix
    ``P`` (3×4) and fills the convex hull of the resulting 2D points.
    Pixels outside the hull are set to 0; pixels inside are 255.

    Parameters
    ----------
    P           : 3×4 DLT camera matrix (maps arena mm → pixel homogeneous).
    arena_pts   : (N, 3) array of 3D corner positions in arena mm coordinates.
    image_shape : (height, width[, channels]) — output mask has this shape.
    pad_px      : expand the hull outward by this many pixels to avoid clipping
                  the animal when it presses against the arena wall.

    Returns
    -------
    mask : uint8 ndarray, shape (height, width), values 0 or 255.
    """
    h, w = image_shape[:2]
    px_pts = []
    for pt in arena_pts:
        Xh = np.append(pt, 1.0)
        p  = P @ Xh
        u, v = p[0] / p[2], p[1] / p[2]
        px_pts.append([u, v])

    pts  = np.array(px_pts, dtype=np.float32)
    hull = cv2.convexHull(pts).reshape(-1, 1, 2)

    # Expand hull outward by pad_px using the centroid
    centroid = pts.mean(axis=0)
    dirs     = hull[:, 0, :] - centroid          # vectors from centroid to hull
    norms    = np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-6
    hull_pad = hull + (dirs / norms * pad_px).reshape(-1, 1, 2)
    hull_pad = hull_pad.astype(np.int32)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [hull_pad], 255)
    return mask


def arena_wall_weight(
        P: np.ndarray,
        arena_pts: np.ndarray,
        image_shape: tuple,
        decay_px: float = 80.0,
) -> np.ndarray:
    """Per-pixel weight map that decreases toward the projected arena walls.

    Uses ``cv2.distanceTransform`` on the arena convex-hull mask to compute
    each pixel's Euclidean distance to the nearest wall edge, then maps that
    distance through an exponential decay:

        weight(x,y) = 1 - exp(-dist(x,y) / decay_px)

    Pixels at the wall get weight ≈ 0; pixels at the arena centre get
    weight ≈ 1.  Multiplying the foreground diff by this map before
    thresholding means wall-adjacent pixels (plexiglas reflections, edge
    artefacts) need a proportionally larger intensity change to be detected
    as foreground.

    Parameters
    ----------
    P          : 3×4 DLT camera matrix.
    arena_pts  : (N, 3) 3D corner positions in arena mm coordinates.
    image_shape: (height, width[, channels]).
    decay_px   : Distance (px) at which the weight reaches ~0.63.
                 Smaller = sharper drop-off at the walls.

    Returns
    -------
    weight : float32 ndarray in [0, 1], shape (height, width).
    """
    # Build the binary arena mask (0/255)
    mask = arena_roi_mask(P, arena_pts, image_shape, pad_px=0)

    # Distance transform: for every *inside* pixel, distance to nearest
    # background pixel (= nearest wall edge).
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)

    # Soft exponential decay so there is no hard edge between high/low weight
    weight = 1.0 - np.exp(-dist / max(decay_px, 1.0))
    return weight.astype(np.float32)


def gabor_bedding_energy(
        image: np.ndarray,
        lambdas: tuple = (8, 12, 16),
        n_orientations: int = 4,
        sigma_factor: float = 1.5,
        ksize: int = 31,
) -> np.ndarray:
    """Compute the per-pixel Gabor energy at bedding-characteristic spatial scales.

    Runs a bank of Gabor filters at several wavelengths (spatial frequencies)
    and orientations, then returns the max response across the full bank.
    This gives a per-pixel measure of how much "bedding-like texture" is
    present at each location.

    Parameters
    ----------
    image         : uint8 or float32 single-channel image.
    lambdas       : Gabor wavelengths in pixels.  Choose to match the bedding
                    fibre scale — for typical NIR arena images with IMX477 at
                    2-4m distance, fibres appear at roughly 8–16 px.
    n_orientations: number of evenly-spaced orientations (default 4 → 0°,
                    45°, 90°, 135°).  Bedding is isotropic so 4 is enough.
    sigma_factor  : sigma = sigma_factor × lambda (controls bandwidth).
    ksize         : kernel size (must be odd; larger = better low-freq resp).

    Returns
    -------
    energy : float32 ndarray, same shape as ``image``, values ≥ 0.
             High values indicate bedding-like texture.
    """
    if image.dtype != np.float32:
        img = image.astype(np.float32)
    else:
        img = image

    energy = np.zeros(img.shape[:2], dtype=np.float32)
    thetas = [np.pi * i / n_orientations for i in range(n_orientations)]

    for lam in lambdas:
        sigma = sigma_factor * lam
        for theta in thetas:
            # Real and imaginary Gabor kernels (quadrature pair)
            kr = cv2.getGaborKernel(
                (ksize, ksize), sigma=sigma, theta=theta,
                lambd=float(lam), gamma=0.5, psi=0.0, ktype=cv2.CV_32F)
            ki = cv2.getGaborKernel(
                (ksize, ksize), sigma=sigma, theta=theta,
                lambd=float(lam), gamma=0.5, psi=np.pi / 2, ktype=cv2.CV_32F)
            # Zero-mean the kernels: OpenCV getGaborKernel has non-zero DC
            # component which causes the filter to respond to mean intensity
            # (making a flat 128-valued patch look textured). Subtracting the
            # mean confines the response to local texture only.
            kr -= kr.mean()
            ki -= ki.mean()

            resp_r = cv2.filter2D(img, cv2.CV_32F, kr)
            resp_i = cv2.filter2D(img, cv2.CV_32F, ki)
            amp    = np.sqrt(resp_r ** 2 + resp_i ** 2)
            np.maximum(energy, amp, out=energy)

    return energy


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
    min_area_px       : discard blobs smaller than this (noise filter).
                        Default 500 px².  Raise to ~1500 to eliminate
                        plexiglas wall reflections (typically < 500 px²).
    max_area_px       : discard blobs larger than this.  Use to reject
                        large bedding-activation artefacts when the whole
                        floor lights up.  None = no upper limit.
    morph_k           : morphological kernel size for opening/closing
    blur_k            : Gaussian blur kernel (0 = skip; overridden by bilateral)
    clahe             : apply CLAHE before background subtraction
    clahe_clip        : CLAHE clip limit (higher = more contrast, more noise)
    clahe_tile        : CLAHE tile grid size (smaller = more local)
    use_green_channel : extract green Bayer channel instead of luminance
    bilateral         : apply bilateral filter instead of Gaussian blur
    bilateral_d       : bilateral filter diameter (neighbourhood size)
    bilateral_sigma   : bilateral sigma for colour and spatial
    roi_mask          : uint8 binary mask (255 = valid, 0 = ignore).
                        Applied to the thresholded diff before connected-
                        components.  Use ``arena_roi_mask()`` to compute
                        one automatically from the DLT projection matrices.
                        Eliminates the arena frame, cables, and everything
                        outside the physical arena interior.
    texture_suppress  : Enable Gabor-based bedding texture suppression.
                        Pre-computes the Gabor energy of the background at
                        bedding-characteristic spatial scales; in each frame
                        pixels whose foreground diff is high BUT whose
                        current Gabor energy is also high (bedding-like
                        texture → disturbed bedding) are weighted down before
                        thresholding.  This discriminates the animal body
                        (smooth dark blob → low Gabor energy) from displaced
                        bedding (same fibrous texture → high Gabor energy).
    texture_lambdas   : Gabor wavelengths (px) targeting the bedding scale.
    texture_alpha     : Suppression strength [0–1].  0 = off, 1 = full.
                        At 1.0 a pixel with maximum bedding-like Gabor energy
                        has its diff value halved; the suppression is
                        multiplicative so it never zeroes out a diff entirely.
    texture_n_orient  : Orientations in Gabor bank (4 for isotropic bedding).
    """

    def __init__(
        self,
        background:        BackgroundModel,
        threshold:         float = 25.0,
        min_area_px:       int   = 500,
        max_area_px:       Optional[int] = None,
        min_solidity:      float = 0.0,
        morph_k:           int   = 7,
        blur_k:            int   = 5,
        clahe:             bool  = False,
        clahe_clip:        float = 2.0,
        clahe_tile:        int   = 8,
        use_green_channel: bool  = False,
        bilateral:         bool  = False,
        bilateral_d:       int   = 9,
        bilateral_sigma:   float = 50.0,
        roi_mask:          Optional[np.ndarray] = None,
        wall_weight:       Optional[np.ndarray] = None,
        texture_suppress:  bool  = False,
        texture_lambdas:   tuple = (8, 12, 16),
        texture_alpha:     float = 0.7,
        texture_n_orient:  int   = 4,
    ):
        self.bg                = background
        self.threshold         = threshold
        self.min_area_px       = min_area_px
        self.max_area_px       = max_area_px
        self.min_solidity      = min_solidity
        self.use_green_channel = use_green_channel
        self.bilateral         = bilateral
        self.bilateral_d       = bilateral_d
        self.bilateral_sigma   = bilateral_sigma
        self._kernel           = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (morph_k, morph_k))
        self._blur_k = blur_k | 1
        # Store as per-camera dict so one detector can serve both cams.
        # roi_mask (cam-0) is passed for backward compat.
        self._roi_masks:   dict = {0: roi_mask, 1: None}
        self._wall_weights: dict = {0: wall_weight, 1: None}

        # Gabor bedding-texture suppression
        # Pre-compute the background's Gabor energy once at init time.
        # This is the per-pixel "how much bedding texture exists here in
        # the background" map, used as a suppression weight in detect().
        self._texture_alpha = texture_alpha if texture_suppress else 0.0
        if texture_suppress:
            bg0_gray = background.bg0
            if bg0_gray.ndim == 3:
                bg0_gray = cv2.cvtColor(
                    bg0_gray.astype(np.uint8), cv2.COLOR_BGR2GRAY
                ).astype(np.float32)
            bg1_gray = background.bg1
            if bg1_gray.ndim == 3:
                bg1_gray = cv2.cvtColor(
                    bg1_gray.astype(np.uint8), cv2.COLOR_BGR2GRAY
                ).astype(np.float32)
            bg0_e = gabor_bedding_energy(bg0_gray, lambdas=texture_lambdas,
                                         n_orientations=texture_n_orient)
            bg1_e = gabor_bedding_energy(bg1_gray, lambdas=texture_lambdas,
                                         n_orientations=texture_n_orient)
            # Normalise to [0, 1] using the 99th percentile so bright
            # LED reflections don't collapse the rest of the range.
            self._bg_gabor: dict = {
                0: bg0_e / (np.percentile(bg0_e, 99) + 1e-6),
                1: bg1_e / (np.percentile(bg1_e, 99) + 1e-6),
            }
        else:
            self._bg_gabor = {}

        # CLAHE instance (shared across frames for efficiency)
        self._clahe = (cv2.createCLAHE(
                           clipLimit=clahe_clip,
                           tileGridSize=(clahe_tile, clahe_tile))
                       if clahe else None)

    # ------------------------------------------------------------------ #

    def set_roi_mask(self, cam: int, mask: "Optional[np.ndarray]") -> None:
        """Set or replace the ROI mask for one camera (0 or 1)."""
        self._roi_masks[cam] = mask

    def set_wall_weight(self, cam: int,
                        weight: "Optional[np.ndarray]") -> None:
        """Set or replace the wall distance weight map for one camera."""
        self._wall_weights[cam] = weight

    # ------------------------------------------------------------------ #

    def _to_channel_gray(self, frame: np.ndarray) -> np.ndarray:
        """Extract the working channel (green or luminance). No CLAHE."""
        if self.use_green_channel and frame.ndim == 3 and frame.shape[2] == 3:
            return frame[:, :, 1].copy().astype(np.uint8)
        return BackgroundModel._to_gray(frame)

    def _to_enhanced_gray(self, frame: np.ndarray) -> np.ndarray:
        """Convert frame to enhanced grayscale (channel + smooth only, no CLAHE).

        CLAHE is applied to the DIFF image in detect(), not here, so that
        it amplifies the animal signal rather than amplifying bedding texture.
        """
        gray = self._to_channel_gray(frame)

        # Smooth before subtraction
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
        # ── Channel extraction (same for frame and background) ───────────
        gray   = self._to_enhanced_gray(frame).astype(np.float32)
        bg_raw = self.bg.bg0 if cam == 0 else self.bg.bg1
        bg     = bg_raw.astype(np.float32)

        if bg.shape != gray.shape:
            bg = cv2.resize(bg, (gray.shape[1], gray.shape[0]),
                            interpolation=cv2.INTER_LINEAR)

        # ── Background uses same channel extraction ───────────────────
        # Re-extract the working channel from the background for
        # consistency (bg_raw is luminance; if green_channel is set
        # all channels are equal so this is a no-op).
        if self.use_green_channel:
            # bg_raw is already single-channel gray — use as-is
            pass

        # ── Diff ─────────────────────────────────────────────────────
        diff = np.abs(gray - bg)

        # ── CLAHE on the DIFF (not the frame) ────────────────────────
        # Applying CLAHE here amplifies the animal blob (high diff signal)
        # relative to the background noise (low diff signal), rather than
        # amplifying bedding texture in the frame before subtraction.
        if self._clahe is not None:
            diff_u8 = np.clip(diff, 0, 255).astype(np.uint8)
            diff    = self._clahe.apply(diff_u8).astype(np.float32)

        # ── Wall distance down-weighting ────────────────────────────────
        # Pixels near the projected arena walls are softly attenuated in
        # the diff image before thresholding.  This means plexiglas
        # reflections and edge artefacts need a proportionally larger
        # intensity change to be flagged as foreground, without creating
        # a hard binary cutoff at the wall.
        _ww = self._wall_weights.get(cam)
        if _ww is not None:
            diff = diff * _ww

        # ── Gabor texture suppression ─────────────────────────────────
        # Where the current frame has high Gabor energy at bedding scales
        # (fibrous texture → probably disturbed bedding, not animal body)
        # the diff is attenuated before thresholding.
        # The animal body is a smooth dark blob → low Gabor energy → NOT
        # suppressed.  Disturbed bedding keeps its fibrous texture → high
        # Gabor energy → suppressed even if the pixel value changed.
        if self._texture_alpha > 0 and cam in self._bg_gabor:
            frame_gabor  = gabor_bedding_energy(gray.astype(np.float32))
            frame_gabor_n = frame_gabor / (np.percentile(frame_gabor, 99) + 1e-6)
            # Use the minimum of frame and background Gabor energy as the
            # suppression weight — this catches displaced bedding (frame
            # energy similar to bg energy) but not novel objects.
            suppress_w   = np.minimum(
                np.clip(frame_gabor_n, 0, 1),
                np.clip(self._bg_gabor[cam], 0, 1))
            diff = diff * (1.0 - self._texture_alpha * suppress_w)

        binary = (diff > self.threshold).astype(np.uint8) * 255
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  self._kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, self._kernel)

        # Arena ROI mask — zero out everything outside the projected arena
        # convex hull.  Eliminates the frame, cables, LED reflections, and
        # any foreground outside the physical recording volume.
        _mask = self._roi_masks.get(cam)
        if _mask is not None:
            binary = cv2.bitwise_and(binary, _mask)

        n, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=8)

        blobs = []
        for i in range(1, n):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < self.min_area_px:
                continue
            if self.max_area_px is not None and area > self.max_area_px:
                continue
            # Solidity filter: reject blobs that are too non-convex
            # (e.g. rat body + cable creates a blob with large hull
            # area relative to the actual pixel area).
            if self.min_solidity > 0:
                blob_mask = (labels == i).astype(np.uint8) * 255
                cnts, _ = cv2.findContours(
                    blob_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if cnts:
                    hull_area = cv2.contourArea(cv2.convexHull(cnts[0]))
                    solidity  = area / (hull_area + 1e-6)
                    if solidity < self.min_solidity:
                        continue
            blobs.append((stats[i], centroids[i]))

        return ForegroundResult(
            mask=binary,
            blobs=blobs,
            frame_gray=gray.astype(np.uint8),
            n_blobs=len(blobs),
            label_map=labels)

    def hull_centroid(
            self,
            result: ForegroundResult,
            cx: float,
            cy: float,
            cable_erosion_px: int = 0,
    ) -> tuple[float, float]:
        """Refine a centroid, optionally stripping an attached cable first.

        Pipeline
        --------
        1. Look up which connected-component label owns (cx, cy).
        2. Extract that blob as a binary mask.
        3. **Cable erosion** (if cable_erosion_px > 0):
               Erode the blob with an ellipse kernel of radius
               cable_erosion_px.  A headstage cable is typically 3–8 px
               wide; the rat body is 60–120 px wide.  Choosing a radius
               between those two values disconnects the cable while
               leaving the body intact.  The largest remaining connected
               component after erosion is the body.
        4. Fit an ellipse to the (possibly eroded) body pixels.
               The ellipse centre sits at the geometric body centre
               regardless of cable direction.
        5. Fall back to convex-hull centroid, then pixel mean.

        Parameters
        ----------
        result           : ForegroundResult from detect().
        cx, cy           : Connected-component centroid of the selected blob.
        cable_erosion_px : Erosion radius in pixels (0 = skip step 3).
                           Good starting point: half the visible cable
                           width in pixels.  Typically 8–15 px for a
                           chronic headstage cable at 1–2 m camera distance.

        Returns
        -------
        (cx_refined, cy_refined) — falls back to (cx, cy) on any failure.
        """
        if result.label_map is None:
            return cx, cy

        # ── Step 1: find the blob label ───────────────────────────────
        ix, iy = int(round(cx)), int(round(cy))
        h, w   = result.label_map.shape
        ix     = max(0, min(w - 1, ix))
        iy     = max(0, min(h - 1, iy))
        lbl    = result.label_map[iy, ix]
        if lbl == 0:
            ys, xs = np.where(result.label_map > 0)
            if len(xs) == 0:
                return cx, cy
            dists   = (xs - ix) ** 2 + (ys - iy) ** 2
            lbl     = result.label_map[ys[np.argmin(dists)], xs[np.argmin(dists)]]

        ys, xs = np.where(result.label_map == lbl)
        if len(xs) < 3:
            return cx, cy

        # ── Step 2: build binary blob mask ────────────────────────────
        blob_mask = (result.label_map == lbl).astype(np.uint8) * 255

        # ── Step 3: cable erosion ─────────────────────────────────────
        if cable_erosion_px > 0:
            r   = int(cable_erosion_px)
            k   = cv2.getStructuringElement(
                      cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
            eroded = cv2.erode(blob_mask, k, iterations=1)

            # Largest connected component of the eroded mask = body
            n_e, lbl_e, stats_e, cent_e = cv2.connectedComponentsWithStats(
                eroded, connectivity=8)
            if n_e > 1:
                # component 0 is background; pick largest foreground one
                body_lbl = 1 + np.argmax(
                    stats_e[1:, cv2.CC_STAT_AREA])
                ys, xs = np.where(lbl_e == body_lbl)
                if len(xs) < 3:
                    # erosion was too aggressive — fall back to full blob
                    ys, xs = np.where(blob_mask > 0)

        # ── Step 4: ellipse-fit centroid ──────────────────────────────
        pts = np.column_stack([xs, ys]).astype(np.float32)
        if len(pts) >= 5:
            try:
                ellipse = cv2.fitEllipse(pts)
                return float(ellipse[0][0]), float(ellipse[0][1])
            except cv2.error:
                pass

        # ── Step 5: hull centroid fallback ────────────────────────────
        hull = cv2.convexHull(pts)
        M    = cv2.moments(hull)
        if M['m00'] < 1e-6:
            return float(pts[:, 0].mean()), float(pts[:, 1].mean())
        return M['m10'] / M['m00'], M['m01'] / M['m00']

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
        centroid_only:           bool  = False,
    ):
        self._ear_var_pct    = ear_variance_percentile
        self._min_ear_area   = min_ear_area_px
        self._centroid_only  = centroid_only

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

        # Centroid-only mode: return ALL blobs sorted by area (largest first),
        # all labeled "animal".
        #
        # Returning all candidates (not just the largest) lets the epipolar
        # matcher use Pass 2 to find the blob in cam0 that is epipolar-
        # consistent with a blob in cam1.  This handles the common case where
        # the largest blob in cam0 is a non-animal foreground object (e.g. an
        # experimenter's hand that shifted position since the background was
        # built) while the actual rat is a smaller blob that nonetheless has
        # a valid epipolar match in cam1.
        if self._centroid_only:
            # Sort blobs largest-first so the epipolar matcher tries them in
            # size order and prefers larger objects when epipolar distance is tied.
            sorted_blobs = sorted(
                fg.blobs,
                key=lambda b: b[0][cv2.CC_STAT_AREA],
                reverse=True)
            return [
                BodyRegion(
                    label="animal",
                    cx=float(centroid[0]),
                    cy=float(centroid[1]),
                    area_px=float(stats[cv2.CC_STAT_AREA]),
                    confidence=1.0)
                for stats, centroid in sorted_blobs
            ]

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
        undistort_match: bool  = True,
    ):
        self.P0 = P0; self.P1 = P1
        self.K0 = K0; self.K1 = K1
        self.dist0 = dist0; self.dist1 = dist1
        self.R = R; self.T = T
        self.max_epipolar_px = max_epipolar_px
        self.undistort_pts   = undistort_pts
        self.undistort_match = undistort_match
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
        matcher = cls(P0=P0, P1=P1, K0=K0, K1=K1,
                      dist0=dist0, dist1=dist1, R=R, T=T,
                      max_epipolar_px=max_epipolar_px)
        # When DLT P matrices are in use, recompute F from them directly.
        # F from K/R/T of the original autocalib is inconsistent with DLT
        # P matrices and causes every epipolar match to fail.
        if "dlt_P0" in cal.files:
            matcher.F = cls._compute_F_from_P(P0, P1)
        return matcher

    @staticmethod
    def _compute_F(K0, K1, R, T) -> np.ndarray:
        """Compute the fundamental matrix F from stereo calibration.

        F = K1^{-T} · [t]× · R · K0^{-1}
        """
        t = T.ravel()
        tx = np.array([[ 0,    -t[2],  t[1]],
                       [ t[2],  0,    -t[0]],
                       [-t[1],  t[0],  0   ]], dtype=np.float64)
        E  = tx @ R.astype(np.float64)
        F  = np.linalg.inv(K1).T @ E @ np.linalg.inv(K0)
        return F / (np.abs(F).max() + 1e-12)

    @staticmethod
    def _compute_F_from_P(P0: np.ndarray, P1: np.ndarray) -> np.ndarray:
        """Compute fundamental matrix F directly from two 3×4 projection matrices.

        Uses the formula:  F = [e']× P1 P0+
        where P0+ is the pseudoinverse of P0 and e' = P1 @ null(P0)
        is the epipole (projection of cam0 centre into cam1).

        This is the correct F when P0/P1 are DLT-estimated projection
        matrices not decomposable into a standard K[R|t] stereo pair.
        """
        # Camera 0 centre: null space of P0
        _, _, Vt = np.linalg.svd(P0)
        C0 = Vt[-1]                         # homogeneous camera centre
        C0 = C0 / C0[3]                     # normalise

        # Epipole in cam1: projection of cam0 centre
        e1 = P1 @ C0                        # (3,) homogeneous
        e1 = e1 / (np.abs(e1).max() + 1e-12)

        # Skew-symmetric matrix of e1
        e1x = np.array([[ 0,    -e1[2],  e1[1]],
                        [ e1[2], 0,     -e1[0]],
                        [-e1[1], e1[0],  0    ]], dtype=np.float64)

        P0_pinv = np.linalg.pinv(P0)
        F = e1x @ P1 @ P0_pinv
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
        prior0:   "tuple[float, float] | None" = None,
        prior1:   "tuple[float, float] | None" = None,
        prior_lambda: float = 0.05,
    ) -> list[tuple[BodyRegion, BodyRegion]]:
        """Return matched (cam0_region, cam1_region) pairs.

        Matching is label-first: if both cameras have a region with the
        same anatomical label, they are matched directly.  Unmatched
        regions fall back to epipolar-nearest matching.

        Parameters
        ----------
        regions0, regions1 : candidate body regions in each camera.
        prior0, prior1     : optional (x, y) pixel centroid of the
                              previous confirmed detection in each
                              camera. When supplied, the global
                              epipolar-nearest score becomes

                                  d_eff = d_epi + lambda * (d0 + d1)

                              where d0/d1 are the pixel distances from
                              each candidate to its prior. This keeps
                              the tracker from jumping to a far-away
                              blob with coincidentally low epipolar
                              distance (e.g. a wall reflection on the
                              opposite side of the arena).
        prior_lambda       : strength of the spatial-prior term
                              (px-epipolar per px-spatial). Default 0.05
                              means a 100 px jump from prior costs the
                              same as 5 px of epipolar error.
        """
        if not regions0 or not regions1:
            return []

        used1: set[int] = set()
        matched: list[tuple[BodyRegion, BodyRegion]] = []

        # Pre-compute undistorted centroids for every candidate so that
        # the epipolar geometry (F is exact only in undistorted coords)
        # is correct.  With ~k1 ≈ -0.24 lenses, matching on raw pixel
        # coordinates near the image periphery absorbs several pixels of
        # distortion error into the max_epipolar_px tolerance.
        if self.undistort_match:
            cx0_arr = np.array([[r.cx, r.cy] for r in regions0])
            cx1_arr = np.array([[r.cx, r.cy] for r in regions1])
            ud0 = self._undistort(cx0_arr, 0)
            ud1 = self._undistort(cx1_arr, 1)
            ud_prior0 = (None if prior0 is None
                         else tuple(self._undistort(
                             np.array([prior0]), 0)[0]))
            ud_prior1 = (None if prior1 is None
                         else tuple(self._undistort(
                             np.array([prior1]), 1)[0]))
        else:
            ud0 = np.array([[r.cx, r.cy] for r in regions0])
            ud1 = np.array([[r.cx, r.cy] for r in regions1])
            ud_prior0 = prior0
            ud_prior1 = prior1

        # Determine if we're in centroid-only mode (all labels identical)
        all_same_label = (
            len(regions0) > 0 and len(regions1) > 0
            and len({r.label for r in regions0}) == 1
            and len({r.label for r in regions1}) == 1
            and next(iter({r.label for r in regions0})) ==
                next(iter({r.label for r in regions1})))

        if all_same_label and (len(regions0) > 1 or len(regions1) > 1):
            # Centroid-only mode: multiple candidates with the same label.
            # Skip label matching entirely and do global epipolar-nearest:
            # find the single best (cam0_blob, cam1_blob) pair across all
            # candidates.  This selects the animal blob (which satisfies the
            # epipolar constraint) over spurious blobs (hand, frame noise)
            # that have no consistent match in the other camera.
            best_d_global = float("inf")
            best_pair = None
            for i0, r0 in enumerate(regions0):
                line = self._epipolar_line(ud0[i0, 0], ud0[i0, 1])
                for i1, r1 in enumerate(regions1):
                    d_epi = self._point_to_line_dist(
                        line, ud1[i1, 0], ud1[i1, 1])
                    if d_epi > self.max_epipolar_px:
                        continue
                    # Optional trajectory prior — penalise candidates far
                    # from the previous confirmed detection in each camera.
                    d_prior = 0.0
                    if ud_prior0 is not None:
                        d_prior += float(np.hypot(
                            ud0[i0, 0] - ud_prior0[0],
                            ud0[i0, 1] - ud_prior0[1]))
                    if ud_prior1 is not None:
                        d_prior += float(np.hypot(
                            ud1[i1, 0] - ud_prior1[0],
                            ud1[i1, 1] - ud_prior1[1]))
                    d_eff = d_epi + prior_lambda * d_prior
                    if d_eff < best_d_global:
                        best_d_global = d_eff
                        best_pair = (r0, r1)
            if best_pair is not None:
                matched.append(best_pair)
            return matched

        # Pass 1: label-based matching WITH epipolar validation.
        # Even when labels agree (e.g. body-part labels in both cameras),
        # we verify epipolar consistency before accepting the match.
        # This prevents mismatched blobs from producing garbage triangulations.
        by_label1 = {r.label: (i, r) for i, r in enumerate(regions1)}
        remaining0 = []
        for i0, r0 in enumerate(regions0):
            if r0.label in by_label1:
                i1, r1 = by_label1[r0.label]
                line = self._epipolar_line(ud0[i0, 0], ud0[i0, 1])
                d    = self._point_to_line_dist(
                    line, ud1[i1, 0], ud1[i1, 1])
                if d <= self.max_epipolar_px:
                    matched.append((r0, r1))
                    used1.add(i1)
                else:
                    remaining0.append((i0, r0))
            else:
                remaining0.append((i0, r0))

        # Pass 2: epipolar nearest for unmatched or epipolar-rejected regions
        for i0, r0 in remaining0:
            line = self._epipolar_line(ud0[i0, 0], ud0[i0, 1])
            best_d, best_i, best_r1 = float("inf"), -1, None
            for i1, r1 in enumerate(regions1):
                if i1 in used1:
                    continue
                d = self._point_to_line_dist(
                    line, ud1[i1, 0], ud1[i1, 1])
                if d < best_d:
                    best_d, best_i, best_r1 = d, i1, r1
            if best_r1 is not None and best_d <= self.max_epipolar_px:
                matched.append((r0, best_r1))
                used1.add(best_i)

        return matched

    def triangulate(
        self,
        matches: list[tuple[BodyRegion, BodyRegion]],
        bounds: "np.ndarray | None" = None,
    ) -> dict[str, np.ndarray]:
        """Triangulate matched region pairs → {label: xyz_mm}.

        Parameters
        ----------
        matches : list of (cam0_region, cam1_region) pairs
        bounds  : optional (6,) array [xmin,xmax,ymin,ymax,zmin,zmax].
                  Points outside bounds + 20% margin are set to NaN
                  (epipolar mismatch rejection).

        Returns
        -------
        dict mapping label string to (3,) world coordinate array (mm).
        Out-of-bounds points are NaN — gap-filling handles them downstream.
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
                                  (pts1[0,0], pts1[0,1]))[:3]
            # Reject points outside arena bounds (epipolar mismatch).
            # 20% margin allows genuine near-boundary detections through.
            if bounds is not None:
                xmin,xmax,ymin,ymax,zmin,zmax = bounds
                dx = (xmax-xmin)*0.20; dy = (ymax-ymin)*0.20
                dz = (zmax-zmin)*0.20
                if not (xmin-dx <= X[0] <= xmax+dx and
                        ymin-dy <= X[1] <= ymax+dy and
                        zmin-dz <= X[2] <= zmax+dz):
                    X = np.full(3, np.nan)
            # Always use cam0 label as canonical name.
            label = r0.label
            result[label] = X
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
            # Apply same pipeline as detect(): CLAHE on diff, not frame
            raw_diff = np.abs(enhanced.astype(np.float32) - bg_arr.astype(np.float32))
            if detector._clahe is not None:
                diff = detector._clahe.apply(
                    np.clip(raw_diff, 0, 255).astype(np.uint8)).astype(np.float32)
            else:
                diff = raw_diff
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
