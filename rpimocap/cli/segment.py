"""
rpimocap-segment — texture-based animal segmentation and 3D tracking
=====================================================================
Detects and tracks body regions (nose, ears, back, tail etc.) through
a stereo TIFF recording without manual annotation per frame.

Pipeline
--------
1. Background model  — median over N evenly-spaced frames
2. Foreground mask   — per-frame background subtraction + morphology
3. Body part labels  — geometric spine-axis analysis (no ML required)
                       or SAM/SAM2 if weights are supplied
4. Stereo matching   — epipolar-constrained region matching
5. Triangulation     — DLT stereo triangulation → 3D coordinates (mm)
6. Temporal output   — HDF5 + viewer JSON in the same format as
                       rpimocap-run, directly compatible with downstream tools

Output
------
    <session>/
    ├── background/
    │   ├── bg.npz             ← background model (reuse with --background-model)
    │   ├── bg_cam0.png        ← background image (for inspection)
    │   └── bg_cam1.png
    └── tracking/
        ├── reconstruction.h5  ← 3D trajectories per body part
        ├── viewer_data.json   ← HTML 3D viewer
        └── detection_stats.csv

Usage
-----
    # Basic (no SAM, optical flow tracking)
    rpimocap-segment \\
        --cam0  cam0_raw.tif \\
        --cam1  cam1_raw.tif \\
        --calib autocalib_refined.npz \\
        --out   output/

    # With arena alignment
    rpimocap-segment \\
        --cam0  cam0_raw.tif --cam1 cam1_raw.tif \\
        --calib autocalib_refined.npz \\
        --align-points align_points.csv \\
        --bounds="-140,140,-215,215,0,388" \\
        --out   output/

    # With SAM2 (better part segmentation)
    rpimocap-segment \\
        --cam0  cam0_raw.tif --cam1 cam1_raw.tif \\
        --calib autocalib_refined.npz \\
        --sam2-checkpoint /path/to/sam2_hiera_large.pt \\
        --out   output/

    # Reuse a saved background model
    rpimocap-segment ... --background-model output/background/bg.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


def _parse_bounds(s: str) -> list[float]:
    return [float(v) for v in s.split(",")]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    # ── Input ────────────────────────────────────────────────────────────────
    io = ap.add_argument_group("Input / Output")
    io.add_argument("--cam0",   required=True,
                    help="Camera 0 video or TIFF stack")
    io.add_argument("--cam1",   required=True,
                    help="Camera 1 video or TIFF stack")
    io.add_argument("--calib",  required=True,
                    help="Calibration .npz (from rpimocap-calibrate or "
                         "rpimocap-autocalib)")
    io.add_argument("--out",    required=True,
                    help="Output directory")
    io.add_argument("--bayer-pattern", default="RGGB",
                    choices=["RGGB","BGGR","GRBG","GBRG"],
                    help="Bayer CFA pattern for raw TIFF stacks (default: RGGB)")

    # ── Background ───────────────────────────────────────────────────────────
    bg = ap.add_argument_group("Background model")
    bg.add_argument("--background-frames", type=int, default=200,
                    metavar="N",
                    help="Number of frames for background estimation "
                         "(default: 200)")
    bg.add_argument("--background-start", type=int, default=0,
                    help="First frame index to use for background "
                         "(default: 0 — use start of video)")
    bg.add_argument("--background-model", default=None, metavar="NPZ",
                    help="Load a pre-computed background model .npz instead "
                         "of computing from scratch")
    bg.add_argument("--background-method", default="median",
                    choices=["median","mean"],
                    help="Background estimation method (default: median)")
    bg.add_argument("--background-extra-cam0", nargs="+", default=[],
                    metavar="TIFF",
                    help="Additional cam0 TIFF files for background estimation. "
                         "Combining multiple sessions makes the median far more "
                         "robust when the animal is present throughout each recording.")
    bg.add_argument("--background-extra-cam1", nargs="+", default=[],
                    metavar="TIFF",
                    help="Additional cam1 TIFF files (must match --background-extra-cam0)")
    bg.add_argument("--flat-field-cam0", default=None, metavar="PATH",
                    help="Flat-field image (PNG / TIFF / NPZ) for cam0, "
                         "applied to every frame to correct NIR vignette. "
                         "Disabled by default — pass an explicit path to "
                         "enable. See rpimocap.detection.vignette.")
    bg.add_argument("--flat-field-cam1", default=None, metavar="PATH",
                    help="Flat-field image for cam1 (same semantics as "
                         "--flat-field-cam0).")
    bg.add_argument("--synthesize-flat-field", action="store_true",
                    help="When no explicit flat-field is supplied, fit a "
                         "smooth radial polynomial to the background model "
                         "and use that as a synthetic flat-field. Less "
                         "accurate than a true rig-calibration capture but "
                         "removes the bulk of the vignette bias.")

    # ── Detection ────────────────────────────────────────────────────────────
    det = ap.add_argument_group("Detection")
    det.add_argument("--threshold", type=float, default=25.0,
                     help="Foreground detection threshold 0-255 "
                          "(default: 25)")
    det.add_argument("--polarity", choices=["either", "bright", "dark"],
                     default="either",
                     help="Background-subtraction polarity. 'either' "
                          "(default) catches both brighter-than-bg and "
                          "darker-than-bg pixels — also catches the "
                          "animal's own cast shadow as foreground. "
                          "'bright' catches only pixels brighter than "
                          "the background → suppresses shadows entirely "
                          "(use for NIR captures where the animal is "
                          "bright fur on dark/textured bedding). "
                          "'dark' is the mirror case.")
    det.add_argument("--min-area", type=int, default=1500,
                    metavar="PX2",
                    help="Minimum blob area px² (default 1500). "
                         "Plexiglas wall reflections are typically "
                         "< 500 px²; rat body is 3000–20000 px².")
    det.add_argument("--max-area", type=int, default=None,
                    metavar="PX2",
                    help="Maximum blob area px² (default: unlimited). "
                         "Discard blobs larger than this — useful for "
                         "rejecting large bedding-activation artefacts.")
    det.add_argument("--cable-erosion", type=int, default=0,
                    metavar="PX",
                    help="Erode the selected rat blob by this radius "
                         "(px) before computing the centroid. "
                         "Disconnects a thin headstage cable "
                         "(typically 3–8 px wide) from the body "
                         "(60–120 px wide). The largest remaining "
                         "component after erosion is used as the body. "
                         "Good starting point: 10–15 px. "
                         "0 = disabled (default).")
    det.add_argument("--body-length", type=float, default=0.0, metavar="MM",
                    help="Expected rat body length nose→tail-base in mm "
                         "(typical 160–220 mm). Enables the anatomical "
                         "Gaussian prior in hull_centroid (step 5): builds "
                         "a rotated body-shaped weight map at the ellipse "
                         "centre, ANDs it with the eroded foreground blob, "
                         "and returns the weighted centroid of the "
                         "intersection. Suppresses outliers at the blob "
                         "boundary (cable tips, bedding-edge artefacts) "
                         "that lie outside the anatomically expected body. "
                         "Requires DLT P matrices in the calibration "
                         "(dlt_P0 / dlt_P1). 0 = disabled (default).")
    det.add_argument("--body-width", type=float, default=70.0, metavar="MM",
                    help="Expected rat body width at widest point in mm "
                         "(typical 55–85 mm). Used by --body-length.")
    det.add_argument("--body-z", type=float, default=0.0, metavar="MM",
                    help="Assumed body height above the arena floor in mm. "
                         "Use 0 for a floor-level centroid (default), "
                         "~50 for a body-centre-of-mass model.")
    det.add_argument("--gabor-refine", action="store_true", default=False,
                    help="Find the rat body outline using Canny edges on "
                         "the Gabor energy map. The body appears as a "
                         "low-energy hole (smooth fur vs fibrous bedding); "
                         "edges on that map trace the real outline in "
                         "texture space, independent of pixel intensity. "
                         "Slots between cable erosion and ellipse fit in "
                         "hull_centroid, and the Gabor-refined mask then "
                         "feeds into the anatomical Gaussian prior. "
                         "Requires a background Gabor model — pass "
                         "--texture-suppress when building the background "
                         "so the model is cached.")
    det.add_argument("--canny-low", type=float, default=30.0, metavar="T",
                    help="Lower Canny hysteresis on the Gabor energy map "
                         "(default 30). Lower = more edges detected; drop "
                         "to 20 if the body contour fragments.")
    det.add_argument("--canny-high", type=float, default=90.0, metavar="T",
                    help="Upper Canny hysteresis (default 90). Higher = "
                         "only strong edges kept; raise to 110 if the "
                         "contour over-segments into bedding fragments.")
    det.add_argument("--kalman-online", action="store_true", default=False,
                    help="Enable an ONLINE per-frame Kalman filter that "
                         "tracks the body XYZ during segmentation. Its "
                         "predicted position seeds the next frame's "
                         "epipolar prior (replacing --trajectory-prior's "
                         "raw last-centroid seed) so the selector is "
                         "biased by velocity, not just position, and "
                         "keeps producing predictions during gap frames. "
                         "Independent of the offline --kalman/--kalman-no-rts "
                         "smoother applied after tracking; use both for "
                         "best results.")
    det.add_argument("--rearing-detection", action="store_true", default=False,
                    help="Enable posture classification from the online "
                         "Kalman state. When the rat is reared (z > 100 mm "
                         "or vz > 200 mm/s, with hysteresis), the next "
                         "frame's hull_centroid receives the vertical-"
                         "posture anatomical prior (~90 × 45 mm) instead "
                         "of the horizontal one (~180 × 70 mm), so step 5 "
                         "doesn't pull the centroid toward a horizontal "
                         "body that isn't there. Requires --kalman-online.")
    det.add_argument("--rearing-z-enter", type=float, default=100.0,
                    metavar="MM",
                    help="Z threshold to enter the reared state (default "
                         "100 mm). Used with --rearing-detection.")
    det.add_argument("--rearing-z-exit", type=float, default=70.0,
                    metavar="MM",
                    help="Z threshold to leave the reared state (default "
                         "70 mm). Hysteresis prevents flicker at the "
                         "boundary.")
    det.add_argument("--sam2-video-checkpoint", type=str, default=None,
                    metavar="PATH",
                    help="Enable SAM2 video propagation for per-frame "
                         "masks. The given checkpoint (e.g. "
                         "sam2_hiera_large.pt) is loaded into a "
                         "SAM2VideoTracker; a pre-pass propagates the "
                         "rat mask across the whole session from a seed "
                         "centroid (taken from bg-subtraction on the "
                         "prompt frame). Per-frame masks are cached to "
                         "disk and consumed by track_sequence instead "
                         "of bg-subtraction. Requires the `sam2` package "
                         "and ~30-100 KB/frame disk space for the cache. "
                         "When unset (default) bg-subtraction is used.")
    det.add_argument("--sam2-video-config", type=str,
                    default="sam2.1_hiera_l.yaml", metavar="CFG",
                    help="SAM2 video config name (default sam2.1_hiera_l.yaml).")
    det.add_argument("--sam2-video-cache-dir", type=str, default=None,
                    metavar="DIR",
                    help="Where to stage the SAM2 mask cache (default: "
                         "<session>/tracking/sam2_masks/). Cache is "
                         "reused if it already exists for the session.")
    det.add_argument("--sam2-video-prompt-frame", type=int, default=0,
                    metavar="N",
                    help="Which frame's bg-sub centroid to use as the "
                         "SAM2 seed prompt (default frame 0).")
    det.add_argument("--min-solidity", type=float, default=0.0,
                    metavar="S",
                    help="Minimum blob solidity = area/hull_area [0–1] "
                         "(default 0 = disabled). A compact rat body is "
                         "~0.6–0.8; a rat+cable blob drops to ~0.3–0.5. "
                         "Set ~0.45 to keep rat body and reject cable-only "
                         "or reflection blobs.")
    det.add_argument("--morph-k", type=int, default=7,
                     help="Morphological kernel size (default: 7)")
    det.add_argument("--max-epipolar-px", type=float, default=8.0,
                     help="Maximum epipolar line distance for stereo "
                          "matching (default: 8 px)")
    det.add_argument("--wall-decay", type=float, default=80.0,
                    metavar="PX",
                    help="Wall distance weight decay in pixels (default 80). "
                         "At this distance from the projected arena wall, "
                         "foreground diff is weighted to ~0.63 of its value. "
                         "Smaller = sharper attenuation at walls. "
                         "Set to 0 to disable wall weighting.")
    det.add_argument("--texture-suppress", action="store_true", default=False,
                    help="Gabor filter bank bedding suppression. "
                         "Attenuates foreground pixels whose current Gabor "
                         "energy matches the background bedding texture, "
                         "reducing false detections from disturbed bedding. "
                         "Recommended when the rat moves bedding around.")
    det.add_argument("--texture-alpha", type=float, default=0.7,
                    metavar="A",
                    help="Gabor suppression strength 0–1 (default 0.7). "
                         "Higher = more bedding suppression, may clip "
                         "animal edges at boundary with bedding.")
    det.add_argument("--texture-lambdas", type=int, nargs="+",
                    default=[8, 12, 16], metavar="PX",
                    help="Gabor wavelengths in px targeting the bedding "
                         "fibre scale (default: 8 12 16 px).")
    det.add_argument("--fur-gabor-min", type=float, default=0.0,
                    metavar="T",
                    help="Require a minimum normalized Gabor energy "
                         "(0-1, after 99th-percentile normalization) "
                         "for a pixel to remain in the foreground "
                         "mask. Suppresses WIDE smooth surfaces — "
                         "acrylic walls, large reflective panels — "
                         "but is ineffective for thin features like a "
                         "tether cable (the Gabor edge response fills "
                         "the entire object width). For thin features "
                         "use --max-aspect-ratio. Typical useful "
                         "values when applicable: 0.03 to 0.15. "
                         "Disabled by default (0.0).")
    det.add_argument("--max-aspect-ratio", type=float, default=None,
                    metavar="R",
                    help="Reject blobs whose long:short axis ratio "
                         "(from minimum bounding rectangle) exceeds "
                         "this value. Tether cables typically have "
                         "aspect 10-30, rat bodies are closer to "
                         "1.5-3. Recommended: 6-10. PRIMARY FIX for "
                         "the failure mode where the cable wins the "
                         "largest-CC pick because the rat is "
                         "fragmented. Disabled by default.")
    det.add_argument("--mahalanobis-k", type=float, default=0.0,
                    metavar="K",
                    help="Use per-pixel Mahalanobis-style "
                         "background subtraction: fg = (frame - "
                         "bg)/max(σ_pixel, σ_floor) > k, where "
                         "σ_pixel is the per-pixel std built into "
                         "the background model. Trouble regions "
                         "(cable mount, headstage specular spots, "
                         "acrylic wall edges) have high σ and need "
                         "a larger excursion to register as "
                         "foreground; stable regions keep effective "
                         "sensitivity ≈ k pixels. With "
                         "--bg-adapt-alpha set, both σ AND μ adapt "
                         "online so illumination drift over a long "
                         "recording is tracked. Requires a bg.npz "
                         "built with this version (older files "
                         "lack the σ field; in that case this flag "
                         "is a silent no-op). Typical values: 3-5. "
                         "Disabled by default (0.0).")
    det.add_argument("--sigma-floor", type=float, default=1.0,
                    metavar="S",
                    help="Floor for per-pixel σ in Mahalanobis "
                         "mode (units: pixel intensity). Prevents "
                         "divide-by-zero / extreme sensitivity at "
                         "pixels that happened to have near-zero "
                         "variance in the bg sample. Default 1.0.")
    det.add_argument("--motion-min", type=float, default=0.0,
                    metavar="PX",
                    help="Require pixels to have at least this much "
                         "frame-to-frame motion (px/frame) to remain "
                         "in the foreground mask. Eliminates "
                         "physically-fixed bright features — cable "
                         "mount hardware, attachment bolts, acrylic "
                         "specular highlights, plexiglass "
                         "reflections — which appear as foreground "
                         "in bg-sub but have zero optical flow "
                         "because they don't move. The rat retains "
                         "motion. Typical useful values: 0.5 to 3.0 "
                         "px/frame. Disabled by default (0.0). "
                         "Adds ~30-50 ms per frame at 2028x1080 "
                         "when --motion-method=flow.")
    det.add_argument("--motion-method", default="flow",
                    choices=["flow", "framediff"],
                    help="Motion estimator. 'flow' = dense optical "
                         "flow (Farneback); true pixel translation, "
                         "static-but-flickering features get flow=0. "
                         "'framediff' = abs(frame - prev); cheap "
                         "(~3 ms) but catches intensity changes too "
                         "(less discriminative). Default 'flow'.")
    det.add_argument("--luminance-correct", action="store_true",
                    default=False,
                    help="Per-frame multiplicative luminance "
                         "correction on the background. Each frame, "
                         "estimate g = median(frame/bg) over the "
                         "non-animal arena pixels and use g*bg as "
                         "the effective background for diff. Handles "
                         "fast IR illumination drift that "
                         "--bg-adapt-alpha is too slow to track. "
                         "Independent of and complementary to "
                         "--mahalanobis-k. Disabled by default.")
    det.add_argument("--luminance-correct-min-bg", type=float,
                    default=10.0, metavar="I",
                    help="Pixels with bg below this intensity are "
                         "excluded from the per-frame g estimate "
                         "(avoid divide-by-tiny). Default 10.0 "
                         "(out of 255).")
    det.add_argument("--luminance-correct-clip-lo", type=float,
                    default=0.5, metavar="R",
                    help="Lower clip on frame/bg ratios before "
                         "taking the median for g estimation. "
                         "Rejects extreme outliers (dark anomalies). "
                         "Default 0.5.")
    det.add_argument("--luminance-correct-clip-hi", type=float,
                    default=2.0, metavar="R",
                    help="Upper clip on frame/bg ratios before "
                         "taking the median for g estimation. "
                         "Rejects animal/cable/specular pixels (which "
                         "are typically 3-5x bg). Default 2.0.")
    det.add_argument("--diff-median-k", type=int, default=0,
                    metavar="K",
                    help="Median filter kernel applied to the diff "
                         "map BEFORE thresholding. Removes "
                         "salt-and-pepper outliers (single-pixel "
                         "bright spots from bedding texture, camera "
                         "noise, residual specular flicker) while "
                         "preserving the rat blob's edges. Must be "
                         "odd; 3 or 5 typical. Default 0 (off).")
    # ── Rat texture bank ──────────────────────────────────────────
    tb = ap.add_argument_group("Rat texture bank (Gabor + Gaussian)")
    tb.add_argument("--use-texture-bank", action="store_true",
                    default=False,
                    help="Enable the rat texture bank. Builds a "
                         "multi-scale Gabor-feature Gaussian model "
                         "from random bootstrap frames, then gates "
                         "blob acceptance by texture similarity. "
                         "Updates online from confident detections "
                         "during tracking; refits trigger version_id "
                         "increments which are logged per-frame.")
    tb.add_argument("--texture-bank-load", type=str, default=None,
                    metavar="PATH",
                    help="Load a pre-trained .npz texture bank "
                         "instead of bootstrapping. Skip the random-"
                         "frame bootstrap phase. Combine with "
                         "--texture-bank-freeze for production runs "
                         "after training.")
    tb.add_argument("--texture-bank-save", type=str, default=None,
                    metavar="PATH",
                    help="Save the final (possibly online-refined) "
                         "bank to this .npz at the end of the run. "
                         "Default: {session}/tracking/texture_bank.npz")
    tb.add_argument("--texture-bank-freeze", action="store_true",
                    default=False,
                    help="Disable online updates. Use bank exactly "
                         "as loaded/bootstrapped. Recommended for "
                         "production runs after training.")
    tb.add_argument("--texture-bank-bootstrap-frames", type=int,
                    default=50, metavar="N",
                    help="Number of random frames to sample for the "
                         "bootstrap phase. Default 50.")
    tb.add_argument("--texture-bank-min-score", type=float, default=0.3,
                    metavar="S",
                    help="Texture similarity threshold for blob "
                         "acceptance. exp(-Mahalanobis²/2D), so 1.0 "
                         "= perfect match, 0 = far. Default 0.3.")
    tb.add_argument("--texture-bank-update-every", type=int, default=100,
                    metavar="N",
                    help="Refit the bank's Gaussian every N online "
                         "samples. Default 100.")
    tb.add_argument("--texture-bank-drift-threshold", type=float,
                    default=0.05, metavar="R",
                    help="Relative-mean-change threshold for "
                         "incrementing version_id on refit. Default "
                         "0.05 (5%% change).")
    tb.add_argument("--no-rotation-invariant-features",
                    action="store_true", default=False,
                    help="Disable rotation-invariant Gabor features. "
                         "By default the bank uses 3 features per "
                         "scale pooled across orientations (max, "
                         "mean, std of |Gabor response|) so the "
                         "learned texture model is invariant to "
                         "which direction the rat is facing. With "
                         "this flag the bank reverts to legacy mode "
                         "(one feature per orientation × scale), "
                         "which is direction-sensitive and known to "
                         "fail as the rat rotates. Only useful when "
                         "loading a legacy saved bank.")
    # ── Bootstrap permissive thresholds ────────────────────────────
    # The bootstrap temp-detector needs to FIND the rat in many
    # frames so we can sample texture patches. Its goal is recall
    # (find rat candidates somewhere) — false positives are tolerable
    # since the largest-CC selection plus patch uniformity filter
    # both reject non-rat blobs. So the bootstrap defaults are much
    # more permissive than the main pass: lower threshold, lower
    # min_area, smaller morph kernel.
    tb.add_argument("--bootstrap-threshold", type=float, default=None,
                    metavar="T",
                    help="Bg-sub threshold for the bootstrap temp "
                         "detector. Default: max(10, --threshold/3) "
                         "— permissive so more frames yield rat "
                         "candidates. Has no effect on the main pass.")
    tb.add_argument("--bootstrap-min-area", type=int, default=None,
                    metavar="PX2",
                    help="Min blob area for bootstrap temp detector. "
                         "Default: max(500, --min-area/5) — "
                         "permissive. No effect on main pass.")
    tb.add_argument("--bootstrap-mahalanobis-k", type=float,
                    default=None, metavar="K",
                    help="Mahalanobis k for bootstrap. Default: "
                         "max(2.0, --mahalanobis-k/2). No effect on "
                         "main pass.")
    tb.add_argument("--bootstrap-morph-k", type=int, default=None,
                    metavar="K",
                    help="Morph kernel size for bootstrap. Default: "
                         "max(5, --morph-k/2). No effect on main "
                         "pass.")
    tb.add_argument("--bootstrap-intensity-percentile", type=float,
                    default=92.0, metavar="P",
                    help="Per-frame intensity percentile for the "
                         "PRIMARY bootstrap rat-finder. The rat is "
                         "white fur under IR and is the brightest "
                         "object in the arena; thresholding at the "
                         "Nth percentile inside the ROI then taking "
                         "the largest CC reliably finds it without "
                         "depending on bg-sub calibration. The "
                         "bg-sub path is used only as fallback. "
                         "Default 92.")
    tb.add_argument("--bootstrap-intensity-morph-k", type=int,
                    default=5, metavar="K",
                    help="Morph-close kernel for the intensity rat-"
                         "finder (consolidates speckle). Default 5.")
    # ── Blob merging ───────────────────────────────────────────────
    mb = ap.add_argument_group("Blob merging (hull-based)")
    mb.add_argument("--merge-blob-distance", type=int, default=0,
                    metavar="PX",
                    help="After per-blob filters, group surviving "
                         "blobs whose centroids are within this many "
                         "pixels and replace each group with the "
                         "convex hull of the union of their pixels. "
                         "Solves the under-segmented rat problem: "
                         "bg-sub fragments the rat into pieces; "
                         "merging reassembles a coherent rat-shaped "
                         "blob. Default 0 (off). Typical: 50-150.")
    mb.add_argument("--merge-blob-dilate", type=int, default=0,
                    metavar="PX",
                    help="Optional dilation radius for each merged "
                         "hull. Grows the hull past the visible "
                         "pixels for robustness when bg-sub "
                         "under-captures the rat's edges. Default 0.")
    # ── Texture-based edge refinement ─────────────────────────────
    er = ap.add_argument_group("Edge refinement (texture-aware)")
    er.add_argument("--edge-refine-texture", action="store_true",
                    default=False,
                    help="After blob merging, grow each blob's "
                         "mask outward into a per-pixel texture-"
                         "score band, restricted to the connected "
                         "components overlapping the original hull. "
                         "Snaps the blob boundary to actual rat "
                         "texture edges rather than the convex hull. "
                         "Requires --use-texture-bank.")
    er.add_argument("--edge-refine-expand-px", type=int, default=30,
                    metavar="PX",
                    help="Maximum pixels the refinement can grow "
                         "the hull outward. Default 30.")
    er.add_argument("--edge-refine-score-threshold", type=float,
                    default=0.15, metavar="S",
                    help="Per-pixel texture similarity threshold "
                         "for inclusion in the refined mask. Lower "
                         "than blob-level texture_min_score since "
                         "per-pixel features have higher variance. "
                         "Typical 0.10-0.25. Default 0.15.")
    er.add_argument("--edge-refine-smooth-window", type=int,
                    default=7, metavar="K",
                    help="Box-filter window for smoothing per-pixel "
                         "Gabor responses before scoring. Brings "
                         "per-pixel features closer to blob-averaged "
                         "training distribution. Odd; default 7.")
    er.add_argument("--edge-refine-canny-barrier", action="store_true",
                    default=False,
                    help="Use Canny intensity edges as hard barriers "
                         "during refinement. Refined mask cannot "
                         "include pixels at Canny-edge locations "
                         "outside the original hull, forcing the "
                         "boundary to snap to intensity "
                         "discontinuities (texture edges of the rat).")
    er.add_argument("--edge-refine-canny-low",  type=int, default=30,
                    metavar="L",
                    help="Canny low threshold for edge barrier. "
                         "Default 30.")
    er.add_argument("--edge-refine-canny-high", type=int, default=90,
                    metavar="H",
                    help="Canny high threshold for edge barrier. "
                         "Default 90.")
    er.add_argument("--edge-refine-canny-dilate", type=int, default=1,
                    metavar="PX",
                    help="Dilation of Canny edges to thicken the "
                         "barrier (single-pixel edges may have gaps). "
                         "Default 1.")
    er.add_argument("--edge-refine-intensity", action="store_true",
                    default=False,
                    help="After the texture-based refinement, expand "
                         "the mask outward using the seed's own "
                         "intensity statistics. The texture bank "
                         "plateaus inside the bright fur interior — "
                         "its score becomes noisy at the rat/bedding "
                         "boundary because boundary pixels have mixed "
                         "texture. Intensity (rat fur ≈ 180-230 vs "
                         "bedding ≈ 80-110) is a much cleaner "
                         "discriminator there. This step takes the "
                         "texture-confirmed seed and grows it out to "
                         "the natural intensity boundary. Geodesic "
                         "constraint keeps disconnected bright "
                         "objects (cable specular highlights, etc) "
                         "out of the result.")
    er.add_argument("--edge-refine-intensity-expand-px", type=int,
                    default=100, metavar="PX",
                    help="Max geodesic distance the intensity "
                         "expansion can grow the seed. Default 100.")
    er.add_argument("--edge-refine-intensity-quantile", type=float,
                    default=0.25, metavar="Q",
                    help="Quantile of seed-interior intensity used "
                         "as the bright/dim threshold for expansion. "
                         "Lower → more permissive (grows into dimmer "
                         "pixels); higher → strict. Default 0.25 (25th "
                         "percentile — the dim shoulder of the fur "
                         "intensity distribution).")
    er.add_argument("--edge-refine-intensity-floor-offset",
                    type=float, default=30.0, metavar="DI",
                    help="Subtract this many intensity units from "
                         "the seed quantile to set the effective "
                         "threshold. Without it, the seed's own "
                         "25th percentile (~210 for bright fur) "
                         "excludes the dimmer rat-edge fur (160-200) "
                         "we need to grow into. With offset 30, the "
                         "threshold reaches ~180 — captures the full "
                         "rat without dipping into bedding (~80-110). "
                         "Default 30.")
    er.add_argument("--edge-refine-intensity-morph-close-k", type=int,
                    default=5, metavar="K",
                    help="Morph-close kernel applied to the candidate "
                         "intensity-mask (consolidates speckles "
                         "before the CC-overlap step). 0 disables. "
                         "Default 5.")
    # ── Patch-based bootstrap ─────────────────────────────────────
    pb = ap.add_argument_group("Patch-based bootstrap")
    pb.add_argument("--patch-bootstrap", action="store_true",
                    default=False,
                    help="During bootstrap, sample many small uniform-"
                         "texture patches inside each detected blob "
                         "instead of computing one feature vector per "
                         "blob. Patches with high intra-patch intensity "
                         "std (boundary patches) are rejected, so the "
                         "bank learns pure rat-fur statistics without "
                         "contamination from the rat-bedding boundary.")
    pb.add_argument("--patch-size",         type=int, default=32,
                    metavar="PX",
                    help="Patch size (square). Default 32.")
    pb.add_argument("--patch-stride",       type=int, default=8,
                    metavar="PX",
                    help="Stride between candidate patch centers. "
                         "Default 8.")
    pb.add_argument("--patch-max-per-blob", type=int, default=20,
                    metavar="N",
                    help="Cap on patches per blob. Default 20.")
    pb.add_argument("--patch-uniformity-max-std", type=float,
                    default=30.0, metavar="S",
                    help="Reject patches whose intra-patch intensity "
                         "std exceeds this. Lower → stricter "
                         "uniformity. Default 30.0 — the white-fur "
                         "rat shows real local variation (fur grain, "
                         "shading); strict thresholds reject "
                         "everything.")
    # ── Artifact masks ────────────────────────────────────────────
    am = ap.add_argument_group("Per-camera artifact masks")
    am.add_argument("--use-artifact-masks", action="store_true",
                    default=False,
                    help="Build per-camera artifact masks during "
                         "bootstrap from Gabor decomposition + "
                         "intensity histogram analysis of the sampled "
                         "frames. Pixels that are consistently bright "
                         "(>top-Nth percentile per frame) AND consistently "
                         "non-rat texture get masked. Suppresses static "
                         "bright artifacts (cable mount, plexiglass "
                         "reflections, ambient bright spots) that bg-sub "
                         "alone cannot reliably eliminate. Requires "
                         "--use-texture-bank.")
    am.add_argument("--artifact-intensity-percentile", type=float,
                    default=95.0, metavar="P",
                    help="Per-frame intensity percentile for the "
                         "'bright' criterion. Default 95.0.")
    am.add_argument("--artifact-texture-score-max", type=float,
                    default=0.10, metavar="S",
                    help="Per-pixel texture score ceiling for the "
                         "'non-rat texture' criterion. Default 0.10.")
    am.add_argument("--artifact-consistency-fraction", type=float,
                    default=0.85, metavar="F",
                    help="Fraction of bootstrap frames a pixel must "
                         "meet BOTH the brightness AND the non-rat-"
                         "texture criteria in to be masked as an "
                         "artifact. Default 0.85 — true static "
                         "artifacts (door reflection, frame rails) "
                         "are present in essentially every frame, "
                         "while pixels the rat happens to visit "
                         "during bootstrap are bright in maybe 30-40% "
                         "of frames. Setting this too low (e.g. 0.5) "
                         "means rat-visited regions get masked, "
                         "and the rat can't be detected when it "
                         "returns there.")
    am.add_argument("--artifact-mask-dilate", type=int, default=5,
                    metavar="PX",
                    help="Dilation of the final artifact mask for "
                         "robustness. Default 5.")
    # ── EdgeMotionRatTracker ROI ───────────────────────────────────
    rt = ap.add_argument_group("Rat tracker ROI (KLT + Kalman hull)")
    rt.add_argument("--use-rat-tracker-roi", action="store_true",
                    default=False,
                    help="Enable the EdgeMotionRatTracker: tracks "
                         "sparse Shi-Tomasi corners with KLT, clusters "
                         "moving corners by space+velocity, builds a "
                         "convex-hull ROI around the rat cluster, and "
                         "Kalman-stabilizes the hull over time. The "
                         "ROI mask is AND'd with the bg-sub binary "
                         "before connected-components labelling. "
                         "Solves the aperture-problem failure of "
                         "Farneback flow (which wipes the rat's "
                         "smooth interior) and the cable-wins-pick "
                         "failure (cable mount lies outside the hull "
                         "so never becomes a CC candidate). "
                         "Disabled by default.")
    rt.add_argument("--rat-body-half-width-px", type=int, default=60,
                    metavar="PX",
                    help="Dilation radius applied to the KLT-derived "
                         "convex hull to produce the full-body ROI "
                         "mask. Roughly the rat's body half-width in "
                         "pixels. Default 60.")
    rt.add_argument("--rat-motion-min", type=float, default=0.5,
                    metavar="PX",
                    help="Minimum per-point KLT flow magnitude "
                         "(px/frame) to consider a point 'moving'. "
                         "Static-but-textured features (e.g. bedding "
                         "fiber endpoints) fall below this. Default 0.5.")
    rt.add_argument("--rat-min-cluster", type=int, default=15,
                    metavar="N",
                    help="Minimum points in a coherent-motion "
                         "cluster to count as a candidate rat. "
                         "Default 15 — high enough to reject small "
                         "spurious clusters from KLT corners along "
                         "the cable wire.")
    rt.add_argument("--rat-max-klt", type=int, default=300,
                    metavar="N",
                    help="Max Shi-Tomasi corners to track per "
                         "camera. Default 300.")
    rt.add_argument("--rat-refresh-every", type=int, default=30,
                    metavar="N",
                    help="Refresh the corner pool every N frames "
                         "regardless of survival, to keep the point "
                         "set adapted as the rat changes pose. "
                         "Default 30.")
    rt.add_argument("--rat-dbscan-eps-xy", type=float, default=40.0,
                    metavar="PX",
                    help="Spatial neighborhood radius for clustering "
                         "moving KLT points (px). Default 40.")
    rt.add_argument("--rat-dbscan-eps-v", type=float, default=2.0,
                    metavar="PX/F",
                    help="Velocity neighborhood radius for clustering "
                         "moving KLT points (px/frame). Default 2.0.")
    rt.add_argument("--rat-cluster-max-aspect-ratio", type=float,
                    default=4.0, metavar="R",
                    help="Reject KLT clusters whose long:short axis "
                         "ratio (from cv2.minAreaRect) exceeds this. "
                         "Cable-wire KLT clusters are highly "
                         "elongated (aspect 10-30 along the wire). "
                         "Rat body KLT clusters are roundish (1.5-3). "
                         "Default 4.0; set negative to disable.")
    rt.add_argument("--rat-seed-roi-radius-px", type=int, default=None,
                    metavar="PX",
                    help="On the first frame, restrict initial "
                         "corner seeding to a disc of this radius "
                         "around the frame center. Helps avoid "
                         "locking onto a cable mount that's near a "
                         "frame edge. Default None (seed whole frame).")
    det.add_argument("--no-roi-mask", action="store_true", default=False,
                    help="Disable the automatic arena ROI mask. "
                         "Use if the mask clips the animal at the walls.")
    det.add_argument("--roi-mode", default="box",
                    choices=["box", "floor", "volume"],
                    help="Arena ROI extent. 'box' (legacy) = hull of all "
                         "8 corners (~3.6x floor area, includes the "
                         "through-wall region past the transparent "
                         "walls). 'floor' = 4 floor corners (footprint). "
                         "'volume' = floor + --roi-max-height-mm band. "
                         "floor/volume are recommended for stereo.")
    det.add_argument("--roi-max-height-mm", type=float, default=120.0,
                    help="Height (mm) of the volume band above the floor "
                         "for --roi-mode volume.")
    det.add_argument("--bg-adapt-alpha", type=float, default=None,
                    metavar="A",
                    help="Enable temporal background adaptation with the "
                         "given EMA weight. At 25 fps, alpha=0.995 gives "
                         "~8 s memory, 0.999 gives ~40 s. Updates only at "
                         "pixels where no animal was detected, so bedding "
                         "that the rat moved earlier is gradually absorbed "
                         "into the new background. Disabled by default.")
    det.add_argument("--bg-adapt-dilate-px", type=int, default=25,
                    metavar="PX",
                    help="Dilation radius (px) applied to the foreground "
                         "mask before background adaptation. The animal's "
                         "shadow / fur halo gets excluded from the update "
                         "to prevent it being baked into the background.")
    det.add_argument("--trajectory-prior", action="store_true",
                    help="Bias the centroid-only blob selector toward "
                         "candidates near the previous frame's confirmed "
                         "detection. Prevents jumps to wall reflections "
                         "or coincidentally well-aligned far blobs.")
    det.add_argument("--trajectory-prior-lambda", type=float, default=0.05,
                    metavar="L",
                    help="Strength of the spatial prior in --trajectory-"
                         "prior mode (px-epipolar per px-spatial). At the "
                         "default L=0.05, a 100 px jump from the prior "
                         "costs the same as 5 px of epipolar error.")
    det.add_argument("--centroid-only", action="store_true",
                     help="Track only the animal centroid (label: 'animal') "
                          "instead of labelling body parts. More robust when "
                          "body-part detection is unreliable.")

    # ── Contrast enhancement ─────────────────────────────────────────────────
    con = ap.add_argument_group("Contrast enhancement (NIR footage)")
    con.add_argument("--clahe", action="store_true",
                     help="Apply CLAHE (adaptive histogram equalisation) before "
                          "background subtraction.  Strongly recommended for NIR "
                          "footage where animal fur blends with bedding.")
    con.add_argument("--clahe-clip", type=float, default=2.0,
                     help="CLAHE clip limit — higher = more contrast enhancement "
                          "but also more noise amplification (default: 2.0)")
    con.add_argument("--clahe-tile", type=int, default=8,
                     help="CLAHE tile grid size — smaller tiles = more local "
                          "enhancement (default: 8)")
    con.add_argument("--green-channel", action="store_true",
                     help="Use the green Bayer channel instead of luminance. "
                          "Green carries ~2x NIR signal on IMX477.  "
                          "Recommended with --bayer-pattern RGGB.")
    con.add_argument("--bilateral", action="store_true",
                     help="Apply bilateral filter instead of Gaussian blur. "
                          "Preserves fur/bedding edges while reducing sensor noise.")
    con.add_argument("--bilateral-d", type=int, default=9,
                     help="Bilateral filter neighbourhood diameter (default: 9)")
    con.add_argument("--bilateral-sigma", type=float, default=50.0,
                     help="Bilateral filter sigma for colour and spatial "
                          "domains (default: 50.0)")

    # ── SAM ──────────────────────────────────────────────────────────────────
    sam = ap.add_argument_group("SAM (optional)")
    sam.add_argument("--sam2-checkpoint", default=None, metavar="PATH",
                     help="Path to SAM2 model weights (.pt). If supplied, "
                          "SAM2 is used for body part segmentation.")
    sam.add_argument("--sam2-config", default="sam2_hiera_large.yaml",
                     help="SAM2 config yaml (default: sam2_hiera_large.yaml)")
    sam.add_argument("--device", default="cuda",
                     help="PyTorch device for SAM2 (default: cuda)")

    # ── Sequence ─────────────────────────────────────────────────────────────
    seq = ap.add_argument_group("Sequence")
    seq.add_argument("--start-frame", type=int, default=0,
                     help="First frame to process (default: 0)")
    seq.add_argument("--end-frame", type=int, default=None,
                     help="Last frame to process (default: end of video)")
    seq.add_argument("--sample-every", type=int, default=1,
                     help="Process every Nth frame (default: 1 = all frames)")
    seq.add_argument("--redetect-every", type=int, default=60,
                     help="Force re-detection every N frames for optical "
                          "flow tracker (default: 60)")

    # ── Arena alignment ───────────────────────────────────────────────────────
    al = ap.add_argument_group("Arena alignment")
    al.add_argument("--align-points", default=None, metavar="CSV",
                    help="Alignment CSV from rpimocap-align "
                         "(expresses output in arena mm frame)")
    al.add_argument("--bounds", default=None,
                    help="Reconstruction bounds xmin,xmax,ymin,ymax,zmin,zmax "
                         "in mm (used for viewer only)")

    # ── Smoothing ─────────────────────────────────────────────────────────────
    sm = ap.add_argument_group("Smoothing")
    sm.add_argument("--smooth-sigma", type=float, default=1.5,
                    help="Gaussian smoothing sigma in frames (default: 1.5, "
                         "0 = off)")
    sm.add_argument("--fill-gaps", type=int, default=5,
                    help="Interpolate gaps up to N frames (default: 5, 0=off)")
    sm.add_argument("--kalman", action="store_true",
                    help="Use a 3D Kalman/RTS smoother instead of Gaussian "
                         "smoothing + linear gap-fill. Adds outlier "
                         "rejection (wall reflections, bad blobs) via "
                         "Mahalanobis gating, and fills gaps with constant-"
                         "velocity predictions clamped to --kalman-max-speed.")
    sm.add_argument("--kalman-fps", type=float, default=25.0, metavar="FPS",
                    help="Frame rate for the Kalman dt (default 25).")
    sm.add_argument("--kalman-max-speed", type=float, default=1000.0,
                    metavar="MM_S",
                    help="Rat maximum speed in mm/s (default 1000); "
                         "initialises velocity covariance.")
    sm.add_argument("--kalman-max-accel", type=float, default=2000.0,
                    metavar="MM_S2",
                    help="Rat maximum acceleration in mm/s² (default 2000); "
                         "sets process noise Q.")
    sm.add_argument("--kalman-noise", type=float, default=8.0, metavar="MM",
                    help="Triangulation measurement noise 1-σ in mm "
                         "(default 8).")
    sm.add_argument("--kalman-outlier-sigma", type=float, default=4.0,
                    metavar="S",
                    help="Mahalanobis threshold for outlier rejection "
                         "(default 4 σ ≈ p<0.003 for χ²(3)).")
    sm.add_argument("--kalman-no-rts", action="store_true",
                    help="Disable RTS backward smoothing pass.")

    # ── Diagnostics ──────────────────────────────────────────────────────────
    dg = ap.add_argument_group("Diagnostics")
    dg.add_argument("--diagnostics", default="/tmp/rpimocap_diag",
                    metavar="DIR",
                    help="Write diagnostic images (background, enhanced frames, "
                         "diff maps, mask overlays) to this directory. "
                         "Default: /tmp/rpimocap_diag  (set to '' to disable)")
    dg.add_argument("--diag-frames", type=int, default=6,
                    help="Number of sample frames to save for diagnostics "
                         "(default: 6, evenly spaced through the video)")

    args = ap.parse_args()

    # ── Imports ───────────────────────────────────────────────────────────────
    from rpimocap.cli.pipeline import open_video
    from rpimocap.detection.segment import (
        BackgroundModel,
        EpipolarMatcher,
        ForegroundDetector,
        GeometricLabeller,
        arena_roi_mask,
    )
    from rpimocap.detection.tracker import SegmentTracker
    from rpimocap.io.export import write_hdf5, write_stats_csv, write_viewer_json
    from rpimocap.reconstruction.kalman import KalmanTracker3D
    from rpimocap.reconstruction.rearing import RearingClassifier
    from rpimocap.reconstruction.triangulate import (
        fill_trajectory_gaps,
        smooth_trajectory,
        trajectory_stats,
    )

    out_dir   = Path(args.out)
    bg_dir    = out_dir / "background"
    track_dir = out_dir / "tracking"
    bg_dir.mkdir(parents=True, exist_ok=True)
    track_dir.mkdir(parents=True, exist_ok=True)

    print("rpimocap-segment")
    print(f"  cam0   : {args.cam0}")
    print(f"  cam1   : {args.cam1}")
    print(f"  calib  : {args.calib}")
    print(f"  out    : {args.out}")

    # ── Open videos ──────────────────────────────────────────────────────────
    cap0 = open_video(args.cam0, bayer_pattern=args.bayer_pattern)
    cap1 = open_video(args.cam1, bayer_pattern=args.bayer_pattern)
    n_frames = int(min(cap0.get(cv2.CAP_PROP_FRAME_COUNT),
                       cap1.get(cv2.CAP_PROP_FRAME_COUNT)))
    vid_w = int(cap0.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap0.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps   = cap0.get(cv2.CAP_PROP_FPS) or 25.0
    print(f"  resolution : {vid_w}x{vid_h}  "
          f"({n_frames} frames @ {fps:.1f} fps)")

    # ── Calibration ──────────────────────────────────────────────────────────
    cal = np.load(args.calib)

    # ── Background model ─────────────────────────────────────────────────────
    print("\n── Background model ─────────────────────────────────────────────")
    bg_npz = bg_dir / "bg.npz"
    if args.background_model and Path(args.background_model).exists():
        print(f"  Loading: {args.background_model}")
        bg = BackgroundModel.from_npz(args.background_model)
    elif bg_npz.exists() and not (args.background_extra_cam0):
        print(f"  Reusing: {bg_npz}")
        bg = BackgroundModel.from_npz(bg_npz)
    else:
        extra0 = args.background_extra_cam0
        extra1 = args.background_extra_cam1
        if extra0 and not extra1:
            # If only cam0 extras given, use the same files for cam1
            extra1 = extra0
        if len(extra1) != len(extra0):
            print("ERROR: --background-extra-cam0 and --background-extra-cam1 "
                  "must have the same number of files")
            sys.exit(1)
        if extra0:
            from rpimocap.io.export import TiffCapture as _TC
            all_caps0 = [cap0] + [_TC(f, bayer_pattern=args.bayer_pattern)
                                   for f in extra0]
            all_caps1 = [cap1] + [_TC(f, bayer_pattern=args.bayer_pattern)
                                   for f in extra1]
            print(f"  Building from {len(all_caps0)} sessions "
                  f"x {args.background_frames} frames each ...")
            bg = BackgroundModel.from_multiple_captures(
                all_caps0, all_caps1,
                n_frames_each=args.background_frames,
                method=args.background_method,
                start_frame=args.background_start,
                verbose=True)
            for c in all_caps0[1:] + all_caps1[1:]:
                c.release()
        else:
            bg = BackgroundModel.from_captures(
                cap0, cap1,
                n_frames=args.background_frames,
                method=args.background_method,
                start_frame=args.background_start,
                verbose=True)
        if args.texture_suppress:
            print("  Caching Gabor bedding-energy maps in bg.npz "
                  "(for --gabor-refine at tracking time) ...")
            bg.compute_gabor(
                lambdas=tuple(args.texture_lambdas),
                n_orientations=4)   # matches ForegroundDetector default
        bg.save(bg_npz)
        if bg.bg_gabor0 is not None:
            print(f"  Saved: {bg_npz}  (incl. Gabor model)")
        else:
            print(f"  Saved: {bg_npz}")

    # Save background images for inspection
    for img, name in [(bg.bg0, "bg_cam0.png"), (bg.bg1, "bg_cam1.png")]:
        cv2.imwrite(str(bg_dir / name),
                    np.clip(img, 0, 255).astype(np.uint8))

    # ── Flat-field (NIR vignette) correction ──────────────────────────────────
    flat0: "np.ndarray | None" = None
    flat1: "np.ndarray | None" = None
    if args.flat_field_cam0 or args.flat_field_cam1 or args.synthesize_flat_field:
        from rpimocap.detection.vignette import (
            apply_flat_field,
            load_flat_field,
            synthesize_flat_field,
        )
        print("\n── Flat-field correction ────────────────────────────")
        if args.flat_field_cam0:
            flat0 = load_flat_field(args.flat_field_cam0)
            print(f"  cam0 flat-field: {args.flat_field_cam0}")
        elif args.synthesize_flat_field:
            flat0 = synthesize_flat_field(bg.bg0)
            print("  cam0 flat-field: synthesized from background")
        if args.flat_field_cam1:
            flat1 = load_flat_field(args.flat_field_cam1)
            print(f"  cam1 flat-field: {args.flat_field_cam1}")
        elif args.synthesize_flat_field:
            flat1 = synthesize_flat_field(bg.bg1)
            print("  cam1 flat-field: synthesized from background")
        # Correct the background itself so the per-frame bg-subtraction is
        # comparing flat-fielded frames against a flat-fielded background.
        if flat0 is not None:
            bg.bg0 = apply_flat_field(bg.bg0, flat0, clip=False).astype(np.float32)
        if flat1 is not None:
            bg.bg1 = apply_flat_field(bg.bg1, flat1, clip=False).astype(np.float32)

    # ── Epipolar matcher ──────────────────────────────────────────────────────
    matcher = EpipolarMatcher.from_calibration(
        cal, max_epipolar_px=args.max_epipolar_px)

    # ── Arena alignment ───────────────────────────────────────────────────────
    align_result = None
    if args.align_points:
        from rpimocap.reconstruction.align import (
            kabsch_align,
            kabsch_align_from_pixels,
            load_align_csv,
        )
        print(f"\n── Arena alignment ({args.align_points}) ─────────────────────")
        try:
            align_pts = load_align_csv(args.align_points)
            _has_px   = any(pt.px0 is not None for pt in align_pts)
            if _has_px:
                align_result = kabsch_align_from_pixels(
                    align_pts, matcher.P0, matcher.P1)
                print(f"  {align_result.n_points} pts  "
                      f"RMSE={align_result.rmse_mm:.2f}mm  "
                      "(re-triangulated from pixel clicks)")
            else:
                align_result = kabsch_align(align_pts)
                print(f"  {align_result.n_points} pts  "
                      f"RMSE={align_result.rmse_mm:.2f}mm  "
                      "(WARNING: no px stored -- re-annotate)")
        except Exception as e:
            print(f"  WARNING: alignment failed — {e}")

    # ── Arena ROI masks ──────────────────────────────────────────────────────
    # Project the 8 known arena corners through each DLT projection matrix
    # to create a convex-hull mask that restricts foreground detection to
    # the physical arena interior.  This eliminates the frame, cables,
    # LED reflections, and bedding disturbance outside the arena.
    _ARENA_CORNERS = np.array([
        [-140, -215,   0], [ 140, -215,   0],
        [ 140,  215,   0], [-140,  215,   0],
        [-140, -215, 388], [ 140, -215, 388],
        [ 140,  215, 388], [-140,  215, 388],
    ], dtype=np.float64)

    roi_mask0 = roi_mask1 = None
    wall_weight0 = wall_weight1 = None
    if not args.no_roi_mask:
        P0_dlt = cal.get("dlt_P0", cal.get("P0", None))
        P1_dlt = cal.get("dlt_P1", cal.get("P1", None))
        if P0_dlt is not None and P1_dlt is not None:
            from rpimocap.detection.segment import arena_roi_corners
            roi_corners = arena_roi_corners(
                _ARENA_CORNERS, mode=getattr(args, "roi_mode", "box"),
                max_height_mm=getattr(args, "roi_max_height_mm", 120.0))
            roi_mask0 = arena_roi_mask(P0_dlt, roi_corners,
                                       (vid_h, vid_w), pad_px=20)
            roi_mask1 = arena_roi_mask(P1_dlt, roi_corners,
                                       (vid_h, vid_w), pad_px=20)
            print(f"  Arena ROI masks computed from DLT projection "
                  f"matrices (roi-mode: {getattr(args,'roi_mode','box')})")
            if args.wall_decay > 0:
                from rpimocap.detection.segment import arena_wall_weight
                wall_weight0 = arena_wall_weight(P0_dlt, _ARENA_CORNERS,
                                                 (vid_h, vid_w),
                                                 decay_px=args.wall_decay)
                wall_weight1 = arena_wall_weight(P1_dlt, _ARENA_CORNERS,
                                                 (vid_h, vid_w),
                                                 decay_px=args.wall_decay)
                print(f"  Wall weight maps (decay={args.wall_decay:.0f}px) computed")
        else:
            print("  WARNING: no DLT P matrices in calib — ROI mask disabled")
    else:
        print("  Arena ROI mask disabled (--no-roi-mask)")

    # ── SAM2 video propagation pre-pass (if enabled) ─────────────────────────
    sam2_mask_cache = None
    if args.sam2_video_checkpoint:
        from rpimocap.detection.sam2_mask_cache import SAM2MaskCache
        from rpimocap.detection.tracker import SAM2VideoTracker

        cache_dir = Path(args.sam2_video_cache_dir
                         or (track_dir / "sam2_masks"))
        cache = SAM2MaskCache(cache_dir)
        if cache.exists:
            print("\n── SAM2 video cache ───────────────────────────────────────────")
            print(f"  Reusing existing cache at {cache_dir}")
            sam2_mask_cache = cache
        else:
            print("\n── SAM2 video pre-pass ─────────────────────────────────────────")
            print(f"  Loading SAM2 video model: {args.sam2_video_checkpoint}")
            svt = SAM2VideoTracker(
                checkpoint=args.sam2_video_checkpoint,
                config=args.sam2_video_config,
                device=args.device)
            if not svt.available:
                print("  WARN: SAM2VideoTracker not available "
                      "(sam2 package missing) — falling back to bg-sub")
            else:
                # Seed prompts: take the bg-sub centroid on the prompt frame
                # from each camera.
                from rpimocap.detection.segment import ForegroundDetector
                det = ForegroundDetector(
                    background=bg, threshold=args.threshold,
                    min_area_px=args.min_area)

                def _frames_iter(cap):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    while True:
                        ok, f = cap.read()
                        if not ok:
                            return
                        yield f

                def _seed_from_bg(cap, cam):
                    cap.set(cv2.CAP_PROP_POS_FRAMES,
                            args.sam2_video_prompt_frame)
                    ok, f = cap.read()
                    if not ok:
                        raise RuntimeError(
                            f"cam{cam}: cannot read prompt frame "
                            f"{args.sam2_video_prompt_frame}")
                    r = det.detect(f, cam)
                    if r.n_blobs == 0:
                        raise RuntimeError(
                            f"cam{cam}: no blobs on prompt frame; "
                            "cannot seed SAM2")
                    # Largest blob centroid
                    ys, xs = np.where(r.label_map == 1)
                    return float(xs.mean()), float(ys.mean())

                p0 = _seed_from_bg(cap0, 0)
                p1 = _seed_from_bg(cap1, 1)
                print(f"  Seed cam0 prompt: ({p0[0]:.0f}, {p0[1]:.0f})")
                print(f"  Seed cam1 prompt: ({p1[0]:.0f}, {p1[1]:.0f})")
                print(f"  Propagating masks → {cache_dir}")
                sam2_mask_cache = SAM2MaskCache.precompute(
                    svt,
                    _frames_iter(cap0), _frames_iter(cap1),
                    prompt0_xy=p0, prompt1_xy=p1,
                    cache_dir=cache_dir,
                    prompt_frame_idx=args.sam2_video_prompt_frame)
                print("  SAM2 propagation complete")

    # ── Texture bank (optional) ──────────────────────────────────────
    # If --use-texture-bank is set, either load a saved bank or
    # bootstrap a new one from random frames before constructing the
    # tracker. The bank is passed to SegmentTracker which forwards
    # it to ForegroundDetector.
    texture_bank = None
    if args.use_texture_bank:
        from rpimocap.detection.rat_texture import RatTextureBank
        if args.texture_bank_load:
            print(f"\n── Loading texture bank: {args.texture_bank_load}")
            texture_bank = RatTextureBank.load(args.texture_bank_load)
            print(f"  Loaded bank: version_id={texture_bank.version_id}, "
                  f"n_samples={texture_bank.n_samples}, "
                  f"D={texture_bank.feature_dim}")
            if args.texture_bank_freeze:
                # Disable online updates: set buffer threshold to infinity
                texture_bank.update_every = 10**9
        else:
            print("\n── Bootstrapping texture bank from random frames")
            from rpimocap.detection.rat_texture import (
                bootstrap_from_random_frames)
            from rpimocap.detection.segment import ForegroundDetector

            texture_bank = RatTextureBank(
                update_every=args.texture_bank_update_every,
                drift_threshold=args.texture_bank_drift_threshold,
                rotation_invariant=not args.no_rotation_invariant_features)
            # Sample random frame indices uniformly across the recording
            import cv2 as _cv2
            n_total = int(min(cap0.get(_cv2.CAP_PROP_FRAME_COUNT),
                              cap1.get(_cv2.CAP_PROP_FRAME_COUNT)))
            n_boot  = min(args.texture_bank_bootstrap_frames, n_total)
            rng = np.random.RandomState(42)
            sample_idx = sorted(rng.choice(
                n_total, size=n_boot, replace=False))
            # Use a temporary detector (NO texture bank) to find blobs.
            # Override threshold / min_area / mahalanobis-k / morph-k
            # with permissive bootstrap defaults — the temp detector
            # needs RECALL (find rat candidates somewhere in most
            # frames); the largest-CC selection plus patch uniformity
            # filter handle the precision side.
            boot_thr   = (float(args.bootstrap_threshold)
                          if args.bootstrap_threshold is not None
                          else max(10.0, args.threshold / 3.0))
            boot_min   = (int(args.bootstrap_min_area)
                          if args.bootstrap_min_area is not None
                          else max(500, args.min_area // 5))
            boot_k     = (float(args.bootstrap_mahalanobis_k)
                          if args.bootstrap_mahalanobis_k is not None
                          else max(2.0, args.mahalanobis_k / 2.0))
            boot_morph = (int(args.bootstrap_morph_k)
                          if args.bootstrap_morph_k is not None
                          else max(5, args.morph_k // 2))
            print(f"  Bootstrap temp-detector (permissive): "
                  f"threshold={boot_thr:.1f} (main {args.threshold}), "
                  f"min_area={boot_min} (main {args.min_area}), "
                  f"mahalanobis_k={boot_k:.1f} (main {args.mahalanobis_k}), "
                  f"morph_k={boot_morph} (main {args.morph_k})")
            tmp_det = ForegroundDetector(
                background=bg,
                threshold=boot_thr,
                min_area_px=boot_min,
                max_area_px=args.max_area,
                min_solidity=args.min_solidity,
                use_green_channel=args.green_channel,
                bilateral=args.bilateral,
                morph_k=boot_morph,
                mahalanobis_k=boot_k,
                sigma_floor=args.sigma_floor,
                luminance_correct=args.luminance_correct,
                diff_median_k=args.diff_median_k,
                polarity=args.polarity,
                roi_mask=(matcher.roi_mask_cam0
                          if hasattr(matcher, "roi_mask_cam0") else None))
            sample_features = []
            # Diagnostics for the bootstrap loop
            boot_frames_with_blob = 0
            boot_total_blobs      = 0
            boot_largest_areas    = []
            boot_patches_per_blob = []
            boot_via_intensity    = 0
            boot_via_bgsub        = 0
            # If --use-artifact-masks, collect grays per camera for later
            # mask construction (after bank is trained).
            collected_gray = {0: [], 1: []}
            # Per-camera ROI masks for the intensity finder
            roi_by_cam = {
                0: (matcher.roi_mask_cam0
                    if hasattr(matcher, "roi_mask_cam0") else None),
                1: (matcher.roi_mask_cam1
                    if hasattr(matcher, "roi_mask_cam1") else None),
            }
            from rpimocap.detection.rat_texture import (
                find_rat_seed_by_intensity)
            for idx in sample_idx:
                cap0.set(_cv2.CAP_PROP_POS_FRAMES, idx)
                cap1.set(_cv2.CAP_PROP_POS_FRAMES, idx)
                ok0, f0 = cap0.read()
                ok1, f1 = cap1.read()
                if not (ok0 and ok1) or f0 is None or f1 is None:
                    continue
                for cam, frame in ((0, f0), (1, f1)):
                    # Run the temp detector first (it does the bayer
                    # demosaic, green-channel extraction, luminance
                    # correction, etc) so we can reuse the same
                    # frame_gray for both finders + artifact masks.
                    r = tmp_det.detect(frame, cam=cam)
                    if args.use_artifact_masks and r.frame_gray is not None:
                        collected_gray[cam].append(r.frame_gray.copy())
                    if r.frame_gray is None:
                        continue
                    gray_u8 = r.frame_gray.astype(np.uint8) \
                        if r.frame_gray.dtype != np.uint8 \
                        else r.frame_gray
                    # ── PRIMARY: intensity-based rat seed ─────────
                    # The rat is white fur under IR — the brightest
                    # object in the arena. Top-percentile thresholding
                    # inside the arena ROI finds it reliably without
                    # depending on bg-sub model calibration.
                    m = find_rat_seed_by_intensity(
                        gray_u8,
                        roi_mask=roi_by_cam.get(cam),
                        intensity_percentile=args.bootstrap_intensity_percentile,
                        min_area_px=boot_min,
                        morph_close_k=args.bootstrap_intensity_morph_k)
                    if m is not None:
                        boot_via_intensity += 1
                    else:
                        # ── FALLBACK: bg-sub largest CC ─────────
                        # Used only when the intensity finder fails
                        # (e.g. very dim frame). Less reliable but
                        # better than skipping the frame entirely.
                        if r.label_map is None:
                            continue
                        n_labels = int(r.label_map.max())
                        if n_labels < 1:
                            continue
                        sizes = np.array(
                            [int((r.label_map == i).sum())
                             for i in range(1, n_labels + 1)],
                            dtype=np.int64)
                        largest_idx = int(np.argmax(sizes)) + 1
                        largest_size = int(sizes[largest_idx - 1])
                        boot_total_blobs += n_labels
                        if largest_size < boot_min:
                            continue
                        m = (r.label_map == largest_idx).astype(np.uint8)
                        boot_via_bgsub += 1
                    boot_frames_with_blob += 1
                    boot_largest_areas.append(int((m > 0).sum()))
                    if args.patch_bootstrap:
                        # Sample uniform-texture patches inside the blob
                        patch_feats = texture_bank.sample_uniform_patches(
                            r.frame_gray, m,
                            patch_size=args.patch_size,
                            stride=args.patch_stride,
                            max_patches=args.patch_max_per_blob,
                            std_max=args.patch_uniformity_max_std,
                            rng_seed=42 + idx + cam)
                        sample_features.extend(patch_feats)
                        boot_patches_per_blob.append(len(patch_feats))
                    else:
                        feats = texture_bank.features_in_blob(
                            r.frame_gray, m)
                        if np.any(feats > 0):
                            sample_features.append(feats)
                            boot_patches_per_blob.append(1)
                        else:
                            boot_patches_per_blob.append(0)
            # Rewind to frame 0 for the main tracking pass
            cap0.set(_cv2.CAP_PROP_POS_FRAMES, 0)
            cap1.set(_cv2.CAP_PROP_POS_FRAMES, 0)
            mode_str = ("uniform patches"
                        if args.patch_bootstrap else "blob averages")
            # Diagnostics on what the bootstrap loop found / rejected
            n_passes = len(sample_idx) * 2  # cam0 + cam1
            print(f"  Bootstrap diagnostics:")
            print(f"    detector passes        : {n_passes}")
            print(f"    rat seed found in pass : "
                  f"{boot_frames_with_blob}/{n_passes} "
                  f"({100*boot_frames_with_blob/max(1, n_passes):.1f}%) "
                  f"[intensity={boot_via_intensity}, "
                  f"bg-sub fallback={boot_via_bgsub}]")
            if boot_largest_areas:
                arr = np.array(boot_largest_areas)
                print(f"    rat seed area px²      : "
                      f"min={int(arr.min())}, median={int(np.median(arr))}, "
                      f"max={int(arr.max())}, n={len(arr)}")
            if args.patch_bootstrap and boot_patches_per_blob:
                ppb = np.array(boot_patches_per_blob)
                rejected = int((ppb == 0).sum())
                print(f"    patches/blob (n={len(ppb)})    : "
                      f"min={int(ppb.min())}, median={int(np.median(ppb))}, "
                      f"max={int(ppb.max())}, mean={ppb.mean():.1f}")
                if rejected > 0:
                    print(f"    blobs yielding 0 patches: "
                          f"{rejected}/{len(ppb)} (uniformity gate "
                          f"likely too strict — try "
                          f"--patch-uniformity-max-std larger, or "
                          f"--patch-size smaller)")
            print(f"  Collected {len(sample_features)} bootstrap "
                  f"samples ({mode_str}) from {len(sample_idx)} "
                  f"random frames")
            # Minimum samples for a stable D-dim Gaussian. With rot-inv
            # mode (D=9 for default 3 scales) we need at least D+1=10
            # samples; require 15 to have some margin.
            min_required = max(15, texture_bank.feature_dim + 5)
            try:
                bootstrap_from_random_frames(
                    texture_bank, sample_features,
                    min_samples=min_required)
                print(f"  Bootstrap complete: version_id="
                      f"{texture_bank.version_id}, "
                      f"n_samples={texture_bank.n_samples}")
            except RuntimeError as e:
                print(f"  WARNING: bootstrap failed ({e}). "
                      f"Disabling texture bank.")
                texture_bank = None
            if texture_bank is not None and args.texture_bank_freeze:
                texture_bank.update_every = 10**9

    # ── Artifact masks (optional, after bank is trained) ──────────
    artifact_mask_cam0 = artifact_mask_cam1 = None
    if (args.use_texture_bank
            and args.use_artifact_masks
            and texture_bank is not None
            and texture_bank.is_ready):
        from rpimocap.detection.rat_texture import build_camera_artifact_mask
        print("\n── Building per-camera artifact masks "
              "(Gabor + intensity histogram)")
        roi0 = (matcher.roi_mask_cam0
                if hasattr(matcher, "roi_mask_cam0") else None)
        roi1 = (matcher.roi_mask_cam1
                if hasattr(matcher, "roi_mask_cam1") else None)
        for cam_idx, (grays, roi_for_cam, label) in enumerate(
                [(collected_gray[0], roi0, "cam0"),
                 (collected_gray[1], roi1, "cam1")]):
            if not grays:
                print(f"  {label}: no bootstrap grays collected; skipping")
                continue
            try:
                m = build_camera_artifact_mask(
                    bank=texture_bank,
                    gray_frames=grays,
                    roi_mask=roi_for_cam,
                    intensity_percentile=args.artifact_intensity_percentile,
                    texture_score_max=args.artifact_texture_score_max,
                    consistency_fraction=args.artifact_consistency_fraction,
                    dilate_px=args.artifact_mask_dilate,
                    smooth_window=args.edge_refine_smooth_window)
            except Exception as e:
                print(f"  {label}: artifact mask build failed: {e}")
                continue
            if m is None:
                print(f"  {label}: artifact mask is None; skipping")
                continue
            n_masked = int((m > 0).sum())
            n_total  = m.size
            print(f"  {label}: artifact mask covers "
                  f"{n_masked} / {n_total} px "
                  f"({100*n_masked/n_total:.2f}%)")
            if cam_idx == 0:
                artifact_mask_cam0 = m
            else:
                artifact_mask_cam1 = m

    # ── Tracker ──────────────────────────────────────────────────────────────
    print("\n── Tracking ────────────────────────────────────────────────────")
    tracker = SegmentTracker(
        background=bg,
        matcher=matcher,
        sam2_checkpoint=args.sam2_checkpoint,
        sam2_config=args.sam2_config,
        device=args.device,
        threshold=args.threshold,
        min_area_px=args.min_area,
        max_area_px=args.max_area,
        min_solidity=args.min_solidity,
        morph_k=args.morph_k,
        redetect_every=args.redetect_every,
        centroid_only=args.centroid_only,
        clahe=args.clahe,
        clahe_clip=args.clahe_clip,
        clahe_tile=args.clahe_tile,
        use_green_channel=args.green_channel,
        bilateral=args.bilateral,
        bilateral_d=args.bilateral_d,
        bilateral_sigma=args.bilateral_sigma,
        roi_mask=roi_mask0,
        wall_weight=wall_weight0,
        cable_erosion_px=args.cable_erosion,
        texture_suppress=args.texture_suppress,
        texture_lambdas=tuple(args.texture_lambdas),
        texture_alpha=args.texture_alpha,
        fur_gabor_min=args.fur_gabor_min,
        max_aspect_ratio=args.max_aspect_ratio,
        mahalanobis_k=args.mahalanobis_k,
        sigma_floor=args.sigma_floor,
        motion_min=args.motion_min,
        motion_method=args.motion_method,
        luminance_correct=args.luminance_correct,
        luminance_correct_min_bg=args.luminance_correct_min_bg,
        luminance_correct_clip_lo=args.luminance_correct_clip_lo,
        luminance_correct_clip_hi=args.luminance_correct_clip_hi,
        diff_median_k=args.diff_median_k,
        texture_bank=texture_bank,
        texture_min_score=args.texture_bank_min_score,
        merge_blob_distance=args.merge_blob_distance,
        merge_blob_dilate=args.merge_blob_dilate,
        edge_refine_texture=args.edge_refine_texture,
        edge_refine_expand_px=args.edge_refine_expand_px,
        edge_refine_score_threshold=args.edge_refine_score_threshold,
        edge_refine_smooth_window=args.edge_refine_smooth_window,
        edge_refine_canny_barrier=args.edge_refine_canny_barrier,
        edge_refine_canny_low=args.edge_refine_canny_low,
        edge_refine_canny_high=args.edge_refine_canny_high,
        edge_refine_canny_dilate=args.edge_refine_canny_dilate,
        edge_refine_intensity=args.edge_refine_intensity,
        edge_refine_intensity_expand_px=args.edge_refine_intensity_expand_px,
        edge_refine_intensity_quantile=args.edge_refine_intensity_quantile,
        edge_refine_intensity_floor_offset=args.edge_refine_intensity_floor_offset,
        edge_refine_intensity_morph_close_k=args.edge_refine_intensity_morph_close_k,
        artifact_mask_cam0=artifact_mask_cam0,
        artifact_mask_cam1=artifact_mask_cam1,
        use_rat_tracker_roi=args.use_rat_tracker_roi,
        rat_body_half_width_px=args.rat_body_half_width_px,
        rat_motion_min=args.rat_motion_min,
        rat_min_cluster=args.rat_min_cluster,
        rat_max_klt=args.rat_max_klt,
        rat_refresh_every=args.rat_refresh_every,
        rat_dbscan_eps_xy=args.rat_dbscan_eps_xy,
        rat_dbscan_eps_v=args.rat_dbscan_eps_v,
        rat_cluster_max_aspect_ratio=(
            None
            if args.rat_cluster_max_aspect_ratio is not None
            and args.rat_cluster_max_aspect_ratio < 0
            else args.rat_cluster_max_aspect_ratio),
        rat_seed_roi_radius_px=args.rat_seed_roi_radius_px,
        polarity=args.polarity,
        bg_adapt_alpha=args.bg_adapt_alpha,
        bg_adapt_dilate_px=args.bg_adapt_dilate_px,
        use_trajectory_prior=args.trajectory_prior,
        trajectory_prior_lambda=args.trajectory_prior_lambda,
        flat_field_cam0=flat0,
        flat_field_cam1=flat1,
        body_length_mm=args.body_length,
        body_width_mm=args.body_width,
        body_z_mm=args.body_z,
        P0=(cal.get("dlt_P0", cal.get("P0", None))
            if args.body_length > 0 else None),
        P1=(cal.get("dlt_P1", cal.get("P1", None))
            if args.body_length > 0 else None),
        gabor_refine=args.gabor_refine,
        canny_low=args.canny_low,
        canny_high=args.canny_high,
        kalman_online=(KalmanTracker3D(
            dt=1.0/float(fps),
            sigma_a=args.kalman_max_accel,
            sigma_z=args.kalman_noise,
            mahalanobis_gate=args.kalman_outlier_sigma)
            if args.kalman_online else None),
        rearing_classifier=(RearingClassifier(
            z_enter=args.rearing_z_enter,
            z_exit=args.rearing_z_exit)
            if (args.rearing_detection and args.kalman_online) else None),
        fps=float(fps),
        sam2_mask_cache=sam2_mask_cache,
        verbose=True)

    # Register cam1 masks on the shared ForegroundDetector
    if roi_mask1 is not None:
        tracker._det.set_roi_mask(1, roi_mask1)
    if wall_weight1 is not None:
        tracker._det.set_wall_weight(1, wall_weight1)

    # ── Diagnostics ──────────────────────────────────────────────────────────
    if args.diagnostics:
        from rpimocap.detection.segment import GeometricLabeller, save_diagnostics
        diag_header = f"\n── Diagnostics -> {args.diagnostics} ──────────────────────"
        print(diag_header)
        # Reset captures to start for diagnostic sampling
        for cap in (cap0, cap1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        save_diagnostics(
            cap0, cap1,
            detector=tracker._det,
            labeller=GeometricLabeller(),
            out_dir=args.diagnostics,
            n_frames=args.diag_frames,
        )
        # Reset again for tracking
        for cap in (cap0, cap1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Parse bounds for epipolar mismatch rejection
    _bounds_arr = None
    if args.bounds:
        try:
            _bounds_arr = np.array([float(v) for v in args.bounds.split(",")])
            if len(_bounds_arr) != 6:
                raise ValueError(f"--bounds needs 6 values, got {len(_bounds_arr)}")
            print(f"  Bounds filter: {_bounds_arr} (rejects out-of-arena triangulations)")
        except Exception:
            print(f"  WARNING: could not parse --bounds '{args.bounds}', no filter applied")

    results = tracker.track_sequence(
        cap0, cap1,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        sample_every=args.sample_every,
        align_result=align_result,
        bounds=_bounds_arr)

    cap0.release()
    cap1.release()

    # ── Convert to skeleton_frames ────────────────────────────────────────────
    skeleton_frames = SegmentTracker.results_to_skeleton_frames(results)

    # ── Capture detection mask BEFORE post-processing ─────────────────────
    # True  = genuine triangulated detection in this frame
    # False = missing (gap) — will be filled by Kalman / interpolation
    # Computed here so it cannot be corrupted by smoothing / Kalman /
    # gap-fill steps below.
    _all_kp = sorted({pt.name for frame in skeleton_frames for pt in frame})
    detected_masks: dict = {}
    for _kp in _all_kp:
        _det = np.zeros(len(skeleton_frames), dtype=bool)
        for _fi, _fr in enumerate(skeleton_frames):
            for _pt in _fr:
                if _pt.name == _kp and not np.isnan(_pt.xyz).any():
                    _det[_fi] = True
        detected_masks[_kp] = _det

    # ── Smoothing / gap fill ─────────────────────────────────────────────────
    print("\n── Post-processing ─────────────────────────────────────────────")
    if args.kalman and skeleton_frames:
        from rpimocap.reconstruction.triangulate import kalman_filter_trajectory
        skeleton_frames = kalman_filter_trajectory(
            skeleton_frames,
            fps=args.kalman_fps,
            max_speed_mm_s=args.kalman_max_speed,
            max_accel_mm_s2=args.kalman_max_accel,
            measurement_noise_mm=args.kalman_noise,
            outlier_sigma=args.kalman_outlier_sigma,
            rts_smooth=not args.kalman_no_rts,
        )
        n_out = sum(1 for frame in skeleton_frames
                    for pt in frame if getattr(pt, 'kalman_outlier', False))
        suffix = "+RTS" if not args.kalman_no_rts else " (forward only)"
        print(f"  Kalman{suffix}: {n_out} outlier frames corrected")
    else:
        if args.smooth_sigma > 0 and skeleton_frames:
            skeleton_frames = smooth_trajectory(
                skeleton_frames, sigma=args.smooth_sigma)
            print(f"  Smoothed (sigma={args.smooth_sigma})")
        if args.fill_gaps > 0 and skeleton_frames:
            skeleton_frames = fill_trajectory_gaps(
                skeleton_frames, max_gap=args.fill_gaps)
            print(f"  Gap-filled (max_gap={args.fill_gaps})")

    # ── Stats ────────────────────────────────────────────────────────────────
    stats = trajectory_stats(skeleton_frames)

    # ── Bounds ───────────────────────────────────────────────────────────────
    bounds = None
    if args.bounds:
        bounds = _parse_bounds(args.bounds)

    # ── Export ───────────────────────────────────────────────────────────────
    print("\n── Exporting ───────────────────────────────────────────────────")

    # Save trained texture bank, if used
    if texture_bank is not None:
        # Flush any pending online samples to the model
        flushed = texture_bank.flush_pending()
        bank_save_path = (
            args.texture_bank_save
            if args.texture_bank_save
            else str(track_dir / "texture_bank.npz"))
        try:
            texture_bank.save(bank_save_path)
            stats_str = texture_bank.stats()
            print(f"  texture_bank.npz  (final version={stats_str['version_id']}, "
                  f"n_samples={stats_str['n_samples']}, "
                  f"buffered={stats_str['buffered']}, "
                  f"flushed_on_save={flushed})")
            print(f"    saved → {bank_save_path}")
        except Exception as e:
            print(f"  WARNING: failed to save texture bank: {e}")

    # Save artifact masks (one PNG per camera) for visual inspection
    for cam_idx, amask in [(0, artifact_mask_cam0),
                            (1, artifact_mask_cam1)]:
        if amask is None:
            continue
        out_path = track_dir / f"artifact_mask_cam{cam_idx}.png"
        try:
            import cv2 as _cv2_save
            _cv2_save.imwrite(str(out_path), amask)
            print(f"  artifact_mask_cam{cam_idx}.png  "
                  f"({int((amask > 0).sum())} px masked)")
        except Exception as e:
            print(f"  WARNING: failed to save cam{cam_idx} "
                  f"artifact mask: {e}")

    write_stats_csv(track_dir / "detection_stats.csv", stats)
    print("  detection_stats.csv")

    write_hdf5(
        track_dir / "reconstruction.h5",
        skeleton_frames,
        detected_masks=detected_masks,
        voxel_frames=None,
        fps=fps,
        metadata={
            "cam0":             args.cam0,
            "cam1":             args.cam1,
            "calib":            args.calib,
            "detector":         "segment",
            "bayer_pattern":    args.bayer_pattern,
            "align_points":     args.align_points,
            "align_rmse_mm":    (align_result.rmse_mm
                                 if align_result else None),
            "sam2_checkpoint":  args.sam2_checkpoint,
        })
    print("  reconstruction.h5")

    # Derive keypoint names and skeleton edges from the tracking results
    kp_names = sorted({pt.name for frame in skeleton_frames for pt in frame})

    # Skeleton connectivity along the spine + ears
    _SPINE = ["nose","head","neck","back","rump","tail_base","tail_tip"]
    _EAR   = [("head","left_ear"), ("head","right_ear")]
    edges  = ([(a, b) for a, b in zip(_SPINE, _SPINE[1:])
                if a in kp_names and b in kp_names]
              + [(a, b) for a, b in _EAR
                 if a in kp_names and b in kp_names])

    write_viewer_json(
        track_dir / "viewer_data.json",
        skeleton_frames,
        keypoint_names=kp_names,
        skeleton_edges=edges,
        bounds=bounds,
        fps=fps,
        voxel_frames=None)
    print("  viewer_data.json")

    # ── Summary ───────────────────────────────────────────────────────────────
    n_det  = sum(1 for r in results if r.detected)
    n_tot  = len(results)
    parts  = set(k for r in results for k in r.xyz)
    print("\n── Done ────────────────────────────────────────────────────────")
    print(f"  Processed : {n_tot} frames")
    print(f"  Detected  : {n_det} ({100*n_det/max(n_tot,1):.1f}%)")
    print(f"  Body parts: {sorted(parts)}")
    print(f"  Output    : {track_dir}/")

    # Per-stage detection drop-off — answers "which pipeline stage
    # is rejecting the rat?". A cliff between two adjacent rows
    # marks the bottleneck (e.g., a sudden drop from "after_aspect"
    # to "after_texture_bank" means the texture-bank gate is the
    # culprit; tune --texture-bank-min-score).
    det = getattr(tracker, "_det", None)
    if det is not None and hasattr(det, "format_pipeline_stats"):
        print("\n" + det.format_pipeline_stats())

    # Pipeline diagnostics — which refinement steps actually fired
    # and how often. Helps tune flag combinations: low gabor_refine
    # success means the texture model isn't seeing a clean low-energy
    # body; low cable_erosion success means cable_erosion_px is too
    # aggressive; many kalman_gap frames means the body is being lost
    # often; etc.
    ss = tracker.step_stats
    if ss:
        print("\n── Pipeline diagnostics ────────────────────────────────────────")
        print(f"  Frames with stereo match : {ss.get('frames_with_match', 0)}"
              f" / {n_tot}")

        def _pct(num_key, den_key):
            num = ss.get(num_key, 0)
            den = ss.get(den_key, 0)
            return f"{num} / {den} ({100*num/den:.1f}%)" if den else "—"

        if ss.get('cable_erosion_attempted', 0):
            print(f"  Cable erosion            : "
                  f"{_pct('cable_erosion_succeeded', 'cable_erosion_attempted')}")
        if ss.get('gabor_refine_attempted', 0):
            print(f"  Gabor refinement         : "
                  f"{_pct('gabor_refine_succeeded', 'gabor_refine_attempted')}")
        if ss.get('anatomical_prior_attempted', 0):
            print(f"  Anatomical prior         : "
                  f"{_pct('anatomical_prior_succeeded', 'anatomical_prior_attempted')}")
        fb_e = ss.get('fallback_ellipse', 0)
        fb_h = ss.get('fallback_hull', 0)
        if fb_e or fb_h:
            print(f"  Fallbacks                : "
                  f"ellipse={fb_e}, hull={fb_h}")
        if ss.get('kalman_with_measurement', 0) or ss.get('kalman_gap', 0):
            print(f"  Online Kalman            : "
                  f"{ss.get('kalman_with_measurement', 0)} measured, "
                  f"{ss.get('kalman_gap', 0)} gap")
        if ss.get('rearing_frames', 0):
            print(f"  Rearing posture          : "
                  f"{ss['rearing_frames']} / {n_tot} "
                  f"({100*ss['rearing_frames']/max(n_tot,1):.1f}%)")
        if ss.get('sam2_mask_hits', 0):
            print(f"  SAM2 mask cache hits     : "
                  f"{ss['sam2_mask_hits']} / {n_tot}")
        if ss.get('bg_adapt_updates', 0):
            print(f"  Background adaptation    : "
                  f"{ss['bg_adapt_updates']} updates")


if __name__ == "__main__":
    main()
