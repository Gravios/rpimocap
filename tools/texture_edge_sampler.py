#!/usr/bin/env python3
"""
texture_edge_sampler.py
=======================
Interactive utility to collect labeled TEXTURE and EDGE statistics by
clicking on random patches of the real frames.

Why this exists
---------------
The texture-distance detector models the background in a dense Gabor
descriptor space, and the research notes point at replacing the
diagonal-Gaussian distance with a *nonparametric* (Parzen/KDE) region
density. Both want real, labeled samples of what each class actually
looks like in that feature space:

  * TEXTURE classes — bedding, rat fur, acrylic wall, ...
  * EDGE classes    — maze edge (sharp, high-contrast rail) vs
                      rat edge (soft fur-to-bedding transition).

This tool shows random patches, cycles through a list of label classes,
and for the CURRENT class:

  * LEFT-CLICK on an example  → save the stats in a window around the
                                click point, then show the next patch.
  * RIGHT-CLICK               → no example here; skip to another patch.

Keyboard:
  click → save a sample (stays on the image)
  SPACE → next image           Tab → next texture class (same kind)
  n → next class (any kind)    p → previous class
  u         → undo last save     s → save CSV now
  r         → new random patch   q / ESC → quit (auto-saves)

The saved stats live in the SAME Gabor descriptor space as the detector,
plus intensity, gradient/structure-tensor, and (for edges) a cross-edge
contrast/sharpness profile — exactly the features that separate a sharp
maze rail from a soft rat outline.

Output
------
  <out>/samples.csv            one row per saved click (appends across
                               sessions). Provenance columns:
                                 session   — session id (--session, else the
                                             cam0 filename stem)
                                 src_file  — source capture file name
                                 cam,frame — camera and frame index
                                 patch_x0,patch_y0,patch_w,patch_h
                                           — the shown patch's origin + size
                                             in the full frame
                                 x,y       — sample (click) coords in the
                                             FULL frame
                                 x_in_patch,y_in_patch
                                           — click coords within the patch
                                 win       — stats-window side length
                               So (session,cam,frame,x,y) uniquely locate a
                               sample and (patch_x0,y0,w,h) reproduce the view.
                               Colour columns (BGR frames): col_{r,g,b}_win
                               (+ _std) and col_rb_win / col_rmb_win are the
                               per-channel mean/std and R/B, (R-B)/(R+B) over
                               the window; col_{r,g,b}_loc and col_rb_loc /
                               col_rmb_loc are the same from a small local
                               average at the click (--color-local, default
                               3px) — use the window colour for broad textures
                               and the local colour for thin ones (cable).
  <out>/patches/<class>_<n>_cam<c>_f<frame>_<x>_<y>.png
                               the patch with the click marked (provenance);
                               the filename encodes cam, frame, and absolute
                               coordinates.

Requires a GUI build of OpenCV (opencv-python, NOT -headless) and a
display (works over X-forwarding).

Example
-------
  python tools/texture_edge_sampler.py \
      --cam0 "$cam0_tif" --cam1 "$cam1_tif" \
      --session 20260214_021722 \
      --bayer-pattern RGGB --green-channel \
      --texture-classes bedding fur wall \
      --edge-classes maze_edge rat_edge \
      --patch-size 500 --win 48 \
      --out "${SUBJECT_DIR}/texsamples"
"""
from __future__ import annotations

import argparse
import csv
import os
from typing import Optional, Sequence

import cv2
import numpy as np

try:                                   # scipy is a core dep
    from scipy.ndimage import map_coordinates
except ImportError:                    # pragma: no cover
    map_coordinates = None


# ────────────────────────────────────────────────────────────────────
#  Feature extraction (testable; no GUI)
# ────────────────────────────────────────────────────────────────────

# Field order for the CSV / record dicts. Kept fixed so sessions append
# cleanly. n_desc Gabor channels are written as desc0..desc{n-1}.
_BASE_FIELDS = [
    # provenance — enough to re-locate and re-sample every click
    "session", "src_file", "cam", "frame",
    "patch_x0", "patch_y0", "patch_w", "patch_h",
    "x", "y", "x_in_patch", "y_in_patch",
    "klass", "kind", "win",
    # intensity stats over the window
    "int_mean", "int_std", "int_min", "int_max",
    "int_p10", "int_p50", "int_p90",
    # colour: per-channel mean+std over the window (broad textures) and a
    # small local average at the click (thin structures like the cable)
    "col_r_win", "col_g_win", "col_b_win",
    "col_r_win_std", "col_g_win_std", "col_b_win_std",
    "col_rb_win", "col_rmb_win",
    "col_r_loc", "col_g_loc", "col_b_loc",
    "col_rb_loc", "col_rmb_loc",
    # gradient / structure-tensor stats over the window
    "grad_mag_mean", "grad_mag_max", "orient_deg", "coherence",
    # cross-edge profile (meaningful for edges; computed for all)
    "edge_contrast", "edge_width_px",
    # higher-order blob geometry (rat vs cable) — from the clicked blob
    "geom_area", "geom_fill", "geom_solidity", "geom_elongation",
]


def intensity_stats(win_gray: np.ndarray) -> dict:
    """Basic intensity statistics over a window."""
    w = win_gray.astype(np.float32)
    p10, p50, p90 = np.percentile(w, [10, 50, 90])
    return {
        "int_mean": float(w.mean()),
        "int_std":  float(w.std()),
        "int_min":  float(w.min()),
        "int_max":  float(w.max()),
        "int_p10":  float(p10),
        "int_p50":  float(p50),
        "int_p90":  float(p90),
    }


def structure_tensor_stats(win_gray: np.ndarray,
                           blur_sigma: float = 1.5) -> dict:
    """Gradient + structure-tensor descriptors over a window.

    Returns mean/max gradient magnitude, the dominant orientation
    (degrees), and coherence in [0, 1] — how strongly oriented the
    local structure is (1 = a clean edge, 0 = isotropic/flat). An edge
    has high coherence; flat bedding has low coherence.
    """
    g = win_gray.astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    # Structure tensor components, smoothed
    k = int(2 * round(3 * blur_sigma) + 1)
    jxx = cv2.GaussianBlur(gx * gx, (k, k), blur_sigma)
    jyy = cv2.GaussianBlur(gy * gy, (k, k), blur_sigma)
    jxy = cv2.GaussianBlur(gx * gy, (k, k), blur_sigma)
    # Average tensor over the window (one orientation/coherence estimate)
    a = float(jxx.mean()); b = float(jyy.mean()); c = float(jxy.mean())
    tr = a + b
    det = a * b - c * c
    disc = max((a - b) * (a - b) + 4 * c * c, 0.0) ** 0.5
    lam1 = 0.5 * (tr + disc)
    lam2 = 0.5 * (tr - disc)
    coherence = 0.0
    if (lam1 + lam2) > 1e-9:
        coherence = float(((lam1 - lam2) / (lam1 + lam2)) ** 2)
    orient = 0.5 * np.degrees(np.arctan2(2 * c, a - b))
    return {
        "grad_mag_mean": float(mag.mean()),
        "grad_mag_max":  float(mag.max()),
        "orient_deg":    float(orient),
        "coherence":     float(coherence),
    }


def cross_edge_profile(gray: np.ndarray, cx: float, cy: float,
                       orient_deg: float, reach: int = 20) -> dict:
    """Sample intensity along the direction PERPENDICULAR to the edge
    (i.e. across it) and characterize the step.

    edge_contrast : the intensity step height across the edge
                    (max − min of the cross-profile). A maze rail is
                    high-contrast; a rat outline is lower.
    edge_width_px : how many px the transition spans from 10%→90% of
                    the step — sharpness. A specular rail is sharp
                    (small width); a soft fur edge is wider.

    The sampling direction is the gradient direction = edge orientation
    + 90°. Uses bilinear interpolation (scipy.map_coordinates).
    """
    if map_coordinates is None:        # pragma: no cover
        return {"edge_contrast": 0.0, "edge_width_px": 0.0}
    # structure_tensor_stats returns the dominant-gradient direction
    # (the eigenvector of the larger eigenvalue), which already points
    # ACROSS the edge. Sample along it to capture the intensity step.
    theta = np.radians(orient_deg)
    dx, dy = np.cos(theta), np.sin(theta)
    rs = np.arange(-reach, reach + 1, dtype=np.float32)
    xs = cx + rs * dx
    ys = cy + rs * dy
    prof = map_coordinates(gray.astype(np.float32),
                           np.vstack([ys, xs]), order=1, mode="nearest")
    contrast = float(prof.max() - prof.min())
    # 10%→90% width
    lo = prof.min() + 0.1 * contrast
    hi = prof.min() + 0.9 * contrast
    above_lo = np.where(prof >= lo)[0]
    above_hi = np.where(prof >= hi)[0]
    width = 0.0
    if contrast > 1e-6 and above_lo.size and above_hi.size:
        # span between first crossing of lo and first crossing of hi
        width = float(abs(above_hi[0] - above_lo[0]) + 1)
    return {"edge_contrast": contrast, "edge_width_px": width}


def gabor_descriptor_at(gray: np.ndarray, cx: int, cy: int,
                        kernels, n_orient: int, n_scales: int,
                        smooth_k: int = 7) -> np.ndarray:
    """The dense rotation-invariant Gabor descriptor sampled at the
    click pixel. Computed on the local window for speed and indexed at
    the center."""
    from rpimocap.detection.texture_distance import dense_gabor_descriptor
    desc = dense_gabor_descriptor(
        gray, kernels, n_orient, n_scales,
        smooth_k=smooth_k, rotation_invariant=True,
        second_layer=False)   # (D, H, W) — fixed 3*n_scales schema
    cy = int(np.clip(cy, 0, desc.shape[1] - 1))
    cx = int(np.clip(cx, 0, desc.shape[2] - 1))
    return desc[:, cy, cx].astype(np.float32)


def blob_geometry_at(gray: np.ndarray, cx: int, cy: int,
                     win: int = 201,
                     thresh_pct: float = 60.0) -> dict:
    """Higher-order geometry of the blob under the click (cx, cy).

    Segments the local window by Otsu (the rat/cable are brighter than
    bedding under IR), takes the connected component containing the
    click, and returns the SAME rotation-invariant shape features the
    tracker ranks on — fill (area/bbox), solidity (area/convex-hull),
    elongation (2nd-moment axis ratio) — plus area. This lets clicked
    rat vs cable samples pre-seed the tracker's shape thresholds
    (min_solidity / max_elongation).

    The window must be large enough to contain both the object AND
    surrounding background so the object is separable; a window smaller
    than the object degenerates (whole window = foreground). Returns
    neutral defaults for a degenerate/no-blob click so records stay
    schema-stable.
    """
    from rpimocap.detection.texture_distance import TextureBlobTracker
    H, W = gray.shape
    half = win // 2
    x0 = int(np.clip(cx - half, 0, max(W - win, 0)))
    y0 = int(np.clip(cy - half, 0, max(H - win, 0)))
    sub = gray[y0:y0 + win, x0:x0 + win]
    if sub.size == 0:
        return {"geom_area": 0.0, "geom_fill": 0.0,
                "geom_solidity": 1.0, "geom_elongation": 1.0}
    sub8 = sub.astype(np.uint8) if sub.dtype != np.uint8 else sub
    # Otsu separates the bright object from darker background; fall back
    # to a percentile if Otsu degenerates (flat window).
    thr_otsu, binm = cv2.threshold(
        sub8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    frac = float((binm > 0).mean())
    if frac > 0.9 or frac < 0.01:                 # Otsu degenerate
        thr = np.percentile(sub8, thresh_pct)
        binm = (sub8 >= thr).astype(np.uint8) * 255
    binm = (binm > 0).astype(np.uint8)
    n_cc, labels, stats, _ = cv2.connectedComponentsWithStats(binm)
    if n_cc <= 1:
        return {"geom_area": 0.0, "geom_fill": 0.0,
                "geom_solidity": 1.0, "geom_elongation": 1.0}
    lx = int(np.clip(cx - x0, 0, win - 1))
    ly = int(np.clip(cy - y0, 0, win - 1))
    idx = int(labels[ly, lx])
    if idx == 0:                                  # click landed on bg
        idx = 1 + int(np.argmax([stats[i, cv2.CC_STAT_AREA]
                                 for i in range(1, n_cc)]))
    fill, solidity, elong = TextureBlobTracker._geom_features(
        labels, idx, stats[idx])
    return {"geom_area": float(stats[idx, cv2.CC_STAT_AREA]),
            "geom_fill": float(fill),
            "geom_solidity": float(solidity),
            "geom_elongation": float(elong)}


def seed_thresholds_from_csv(csv_path: str,
                             rat_classes=("fur", "rat"),
                             cable_classes=("cable", "wire", "tether",
                                            "headstage"),
                             margin: float = 0.5) -> dict:
    """Turn labeled clicks into pre-seeded tracker shape thresholds.

    Reads the sampler CSV, splits the geometry features (solidity,
    elongation, fill) into rat vs cable groups by class name, and
    proposes reject thresholds placed between the two distributions:

      min_solidity  : below the rat's solidity spread (rejects the
                      concave cable+headstage).
      max_elongation: above the rat's elongation spread (rejects the
                      thin cable).

    `margin` (in pooled-std units) sets how far the threshold sits from
    the rat distribution toward the cable one. Returns a dict with the
    proposed thresholds and the per-class stats, plus a d-prime per
    feature so you can see which shape feature separates best.
    """
    rows = []
    with open(csv_path) as fh:
        for r in csv.DictReader(fh):
            rows.append(r)

    def group(names):
        out = {"solidity": [], "elongation": [], "fill": []}
        for r in rows:
            if r.get("klass", "").lower() in names:
                for k, col in (("solidity", "geom_solidity"),
                               ("elongation", "geom_elongation"),
                               ("fill", "geom_fill")):
                    try:
                        v = float(r.get(col, ""))
                    except (TypeError, ValueError):
                        continue
                    if v > 0 or k == "solidity":
                        out[k].append(v)
        return {k: np.asarray(v, float) for k, v in out.items()}

    rat = group(set(c.lower() for c in rat_classes))
    cab = group(set(c.lower() for c in cable_classes))

    def dprime(a, b):
        if len(a) < 2 or len(b) < 2:
            return float("nan")
        return abs(a.mean() - b.mean()) / np.sqrt(
            0.5 * (a.var() + b.var()) + 1e-9)

    result = {"n_rat": len(rat["solidity"]),
              "n_cable": len(cab["solidity"]),
              "features": {}}
    for k in ("solidity", "elongation", "fill"):
        result["features"][k] = {
            "rat_mean": float(rat[k].mean()) if len(rat[k]) else None,
            "rat_std": float(rat[k].std()) if len(rat[k]) else None,
            "cable_mean": float(cab[k].mean()) if len(cab[k]) else None,
            "cable_std": float(cab[k].std()) if len(cab[k]) else None,
            "dprime": dprime(rat[k], cab[k]),
        }

    # Propose thresholds only if both groups are populated
    if len(rat["solidity"]) >= 2 and len(cab["solidity"]) >= 2:
        result["min_solidity"] = float(
            rat["solidity"].mean() - margin * rat["solidity"].std())
        result["max_elongation"] = float(
            rat["elongation"].mean() + margin * rat["elongation"].std())
    return result


def color_stats(color_patch, x0: int, y0: int, win: int,
                cx: int, cy: int, local: int = 3) -> dict:
    """Color statistics for a click, in two complementary supports.

    Frames are BGR (the capture demosaics to BGR). Two representations are
    captured because which one is informative depends on whether the target
    fills the window:

    * **window** (the same ``win``x``win`` box the texture/intensity stats use):
      per-channel mean + std, and R/B and (R-B)/(R+B) from the means. Best for
      broad textures (fur, bedding) that fill the window; DILUTED toward the
      background for thin structures.
    * **local** (a small ``local``x``local`` average at the click, default 3):
      per-channel mean and the two ratios. Preserves the colour of thin
      structures (e.g. a cable) that a large window averages away.

    Returns ``{}`` if ``color_patch`` is not a 3-channel image.
    """
    if (color_patch is None or color_patch.ndim != 3
            or color_patch.shape[2] < 3):
        return {}
    cp = color_patch.astype(np.float32)
    w = cp[y0:y0 + win, x0:x0 + win]
    b, g, r = w[..., 0], w[..., 1], w[..., 2]
    rw, gw, bw = float(r.mean()), float(g.mean()), float(b.mean())
    h = local // 2
    ly0, ly1 = max(0, cy - h), min(cp.shape[0], cy + h + 1)
    lx0, lx1 = max(0, cx - h), min(cp.shape[1], cx + h + 1)
    lo = cp[ly0:ly1, lx0:lx1]
    lr, lg, lb = float(lo[..., 2].mean()), float(lo[..., 1].mean()), \
        float(lo[..., 0].mean())
    return {
        "col_r_win": rw, "col_g_win": gw, "col_b_win": bw,
        "col_r_win_std": float(r.std()), "col_g_win_std": float(g.std()),
        "col_b_win_std": float(b.std()),
        "col_rb_win": rw / (bw + 1e-6),
        "col_rmb_win": (rw - bw) / (rw + bw + 1e-6),
        "col_r_loc": lr, "col_g_loc": lg, "col_b_loc": lb,
        "col_rb_loc": lr / (lb + 1e-6),
        "col_rmb_loc": (lr - lb) / (lr + lb + 1e-6),
    }


def extract_record(gray: np.ndarray, cx: int, cy: int,
                   klass: str, kind: str, win: int,
                   kernels, n_orient: int, n_scales: int,
                   cam: int, frame_idx: int,
                   smooth_k: int = 7,
                   color_patch=None, color_local: int = 3) -> dict:
    """Build one labeled record for a click at (cx, cy) on `gray`.

    `gray` here is the full patch (already cropped). (cx, cy) are in
    patch coordinates. `kind` is 'texture' or 'edge'. `color_patch`, if given,
    is the same patch in BGR — colour stats are added over the identical
    window plus a `color_local`x`color_local` average at the click.
    """
    H, W = gray.shape
    half = win // 2
    x0 = int(np.clip(cx - half, 0, max(W - win, 0)))
    y0 = int(np.clip(cy - half, 0, max(H - win, 0)))
    window = gray[y0:y0 + win, x0:x0 + win]

    rec: dict = {
        "cam": cam, "frame": frame_idx, "x": int(cx), "y": int(cy),
        "klass": klass, "kind": kind, "win": win,
    }
    rec.update(intensity_stats(window))
    rec.update(color_stats(color_patch, x0, y0, win, cx, cy, local=color_local))
    st = structure_tensor_stats(window)
    rec.update(st)
    rec.update(cross_edge_profile(gray, cx, cy, st["orient_deg"]))
    # Higher-order blob geometry (rat vs cable shape) — only meaningful
    # for texture-kind clicks on an object; edges get neutral defaults.
    if kind == "texture":
        rec.update(blob_geometry_at(gray, cx, cy, win=max(win * 2 + 1, 81)))
    else:
        rec.update({"geom_area": 0.0, "geom_fill": 0.0,
                    "geom_solidity": 1.0, "geom_elongation": 1.0})
    desc = gabor_descriptor_at(
        gray, cx, cy, kernels, n_orient, n_scales, smooth_k=smooth_k)
    for i, v in enumerate(desc):
        rec[f"desc{i}"] = float(v)
    rec["_n_desc"] = int(len(desc))
    return rec


# ────────────────────────────────────────────────────────────────────
#  Patch sampling (testable; no GUI)
# ────────────────────────────────────────────────────────────────────


def sample_patch(frame_gray: np.ndarray, patch_size: int,
                 rng: np.random.RandomState) -> tuple:
    """Crop a random patch_size×patch_size window from a frame.
    Returns (patch, x0, y0). If the frame is smaller than patch_size,
    returns the whole frame."""
    H, W = frame_gray.shape
    pw = min(patch_size, W)
    ph = min(patch_size, H)
    x0 = int(rng.randint(0, max(W - pw, 0) + 1))
    y0 = int(rng.randint(0, max(H - ph, 0) + 1))
    return frame_gray[y0:y0 + ph, x0:x0 + pw], x0, y0


def write_csv(path: str, records: Sequence[dict], n_desc: int) -> None:
    """Write/append records to a CSV with a stable header (base fields
    + desc0..desc{n_desc-1})."""
    fields = list(_BASE_FIELDS) + [f"desc{i}" for i in range(n_desc)]
    exists = os.path.exists(path)
    if exists:
        with open(path, newline="") as fh:
            header = fh.readline().strip().split(",")
        if header and header != fields:
            raise SystemExit(
                f"{path} was written by a different sampler version "
                f"(its columns don't match the current provenance schema). "
                f"Move it aside or point --out at a fresh directory so the "
                f"session/patch/coordinate columns can be written cleanly.")
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        if not exists:
            w.writeheader()
        for r in records:
            w.writerow(r)


# ────────────────────────────────────────────────────────────────────
#  Interactive shell (OpenCV highgui)
# ────────────────────────────────────────────────────────────────────


def _to_gray(frame: np.ndarray, use_green: bool) -> np.ndarray:
    if frame.ndim == 2:
        return frame
    if use_green and frame.shape[2] == 3:
        return frame[:, :, 1]
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


class _Session:
    """Holds interactive state and handles mouse clicks."""

    def __init__(self, args, caps, kernels, n_orient, n_scales,
                 classes, kinds):
        self.args = args
        self.caps = caps                 # {cam: (cap, n_frames)}
        self.kernels = kernels
        self.n_orient = n_orient
        self.n_scales = n_scales
        self.classes = classes           # list of class names
        self.kinds = kinds               # parallel list 'texture'/'edge'
        self.ci = 0                      # current class index
        self.rng = np.random.RandomState(args.seed)
        self.records: list[dict] = []
        self.n_desc = None
        self.patch = None                # current patch gray
        self.color_patch = None          # current patch BGR (for colour stats)
        self.cam = 0
        self.frame_idx = 0
        self.px0 = self.py0 = 0          # patch origin in the frame
        self.pw = self.ph = 0            # patch size in the frame
        self.disp = None                 # BGR display image
        self.saved_count = {c: 0 for c in classes}
        # provenance: session id + per-cam source file paths
        self.session = (args.session or
                        os.path.splitext(os.path.basename(args.cam0))[0])
        self.src_files = {0: args.cam0}
        if getattr(args, "cam1", None):
            self.src_files[1] = args.cam1
        os.makedirs(os.path.join(args.out, "patches"), exist_ok=True)
        self.csv_path = os.path.join(args.out, "samples.csv")

    # ── patch loading ──────────────────────────────────────────────

    def new_patch(self):
        self.cam = int(self.rng.choice(list(self.caps.keys())))
        cap, n_frames = self.caps[self.cam]
        self.frame_idx = int(self.rng.randint(0, n_frames))
        cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            return
        gray = _to_gray(frame, self.args.green_channel)
        self.patch, self.px0, self.py0 = sample_patch(
            gray, self.args.patch_size, self.rng)
        self.ph, self.pw = self.patch.shape[:2]
        # keep the same crop in colour (BGR) for the colour stats; None if the
        # source is single-channel
        if frame.ndim == 3 and frame.shape[2] >= 3:
            self.color_patch = frame[self.py0:self.py0 + self.ph,
                                     self.px0:self.px0 + self.pw]
        else:
            self.color_patch = None
        self._render()

    # ── rendering ──────────────────────────────────────────────────

    def _render(self, marker=None):
        if self.patch is None:
            return
        g = self.patch
        # contrast stretch for visibility
        lo, hi = np.percentile(g, [1, 99])
        vis = np.clip((g.astype(np.float32) - lo) / (hi - lo + 1e-6)
                      * 255, 0, 255).astype(np.uint8)
        disp = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
        klass = self.classes[self.ci]
        kind = self.kinds[self.ci]
        col = (0, 220, 0) if kind == "texture" else (0, 165, 255)
        cv2.putText(disp, f"[{kind}] {klass}", (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
        cv2.putText(disp,
                    f"cam{self.cam} f{self.frame_idx}  "
                    f"saved={self.saved_count[klass]}  "
                    f"L=save R=skip n=next u=undo q=quit",
                    (8, disp.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        if marker is not None:
            cv2.drawMarker(disp, marker, col, cv2.MARKER_CROSS, 18, 2)
        self.disp = disp
        cv2.imshow(self.args.window, disp)

    # ── mouse ──────────────────────────────────────────────────────

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self._save_click(x, y)
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.new_patch()             # skip

    def _save_click(self, x, y):
        if self.patch is None:
            return
        klass = self.classes[self.ci]
        kind = self.kinds[self.ci]
        rec = extract_record(
            self.patch, x, y, klass, kind, self.args.win,
            self.kernels, self.n_orient, self.n_scales,
            cam=self.cam, frame_idx=self.frame_idx,
            smooth_k=self.args.smooth_k,
            color_patch=self.color_patch, color_local=self.args.color_local)
        # provenance: session, source file, patch origin/size in the frame,
        # click coords both within the patch and in the FULL frame.
        rec["session"] = self.session
        rec["src_file"] = os.path.basename(self.src_files.get(self.cam, ""))
        rec["patch_x0"], rec["patch_y0"] = int(self.px0), int(self.py0)
        rec["patch_w"], rec["patch_h"] = int(self.pw), int(self.ph)
        rec["x_in_patch"], rec["y_in_patch"] = int(x), int(y)
        rec["x"] = int(self.px0 + x)     # absolute frame coordinates
        rec["y"] = int(self.py0 + y)
        if self.n_desc is None:
            self.n_desc = rec["_n_desc"]
        self.records.append(rec)
        self.saved_count[klass] += 1
        # provenance patch png with the click marked; filename encodes the
        # exact location so it can be traced without opening the CSV.
        self._render(marker=(x, y))
        n = self.saved_count[klass]
        pth = os.path.join(self.args.out, "patches",
                           f"{klass}_{n:04d}_cam{self.cam}_f{self.frame_idx}"
                           f"_{rec['x']}_{rec['y']}.png")
        cv2.imwrite(pth, self.disp)
        cv2.waitKey(120)                 # brief flash of the marker
        # Stay on the SAME patch so multiple samples can be taken from
        # one image; advance only when the user presses SPACE.
        self._render()

    # ── persistence ────────────────────────────────────────────────

    def flush(self):
        if not self.records:
            return
        nd = self.n_desc or 0
        write_csv(self.csv_path, self.records, nd)
        print(f"  wrote {len(self.records)} records → {self.csv_path}")
        self.records.clear()

    def undo(self):
        if self.records:
            r = self.records.pop()
            self.saved_count[r["klass"]] = max(
                0, self.saved_count[r["klass"]] - 1)
            print(f"  undid last {r['klass']} sample")
            self._render()

    # ── class navigation ───────────────────────────────────────────

    def next_class(self, step=1):
        self.ci = (self.ci + step) % len(self.classes)
        self._render()

    def next_class_of_kind(self, step=1):
        """Advance to the next class of the SAME kind as the current one
        (texture→texture, edge→edge). Bound to Tab so switching between
        texture label classes skips over the edge classes. Falls back to
        plain cycling if only one class shares the current kind."""
        cur_kind = self.kinds[self.ci]
        n = len(self.classes)
        for k in range(1, n + 1):
            j = (self.ci + step * k) % n
            if self.kinds[j] == cur_kind:
                self.ci = j
                self._render()
                return
        # only one class of this kind — nothing to switch to
        self._render()


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed-thresholds", metavar="LABELED_CSV",
                    default=None,
                    help="Aggregation mode (no GUI): read a sampler CSV "
                         "and print pre-seeded tracker shape thresholds "
                         "(min_solidity / max_elongation) derived from "
                         "the labeled rat vs cable geometry. Exits after.")
    ap.add_argument("--rat-classes", nargs="+",
                    default=["fur", "rat"],
                    help="Class names counted as rat for --seed-thresholds.")
    ap.add_argument("--cable-classes", nargs="+",
                    default=["cable", "wire", "tether", "headstage"],
                    help="Class names counted as cable/artifact for "
                         "--seed-thresholds.")
    ap.add_argument("--seed-margin", type=float, default=0.5,
                    help="Threshold placement (pooled-std units) from the "
                         "rat distribution toward the cable one.")
    ap.add_argument("--cam0", default=None,
                    help="Camera-0 TIFF (required for sampling; omit for "
                         "--seed-thresholds).")
    ap.add_argument("--cam1", default=None,
                    help="Optional second camera TIFF.")
    ap.add_argument("--bayer-pattern", default="RGGB")
    ap.add_argument("--green-channel", action="store_true",
                    default=False)
    ap.add_argument("--texture-classes", nargs="+",
                    default=["bedding", "fur", "wall"],
                    help="Texture class names to cycle through.")
    ap.add_argument("--edge-classes", nargs="+",
                    default=["maze_edge", "rat_edge"],
                    help="Edge class names to cycle through.")
    ap.add_argument("--patch-size", type=int, default=500,
                    help="Side length (px) of the random patch shown.")
    ap.add_argument("--win", type=int, default=48,
                    help="Side length (px) of the stats window around "
                         "the click point.")
    ap.add_argument("--scales", type=int, nargs="+",
                    default=[5, 9, 13],
                    help="Gabor scales (must match what you intend to "
                         "use in the detector for comparable stats).")
    ap.add_argument("--n-orientations", type=int, default=8)
    ap.add_argument("--color-local", type=int, default=3,
                    help="Side length (px) of the small local colour average "
                         "at the click; keep small (3-5) so thin structures "
                         "like the cable aren't averaged into the background.")
    ap.add_argument("--smooth-k", type=int, default=7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--window", default="texture_edge_sampler")
    ap.add_argument("--out", default=None,
                    help="Output directory for samples.csv + provenance "
                         "patches (required for sampling; not used by "
                         "--seed-thresholds).")
    ap.add_argument("--session", default=None,
                    help="Session id recorded with every sample (e.g. the "
                         "capture timestamp 20260214_021722). Defaults to the "
                         "cam0 filename stem; set it so frame indices are "
                         "unambiguous across sessions.")
    args = ap.parse_args(argv)

    # Aggregation mode: derive shape thresholds from a labeled CSV, no GUI.
    if args.seed_thresholds:
        import json
        res = seed_thresholds_from_csv(
            args.seed_thresholds,
            rat_classes=args.rat_classes,
            cable_classes=args.cable_classes,
            margin=args.seed_margin)
        print(f"labeled samples: rat={res['n_rat']} "
              f"cable={res['n_cable']}")
        print("per-feature separation (rat vs cable):")
        for k, f in res["features"].items():
            if f["rat_mean"] is None or f["cable_mean"] is None:
                print(f"  {k:11}: (insufficient samples)")
                continue
            print(f"  {k:11}: rat {f['rat_mean']:.2f}±{f['rat_std']:.2f}"
                  f"  cable {f['cable_mean']:.2f}±{f['cable_std']:.2f}"
                  f"  d'={f['dprime']:.2f}")
        if "min_solidity" in res:
            print("\nproposed tracker thresholds (pass to the probe):")
            print(f"  --track-min-solidity {res['min_solidity']:.2f} "
                  f"--track-max-elongation {res['max_elongation']:.1f}")
        else:
            print("\n(need >=2 rat AND >=2 cable samples to propose "
                  "thresholds; label some cable/headstage clicks.)")
        return

    if not args.cam0:
        ap.error("--cam0 is required for sampling "
                 "(or use --seed-thresholds for aggregation).")
    if not args.out:
        ap.error("--out is required for sampling "
                 "(or use --seed-thresholds for aggregation).")

    os.makedirs(args.out, exist_ok=True)

    from rpimocap.io.export import TiffCapture
    from rpimocap.detection.rat_texture import build_gabor_kernels

    orientations = [i * np.pi / args.n_orientations
                    for i in range(args.n_orientations)]
    n_orient = len(orientations)
    n_scales = len(args.scales)
    kernels = build_gabor_kernels(orientations, args.scales)

    caps = {}
    cap0 = TiffCapture(args.cam0, bayer_pattern=args.bayer_pattern)
    caps[0] = (cap0, int(cap0.get(cv2.CAP_PROP_FRAME_COUNT)))
    if args.cam1:
        cap1 = TiffCapture(args.cam1, bayer_pattern=args.bayer_pattern)
        caps[1] = (cap1, int(cap1.get(cv2.CAP_PROP_FRAME_COUNT)))

    classes = list(args.texture_classes) + list(args.edge_classes)
    kinds = (["texture"] * len(args.texture_classes)
             + ["edge"] * len(args.edge_classes))

    print("Classes:")
    for c, k in zip(classes, kinds):
        print(f"  [{k}] {c}")
    print("\nControls: LEFT-click=save  RIGHT-click=skip  "
          "SPACE=next image  Tab=next texture class  n=next(any)  "
          "p=prev  u=undo  s=save  r=next image  "
          "q/ESC=quit\n")

    sess = _Session(args, caps, kernels, n_orient, n_scales,
                    classes, kinds)
    # WINDOW_GUI_NORMAL disables the Qt "expanded" GUI (toolbar, status
    # bar, and the right-click context menu). Without it, right-click —
    # which we use to skip to a new patch — also pops the Qt context
    # menu. Fall back to plain WINDOW_NORMAL on non-Qt builds where the
    # flag isn't defined.
    try:
        flags = cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL
    except AttributeError:                     # pragma: no cover
        flags = cv2.WINDOW_NORMAL
    cv2.namedWindow(args.window, flags)
    cv2.setMouseCallback(args.window, sess.on_mouse)
    sess.new_patch()

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key in (ord("q"), 27):        # q or ESC
            break
        elif key == ord(" "):            # SPACE → next image/patch
            sess.new_patch()
        elif key == 9:                   # Tab → next texture class (same kind)
            sess.next_class_of_kind(+1)
        elif key == ord("n"):            # n → next class (any kind)
            sess.next_class(+1)
        elif key == ord("p"):
            sess.next_class(-1)
        elif key == ord("u"):
            sess.undo()
        elif key == ord("s"):
            sess.flush()
        elif key == ord("r"):            # r → also next image (kept)
            sess.new_patch()
        # window closed?
        if cv2.getWindowProperty(args.window,
                                 cv2.WND_PROP_VISIBLE) < 1:
            break

    sess.flush()
    cv2.destroyAllWindows()
    # session summary
    total = sum(sess.saved_count.values())
    print(f"\nDone. {total} samples this session:")
    for c in classes:
        print(f"  {c}: {sess.saved_count[c]}")
    print(f"CSV: {sess.csv_path}")


if __name__ == "__main__":
    main()
