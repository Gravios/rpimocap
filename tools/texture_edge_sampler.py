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
  n / SPACE → next class        p → previous class
  u         → undo last save     s → save CSV now
  r         → new random patch   q / ESC → quit (auto-saves)

The saved stats live in the SAME Gabor descriptor space as the detector,
plus intensity, gradient/structure-tensor, and (for edges) a cross-edge
contrast/sharpness profile — exactly the features that separate a sharp
maze rail from a soft rat outline.

Output
------
  <out>/samples.csv            one row per saved click (appends across
                               sessions)
  <out>/patches/<class>_<n>.png  the patch with the click marked
                               (provenance)

Requires a GUI build of OpenCV (opencv-python, NOT -headless) and a
display (works over X-forwarding).

Example
-------
  python tools/texture_edge_sampler.py \
      --cam0 "$cam0_tif" --cam1 "$cam1_tif" \
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
    "cam", "frame", "x", "y", "klass", "kind", "win",
    # intensity stats over the window
    "int_mean", "int_std", "int_min", "int_max",
    "int_p10", "int_p50", "int_p90",
    # gradient / structure-tensor stats over the window
    "grad_mag_mean", "grad_mag_max", "orient_deg", "coherence",
    # cross-edge profile (meaningful for edges; computed for all)
    "edge_contrast", "edge_width_px",
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
        smooth_k=smooth_k, rotation_invariant=True)   # (D, H, W)
    cy = int(np.clip(cy, 0, desc.shape[1] - 1))
    cx = int(np.clip(cx, 0, desc.shape[2] - 1))
    return desc[:, cy, cx].astype(np.float32)


def extract_record(gray: np.ndarray, cx: int, cy: int,
                   klass: str, kind: str, win: int,
                   kernels, n_orient: int, n_scales: int,
                   cam: int, frame_idx: int,
                   smooth_k: int = 7) -> dict:
    """Build one labeled record for a click at (cx, cy) on `gray`.

    `gray` here is the full patch (already cropped). (cx, cy) are in
    patch coordinates. `kind` is 'texture' or 'edge'.
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
    st = structure_tensor_stats(window)
    rec.update(st)
    rec.update(cross_edge_profile(gray, cx, cy, st["orient_deg"]))
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
        self.cam = 0
        self.frame_idx = 0
        self.px0 = self.py0 = 0          # patch origin in the frame
        self.disp = None                 # BGR display image
        self.saved_count = {c: 0 for c in classes}
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
            smooth_k=self.args.smooth_k)
        # record absolute frame coordinates too
        rec["x"] = int(self.px0 + x)
        rec["y"] = int(self.py0 + y)
        if self.n_desc is None:
            self.n_desc = rec["_n_desc"]
        self.records.append(rec)
        self.saved_count[klass] += 1
        # provenance patch png with the click marked
        self._render(marker=(x, y))
        n = self.saved_count[klass]
        pth = os.path.join(self.args.out, "patches",
                           f"{klass}_{n:04d}.png")
        cv2.imwrite(pth, self.disp)
        cv2.waitKey(120)                 # brief flash of the marker
        self.new_patch()

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


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cam0", required=True)
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
    ap.add_argument("--smooth-k", type=int, default=7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--window", default="texture_edge_sampler")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

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
          "n/SPACE=next class  p=prev  u=undo  s=save  r=new patch  "
          "q/ESC=quit\n")

    sess = _Session(args, caps, kernels, n_orient, n_scales,
                    classes, kinds)
    cv2.namedWindow(args.window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(args.window, sess.on_mouse)
    sess.new_patch()

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key in (ord("q"), 27):        # q or ESC
            break
        elif key in (ord("n"), ord(" ")):
            sess.next_class(+1)
        elif key == ord("p"):
            sess.next_class(-1)
        elif key == ord("u"):
            sess.undo()
        elif key == ord("s"):
            sess.flush()
        elif key == ord("r"):
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
