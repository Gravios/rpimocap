# Topological Rat Detection

A hand-crafted 2D detector for a white-furred rat on granular bedding under
infrared illumination, with epipolar-consistent stereo triangulation to 3D.
It lives in `rpimocap.detection.topo_detect` and has a command-line driver at
`tools/topo_track.py` (wrapper: `tools/topo_track.sh`).

**Best for:** a low-contrast subject whose *texture* differs from the
background even when its *brightness* does not — a smooth-furred rat on
countable-grain bedding. No labelled training data required. For an eventual
keypoint pose (whisker-scale), pair this with a learned detector
(`SLEAP`/`DLC` via [detectors.md](detectors.md)); this module solves the
body-localization and stereo-correspondence problem that a blob detector
cannot.

---

## Why not intensity?

Every intensity- or energy-based feature we evaluated plateaus at roughly
**1–2σ** of separation between the rat and the bedding: raw brightness, Gabor
energy at all scales, Laplacian-of-Gaussian amplitude, and median high-pass.
The white fur and the wood-shaving bedding are similar in brightness and even
in local contrast — and on the lower-contrast camera the rat measures as
*less* anomalous than the bedding. A background-subtraction centroid
(`CentroidPoseDetector`) also struggles: shadows, reflections, and the tether
cable all produce foreground blobs.

The discriminating fact is **topological, not photometric**.

## The topological insight

Bedding is a field of *many countable grains*; the rat's fur is *internally
smooth*. Measuring grain **density** (how many discrete texture features are
present) rather than grain **amplitude** (how strong they are) separates the
rat at **−4 to −7σ** and holds on both cameras. Counting is robust where
amplitude is not: the rat has few grain-scale features, bedding has many.

## Pipeline

```
              per view                                         stereo
  ┌───────────────────────────────────────────┐        ┌──────────────────┐
  gray ─▶ median bandpass ─▶ grain-count map ──┼─▶ localize (low = rat)   │
                     │                          │   └─ seed blob + top-K   │
                     │                          │      candidates ─────────┼─▶ epipolar
                     ├─▶ body-scale −LoG ───────┼─▶ centroid (on the body) │   match
                     │   (+ cable-suppressed    │                          │   (best_stereo_point)
                     │    fur map, optional)    │                          │      │
                     └─▶ circle-grow segment ───┼─▶ silhouette mask        │      ▼
                         (texture barrier)      │                          │   triangulate + gate
  ┌───────────────────────────────────────────┘        └──────────────────┘   → 3D point
```

Everything downstream of the median bandpass lives in that one representation.

---

## The median bandpass space

```python
mbp = median_bandpass(gray, small_k=3, large_k=21)
```

`medianBlur(k=3) − medianBlur(k=21)`. The median (not Gaussian) difference is
robust to the bright rat and to speckle, and the small/large kernels straddle
the grain scale: bedding becomes a busy field of grains, smooth fur cancels to
≈0. The median bandpass separates rat/bedding ~1.5× better than a Gaussian
difference-of-Gaussians.

## Localization — the grain-count map

```python
gc = grain_count_map(mbp, patch=112, peak_frac=0.5)
```

Counts **strict local maxima** of the median bandpass inside a `patch`×`patch`
window, computed as a single box filter of the strict-maxima map — O(N) for
any patch size, no per-patch labeling. The result is **low over the smooth
rat, high over bedding**.

- **Strict** maxima (`peak > neighbourhood min`) are essential: a naive
  `peak == maximum_filter` flags every pixel of a flat region as a maximum,
  which *inverts* the signal over the smooth rat.
- A larger `patch` tightens the bedding distribution (bigger separation) at
  the cost of localization resolution. ~112 px is the sweet spot for a
  ~250 px rat.

The rat is the largest low-grain blob inside the arena floor ROI. The top
`max_candidates` blobs (best first) are kept as `Detection.candidates` for the
stereo match.

## Centroid — body-scale −LoG, and cable suppression

The default centroid is the peak of a **body-scale Mexican hat** (`body_blob`,
σ≈80 px): a bright, body-sized blob produces a positive peak at its centre, so
the centroid sits on the animal rather than drifting toward the thin cable.

For a sharper centroid that actively suppresses the cable, enable
**cable suppression** (`cable_suppress=True`):

```python
mix = cable_suppressed_map(gray, mbp, floor_mask,
                           illum_sigma=201.0, barrier_sigma=32.0)
```

The `|Laplacian|` texture barrier cannot separate the rat from the tether
(both are smooth), but *intensity* can — the rat is bright, the cable dark.
`cable_suppressed_map` mixes the **inverted, illumination-flattened,
floor-normalized intensity** 50/50 with a σ-32 smoothed `|Laplacian|` barrier:

- the rat is low in **both** terms (smooth *and* bright),
- the cable is low in the barrier but **high** in the inverted intensity (it's
  dark), so it averages up into the bedding cluster and stops dragging the
  centroid.

The pieces were swept on real frames: illumination map σ=201 (wider than the
rat so it isn't muted), normalization to the *floor* range (so the bright
rails and gloves don't set the scale), and a σ=32 barrier — the value that
maximized the rat's separation while minimizing the centroid offset (measured
~64–67 px vs the −LoG peak's ~70–78 px). Cable suppression is **frame-
dependent** — it helps where the cable is the dominant confound but can pick a
worse candidate elsewhere — so it is opt-in.

## Segmentation — growing circles against a texture barrier

```python
mask = circle_grow_segment(seed_blob, barrier, floor_mask, barrier_pct=55.0)
```

Circles are seeded across the localized blob and grown outward to the distance
to the nearest barrier (a distance transform of the smooth region); their
union is the silhouette. Because the barrier is **texture, not brightness**,
this holds on the low-contrast camera where a brightness flood-fill leaks.
`barrier_pct` is the one knob: raise it to let circles reach further before
the barrier stops them.

### Barrier choices (`seg_barrier=`)

| Value | Barrier | Character |
|-------|---------|-----------|
| `grain` (default) | grain-count map | robust texture boundary |
| `laplacian` | `Gaussian(\|Laplacian(mbp)\|)` — σ-robust energy | contiguous; forgiving σ |
| `both` | grain **AND** laplacian | tightest; closes single-measure leaks |
| `fur` | the cable-suppressed map | limits the border to the **bright + smooth body**, drops the cable; fills a weak-contrast view but can over-grow where other bright-smooth patches exist |
| `grain+fur` | grain **AND** fur | reins the fur's over-grow at the cost of re-under-segmenting |

The two cameras tend to have opposite tendencies (one over-grows, one under),
so the barrier is selectable rather than fixed. For an under-segmenting view,
reach for `fur` before raising `barrier_pct`.

## Stereo — epipolar-consistent matching

```python
result = detect_stereo(gray0, gray1, floor0, floor1, dlt_P0, dlt_P1,
                       max_epipolar_px=60.0, max_reproj_px=60.0)
```

Rather than blindly triangulating each view's single best blob — which fails
when the two do not correspond — `detect_stereo` passes the per-view
**candidate lists** to the reconstruction epipolar matcher
(`best_stereo_point`). The chosen pair must (a) lie on each other's epipolar
line within `max_epipolar_px`, (b) triangulate inside the arena, and
(c) reproject within `max_reproj_px`. Single-view false positives — including
the floor reflection, which has no consistent partner — are dropped.

Tolerances default **loose (60 px)** because the topology centroid is the
rat's approximate centre (~70 px), not a sub-pixel keypoint. Tighten them once
a sharper centroid (`cable_suppress=True`) is in use; a tighter
`--max-reproj-px` (15–20) also rejects implausible "ceiling ghost" matches.

The returned `StereoResult` reports the **matched** pixel points (`pt0`,
`pt1`) — the centroids that actually produced the 3D point — which are what
the CLI logs and the overlay draws, so the drawn dot, the CSV, and the 3D
point always agree.

---

## Python API

| Function | Purpose |
|----------|---------|
| `median_bandpass(gray, small_k=3, large_k=21)` | the base representation |
| `grain_peaks(mbp, peak_frac=0.5)` | strict-maxima grain map |
| `grain_count_map(mbp, patch=112, peak_frac=0.5)` | grain density (rat = low) |
| `body_blob(gray, sigma=80.0)` | body-scale −LoG (centroid) |
| `laplacian_magnitude(mbp, sigma=3.0)` | σ-robust texture-energy barrier |
| `combine_barriers(maps, floor_mask)` | z-score + elementwise-max (AND-of-lows) |
| `cable_suppressed_map(gray, mbp, floor_mask, illum_sigma=201, barrier_sigma=32)` | fur map; rat = low, cable → bedding |
| `circle_grow_segment(seed_blob, barrier, floor_mask, barrier_pct=55)` | silhouette |
| `build_floor_mask(dlt_P, arena_corners, image_shape, mode="volume", max_height_mm=260)` | projected arena ROI |
| `detect(gray, floor_mask, ...) -> Detection` | single-view detection |
| `detect_stereo(gray0, gray1, floor0, floor1, dlt_P0, dlt_P1, ...) -> StereoResult` | epipolar stereo → 3D |

```python
@dataclass
class Detection:
    found: bool
    centroid: tuple[float, float] | None   # (x, y) px — best candidate
    mask: np.ndarray                       # segmented silhouette (uint8)
    seed_blob: np.ndarray                  # low-grain-count localization
    separation: float                      # rat vs bedding grain sigma (< 0)
    candidates: list                       # top-K (x, y), best first

@dataclass
class StereoResult:
    point: np.ndarray | None               # (3,) arena-mm 3D point
    accepted: bool                         # epipolar-consistent + in-arena
    reproj_err: float                      # max per-view reprojection error (px)
    det0: Detection
    det1: Detection
    pt0: tuple[float, float] | None        # cam0 centroid that produced point
    pt1: tuple[float, float] | None        # cam1 centroid that produced point
```

### Minimal example

```python
import numpy as np
from rpimocap.detection.topo_detect import detect_stereo, build_floor_mask

cal = np.load("calib_from_corners.npz")            # arena-registered DLT
P0, P1 = cal["dlt_P0"], cal["dlt_P1"]
arena = np.array([[-140,-215,0],[140,-215,0],[140,215,0],[-140,215,0],
                  [-140,-215,388],[140,-215,388],[140,215,388],[-140,215,388]], float)
fl0 = build_floor_mask(P0, arena, gray0.shape, mode="floor")
fl1 = build_floor_mask(P1, arena, gray1.shape, mode="floor")

R = detect_stereo(gray0, gray1, fl0, fl1, P0, P1, cable_suppress=True)
if R.accepted:
    print("3D:", R.point, "reproj", R.reproj_err, "px")
```

> **Calibration:** `topo_track` needs the **arena-registered DLT** matrices
> `dlt_P0`/`dlt_P1` (arena mm → pixel, e.g. `calib_from_corners.npz`). The
> standard projection matrices in `autocalib.npz` use a different coordinate
> frame and will make every triangulation wrong.

---

## Command line — `topo_track`

```bash
tools/topo_track.sh CAM0.tif CAM1.tif CALIB.npz [OUT.csv] [flags...]
```

The wrapper takes the two camera TIFFs, the calibration, and an optional
output path positionally; remaining flags pass through to `tools/topo_track.py`
(it runs from a plain checkout — the repo is put on `PYTHONPATH`). Each frame's
matched centroids are triangulated and gated, writing a track CSV:

```
frame, found, cam0_cx, cam0_cy, cam1_cx, cam1_cy, X_mm, Y_mm, Z_mm,
accepted, reproj_px, sep0, sep1
```

`cam*_cx/cy` are the **matched** pair behind `X_mm` (for accepted rows), so the
CSV, the 3D point, and the overlay always describe the same detection.

### Flags

| Flag | Default | Meaning |
|------|---------|---------|
| `--roi-mode` | `volume` | `box` \| `floor` \| `volume` arena ROI |
| `--roi-max-height-mm` | `260` | volume ROI height band |
| `--patch` | `112` | grain-count window px (detection scale) |
| `--blob-sigma` | `80` | body-scale −LoG σ for the centroid |
| `--seg-barrier` | `grain` | `grain` \| `laplacian` \| `both` \| `fur` \| `grain+fur` |
| `--barrier-pct` | `55` | segmentation barrier percentile |
| `--barrier-sigma` | `3` | Gaussian σ for the Laplacian barrier |
| `--cable-suppress` | off | cable-suppressed (invert-mix) centroid |
| `--max-epipolar-px` | `60` | max symmetric epipolar distance for a match |
| `--max-reproj-px` | `60` | max reprojection error for a match |
| `--overlay-dir` | — | write a per-frame detection overlay PNG here |
| `--start` / `--stride` / `--max-frames` | `0` / `1` / all | frame range |

### Inspecting a run

```bash
tools/topo_track.sh raw/cam0.tif raw/cam1.tif calib_from_corners.npz out.csv \
    --roi-mode floor --overlay-dir overlays --stride 2 --max-frames 1000
```

`--overlay-dir` writes `overlays/frame_NNNNNN.png`: cam0 | cam1 side by side
with the mask (green), the matched centroid (cyan; grey if a view had no
accepted match), the candidate blobs (orange), and a banner with the 3D point
/ accepted flag / reprojection error. The CSV says *what* was accepted; the
overlays show *why* — a bad match looks bad because the dot sits exactly where
the 3D point came from.

---

## Validation & known limitations

Validated on real session 021722 frames (calibration
`calib_from_corners.npz`, anchor: arena (0, 195) → cam0 px (1385, 630), cam1 px
(655, 615)):

- **−4 to −7σ** rat/bedding separation; grain-count localizes on both cameras.
- **10/10** detection over the grooming dwell sequence, centroid stable to
  ~1 px across frames on both views.
- Cable-suppressed centroids ~64–67 px from the anchor (vs ~70–78 px).
- Stereo: the two views are epipolar-consistent to ~5 px; the matcher picks
  the rat and drops false candidates; the triangulated point sits near the
  anchor in x/y.

Known limitations:

- **Depth (z) is the soft axis.** A ~60 px centroid error projects mostly into
  depth, so x/y land near the anchor while z can be off by tens of mm. The
  reprojection error confirms the *pair* is consistent; z accuracy is a
  centroid-precision limit, not a matching error.
- **Dwell only.** All real-frame validation is on a stationary (grooming)
  sequence. Per-frame detection is independent, so it should carry to motion,
  but tracking a *moving* rat is unconfirmed — run a short moving clip through
  the CLI with `--overlay-dir` and scrub the overlays to confirm.
- **`--cable-suppress` is frame-dependent.** It sharpens the centroid where
  the cable dominates but can select a worse candidate elsewhere (an
  implausible high/"ceiling" match). Leave it off by default and A/B per
  session; a tighter `--max-reproj-px` rejects such ghosts.
- **Compute** is ~0.5 s/frame, dominated by the full-resolution median. Fine
  for a batch over ~12 k frames (~100 min); trimmable with a downsampled
  median if needed.
- **Body only.** This detector returns a body centroid and silhouette, not a
  keypoint skeleton. For whisker-scale pose, feed the localized crop to a
  learned detector.
