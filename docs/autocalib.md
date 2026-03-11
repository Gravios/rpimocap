# Self-Calibration

`rpimocap-autocalib` estimates camera intrinsics (focal length) from the
subject's own motion inside a known arena, with no calibration target. The
output `.npz` is drop-in compatible with `rpimocap-run`.

---

## When to use self-calibration

| Scenario | Use autocalib? |
|----------|---------------|
| Checkerboard unavailable | Yes |
| Retrospective calibration from existing footage | Yes |
| Cameras repositioned mid-session | Yes, per session |
| Strong barrel distortion (action cameras, wide-angle) | Not ideal — use checkerboard |
| Cross-validating a checkerboard calibration | Yes, with `--compare` |

Self-calibration assumes zero lens distortion. For cameras with moderate
distortion (standard C-mount lenses, typical webcams) this is acceptable;
for wide-angle or fisheye lenses, use the result as an initial guess for
`rpimocap-calibrate`.

---

## Algorithm overview

The pipeline runs three sequential stages, all implemented in
`rpimocap.calibration.autocalib`.

### Stage 1 — Feature matching and F estimation

`autocalib_features.py` / `rpimocap.calibration.autocalib.features`

For a sample of frame pairs drawn from both videos:

1. **SIFT extraction** — 4 000 features per frame, contrast threshold 0.025.
2. **Lowe ratio test** — threshold 0.72; rejects ambiguous matches.
3. **Spatial grid filter** — 4×4 cells, max 30 matches/cell, so the
   correspondence cloud spans the whole frame rather than clustering in
   one textured region.
4. **RANSAC fundamental matrix estimation** — threshold 1.5 px, confidence
   0.9995. Produces a `FundamentalEstimate` with an inlier count, inlier
   ratio, and mean Sampson distance.
5. **Quality scoring** — a weighted combination of inlier count (35%),
   inlier ratio (35%), and Sampson distance (30%). Estimates below a
   quality threshold are discarded.
6. **Temporal diversity** — when subsampling, frame pairs are selected to
   maximise temporal spread, so diverse camera motion is captured.

### Stage 2 — Kruppa focal estimation

`autocalib_kruppa.py` / `rpimocap.calibration.autocalib.kruppa`

Given shared cameras (K₀ = K₁), zero skew, square pixels, and principal
point at the image centre, the intrinsic matrix has **one degree of
freedom**: the focal length *f*.

For each fundamental matrix F and candidate *f*:

1. Form K(*f*) and compute E = Kᵀ F K.
2. Evaluate the essential-matrix constraint (Kruppa cost):

   ```
   cost = ‖2EEᵀE − tr(EEᵀ)·E‖_F / ‖E‖³
   ```

   This is identically zero for a valid essential matrix. The cost is
   minimised by the *f* that is most consistent with the epipolar geometry.

3. A 1D scan over *f* ∈ [0.4×diag, 3.5×diag] followed by Brent's method
   finds the cost minimum to sub-pixel precision.

4. A quality-weighted average across all frame-pair estimates gives a robust
   pooled focal length.

### Stage 3 — Arena metric refinement

If `--bounds` is supplied:

1. The best E is decomposed into R and t (with cheirality test).
2. All inlier correspondences are triangulated using DLT.
3. The 2nd–98th percentile extent of the point cloud is compared to the
   known arena span.
4. A metric scale correction is applied:

   ```
   f_refined = f_kruppa × √(arena_span / reconstruction_span)
   ```

   The median of per-axis scale factors is used for robustness.

### Optional Stage 4 — Bundle refinement

`--refine-bundle` runs Nelder-Mead optimisation of the mean reprojection
error over all inlier correspondences, starting from the Kruppa estimate.
This adds ~10 s and rarely improves by more than 5 px on well-behaved data.

---

## Command reference

```
rpimocap-autocalib [OPTIONS]

Required
  --cam0    PATH              Camera 0 video
  --cam1    PATH              Camera 1 video
  --bounds  xmin,xmax,...     Arena extents in mm (6 floats)

Optional
  --n-frames    INT           Maximum frames to sample (default: 2000)
  --min-pairs   INT           Minimum F estimates required (default: 20)
  --refine-bundle             Run Nelder-Mead bundle refinement after Kruppa
  --compare     PATH          Checkerboard .npz to cross-validate against
  --out         PATH          Output .npz (default: autocalib.npz)
  --report      PATH          Write HTML validation report to this path
  --quiet                     Suppress progress output
```

### Bounds format

```
--bounds "xmin,xmax,ymin,ymax,zmin,zmax"
```

All values in millimetres. Example for a 600 mm × 400 mm × 350 mm arena:

```
--bounds "-300,300,-200,200,0,350"
```

The bounds define the world coordinate system. The origin is typically
the floor-centre of the arena.

---

## Output .npz schema

Identical to `rpimocap-calibrate` output so the two are interchangeable:

| Key | Shape | Description |
|-----|-------|-------------|
| `K0`, `K1` | (3, 3) | Shared intrinsic matrix (K0 == K1) |
| `dist0`, `dist1` | (1, 5) | Zero distortion coefficients |
| `R`, `T` | (3,3), (3,) | Stereo extrinsics from best E decomposition |
| `E`, `F` | (3, 3) | Essential / fundamental matrices |
| `P0`, `P1` | (3, 4) | Projection matrices (not rectified) |
| `image_size` | (2,) | (width, height) in pixels |
| `f_px` | scalar | Final focal length estimate in pixels |
| `f_metric` | scalar or None | Metric-refined focal length |
| `n_estimates` | int | Number of F estimates used |
| `mean_residual` | scalar | Mean Kruppa cost at the solution |

---

## Validation report

`--report autocalib_report.html` generates a standalone HTML page (no
external dependencies) containing:

- **Cost landscape** — the 1D Kruppa cost scan showing the minimum at the
  estimated focal length.
- **Per-frame quality scatter** — inlier count vs. Sampson distance coloured
  by quality score; reveals whether estimates cluster or scatter.
- **Inlier / Sampson scatter** — distribution of match quality across the
  frame population.
- **Quality gates** — pass/fail summary:
  - ≥ 20 valid F estimates
  - Plausible FOV (focal length within the scan range)
  - Mean essential residual < 0.05
  - Metric refinement applied (if bounds were supplied)

Open in any browser; no server required.

---

## Cross-validation with a checkerboard calibration

```bash
rpimocap-autocalib \
    --cam0 session_cam0.mp4 \
    --cam1 session_cam1.mp4 \
    --bounds "-300,300,-200,200,0,350" \
    --compare calibration.npz \
    --out autocalib.npz \
    --report comparison_report.html
```

The report will include a table comparing focal length and principal point
between the two methods. Agreement within 2–5% is typical for standard
lenses. Larger discrepancies usually indicate distortion that the
self-calibration model cannot capture.

---

## Library API

```python
from rpimocap.calibration.autocalib import (
    CrossViewMatcher,
    sample_frame_pairs,
    filter_estimates,
    run_focal_estimation,
)
import cv2

cap0 = cv2.VideoCapture("cam0.mp4")
cap1 = cv2.VideoCapture("cam1.mp4")

matcher = CrossViewMatcher()
pairs = sample_frame_pairs(cap0, cap1, n_pairs=60)
estimates = [matcher.match(f0, f1, idx) for idx, f0, f1 in pairs]
estimates = filter_estimates(estimates, min_quality=0.35)

result = run_focal_estimation(
    estimates,
    image_size=(1280, 720),
    arena_bounds=((-300, 300), (-200, 200), (0, 350)),
    verbose=True,
)

print(f"f = {result.f_px:.1f} px")
print(result.K)
```

---

## Troubleshooting

**"Not enough F estimates (got N, need ≥ 20)"**
— The subject lacks texture (white fur, uniform coat), the arena lighting
is too uniform, or the subject barely moves in frame. Try increasing
`--n-frames`, improving arena contrast, or switching to a checkerboard.

**Focal length estimate wildly wrong (< 400 px or > 5000 px for a 1280-wide sensor)**
— Check that the two video files are genuinely synchronised. If cam0 and
cam1 are offset by even a fraction of a second, the correspondences will
be incorrect and the fundamental matrices degenerate.

**High essential residual (> 0.15) despite low Sampson distance**
— The cameras may not share the same focal length (e.g. different zoom
settings). Self-calibration assumes K₀ = K₁; mismatched zoom will
produce a biased estimate. Set both cameras to the same focal length.

**Arena metric refinement not applied**
— Triangulation from a near-degenerate E (small baseline or near-planar
scene) fails the cheirality test. Increase the physical camera separation
or ensure the subject moves in 3D rather than primarily in a plane.
