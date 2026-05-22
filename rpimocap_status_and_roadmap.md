# rpimocap — Status and Roadmap

_Version 0.5.0 — 152 passing tests_

## Current state

The pipeline supports a full stereo capture → calibrate → align →
detect → triangulate → smooth → export workflow, with optional
refraction correction through acrylic arena walls and a rich set of
detection / centroid-refinement / outlier-rejection tools layered on
top.

### Shipped components

| Module                                          | Status   | Tests |
|-------------------------------------------------|----------|------:|
| `rpimocap-align` GUI (Kabsch + plumb-line)      | shipped  | —     |
| `TiffCapture` (streaming multi-frame TIFF)      | shipped  | covered |
| Multi-Pi coordinator (SSH, session-IDs)         | shipped  | —     |
| `rpimocap-calibrate`, `rpimocap-autocalib`      | shipped  | 23    |
| Triangulation (DLT + trajectory utilities)      | shipped  | 18    |
| Voxel carving                                   | shipped  | covered |
| Planar refraction correction (opt-in)           | shipped  | 26    |
| Kalman tracker (online, per-frame)              | shipped  | 7     |
| Kalman / RTS smoother (offline, trajectory)     | shipped  | 6     |
| Temporal background adaptation                  | shipped  | 4     |
| Trajectory-constrained blob selection           | shipped  | 2     |
| Undistort-before-epipolar matching              | shipped  | 2     |
| Anatomical Gaussian shape prior                 | shipped  | 8     |
| Gabor-edge body contour                         | shipped  | 9     |
| NIR vignette / flat-field correction            | shipped  | 11    |
| Rearing-state classifier                        | shipped  | 9     |
| SAM2 video-propagation scaffold                 | scaffold | 4     |
| Detected-frame mask in HDF5                     | shipped  | 5     |
| Pose detection (`CentroidPoseDetector`)         | placeholder | — |

### `hull_centroid` pipeline

The centroid refinement pipeline is now six steps deep:

```
1. Find the connected-component label owning (cx, cy)
2. Extract that blob as a binary mask
3. Cable erosion                (cable_erosion_px > 0)
3b. Gabor-edge body contour     (gabor_refine=True)
4. Ellipse fit on body pixels → orientation θ + first centroid
5. Anatomical Gaussian prior    (P + body_length_mm > 0)
6. Hull-centroid fallback
```

Steps 3b and 5 are the substantive 0.5.0 additions for the bedding-
disturbance and cable-contamination edge cases.

---

## Roadmap

### ✅ Phase 1 — End-to-end field validation
Accuracy gates: reprojection RMSE < 3 px, triangulated arena
dimensions within 5 mm of physical measurement.

### ✅ Phase 2 — Planar refraction correction (v0.3.0)
Acrylic walls modelled as parallel-faced slabs. Calibration unmodified
because corners sit on the outer face. **Off by default** since the
standard rig has no plexiglass between cameras and animal.

### ✅ Phase 2.5 — Centroid robustness pass (v0.5.0)
Anatomical prior, Gabor body contour, Kalman/RTS smoother, detected
mask, temporal background adaptation, trajectory-constrained selector,
undistort-before-epipolar, NIR vignette correction.

### ⏳ Phase 3 — SLEAP whisker detection
Replace `CentroidPoseDetector` placeholder with a SLEAP-trained pose
model for whisker-scale tracking. Triangulation contract already in
place — this is a detection-backend swap. **Most scientifically
impactful next step.**

### ⏳ Phase 4 — Performance / GPU acceleration
Profile and accelerate processing of 50 GB+ TIFF sessions. Target
hardware: NVIDIA 5070 Ti. Likely hotspots: voxel-carving silhouette
projection, per-frame Gabor energy, batch detection.

### ⏳ Phase 5 — N-view generalisation
Extend stereo → 3–4 camera multi-view DLT. Closest-point-of-two →
least-squares-of-N. Refraction module already operates per-ray, so
wall finding generalises naturally.

### Pending integration (small follow-ups)

These ship the APIs but the per-frame wiring is open:

- **SAM2VideoTracker (0007)** — wire into `SegmentTracker._process_frame`
  as an alternative to optical-flow seeding (~30 lines).
- **`KalmanTracker3D` (0003)** — online filter; wire into the detector
  loop alongside the offline Kalman/RTS smoother.
- **Rearing classifier (0008)** — hook into `GeometricLabeller` so the
  vertical-posture anatomical prior is swapped in during rear bouts.

---

## Notable design choices

- **Calibration corners on the outer wall face.** Lets refraction be
  applied at triangulation time without re-calibrating; absent the
  refraction model the pipeline produces valid (if slightly biased)
  3D points.
- **Parallel-slab refraction is single-shot.** Because the refracted
  in-arena ray is parallel to the original camera ray, the wall a ray
  crosses is determined by the camera observation alone, not by the
  unknown 3D point. Refractive triangulation is therefore a one-step
  closest-point-of-two-shifted-lines computation rather than an
  iterative solve.
- **Detected mask captured pre-post-processing.** The CLI computes the
  `detected_masks` dict immediately after `results_to_skeleton_frames()`
  and before any smoothing / Kalman / gap-fill, so post-processing
  modifications cannot corrupt the mask.
- **Numbered-patch workflow.** Feature additions land as
  `git format-patch`-compatible patches numbered consecutively
  (`patches/0001-...` etc.). The CHANGELOG and this status doc map
  patch numbers → user-visible changes.
