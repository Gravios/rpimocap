# Changelog

All notable changes to `rpimocap`. Entries are grouped by release;
within a release each bullet references the underlying patch number
(see `git log --oneline` for the matching commit).

## [0.5.0] — 2026-05-22

The "post-v0.3.0 features" release. Bundles two patch series:

- **0001–0009** — future-feature series rooted at upstream HEAD
  before refraction. Tracker/detector additions and infrastructure.
- **0038–0041** — body-refine pipeline and HDF5 schema additions.

Test suite grew from 85 (0.3.0 baseline) to 152 passing.

### Centroid refinement

- **0038** — anatomical Gaussian shape prior (step 5 of `hull_centroid`).
  Uses the DLT projection matrix `P` to compute pixel/mm scale at the
  ellipse-fit position, builds a rotated body-shaped weight map at
  the known `body_length_mm × body_width_mm` dimensions, ANDs it with
  the foreground blob, and returns the weighted-centroid of the
  intersection. Suppresses outliers from cable tips and bedding-edge
  artefacts. New CLI: `--body-length MM`, `--body-width MM`, `--body-z MM`.
- **0039** — Gabor-edge body contour (step 3b of `hull_centroid`).
  Finds the rat outline in texture space using Canny edges on the
  Gabor energy map. Updates both the body-pixel list (consumed by
  step 4 ellipse fit) and the eroded blob mask (consumed by step 5
  anatomical prior), so the prior sees the texture-refined body
  region. New CLI: `--gabor-refine`, `--canny-low T`, `--canny-high T`.
  New `gabor_energy` field on `ForegroundResult`. New method
  `ForegroundDetector.gabor_body_contour()`.

### Detection improvements

- **0004** — EMA temporal background adaptation. `BackgroundModel.update()`
  EMA-updates the background outside a foreground mask; bedding the
  rat moved earlier in the recording gradually becomes background.
  New CLI: `--bg-adapt-alpha A`, `--bg-adapt-dilate-px PX`.
- **0005** — trajectory-constrained blob selection.
  `EpipolarMatcher.match()` now accepts `prior0`/`prior1`/`prior_lambda`
  kwargs to bias candidate scoring toward last-frame centroid. Prevents
  the selector from jumping to a wall reflection with coincidentally
  low epipolar distance. New CLI: `--trajectory-prior`,
  `--trajectory-prior-lambda L`.
- **0006** — undistort centroids before epipolar matching.
  `EpipolarMatcher.match()` now undistorts centroids once upfront and
  uses undistorted coords for line construction, distance, and the
  prior penalty. Reduces apparent epipolar distance from several pixels
  to <1 px on a k1=-0.24 lens. New ctor flag `undistort_match=True`.
- **0009** — NIR vignette / flat-field correction. New module
  `rpimocap.detection.vignette` with `load_flat_field`, `apply_flat_field`,
  `synthesize_flat_field`. Wired symmetrically into both background
  construction and per-frame correction. New CLI: `--flat-field-cam0`,
  `--flat-field-cam1`, `--synthesize-flat-field`.

### Trajectory refinement

- **0003** — `KalmanTracker3D` per-frame online filter. 6-state
  constant-velocity, continuous white-acceleration process noise,
  Mahalanobis-gate outlier rejection. For use inside detector loops.
- **0040** — trajectory-level Kalman/RTS smoother
  (`kalman_filter_trajectory`). Constant-velocity Kalman + RTS
  backward pass. Replaces Gaussian smooth + linear gap-fill with
  physics-constrained gap fill and outlier rejection. New
  `kalman_outlier` field on `Point3D`. New CLI: `--kalman`,
  `--kalman-fps`, `--kalman-max-speed`, `--kalman-max-accel`,
  `--kalman-noise`, `--kalman-outlier-sigma`, `--kalman-no-rts`.
- **0008** — `RearingClassifier` with hysteresis. New module
  `rpimocap.reconstruction.rearing`. Turns Kalman state
  `[x, y, z, vx, vy, vz]` into a `PostureState(reared, body_length_mm,
  body_width_mm, ...)`. Not yet wired into `GeometricLabeller`.

### Tracking infrastructure

- **0007** — `SAM2VideoTracker` scaffold. New class wrapping the SAM2
  video predictor (separate from `SAM2Tracker` which uses the image
  predictor). One initial annotation propagates through the clip via
  SAM2's appearance-and-motion model. Graceful fallback when sam2
  isn't installed. Not yet wired into `SegmentTracker._process_frame`.

### Output schema

- **0041** — `/skeleton/<name>/detected` boolean dataset in
  `reconstruction.h5`. True iff the frame had a genuine triangulated
  detection BEFORE post-processing. Captured by the CLI immediately
  after triangulation; persists through smoothing/Kalman/gap-fill so
  downstream analysis can filter `xyz[det]` to real detections.

### Refraction (default behaviour change)

- **0002** — refraction correction is now opt-in. The standard rpimocap
  rig has no plexiglass between the cameras and the animal, so
  straight-ray DLT is correct by default. The previous single
  `--refraction-config JSON` switch is split into `--refraction-config
  JSON` + `--enable-refraction`; both must be set for refraction to
  apply. Off by default. **Behaviour change vs 0.3.0.**

## [0.3.0] — 2026-05-21

The "planar refraction correction" release.

- **0001** — Snell-law vector refraction, parallel-faced slab geometry,
  `ArenaRefractionModel` with first-hit wall search, iterative
  `triangulate_refracted` solver (single-shot in parallel-slab case),
  JSON config I/O, `triangulate_keypoints` extended with opt-in
  `arena_model` kwarg. Inside a ±140 × ±215 × 0–388 mm PMMA arena
  (n=1.49) with a 600 mm-baseline stereo pair viewing through one
  wall, refractive triangulation recovers known interior points to
  < 0.05 mm vs ~2 mm bias for straight-ray DLT.

## [0.2.0] and earlier

Pre-CHANGELOG. See `git log` and the project status doc for context.
