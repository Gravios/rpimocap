# rpimocap — Status and Roadmap

_Version 0.3.0 (Phase 2 complete)_

## Current state

The pipeline now supports a full stereo capture → calibrate → align →
triangulate → export workflow, with optional refraction correction
through acrylic arena walls. Test suite: **85 passing**.

### Shipped components

| Module                                          | Status      | Tests |
|-------------------------------------------------|-------------|-------|
| `rpimocap-align` GUI (Kabsch + plumb-line)      | shipped     | —     |
| `TiffCapture` (streaming multi-frame TIFF)      | shipped     | covered |
| Multi-Pi coordinator (SSH, session-IDs)         | shipped     | —     |
| `rpimocap-calibrate`, `rpimocap-autocalib`      | shipped     | 23    |
| `triangulate.py` DLT + trajectory utilities     | shipped     | 18    |
| `voxel.py` carving                              | shipped     | covered |
| **Phase 2: planar refraction correction**       | **new**     | **26**|
| Pose detection (`CentroidPoseDetector`)         | placeholder | —     |

### Phase 2 deliverables (this milestone)

- `rpimocap/reconstruction/refraction.py` — Snell-law vector refraction,
  parallel-slab `RefractivePlane`, `ArenaRefractionModel` with first-hit
  wall search, JSON config I/O, world-space ray construction via
  `pixel_to_world_ray`, and the iterative `triangulate_refracted` solver
  (single-shot in the parallel-slab case).
- `rpimocap/reconstruction/triangulate.py` — `triangulate_keypoints`
  extended with an optional `arena_model` kwarg and per-camera
  intrinsics/extrinsics; falls back silently to straight-ray DLT when
  any of the required parameters are missing.
- `rpimocap/cli/pipeline.py` — new `--refraction-config JSON` flag that
  loads an arena model and routes every triangulation through the
  refractive solver.
- `tests/test_refraction.py` — 26 tests covering Snell vector form, slab
  geometry, ray construction, single-point and multi-point recovery
  inside a realistic ±140 × ±215 × 0–388 mm enclosure, plus the
  high-level `triangulate_keypoints` integration path.
- `docs/refraction.md` — physics derivation, API reference, config-file
  example, and a worked end-to-end pipeline invocation.

### Key result

Inside a PMMA box arena (6 mm walls, n = 1.49) with a 600 mm-baseline
stereo pair viewing through one wall, refractive triangulation recovers
known interior points to **< 0.05 mm**, versus a **~2 mm** apparent-
position bias from straight-ray DLT on the same observations.

---

## Roadmap

### ✅ Phase 1 — End-to-end field validation
Activity, not code. Target accuracy gates:
- reprojection RMSE < 3 px
- triangulated arena dimensions within 5 mm of physical measurement

### ✅ Phase 2 — Planar refraction correction (this release)
Acrylic walls (n ≈ 1.49) modelled as parallel-faced slabs; calibration
unmodified because corners sit on the outer face.

**OFF by default.** The current rpimocap rig has no plexiglass between
the cameras and the animal. The module is shipped and tested for future
setups where an acrylic wall is interposed; opt in with both
`--enable-refraction` and `--refraction-config`.

### ⏳ Phase 3 — Pose detection
Replace the `CentroidPoseDetector` placeholder with a SLEAP-trained
model for whisker-scale tracking. The triangulation contract
(`Pose2DResult` → `triangulate_keypoints`) is already in place.

### ⏳ Phase 4 — Performance / GPU acceleration
Profile and accelerate processing of 50 GB+ TIFF sessions. Target
hardware: NVIDIA 5070 Ti. Likely candidates: voxel carving silhouette
projection, batch detection, refraction-aware projection if it becomes
a hot path.

### ⏳ Phase 5 — N-view generalisation
Extend from stereo to 3–4 camera multi-view DLT. Refraction module
already operates per-ray, so the same wall-finding logic generalises
naturally; the intersection step changes from closest-point-of-two
to closest-point-of-N (least-squares on the over-determined system).

---

## Notable design choices

- **Calibration corners are on the outer wall face.** This keeps the
  refraction model purely additive: it can be applied at triangulation
  time without re-calibrating, and absent the model the pipeline still
  produces valid (if slightly biased) 3D points.
- **Parallel-slab invariant.** Because the refracted in-arena ray is
  parallel to the original camera ray, the wall a ray crosses is
  determined entirely by the camera observation, not by the unknown 3D
  point. Refractive triangulation is therefore single-shot, not
  iterative — much simpler and stabler than the general curved-surface
  case.
- **Silent fallback when refraction params are incomplete.**
  `triangulate_keypoints` accepts the new kwargs as optional, so any
  existing caller continues to work unchanged.
