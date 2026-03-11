# API Reference

Complete reference for all public symbols in `rpimocap`. Import paths
use the shortest available route via sub-package `__init__.py` re-exports.

---

## rpimocap.calibration

```python
from rpimocap.calibration import (
    detect_corners_paired,
    calibrate_intrinsics,
    calibrate_stereo,
    stereo_rectify,
    validate_epipolar,
)
```

---

### `detect_corners_paired(cap0, cap1, pattern, skip, max_pairs) → list[FramePair]`

Scan two synchronised video captures for frames where a checkerboard is
detected in both cameras simultaneously.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cap0`, `cap1` | `cv2.VideoCapture` | — | Open video captures |
| `pattern` | `tuple[int,int]` | — | Inner corners (cols, rows) |
| `skip` | `int` | 8 | Sample every N-th frame |
| `max_pairs` | `int` | 80 | Stop after this many valid pairs |

Returns a list of `FramePair` objects, each holding the pixel coordinates
of detected inner corners in both cameras.

---

### `calibrate_intrinsics(pairs, pattern, square_mm, rational) → IntrinsicResult`

Estimate per-camera intrinsics from a list of paired detections.

| Parameter | Type | Description |
|-----------|------|-------------|
| `pairs` | `list[FramePair]` | Output of `detect_corners_paired` |
| `pattern` | `tuple[int,int]` | Inner corners (cols, rows) |
| `square_mm` | `float` | Physical square edge length in mm |
| `rational` | `bool` | Use 8-coefficient rational distortion model |

Returns an `IntrinsicResult` with fields `K0`, `K1`, `dist0`, `dist1`,
`rms0`, `rms1`.

---

### `calibrate_stereo(intrinsics, pairs, pattern, square_mm) → StereoResult`

Estimate the stereo extrinsics (R, T) using fixed intrinsics.

Returns a `StereoResult` with all fields in the `.npz` output schema
(see [calibration.md](calibration.md)).

---

### `stereo_rectify(stereo) → RectifyResult`

Compute rectification maps from a `StereoResult`. Returns a `RectifyResult`
with `R0`, `R1`, `P0`, `P1`, `Q`, `map0x/y`, `map1x/y`.

---

### `validate_epipolar(stereo, pairs) → float`

Compute mean symmetric epipolar distance over all inlier pairs.
Returns the distance in pixels; values below 1.0 px are acceptable.

---

## rpimocap.calibration.autocalib

```python
from rpimocap.calibration.autocalib import (
    FundamentalEstimate,
    CrossViewMatcher,
    sample_frame_pairs,
    filter_estimates,
    sampson_distance,
    KruppaResult,
    EssentialDecomposition,
    make_K,
    cost_for_f,
    essential_constraint_residual,
    estimate_focal_kruppa,
    decompose_essential,
    metric_scale_refinement,
    bundle_refine_focal,
    run_focal_estimation,
    generate_report,
)
```

---

### `FundamentalEstimate`

Dataclass holding one RANSAC fundamental matrix estimate.

| Field | Type | Description |
|-------|------|-------------|
| `frame_idx` | `int` | Source frame index |
| `F` | `ndarray (3,3)` | Fundamental matrix |
| `pts0`, `pts1` | `ndarray (N,2)` | RANSAC inlier pixel coordinates |
| `n_inliers` | `int` | Number of inliers |
| `inlier_ratio` | `float` | Inliers / total matches |
| `mean_sampson` | `float` | Mean Sampson distance of inliers (px) |
| `quality` | `float` | Composite quality score in [0, 1] (computed) |

---

### `CrossViewMatcher`

SIFT-based cross-view matcher. Stateless; can be reused across frame pairs.

```python
matcher = CrossViewMatcher(
    n_features=4000,
    contrast_threshold=0.025,
    ratio_threshold=0.72,
    grid_cells=4,
    max_per_cell=30,
)
est = matcher.match(frame0, frame1, frame_idx=42)
# Returns FundamentalEstimate or None if RANSAC fails
```

---

### `sample_frame_pairs(cap0, cap1, n_pairs, seed) → list[tuple]`

Draw temporally-diverse frame pairs from two video captures.

Returns a list of `(frame_idx, frame0, frame1)` tuples.

---

### `filter_estimates(estimates, min_quality, top_n) → list[FundamentalEstimate]`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `estimates` | `list[FundamentalEstimate]` | — | Pool to filter |
| `min_quality` | `float` | 0.3 | Discard estimates below this quality |
| `top_n` | `int \| None` | None | Keep only the top N by quality |

---

### `sampson_distance(F, pts0, pts1) → ndarray (N,)`

Compute the symmetric Sampson distance for each correspondence.
Values below ~2 px indicate geometrically consistent matches.

---

### `KruppaResult`

Dataclass containing all focal length estimation outputs.

| Field | Type | Description |
|-------|------|-------------|
| `f_px` | `float` | Final focal length in pixels |
| `K` | `ndarray (3,3)` | Intrinsic matrix |
| `f_kruppa` | `float` | Focal length from Kruppa solve only |
| `f_metric` | `float \| None` | After arena metric refinement |
| `scale_factor` | `float \| None` | Applied scale correction |
| `n_estimates_used` | `int` | F estimates included in the solve |
| `mean_essential_residual` | `float` | Mean Kruppa cost at solution |
| `f_search_vals` | `ndarray` | Focal length values from 1D scan |
| `f_search_costs` | `ndarray` | Corresponding costs |

---

### `make_K(f, cx, cy) → ndarray (3,3)`

Build a pinhole intrinsic matrix with zero skew and square pixels.

---

### `cost_for_f(f, estimates, cx, cy) → float`

Evaluate the quality-weighted mean Kruppa essential-constraint cost
at a given focal length.

---

### `essential_constraint_residual(E) → float`

Measure how far E deviates from a valid essential matrix:

```
‖2EEᵀE − tr(EEᵀ)·E‖_F / ‖E‖³
```

Zero for a valid essential matrix; high for degenerate or noise-corrupted E.

---

### `estimate_focal_kruppa(estimates, image_size, scan_steps, verbose) → tuple`

Run the 1D cost scan followed by Brent refinement.

Returns `(f_px, scan_f_array, scan_cost_array)`.

---

### `decompose_essential(E, pts0, pts1, K) → tuple[R, t] | None`

Decompose E into (R, t) using the cheirality test (most points in front
of both cameras). Returns `None` if cheirality test fails.

---

### `metric_scale_refinement(E, estimates, image_size, K, arena_bounds) → float`

Triangulate inlier correspondences and compute the scale factor to match
the known arena dimensions. Returns `f_refined`.

---

### `bundle_refine_focal(f0, estimates, image_size) → float`

Nelder-Mead minimisation of mean reprojection error.
Returns refined focal length in pixels.

---

### `run_focal_estimation(estimates, image_size, arena_bounds, refine_bundle, verbose) → KruppaResult`

High-level entry point: runs all three stages and returns a `KruppaResult`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `estimates` | `list[FundamentalEstimate]` | — | Pool of F estimates |
| `image_size` | `tuple[int,int]` | — | (width, height) in pixels |
| `arena_bounds` | nested tuple or None | None | Physical arena for metric scaling |
| `refine_bundle` | `bool` | False | Run bundle refinement |
| `verbose` | `bool` | True | Print progress |

---

### `generate_report(result, estimates, image_size, checkerboard_npz, out_path)`

Write a standalone HTML validation report.

| Parameter | Type | Description |
|-----------|------|-------------|
| `result` | `KruppaResult` | Focal estimation result |
| `estimates` | `list[FundamentalEstimate]` | Full estimate pool |
| `image_size` | `tuple[int,int]` | (width, height) |
| `checkerboard_npz` | `str \| None` | Path to compare against |
| `out_path` | `str` | Output HTML file path |

---

## rpimocap.detection

```python
from rpimocap.detection import (
    Keypoint2D, Pose2DResult, PoseDetector2D,
    MediaPipePoseDetector, CentroidPoseDetector, CSVPoseDetector,
)
```

See [detectors.md](detectors.md) for full usage documentation.

### `Keypoint2D`

Dataclass: `name`, `x`, `y`, `confidence`, `as_array() → ndarray (2,)`.

### `Pose2DResult`

Dataclass: `frame_idx`, `detected`, `keypoints`.
Method: `by_name() → dict[str, Keypoint2D]`.

### `PoseDetector2D`

Abstract base class. Subclass and implement `keypoint_names`,
`skeleton_edges`, `detect(frame, frame_idx)`.

### `MediaPipePoseDetector(model_complexity, min_detection_conf, min_tracking_conf)`

### `CentroidPoseDetector(history, var_threshold, min_area, morph_ksize)`

### `CSVPoseDetector(csv_path, fmt, min_likelihood, skeleton_edges, frame_index_offset)`

`fmt` is `"dlc"` or `"sleap"`.

---

## rpimocap.reconstruction

```python
from rpimocap.reconstruction import (
    # triangulate
    Point3D,
    triangulate_dlt,
    reprojection_error,
    triangulate_keypoints,
    build_trajectory_dict,
    smooth_trajectory,
    fill_trajectory_gaps,
    trajectory_stats,
    # voxel
    VoxelGrid,
    build_voxel_grid,
    voxel_centers,
    project_points_batch,
    make_bg_subtractor,
    extract_silhouette,
    carve_frame,
    apply_carving,
    occupied_centers,
    surface_centers,
    grid_to_mesh,
)
```

---

### `Point3D`

Dataclass: `name`, `xyz` `(ndarray (3,))`, `confidence`, `reprojection_error`.
Method: `as_list() → list[float]`.

---

### `triangulate_dlt(P0, P1, pt0, pt1) → ndarray (4,)`

DLT triangulation. Returns homogeneous 4-vector with w=1.

---

### `reprojection_error(P, X, pt) → float`

Pixel-space reprojection error for one projection.

---

### `triangulate_keypoints(P0, P1, result0, result1, min_confidence, max_reprojection_px) → list[Point3D | None]`

Batch DLT with confidence and reprojection-error filtering.

---

### `build_trajectory_dict(frames) → dict[str, ndarray (n_frames, 3)]`

Convert `list[list[Point3D]]` to a per-landmark dict of (n_frames, 3) arrays.
Missing detections are `NaN`.

---

### `smooth_trajectory(frames, sigma) → list[list[Point3D]]`

Gaussian temporal smoothing, NaN-aware.

---

### `fill_trajectory_gaps(frames, max_gap) → list[list[Point3D]]`

Linear interpolation over runs of ≤ `max_gap` missing frames.

---

### `trajectory_stats(frames) → dict[str, dict]`

Per-landmark statistics dict with keys `detection_rate`,
`mean_reprojection_px`, `std_reprojection_px`.

---

### `VoxelGrid`

Dataclass: `occupancy` `(ndarray bool (nx,ny,nz))`, `origin (ndarray (3,))`,
`voxel_size (float)`, `shape`.

Properties: `n_occupied`, `bounds_mm`.

---

### `build_voxel_grid(bounds, voxel_size) → VoxelGrid`

Allocate a fully-occupied grid. `bounds` is
`((xmin,xmax),(ymin,ymax),(zmin,zmax))` in mm.

---

### `voxel_centers(grid) → ndarray (nx*ny*nz, 3)`

World-space centre coordinates of all voxels (occupied or not).

---

### `project_points_batch(P, pts3d) → ndarray (N, 2)`

Batch projection of (N, 3) world points through projection matrix P.

---

### `make_bg_subtractor(history, var_threshold, detect_shadows) → cv2.BackgroundSubtractorMOG2`

---

### `extract_silhouette(frame, bg_subtractor, morph_ksize, min_area_px, dilate_px) → ndarray (H,W) uint8`

Foreground mask. 255 = foreground, 0 = background.

---

### `carve_frame(grid, P0, P1, mask0, mask1, image_size, chunk_size) → ndarray bool (nx,ny,nz)`

Project occupied voxels into both cameras; mark as False any voxel that
falls outside either foreground mask. Returns updated occupancy array
without mutating `grid`.

---

### `apply_carving(grid, new_occupancy) → VoxelGrid`

Return a new VoxelGrid with the given occupancy (original unchanged).

---

### `occupied_centers(grid) → ndarray (n_occupied, 3)`

World-space centres of all occupied voxels.

---

### `surface_centers(grid) → ndarray (n_surface, 3)`

World-space centres of occupied voxels that have at least one free
neighbour (surface voxels only).

---

### `grid_to_mesh(grid) → tuple[ndarray, ndarray, ndarray]`

Marching Cubes mesh extraction. Returns `(vertices, faces, normals)`.
All in world-space mm.

---

## rpimocap.io

```python
from rpimocap.io import (
    write_ply_pointcloud,
    write_ply_mesh,
    write_ply_skeleton_frame,
    write_hdf5,
    write_viewer_json,
    write_stats_csv,
)
```

---

### `write_ply_pointcloud(path, points, colors)`

| Parameter | Type | Description |
|-----------|------|-------------|
| `path` | `str \| Path` | Output file path |
| `points` | `ndarray (N,3)` | World-space coordinates |
| `colors` | `ndarray (N,3) uint8 \| None` | RGB colours (None → grey) |

---

### `write_ply_mesh(path, vertices, faces, vertex_colors)`

| Parameter | Type | Description |
|-----------|------|-------------|
| `vertices` | `ndarray (V,3)` | Vertex positions |
| `faces` | `ndarray (F,3) int` | Triangle indices |
| `vertex_colors` | `ndarray (V,3) uint8 \| None` | Per-vertex RGB |

---

### `write_ply_skeleton_frame(path, frame_points)`

Write a per-frame skeleton as a PLY point cloud.
`frame_points` is a `list[Point3D]`.

---

### `write_hdf5(path, skeleton_frames, voxel_frames, fps, metadata)`

| Parameter | Type | Description |
|-----------|------|-------------|
| `skeleton_frames` | `list[list[Point3D]]` | Per-frame detection results |
| `voxel_frames` | `list[ndarray (N,3)] \| None` | Per-frame point clouds |
| `fps` | `float` | Video frame rate |
| `metadata` | `dict \| None` | Stored as JSON in root attributes |

---

### `write_viewer_json(path, skeleton_frames, keypoint_names, skeleton_edges, fps, voxel_frames, voxel_downsample)`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `voxel_frames` | `list \| None` | None | Omit voxels from viewer |
| `voxel_downsample` | `int` | 1 | Keep every N-th voxel point |

---

### `write_stats_csv(path, stats)`

`stats` is the dict returned by `trajectory_stats`.

---

## rpimocap.viewer

```python
from rpimocap.viewer import viewer_html_path, deploy_viewer
```

### `viewer_html_path() → Path`

Return the absolute filesystem path to `viewer/assets/index.html`.

### `deploy_viewer(dest_dir, viewer_json) → Path`

Copy the viewer HTML to `dest_dir/index.html` and optionally copy
`viewer_json` to `dest_dir/data/viewer_data.json`.
Returns the path of the deployed `index.html`.
