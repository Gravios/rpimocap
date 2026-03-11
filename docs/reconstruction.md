# 3D Reconstruction

This document covers the two reconstruction stages: sparse skeleton
triangulation (`rpimocap.reconstruction.triangulate`) and dense volumetric
hull recovery via voxel carving (`rpimocap.reconstruction.voxel`).

---

## Sparse triangulation

### DLT triangulation

`triangulate_dlt(P0, P1, pt0, pt1)` solves for the 3D point X that best
satisfies both projection equations:

```
λ₀ pt0 = P0 X
λ₁ pt1 = P1 X
```

using the **Direct Linear Transform** (Hartley & Zisserman §12.2). The two
equations are stacked into a 4×4 homogeneous system and the solution is the
right singular vector corresponding to the smallest singular value (via SVD).
The returned 4-vector is normalised so that the homogeneous coordinate w = 1.

DLT is preferred over the algebraic method (`cv2.triangulatePoints`) because
it is numerically stable and naturally handles near-degenerate configurations.

```python
from rpimocap.reconstruction import triangulate_dlt, reprojection_error
import numpy as np

P0 = np.load("calibration.npz")["P0"]
P1 = np.load("calibration.npz")["P1"]

pt0 = (640.2, 351.8)   # pixel coords in camera 0
pt1 = (598.4, 351.8)   # pixel coords in camera 1 (same scan line if rectified)

Xh = triangulate_dlt(P0, P1, pt0, pt1)
X  = Xh[:3]   # world coords in mm
print(f"3D point: {X}")

err = reprojection_error(P0, Xh, pt0)
print(f"Reprojection error cam0: {err:.2f} px")
```

### Batch triangulation and filtering

`triangulate_keypoints` runs DLT for every matched landmark pair from one
frame, filtering by:

- **Minimum confidence** — both detectors must report confidence ≥ threshold.
- **Maximum reprojection error** — points whose mean reprojection error
  exceeds the threshold (default 20 px) are discarded and recorded as `NaN`.

```python
from rpimocap.reconstruction import triangulate_keypoints

points3d = triangulate_keypoints(
    P0, P1,
    result_cam0, result_cam1,     # Pose2DResult objects
    min_confidence=0.5,
    max_reprojection_px=15.0,
)
# Returns list[Point3D | None], aligned with keypoint_names
```

### Trajectory smoothing

Raw triangulated trajectories contain frame-to-frame jitter from detection
noise. `smooth_trajectory` applies a **Gaussian temporal filter** (via
`scipy.ndimage.gaussian_filter1d`) independently to each spatial coordinate
of each landmark. NaN frames (missed detections) are excluded from the
Gaussian kernel and remain NaN after smoothing — they are not smeared into
adjacent valid frames.

```python
from rpimocap.reconstruction import smooth_trajectory

smoothed = smooth_trajectory(frames, sigma=1.5)
# sigma=1.5 is a good default; larger values produce more heavily smoothed trajectories
```

### Gap filling

`fill_trajectory_gaps` linearly interpolates over short runs of NaN frames:

```python
from rpimocap.reconstruction import fill_trajectory_gaps

filled = fill_trajectory_gaps(frames, max_gap=10)
# Gaps of ≤ 10 consecutive missing frames are filled by linear interpolation.
# Longer gaps remain NaN.
```

Set `max_gap` to match your expected maximum occlusion duration in frames.
For typical rodent behaviour at 120 fps with bilateral cameras, 10–15 frames
(80–125 ms) is a reasonable default.

### Detection statistics

```python
from rpimocap.reconstruction import trajectory_stats

stats = trajectory_stats(frames)
for name, s in stats.items():
    print(f"{name}: {s['detection_rate']:.1%} detected, "
          f"mean reprojection error {s['mean_reprojection_px']:.2f} px")
```

---

## Dense voxel carving

Voxel carving reconstructs a volumetric hull of the subject by projecting
a 3D occupancy grid into each camera and carving away voxels that fall
**outside** the foreground silhouette.

### The VoxelGrid

```python
from rpimocap.reconstruction import build_voxel_grid, occupied_centers

bounds = ((-200.0, 200.0), (-200.0, 200.0), (0.0, 400.0))  # mm
grid = build_voxel_grid(bounds, voxel_size=8.0)
# grid.shape: (50, 50, 50) occupancy grid, all True initially

pts = occupied_centers(grid)   # (N, 3) world-space centres of occupied voxels
print(f"{grid.n_occupied} occupied voxels")
```

`voxel_size` controls the trade-off between resolution and speed:

| voxel_size | Grid cells (400³ arena) | Carving time/frame (approx) |
|------------|-------------------------|------------------------------|
| 16 mm | 25 × 25 × 25 = 15 625 | ~5 ms |
| 8 mm | 50 × 50 × 50 = 125 000 | ~40 ms |
| 4 mm | 100 × 100 × 100 = 1 000 000 | ~350 ms |

On the Ryzen 9800X3D, the chunked projection batching keeps 8 mm at real-time
rates for 120 fps video.

### Silhouette extraction

```python
from rpimocap.reconstruction import make_bg_subtractor, extract_silhouette

bg0 = make_bg_subtractor(history=200, var_threshold=40)
bg1 = make_bg_subtractor(history=200, var_threshold=40)

# First pass: feed frames to build the background model (~30 s of video)
for frame in calibration_frames:
    bg0.apply(frame, learningRate=0.01)

# Extraction during reconstruction
mask0 = extract_silhouette(
    frame0, bg0,
    morph_ksize=5,      # morphological open/close kernel (removes noise)
    min_area_px=500,    # discard blobs smaller than this (shadows, reflections)
    dilate_px=8,        # dilate mask to include the subject outline
)
# mask0: (H, W) uint8, 255 = foreground, 0 = background
```

`var_threshold` controls sensitivity. Lower values (20–30) detect subtle
foreground; higher values (50–80) require larger motion and reject slow
drifts. For uniform-coat animals in a clean arena, 40 is a good default.

### Carving a frame

```python
from rpimocap.reconstruction import carve_frame, apply_carving

new_occupancy = carve_frame(
    grid, P0, P1,
    mask0, mask1,
    image_size=(1280, 720),
    chunk_size=4096,    # voxels processed per batch (memory/speed trade-off)
)
grid = apply_carving(grid, new_occupancy)
print(f"After carving: {grid.n_occupied} occupied voxels")
```

`carve_frame` projects all occupied voxel centres into both cameras in chunks
of `chunk_size`. Any voxel whose projection falls outside the foreground mask
in **either** camera is marked as free space. `apply_carving` returns a new
grid with the updated occupancy (the original is not mutated).

### Mesh extraction

```python
from rpimocap.reconstruction import grid_to_mesh
from rpimocap.io import write_ply_mesh

vertices, faces, normals = grid_to_mesh(grid)
write_ply_mesh("output/mesh.ply", vertices, faces)
```

`grid_to_mesh` uses scikit-image's Marching Cubes algorithm to extract a
triangle mesh from the occupancy grid. The mesh is in world-space millimetres.

### Accumulated carving across a session

For a coarser but stable whole-session hull, accumulate carvings:

```python
# Initialise once
grid = build_voxel_grid(bounds, voxel_size=8.0)

for frame0, frame1 in frame_iterator:
    mask0 = extract_silhouette(frame0, bg0)
    mask1 = extract_silhouette(frame1, bg1)
    new_occ = carve_frame(grid, P0, P1, mask0, mask1, image_size)
    grid = apply_carving(grid, new_occ)
    # voxels are only ever removed, never re-added

pts = occupied_centers(grid)   # tight bounding hull of the subject's reach envelope
```

---

## Coordinate system

All 3D coordinates are in the world frame established by the calibration.
The physical units are millimetres by convention (matching the `--square`
argument to `rpimocap-calibrate`).

The `--bounds` argument to `rpimocap-run` defines the arena in this world
frame. A typical setup:

```
X: left/right  (-300 to 300 mm)
Y: front/back  (-200 to 200 mm)
Z: up           (0 to 350 mm)
```

Camera 0 is the reference frame; camera 1's position and orientation is given
by the extrinsic parameters R and T in the calibration file.

---

## Pipeline integration

The full frame-loop in `rpimocap.cli.pipeline` integrates these stages:

```
For each frame:
  ├─ Detect 2D keypoints in both cameras (detect2d)
  ├─ Triangulate matched keypoints (triangulate_keypoints)
  ├─ Extract silhouettes (extract_silhouette) [optional]
  └─ Carve voxel grid (carve_frame) [optional]

Post-loop:
  ├─ Smooth trajectories (smooth_trajectory)
  ├─ Fill gaps (fill_trajectory_gaps)
  ├─ Compute stats (trajectory_stats)
  └─ Export (write_hdf5, write_viewer_json, write_ply_*)
```
