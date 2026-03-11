# Export Formats

`rpimocap.io` writes four output formats. All are produced automatically
by `rpimocap-run`; each can also be called directly from Python.

---

## HDF5 archive

The primary output: a single file containing the full trajectory and,
optionally, per-frame voxel point clouds.

### File structure

```
reconstruction.h5
│
├── Attributes
│   ├── keypoint_names  : JSON array of landmark names
│   ├── skeleton_edges  : JSON array of [src, dst] pairs
│   ├── fps             : float
│   ├── n_frames        : int
│   └── metadata        : JSON dict (calibration path, detector, CLI flags, ...)
│
├── skeleton/
│   ├── nose/
│   │   ├── xyz         : float32 (n_frames, 3)  — NaN where not detected
│   │   └── confidence  : float32 (n_frames,)
│   ├── left_shoulder/
│   │   └── ...
│   └── ...
│
└── voxels/             (present only if --voxel carving was enabled)
    ├── frame_000000    : float32 (N, 3)  — world-space point cloud
    ├── frame_000001    : float32 (M, 3)
    └── ...
```

### Reading in Python

```python
import h5py
import numpy as np

with h5py.File("output/reconstruction.h5", "r") as f:
    # Metadata
    import json
    names  = json.loads(f.attrs["keypoint_names"])   # e.g. ["nose", "left_ear", ...]
    fps    = float(f.attrs["fps"])
    n      = int(f.attrs["n_frames"])

    # 3D trajectory for one landmark: (n_frames, 3), NaN = not detected
    nose   = f["skeleton/nose/xyz"][:]
    conf   = f["skeleton/nose/confidence"][:]

    # All landmarks at once
    traj = {name: f[f"skeleton/{name}/xyz"][:] for name in names}

    # Voxel cloud for a specific frame
    if "voxels/frame_000050" in f:
        cloud = f["voxels/frame_000050"][:]   # (N, 3) mm

    # Boolean detected mask
    detected = ~np.isnan(nose).any(axis=1)
    print(f"Nose detected in {detected.sum()}/{n} frames ({detected.mean():.1%})")
```

### Reading in MATLAB

```matlab
% Attributes
names = jsondecode(h5readatt('reconstruction.h5', '/', 'keypoint_names'));
fps   = double(h5readatt('reconstruction.h5', '/', 'fps'));

% Trajectory: MATLAB returns [3 × n_frames] (column-major)
xyz = h5read('reconstruction.h5', '/skeleton/nose/xyz');  % [3, n_frames]
xyz = xyz';                                               % → [n_frames, 3]

% Replace NaN with interpolated values
valid = ~any(isnan(xyz), 2);
t     = 1:size(xyz, 1);
for dim = 1:3
    xyz(~valid, dim) = interp1(t(valid), xyz(valid, dim), t(~valid), 'linear', NaN);
end
```

### Writing from Python

```python
from rpimocap.io import write_hdf5

write_hdf5(
    path="output/reconstruction.h5",
    skeleton_frames=frames,          # list[list[Point3D]]
    voxel_frames=voxel_clouds,       # list[np.ndarray (N,3)] or None
    fps=120.0,
    metadata={"calibration": "calibration.npz", "detector": "dlc"},
)
```

---

## Three.js viewer JSON

`viewer_data.json` is consumed by the bundled `viewer/assets/index.html`.
It is compact JSON (no pretty-printing) with the following structure:

```json
{
  "fps": 120.0,
  "n_frames": 14400,
  "keypoint_names": ["nose", "left_ear", ...],
  "skeleton_edges": [["nose", "neck"], ...],
  "skeleton": {
    "nose": [[x0,y0,z0], [x1,y1,z1], ...],
    "left_ear": [[x0,y0,z0], null, [x2,y2,z2], ...],
    ...
  },
  "voxels": [
    [[x,y,z], ...],
    null,
    [[x,y,z], ...],
    ...
  ]
}
```

`null` entries in trajectories or the voxels array represent frames with
no detection or empty voxel sets.

### Writing from Python

```python
from rpimocap.io import write_viewer_json

write_viewer_json(
    path="output/viewer_data.json",
    skeleton_frames=frames,
    keypoint_names=detector.keypoint_names,
    skeleton_edges=detector.skeleton_edges,
    fps=120.0,
    voxel_frames=voxel_clouds,   # None to skip volumetric data
    voxel_downsample=4,          # keep every 4th point cloud point for smaller file
)
```

`voxel_downsample` controls file size. For 8 mm voxels in a 600 mm arena,
each frame can contain O(10 000) occupied voxel centres — downsampling by 4
reduces viewer JSON size by ~75% with minimal visual impact.

---

## PLY files

PLY (Polygon File Format / Stanford Triangle Format) is the most portable
3D format, readable by MeshLab, Blender, CloudCompare, and MATLAB's
`pcread` / `stlread`.

rpimocap writes three PLY variants:

### Point cloud (per-frame voxels)

```python
from rpimocap.io import write_ply_pointcloud
import numpy as np

pts    = np.random.randn(5000, 3) * 100   # (N, 3) world coords
colors = np.full((5000, 3), 128, dtype=np.uint8)   # grey

write_ply_pointcloud("output/ply/volume/frame_000042.ply", pts, colors)
```

### Skeleton point cloud (per-frame keypoints)

```python
from rpimocap.io import write_ply_skeleton_frame

write_ply_skeleton_frame(
    "output/ply/skeleton/frame_000042.ply",
    frame_points,    # list[Point3D]
)
```

### Triangle mesh (Marching Cubes output)

```python
from rpimocap.io import write_ply_mesh
from rpimocap.reconstruction import build_voxel_grid, carve_frame, apply_carving, grid_to_mesh

# ... carve grid across frames ...
vertices, faces, _ = grid_to_mesh(grid)
write_ply_mesh("output/mesh.ply", vertices, faces)
```

All PLY files are ASCII format for maximum compatibility. Binary PLY would
be ~3× smaller; use CloudCompare's "Save as binary PLY" if file size matters.

---

## Detection statistics CSV

`detection_stats.csv` is written automatically by the pipeline:

```csv
landmark,n_frames,n_detected,detection_rate,mean_reprojection_px,std_reprojection_px
nose,14400,13847,0.9616,2.14,0.83
left_ear,14400,12309,0.8548,3.01,1.24
...
```

Read in Python:

```python
import pandas as pd
stats = pd.read_csv("output/detection_stats.csv")
print(stats.sort_values("detection_rate").to_string())
```

---

## Deploying the viewer

```python
from rpimocap.viewer import deploy_viewer

# Copies index.html and optionally the JSON into a ready-to-serve directory
deploy_viewer(
    dest_dir="output/viewer",
    viewer_json="output/viewer_data.json",
)

# Then serve:
# cd output/viewer && python -m http.server 8080
```

Or manually:

```bash
mkdir -p output/viewer/data
python -c "from rpimocap.viewer import viewer_html_path; print(viewer_html_path())"
# Copy that path to output/viewer/index.html
cp output/viewer_data.json output/viewer/data/viewer_data.json
```

The viewer auto-loads `./data/viewer_data.json` on startup and also accepts
drag-and-drop of any `viewer_data.json` file.

---

## File size guide

For a 120 fps, 2-minute session with 33 landmarks and 8 mm voxels:

| Format | Approximate size |
|--------|-----------------|
| `reconstruction.h5` (skeleton only) | 15–25 MB |
| `reconstruction.h5` (skeleton + voxels) | 200–600 MB |
| `viewer_data.json` (skeleton only) | 8–15 MB |
| `viewer_data.json` (skeleton + voxels, downsample=4) | 40–120 MB |
| PLY skeleton frames (all 14 400) | 50–150 MB |
| PLY volume frames (every 5th, 2880 files) | 300 MB–1 GB |

For long sessions, consider `--no-voxel` to skip volumetric reconstruction
and `--ply-stride 10` to thin the PLY output.
