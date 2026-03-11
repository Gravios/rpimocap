# Quickstart

This guide walks through a complete reconstruction from two synchronised
video files in five steps. Each step can be run independently once its
prerequisite output exists.

## Prerequisites

- Two synchronised video files: `cam0.mp4`, `cam1.mp4`
- Physical arena dimensions in millimetres
- rpimocap installed: `pip install -e .`

---

## Step 1 — Calibrate the cameras

Shoot a rigid checkerboard target (printed or physical) while it is fully
visible in **both cameras simultaneously**. Move it through tilts, rotations,
and positions across the full field of view. Aim for 40–60 paired detections.

```bash
rpimocap-calibrate \
    --cam0    calib_cam0.mp4 \
    --cam1    calib_cam1.mp4 \
    --pattern 9x6 \
    --square  25.0 \
    --out     calibration.npz
```

**`--pattern`** is the number of *inner corners*, not squares. A 10×7
physical board has 9×6 inner corners.

**`--square`** is the physical edge length of one square in millimetres.
Getting this right sets the metric scale of the 3D reconstruction.

Quality targets printed at the end of calibration:

| Metric | Good | Acceptable |
|--------|------|------------|
| Intrinsic RMS | < 0.5 px | < 1.0 px |
| Stereo RMS | < 1.0 px | < 1.5 px |
| Epipolar distance | < 0.5 px | < 1.0 px |

If you cannot shoot a checkerboard, use self-calibration instead
(see [autocalib.md](autocalib.md)).

---

## Step 2 — (Optional) Self-calibration

If a checkerboard is unavailable, estimate intrinsics from the subject's
motion and the known arena:

```bash
rpimocap-autocalib \
    --cam0   cam0.mp4 \
    --cam1   cam1.mp4 \
    --bounds "-300,300,-200,200,0,400" \
    --out    autocalib.npz \
    --report autocalib_report.html
```

Open `autocalib_report.html` in any browser to inspect convergence and
quality gates. Use `autocalib.npz` in place of `calibration.npz` in
subsequent steps.

---

## Step 3 — Run the reconstruction pipeline

```bash
rpimocap-run \
    --cam0   cam0.mp4 \
    --cam1   cam1.mp4 \
    --calib  calibration.npz \
    --bounds "-300,300,-200,200,0,400" \
    --out    output/
```

**`--bounds`** defines the physical arena in mm as
`xmin,xmax,ymin,ymax,zmin,zmax`. The tighter you make this, the faster
voxel carving runs and the cleaner the resulting hull.

Output produced:

```
output/
├── reconstruction.h5        # full skeleton + voxel trajectories
├── viewer_data.json         # Three.js viewer payload
├── detection_stats.csv      # per-landmark detection rates
└── ply/
    ├── skeleton/            # per-frame 3D keypoint PLY files
    └── volume/              # per-frame voxel point cloud PLY files
```

---

## Step 4 — View the reconstruction

```bash
# Copy the viewer HTML next to the data file
mkdir -p output/viewer/data
cp rpimocap/viewer/assets/index.html output/viewer/index.html
cp output/viewer_data.json output/viewer/data/viewer_data.json

# Serve locally (required for the fetch() call in the viewer)
cd output/viewer
python -m http.server 8080
```

Open `http://localhost:8080` in any modern browser.

Or use the helper from Python:

```python
from rpimocap.viewer import deploy_viewer
deploy_viewer("output/viewer", viewer_json="output/viewer_data.json")
# then: python -m http.server 8080 (in output/viewer/)
```

**Viewer keyboard shortcuts**

| Key | Action |
|-----|--------|
| Space | Play / pause |
| ← / → | Step one frame |
| Home / End | Jump to first / last frame |
| Left drag | Orbit |
| Scroll | Zoom |
| Right drag | Pan |

---

## Step 5 — Load the data in Python / MATLAB

### Python

```python
import h5py
import numpy as np

with h5py.File("output/reconstruction.h5", "r") as f:
    names    = list(f.attrs["keypoint_names"])
    n_frames = f.attrs["n_frames"]
    fps      = f.attrs["fps"]

    # Per-landmark 3D trajectory: (n_frames, 3), NaN = not detected
    nose_xyz = f["skeleton/nose/xyz"][:]
    nose_conf = f["skeleton/nose/confidence"][:]

    # Voxel point cloud for frame 100 (if voxel carving was enabled)
    if "voxels/frame_000100" in f:
        pts3d = f["voxels/frame_000100"][:]   # (N, 3)
```

### MATLAB

```matlab
info     = h5info('output/reconstruction.h5');
xyz      = h5read('output/reconstruction.h5', '/skeleton/nose/xyz');  % [3, n_frames]
conf     = h5read('output/reconstruction.h5', '/skeleton/nose/confidence');
n_frames = double(h5readatt('output/reconstruction.h5', '/', 'n_frames'));
fps      = double(h5readatt('output/reconstruction.h5', '/', 'fps'));
```

---

## Common options

### Faster runs during development

```bash
rpimocap-run ... \
    --n-frames   500 \      # process only the first 500 frames
    --no-voxel   \          # skip voxel carving entirely
    --voxel-size 15.0       # coarser voxel grid (default: 8.0)
```

### Rodent tracking with DLC

```bash
rpimocap-run \
    --cam0     cam0.mp4 \
    --cam1     cam1.mp4 \
    --calib    calibration.npz \
    --detector dlc \
    --dlc0     cam0_DLC_scored.csv \
    --dlc1     cam1_DLC_scored.csv \
    --bounds   "-300,300,-200,200,0,400" \
    --out      output/
```

### Save per-frame PLY meshes (Marching Cubes)

```bash
rpimocap-run ... \
    --save-ply-volume \
    --save-ply-skeleton \
    --save-mesh \
    --ply-stride 5          # every 5th frame to manage file count
```
