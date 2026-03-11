# rpimocap

**Raspberry Pi multi-camera 3D motion capture.**

Reconstruct a 3D pose skeleton and volumetric hull of a moving subject in a
constrained space from two synchronous video perspectives. Designed for
neuroscience behavioural experiments (rodents, humans) but applicable to any
tracked subject in a known arena.

```
cam0.mp4 ─┐                        ┌─ reconstruction.h5  (skeleton + voxels)
           ├─ rpimocap-run ─────────┤─ viewer_data.json   (Three.js viewer)
cam1.mp4 ─┘                        └─ detection_stats.csv
```

---

## Features

- **Stereo calibration** from a checkerboard target with quality reporting
- **Self-calibration** from subject motion using Kruppa equations + arena
  metric refinement (no checkerboard needed)
- **Pluggable 2D detection**: MediaPipe (human), background-subtraction
  centroid (rodent), DLC/SLEAP CSV import, or custom backend
- **DLT triangulation** with reprojection-error filtering, Gaussian
  trajectory smoothing, and gap interpolation
- **Voxel carving** from silhouette masks into a 3D occupancy grid,
  Marching Cubes mesh extraction
- **Multiple export formats**: HDF5 archive, PLY point clouds / meshes,
  viewer JSON
- **Interactive Three.js viewer** — timeline scrubber, layer toggles,
  drag-and-drop JSON loading
- **Complete test suite** — 41 unit tests, no video files required

---

## Quick install

```bash
git clone https://github.com/yourlab/rpimocap
cd rpimocap
pip install -e .
```

Python 3.11 recommended. See [docs/installation.md](docs/installation.md)
for full platform notes and optional dependencies.

---

## Five-minute example

```bash
# 1. Calibrate
rpimocap-calibrate \
    --cam0 calib_cam0.mp4 --cam1 calib_cam1.mp4 \
    --pattern 9x6 --square 25.0 --out calibration.npz

# 2. Reconstruct
rpimocap-run \
    --cam0 session_cam0.mp4 --cam1 session_cam1.mp4 \
    --calib calibration.npz \
    --bounds "-300,300,-200,200,0,400" \
    --out output/

# 3. View
python -c "from rpimocap.viewer import deploy_viewer; deploy_viewer('output/viewer', 'output/viewer_data.json')"
cd output/viewer && python -m http.server 8080
# open http://localhost:8080
```

---

## Package structure

```
rpimocap/
├── rpimocap/
│   ├── calibration/
│   │   ├── checkerboard.py       Checkerboard stereo calibration
│   │   └── autocalib/
│   │       ├── features.py       SIFT cross-view matching
│   │       ├── kruppa.py         Kruppa focal estimation
│   │       └── report.py         HTML validation report
│   ├── detection/
│   │   └── detectors.py          PoseDetector2D + four backends
│   ├── reconstruction/
│   │   ├── triangulate.py        DLT, smoothing, gap fill
│   │   └── voxel.py              Silhouette extraction, voxel carving
│   ├── io/
│   │   └── export.py             PLY, HDF5, JSON, CSV writers
│   ├── viewer/
│   │   └── assets/index.html     Three.js viewer
│   └── cli/
│       ├── calibrate.py          rpimocap-calibrate entry point
│       ├── autocalib.py          rpimocap-autocalib entry point
│       └── pipeline.py           rpimocap-run entry point
└── tests/
    ├── test_calibration.py       13 tests — Kruppa, make_K, decompose_E
    ├── test_reconstruction.py    16 tests — DLT, trajectory, voxel
    └── test_autocalib.py         12 tests — features, filtering, report
```

---

## CLI reference

### `rpimocap-calibrate`

```
--cam0       PATH          Camera 0 video
--cam1       PATH          Camera 1 video
--pattern    WxH           Inner corners, e.g. 9x6
--square     FLOAT         Square size in mm
--skip       INT           Sample every N frames (default: 8)
--max-pairs  INT           Max paired detections (default: 80)
--no-rational              Use 5-coeff distortion (default: rational 8-coeff)
--out        PATH          Output .npz (default: calibration.npz)
```

### `rpimocap-autocalib`

```
--cam0           PATH      Camera 0 video
--cam1           PATH      Camera 1 video
--bounds         BOUNDS    xmin,xmax,ymin,ymax,zmin,zmax in mm
--n-frames       INT       Frames to sample (default: 2000)
--refine-bundle            Nelder-Mead bundle refinement
--compare        PATH      Checkerboard .npz to compare against
--out            PATH      Output .npz (default: autocalib.npz)
--report         PATH      HTML report output path
```

### `rpimocap-run`

```
--cam0             PATH    Camera 0 video (required)
--cam1             PATH    Camera 1 video (required)
--calib            PATH    Calibration .npz (required)
--bounds           BOUNDS  xmin,xmax,ymin,ymax,zmin,zmax in mm (required)
--detector         STR     centroid | mediapipe | dlc | sleap (default: centroid)
--dlc0/--dlc1      PATH    DLC CSV for each camera
--sleap0/--sleap1  PATH    SLEAP CSV for each camera
--voxel-size       FLOAT   Voxel edge length in mm (default: 8.0)
--no-voxel                 Skip voxel carving (skeleton only)
--smooth-sigma     FLOAT   Gaussian smoothing sigma in frames (default: 1.5)
--fill-gaps        INT     Maximum gap to interpolate in frames (default: 10)
--max-repr-error   FLOAT   Reprojection error filter threshold px (default: 20)
--rectify                  Apply stereo rectification before detection
--n-frames         INT     Process only this many frames
--save-ply-skeleton        Write per-frame skeleton PLY files
--save-ply-volume          Write per-frame voxel PLY files
--save-mesh                Write a single Marching Cubes mesh PLY
--ply-stride       INT     Write every N-th PLY frame (default: 1)
--out              PATH    Output directory (default: output/)
```

---

## Python API

```python
import numpy as np
from rpimocap.calibration.autocalib import run_focal_estimation, CrossViewMatcher
from rpimocap.detection import CentroidPoseDetector
from rpimocap.reconstruction import (
    triangulate_keypoints, smooth_trajectory, fill_trajectory_gaps,
    build_voxel_grid, extract_silhouette, carve_frame, apply_carving,
)
from rpimocap.io import write_hdf5, write_viewer_json
from rpimocap.viewer import deploy_viewer

cal = np.load("calibration.npz")
P0, P1 = cal["P0"], cal["P1"]

detector0 = CentroidPoseDetector()
detector1 = CentroidPoseDetector()
grid = build_voxel_grid(((-300,300),(-200,200),(0,400)), voxel_size=8.0)

frames = []
for i, (f0, f1) in enumerate(frame_iterator):
    r0 = detector0.detect(f0, i)
    r1 = detector1.detect(f1, i)
    pts3d = triangulate_keypoints(P0, P1, r0, r1)
    frames.append(pts3d)

    mask0 = extract_silhouette(f0, bg0)
    mask1 = extract_silhouette(f1, bg1)
    grid  = apply_carving(grid, carve_frame(grid, P0, P1, mask0, mask1, (1280,720)))

frames = smooth_trajectory(frames, sigma=1.5)
frames = fill_trajectory_gaps(frames, max_gap=10)

write_hdf5("output/reconstruction.h5", frames, fps=120.0)
write_viewer_json("output/viewer_data.json", frames,
                  detector0.keypoint_names, detector0.skeleton_edges, fps=120.0)
deploy_viewer("output/viewer", "output/viewer_data.json")
```

---

## Documentation

| Document | Contents |
|----------|----------|
| [docs/installation.md](docs/installation.md) | System requirements, platform notes, venv setup |
| [docs/quickstart.md](docs/quickstart.md) | End-to-end walkthrough with all five steps |
| [docs/calibration.md](docs/calibration.md) | Checkerboard algorithm, quality targets, shooting guide |
| [docs/autocalib.md](docs/autocalib.md) | Kruppa self-calibration theory, API, troubleshooting |
| [docs/reconstruction.md](docs/reconstruction.md) | DLT triangulation, voxel carving internals |
| [docs/detectors.md](docs/detectors.md) | All four detector backends, custom detector guide |
| [docs/export.md](docs/export.md) | HDF5 schema, PLY variants, viewer JSON format |
| [docs/api.md](docs/api.md) | Complete public API reference |

---

## Running the tests

```bash
pip install -e ".[dev]"
python -m pytest tests/ -v
```

Expected output: 41 passed in ~3 s. No video files, GPU, or MediaPipe
installation required.

---

## Recommended workflow

```
┌─────────────────────────────────────────────────────┐
│  Calibration (once per camera setup)                │
│                                                     │
│  Have checkerboard? ──Yes──► rpimocap-calibrate     │
│          │                         │               │
│          No                        ▼               │
│          ▼               calibration.npz           │
│  rpimocap-autocalib ──────────────►                │
└─────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────┐
│  Per-session reconstruction                         │
│                                                     │
│  rpimocap-run --detector dlc --dlc0 cam0.csv ...   │
│                    │                               │
│                    ▼                               │
│     output/reconstruction.h5                       │
│     output/viewer_data.json                        │
└─────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────┐
│  Analysis                                           │
│                                                     │
│  Python/MATLAB: h5py / h5read                      │
│  Visualisation: browser viewer / MeshLab / Blender │
└─────────────────────────────────────────────────────┘
```

---

## License

MIT — see `LICENSE`.
