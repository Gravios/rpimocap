# Installation

## Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Python | 3.10 | 3.11 |
| OpenCV | 4.8 | 4.10+ |
| NumPy | 1.24 | 2.x |
| SciPy | 1.11 | 1.13+ |
| scikit-image | 0.21 | 0.24+ |
| h5py | 3.9 | 3.11+ |

Python 3.11 is the recommended runtime. The scientific Python stack (NumPy,
SciPy, scikit-image) is most mature there and the CPython interpreter
improvements over 3.10 are measurable in the voxel carving inner loops.
Python 3.12 works but has no meaningful advantage for this workload.
Python 3.13 is not yet supported due to incomplete scipy/scikit-image wheel
availability.

## Standard install

```bash
# Clone or unpack the source
tar xzf rpimocap-0.1.0.tar.gz
cd rpimocap

# Create a virtual environment (strongly recommended)
python3.11 -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows

# Install with all runtime dependencies
pip install -e .
```

## Optional backends

### MediaPipe (human pose, 33 landmarks)

Required only when using `--detector mediapipe` in the pipeline or
instantiating `MediaPipePoseDetector` directly.

```bash
pip install -e ".[mediapipe]"
```

### DeepLabCut / SLEAP

No additional Python package is required — rpimocap reads the CSV export
files that DLC and SLEAP produce. Run your tracking framework normally,
export to CSV, then pass the files via `--dlc0` / `--dlc1` or
`--sleap0` / `--sleap1`.

## Development install

Includes pytest, ruff (linter), and mypy (type checker):

```bash
pip install -e ".[dev]"
python -m pytest tests/ -v        # run the test suite
```

## Verifying the install

```bash
rpimocap-calibrate --help
rpimocap-autocalib --help
rpimocap-run       --help
```

All three commands should print their usage without error.

```python
# Quick import smoke test
import rpimocap
print(rpimocap.__version__)   # 0.1.0

from rpimocap.reconstruction import triangulate_dlt, VoxelGrid
from rpimocap.calibration.autocalib import make_K, run_focal_estimation
from rpimocap.io import write_hdf5
```

## Platform notes

### Linux (Ubuntu 22.04 / 24.04)

Fully supported and the primary development platform. Ensure the system
OpenCV is not shadowing the pip-installed version:

```bash
python -c "import cv2; print(cv2.__version__, cv2.__file__)"
# Should point into your venv, not /usr/lib/python3/
```

If running on a machine with an NVIDIA GPU (RTX 5070 Ti or similar),
use the CUDA-enabled OpenCV build for accelerated background subtraction:

```bash
pip install opencv-contrib-python  # instead of opencv-python
```

### Raspberry Pi 5

rpimocap runs on Pi 5 for capture coordination but **reconstruction should
be performed on a desktop workstation**. The voxel carving stage is
memory-bound and benefits from the large L3 cache on desktop CPUs
(e.g. Ryzen 9800X3D with 3D V-Cache).

Install headless OpenCV on Pi to avoid pulling in Qt/GTK:

```bash
pip install opencv-python-headless
```

### macOS

Supported. Use `brew install python@3.11` and a venv as above.
The Metal-accelerated OpenCV build is not required and not used.

### Windows

Supported but not tested as a primary target. Use a virtual environment
and the `opencv-python` wheel from PyPI. The `viewer/assets/index.html`
works in any browser regardless of OS.
