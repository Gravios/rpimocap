# Camera Calibration

rpimocap supports two calibration routes: the classical checkerboard method
(`rpimocap-calibrate`) and self-calibration from subject motion
(`rpimocap-autocalib`). Both produce a `.npz` file in the same schema that
can be consumed by `rpimocap-run`.

---

## Checkerboard calibration

### How it works

rpimocap uses OpenCV's `stereoCalibrate` pipeline:

1. **Corner detection** — `cv2.findChessboardCorners` + subpixel refinement
   (`cornerSubPix`) locates the inner-corner grid in each frame.
2. **Intrinsic calibration** — per-camera `calibrateCamera` minimises
   reprojection error over object→image correspondences to recover the 3×3
   intrinsic matrix K and distortion coefficients.
3. **Stereo calibration** — `stereoCalibrate` (with `CALIB_FIX_INTRINSIC`)
   estimates the rotation R and translation T between the two camera frames.
4. **Stereo rectification** — `stereoRectify` computes the rectification
   transforms R0, R1 and new projection matrices P0, P1. The rectified
   projection matrices satisfy the epipolar constraint that corresponding
   points lie on the same horizontal scan line, which is required for
   dense disparity (though rpimocap uses sparse triangulation and does not
   strictly require rectification).

### Distortion model

By default rpimocap uses OpenCV's **rational distortion model**
(`CALIB_RATIONAL_MODEL`), which has 8 coefficients (k1–k6, p1, p2) and
handles wide-angle lenses and action cameras well. Disable it with
`--no-rational` for standard lenses if the rational model overfits with
fewer than 30 calibration frames.

### Shooting a good calibration dataset

- **Coverage** — The board must cover all corners of the field of view.
  Skewing the board diagonally is more informative than holding it flat.
- **Diversity** — Vary tilt angle (±45°), rotation, and depth. The more
  diverse the pose distribution, the better-conditioned the solve.
- **Synchronisation** — Both cameras must show the same board position in
  each paired frame. Use a hardware trigger or export frame-matched video
  from your capture system. Frames where the board is partially visible in
  either camera are silently skipped.
- **Quantity** — 40–80 valid pairs is the sweet spot. More is rarely helpful
  and slows down the solve.
- **Lighting** — Even, diffuse light. Specular reflections on glossy boards
  destroy corner detection.
- **Motion blur** — Move slowly. If the board blurs at your shutter speed,
  it will not detect.

### Command reference

```
rpimocap-calibrate [OPTIONS]

Required
  --cam0    PATH      Camera 0 source (video file or image directory)
  --cam1    PATH      Camera 1 source
  --pattern WxH       Inner corner count, e.g. 9x6
  --square  FLOAT     Physical square edge length in mm

Optional
  --skip    INT       Sample every N frames from the source (default: 8)
  --max-pairs INT     Maximum paired detections to collect (default: 80)
  --no-rational       Use standard (5-coeff) distortion instead of rational
  --out     PATH      Output .npz path (default: calibration.npz)
  --quiet             Suppress per-pair progress output
```

### Output .npz schema

| Key | Shape | Description |
|-----|-------|-------------|
| `K0`, `K1` | (3, 3) | Intrinsic matrices |
| `dist0`, `dist1` | (1, 8) | Distortion coefficients |
| `R`, `T` | (3,3), (3,) | Stereo extrinsics (cam0→cam1) |
| `E`, `F` | (3, 3) | Essential and fundamental matrices |
| `P0`, `P1` | (3, 4) | Rectified projection matrices |
| `R0`, `R1` | (3, 3) | Rectification rotation transforms |
| `Q` | (4, 4) | Disparity-to-depth re-projection matrix |
| `map0x`, `map0y` | (H, W) | Undistort+rectify maps for cam0 |
| `map1x`, `map1y` | (H, W) | Undistort+rectify maps for cam1 |
| `image_size` | (2,) | (width, height) in pixels |
| `rms0`, `rms1` | scalar | Per-camera intrinsic RMS reprojection error |
| `rms_stereo` | scalar | Stereo RMS reprojection error |
| `epi_dist` | scalar | Mean symmetric epipolar distance |

### Quality interpretation

**Intrinsic RMS** measures how well the model fits the calibration frames
after optimisation. Values below 0.5 px are excellent. Values above 1.5 px
suggest insufficient pose diversity, a warped/non-flat board, or
corner-detection failures.

**Stereo RMS** measures how consistently both cameras agree on the 3D
positions of the calibration points. Values above 1.0 px often indicate
poor frame synchronisation (the board moved between the two exposures) or
lens distortion that is not well modelled.

**Epipolar distance** is the ground-truth geometric constraint: for matched
points, `p1ᵀ F p0 = 0`. A mean symmetric epipolar distance above 1 px
means the recovered fundamental matrix is inaccurate, which will degrade
triangulation.

### Using calibration in Python

```python
import numpy as np

cal = np.load("calibration.npz")

K0   = cal["K0"]          # (3, 3) camera 0 intrinsics
K1   = cal["K1"]          # (3, 3) camera 1 intrinsics
R, T = cal["R"], cal["T"] # stereo extrinsics
P0   = cal["P0"]          # (3, 4) projection matrix cam0 (rectified)
P1   = cal["P1"]          # (3, 4) projection matrix cam1 (rectified)

# Apply rectification to a frame
import cv2
frame_rect = cv2.remap(frame, cal["map0x"], cal["map0y"], cv2.INTER_LINEAR)
```

---

## Self-calibration

See [autocalib.md](autocalib.md) for the full self-calibration guide, which
estimates intrinsics from subject motion without any calibration target.

### When to use which method

| Situation | Recommended method |
|-----------|-------------------|
| Checkerboard available, < 5 min to shoot | Checkerboard |
| Cameras will be repositioned between experiments | Checkerboard (each time) |
| Retrospective calibration from existing footage | Self-calibration |
| Strong wide-angle distortion (action cameras) | Checkerboard (rational model) |
| Identical consumer cameras, arena dimensions known | Self-calibration |
| Cross-validating another calibration | Both (compare K values) |
