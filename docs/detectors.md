# Detection Backends

All pose detectors in `rpimocap.detection` share a common interface defined
by the abstract base class `PoseDetector2D`. This makes it straightforward
to swap backends without changing the reconstruction pipeline.

---

## The PoseDetector2D interface

```python
from rpimocap.detection import PoseDetector2D, Pose2DResult, Keypoint2D
```

Every backend must implement three properties/methods:

| Member | Type | Description |
|--------|------|-------------|
| `keypoint_names` | `list[str]` | Ordered list of landmark identifiers |
| `skeleton_edges` | `list[tuple[str, str]]` | Pairs of names defining connected joints |
| `detect(frame, frame_idx)` | `Pose2DResult` | Run detection on one BGR frame |

And one optional method:

| Member | Default | Description |
|--------|---------|-------------|
| `detect_batch(frames, start_idx)` | Calls `detect` in a loop | Vectorised multi-frame inference |
| `close()` | No-op | Release GPU handles, file handles, etc. |

### Keypoint2D

```python
@dataclass
class Keypoint2D:
    name: str
    x: float          # pixel column
    y: float          # pixel row
    confidence: float # 0.0–1.0
```

### Pose2DResult

```python
@dataclass
class Pose2DResult:
    frame_idx: int
    detected: bool
    keypoints: list[Keypoint2D]

    def by_name(self) -> dict[str, Keypoint2D]: ...
```

---

## MediaPipePoseDetector

**Best for:** human subjects, laboratory or clinical motion analysis.

Wraps Google's MediaPipe BlazePose model, which provides 33 body landmarks
with sub-centimetre accuracy in good lighting conditions.

### Installation

```bash
pip install "rpimocap[mediapipe]"
# or directly:
pip install mediapipe>=0.10
```

### Usage

```python
from rpimocap.detection import MediaPipePoseDetector

detector = MediaPipePoseDetector(
    model_complexity=1,    # 0=lite, 1=full, 2=heavy
    min_detection_conf=0.5,
    min_tracking_conf=0.5,
)

result = detector.detect(frame_bgr, frame_idx=42)
if result.detected:
    kps = result.by_name()
    nose_x, nose_y = kps["nose"].x, kps["nose"].y
    print(f"Nose at ({nose_x:.1f}, {nose_y:.1f}), confidence {kps['nose'].confidence:.2f}")

detector.close()  # releases the model
```

### Landmark names

The 33 BlazePose landmarks in order:

```
nose, left_eye_inner, left_eye, left_eye_outer, right_eye_inner,
right_eye, right_eye_outer, left_ear, right_ear, mouth_left,
mouth_right, left_shoulder, right_shoulder, left_elbow, right_elbow,
left_wrist, right_wrist, left_pinky, right_pinky, left_index,
right_index, left_thumb, right_thumb, left_hip, right_hip,
left_knee, right_knee, left_ankle, right_ankle, left_heel,
right_heel, left_foot_index, right_foot_index
```

### Notes

- Uses the `visibility` field as the confidence score; landmarks behind
  the body or outside the frame have low visibility.
- `model_complexity=2` is marginally more accurate but ~3× slower.
- Requires `pip install mediapipe` — not installed by default to keep the
  base install lightweight.

---

## CentroidPoseDetector

**Best for:** rodent tracking, any animal with a uniform or textured coat,
fast prototyping without any labelled data.

Uses MOG2 background subtraction to extract the foreground blob, then
fits an ellipse to compute the body axis.

### Usage

```python
from rpimocap.detection import CentroidPoseDetector

detector = CentroidPoseDetector(
    history=200,          # frames for background model
    var_threshold=40,     # MOG2 sensitivity
    min_area=300,         # minimum foreground blob area in px²
    morph_ksize=5,        # morphological open/close kernel size
)

# Feed some frames to build the background model first
for warm_up_frame in first_30_seconds:
    detector.detect(warm_up_frame, frame_idx=-1)   # output discarded

# Now use it
result = detector.detect(frame, frame_idx=0)
kps = result.by_name()
cx, cy = kps["centroid"].x, kps["centroid"].y
```

### Keypoints returned

| Name | Description |
|------|-------------|
| `centroid` | Centroid of the foreground blob |
| `head` | First ellipse axis endpoint (assumed head direction) |
| `tail` | Second ellipse axis endpoint |

**Head/tail ambiguity:** the detector cannot distinguish head from tail.
Post-process the trajectory using a smoothness prior or an additional
marker if directionality matters.

### Notes

- Zero training required — works immediately on novel subjects.
- Sensitive to lighting changes, shadows, and reflections. Tune
  `var_threshold` and `morph_ksize` for your arena.
- The background model is initialised from the first `history` frames,
  so keep the subject out of frame at the start of recording or pass a
  background-only clip to `warm_up()`.

---

## CSVPoseDetector

**Best for:** integrating pre-computed DeepLabCut or SLEAP trajectories.

Reads CSV exports from DLC or SLEAP and returns keypoints for each frame
without running any inference.

### DeepLabCut

```python
from rpimocap.detection import CSVPoseDetector

detector = CSVPoseDetector(
    csv_path="session_cam0DLC_resnet50_shuffled1.csv",
    fmt="dlc",
    min_likelihood=0.6,    # DLC's p-cut; low-confidence points get confidence=0
    skeleton_edges=[("nose", "neck"), ("neck", "body"), ("body", "tail_base")],
)

result = detector.detect(frame=None, frame_idx=100)
# frame is ignored — all data comes from the CSV
```

The `frame` argument is unused and can be `None`. This allows the
CSVPoseDetector to be used even without the original video files, as long as
the CSV spans the required frame indices.

### SLEAP

```python
detector = CSVPoseDetector(
    csv_path="session_cam0.analysis.csv",
    fmt="sleap",
    min_likelihood=0.5,
)
```

SLEAP exports include a `score` column that maps to the confidence field.

### CSV format expectations

**DLC format** (`fmt="dlc"`):

```
scorer,       DLC_scorer, DLC_scorer, DLC_scorer, DLC_scorer, ...
bodyparts,    nose,       nose,       neck,       neck,       ...
coords,       x,          y,          x,          y,          ...
0,            412.3,      298.1,      435.2,      310.4,      ...
```

The top three header rows are the standard DLC multi-index CSV. Frame
indices are in the first column (0-based).

**SLEAP format** (`fmt="sleap"`):

```
frame_idx, instance, node, x, y, score
0,         0,        nose, 412.3, 298.1, 0.95
0,         0,        neck, 435.2, 310.4, 0.88
```

Only the first track/instance is used for single-animal experiments.

### Notes

- The `detect_batch` method on `CSVPoseDetector` is O(1) per frame — it
  does a single DataFrame lookup rather than looping.
- Frame indices in the CSV must match the frame indices passed to `detect`.
  If your CSV uses 1-based indexing, pass `frame_index_offset=-1` at
  construction.
- DLC's `p-cut` (likelihood threshold) is separate from rpimocap's
  `min_confidence` in `triangulate_keypoints`. Apply both for belt-and-
  braces filtering.

---

## Writing a custom detector

Subclass `PoseDetector2D` and implement the three abstract members:

```python
from rpimocap.detection.detectors import PoseDetector2D, Pose2DResult, Keypoint2D
import numpy as np
import cv2

class ThresholdBlobDetector(PoseDetector2D):
    """Simple colour-threshold blob detector for a brightly coloured marker."""

    def __init__(self, lower_hsv, upper_hsv):
        self._lower = np.array(lower_hsv, np.uint8)
        self._upper = np.array(upper_hsv, np.uint8)

    @property
    def keypoint_names(self) -> list[str]:
        return ["marker"]

    @property
    def skeleton_edges(self) -> list[tuple[str, str]]:
        return []   # single-point; no edges

    def detect(self, frame: np.ndarray, frame_idx: int = 0) -> Pose2DResult:
        hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self._lower, self._upper)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return Pose2DResult(frame_idx=frame_idx, detected=False, keypoints=[])
        c = max(cnts, key=cv2.contourArea)
        M = cv2.moments(c)
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        conf = min(1.0, cv2.contourArea(c) / 500)   # area-based confidence
        kp = Keypoint2D(name="marker", x=cx, y=cy, confidence=conf)
        return Pose2DResult(frame_idx=frame_idx, detected=True, keypoints=[kp])
```

Pass an instance to `rpimocap-run` via `--detector custom` and
`--detector-module path/to/module.py:ThresholdBlobDetector` (not yet
supported by the CLI — instantiate programmatically via `pipeline.run()`).

---

## Choosing a detector

| Factor | MediaPipe | Centroid | CSV (DLC/SLEAP) |
|--------|-----------|----------|-----------------|
| Training data needed | No | No | Yes (DLC/SLEAP model) |
| GPU for inference | Optional | No | No (CSV already computed) |
| Species | Human | Any | Any |
| Landmark count | 33 | 3 | Model-dependent |
| Robustness to occlusion | High | Medium | Model-dependent |
| Setup time | Minutes | Seconds | Hours–days (training) |
| Reconstruction accuracy | High | Low | High |
