"""
detect2d.py — 2D keypoint detection with swappable backends
============================================================
Provides a base class PoseDetector2D and three concrete implementations:

  MediaPipePoseDetector  — Human pose, 33 landmarks (MediaPipe Pose)
  CentroidPoseDetector   — Single-subject centroid + ellipse axis (any species)
  CSVPoseDetector        — Load pre-computed keypoints from DLC / SLEAP CSV

For rodent subjects, the recommended path is to run DeepLabCut or SLEAP
offline, export keypoints to CSV, and use CSVPoseDetector here.  The
CentroidPoseDetector provides a zero-training fallback.

All detectors return Pose2DResult objects with a list of Keypoint2D,
keeping the triangulation stage detector-agnostic.
"""

from __future__ import annotations

import csv
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


# --------------------------------------------------------------------------- #
#  Data classes                                                                #
# --------------------------------------------------------------------------- #

@dataclass
class Keypoint2D:
    """A single 2D keypoint detection in pixel space.

        Attributes
        ----------
        name       : landmark identifier (e.g. ``"nose"``, ``"left_shoulder"``)
        x, y       : pixel coordinates
        confidence : detector confidence in [0, 1] (visibility for MediaPipe,
                     likelihood for DLC/SLEAP, 1.0 for the centroid detector)
        """
    name: str
    x: float
    y: float
    confidence: float = 1.0

    def as_array(self) -> np.ndarray:
        """Return (x, y) as a (2,) float64 NumPy array."""
        return np.array([self.x, self.y], dtype=np.float64)


@dataclass
class Pose2DResult:
    """Detection result for a single frame from a single camera.

        Attributes
        ----------
        frame_idx : index of the source video frame
        detected  : True if at least one keypoint was found
        keypoints : list of Keypoint2D, may be empty if detected is False
        """
    frame_idx: int
    detected: bool
    keypoints: list[Keypoint2D] = field(default_factory=list)

    def by_name(self) -> dict[str, Keypoint2D]:
        """Return a ``{name: Keypoint2D}`` dict for O(1) lookup by landmark name."""
        return {kp.name: kp for kp in self.keypoints}


# --------------------------------------------------------------------------- #
#  Abstract base                                                               #
# --------------------------------------------------------------------------- #

class PoseDetector2D(ABC):
    """Base class for 2D pose detectors."""

    @property
    @abstractmethod
    def keypoint_names(self) -> list[str]:
        """Ordered list of landmark names this detector produces."""
        ...

    @property
    def skeleton_edges(self) -> list[tuple[str, str]]:
        """Name-pairs defining skeleton connectivity. Override in subclasses."""
        return []

    @abstractmethod
    def detect(self, frame: np.ndarray, frame_idx: int) -> Pose2DResult:
        """Run detection on a single BGR frame. Must be thread-safe."""
        ...

    def detect_batch(self, frames: list[np.ndarray],
                     start_idx: int = 0) -> list[Pose2DResult]:
        """Run detection on a list of BGR frames, returning one result per frame.

            Parameters
            ----------
            frames    : list of BGR uint8 arrays
            start_idx : frame index assigned to the first element
            """
        return [self.detect(f, start_idx + i) for i, f in enumerate(frames)]

    def close(self):
        """Release resources (e.g. GPU handles)."""
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# --------------------------------------------------------------------------- #
#  MediaPipe (human, 33 landmarks)                                             #
# --------------------------------------------------------------------------- #

class MediaPipePoseDetector(PoseDetector2D):
    """
    Human pose detection via MediaPipe Pose.
    Requires: pip install mediapipe

    Model complexity 2 gives the highest accuracy at the cost of inference
    speed. Set model_complexity=1 for faster processing on embedded hardware.
    """

    _NAMES = [
        "nose",
        "left_eye_inner", "left_eye", "left_eye_outer",
        "right_eye_inner", "right_eye", "right_eye_outer",
        "left_ear", "right_ear",
        "mouth_left", "mouth_right",
        "left_shoulder", "right_shoulder",
        "left_elbow", "right_elbow",
        "left_wrist", "right_wrist",
        "left_pinky", "right_pinky",
        "left_index", "right_index",
        "left_thumb", "right_thumb",
        "left_hip", "right_hip",
        "left_knee", "right_knee",
        "left_ankle", "right_ankle",
        "left_heel", "right_heel",
        "left_foot_index", "right_foot_index",
    ]

    _EDGES = [
        # Face
        ("nose", "left_eye"),
        ("nose", "right_eye"),
        ("left_eye", "left_ear"),
        ("right_eye", "right_ear"),
        # Torso
        ("left_shoulder", "right_shoulder"),
        ("left_shoulder", "left_hip"),
        ("right_shoulder", "right_hip"),
        ("left_hip", "right_hip"),
        # Left arm
        ("left_shoulder", "left_elbow"),
        ("left_elbow", "left_wrist"),
        # Right arm
        ("right_shoulder", "right_elbow"),
        ("right_elbow", "right_wrist"),
        # Left leg
        ("left_hip", "left_knee"),
        ("left_knee", "left_ankle"),
        ("left_ankle", "left_heel"),
        ("left_ankle", "left_foot_index"),
        # Right leg
        ("right_hip", "right_knee"),
        ("right_knee", "right_ankle"),
        ("right_ankle", "right_heel"),
        ("right_ankle", "right_foot_index"),
    ]

    def __init__(self,
                 model_complexity: int = 2,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        try:
            import mediapipe as mp
        except ImportError:
            raise ImportError("pip install mediapipe")
        self._pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    @property
    def keypoint_names(self):
        """Return the 33 MediaPipe BlazePose landmark names in order."""
        return self._NAMES

    @property
    def skeleton_edges(self):
        """Return MediaPipe BlazePose skeleton connectivity (33 joints, 32 edges)."""
        return self._EDGES

    def detect(self, frame: np.ndarray, frame_idx: int) -> Pose2DResult:
        """Run MediaPipe BlazePose on one BGR frame and return a Pose2DResult."""
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self._pose.process(rgb)
        if not res.pose_landmarks:
            return Pose2DResult(frame_idx, False)
        kps = [
            Keypoint2D(
                name=name,
                x=lm.x * w,
                y=lm.y * h,
                confidence=lm.visibility,
            )
            for name, lm in zip(self._NAMES, res.pose_landmarks.landmark)
        ]
        return Pose2DResult(frame_idx, True, kps)

    def close(self):
        """Release any resources held by the detector (e.g. GPU handles, file handles)."""
        self._pose.close()


# --------------------------------------------------------------------------- #
#  Centroid detector (species-agnostic, background subtraction)               #
# --------------------------------------------------------------------------- #

class CentroidPoseDetector(PoseDetector2D):
    """
    Background-subtraction based detector producing a centroid, head estimate,
    and tail estimate from the subject's silhouette ellipse major axis.

    Suitable as a zero-training fallback for rodents and other compact subjects.
    For production rodent tracking, replace with DeepLabCut / SLEAP outputs
    loaded via CSVPoseDetector.

    Parameters
    ----------
    bg_history   : MOG2 background history length (frames)
    var_threshold: MOG2 variance threshold (increase if noisy)
    morph_ksize  : Morphological kernel for mask cleanup
    min_area_px  : Minimum contour area in pixels; smaller blobs are ignored
    """

    _NAMES = ["centroid", "head", "tail"]
    _EDGES = [("head", "centroid"), ("centroid", "tail")]

    def __init__(self,
                 bg_history: int = 300,
                 var_threshold: float = 40.0,
                 morph_ksize: int = 7,
                 min_area_px: int = 400):
        self._bg = cv2.createBackgroundSubtractorMOG2(
            history=bg_history,
            varThreshold=var_threshold,
            detectShadows=False,
        )
        self._morph_ksize = morph_ksize
        self._min_area = min_area_px

    @property
    def keypoint_names(self):
        """Return the three centroid-detector landmark names: centroid, head, tail."""
        return self._NAMES

    @property
    def skeleton_edges(self):
        """Return the two body-axis edges: head→centroid and centroid→tail."""
        return self._EDGES

    def detect(self, frame: np.ndarray, frame_idx: int) -> Pose2DResult:
        """Run MOG2 background subtraction and return centroid + head/tail axis points."""
        fg = self._bg.apply(frame)
        k = self._morph_ksize
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel)

        cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return Pose2DResult(frame_idx, False)

        c = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(c) < self._min_area:
            return Pose2DResult(frame_idx, False)

        M = cv2.moments(c)
        if M["m00"] == 0:
            return Pose2DResult(frame_idx, False)
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]

        kps = [Keypoint2D("centroid", cx, cy, 1.0)]

        if len(c) >= 5:
            (ex, ey), (MA, ma), angle = cv2.fitEllipse(c)
            rad = np.deg2rad(angle)
            half = MA / 2.0
            dx, dy = half * np.cos(rad), half * np.sin(rad)
            kps.append(Keypoint2D("head", ex + dx, ey + dy, 0.75))
            kps.append(Keypoint2D("tail", ex - dx, ey - dy, 0.75))

        return Pose2DResult(frame_idx, True, kps)


# --------------------------------------------------------------------------- #
#  CSV loader (DLC / SLEAP output)                                             #
# --------------------------------------------------------------------------- #

class CSVPoseDetector(PoseDetector2D):
    """
    Load pre-computed 2D keypoints exported from DeepLabCut or SLEAP.

    DeepLabCut CSV format (multi-animal or single-animal):
        Row 0: scorer
        Row 1: individuals (optional)
        Row 2: bodyparts
        Row 3: coords  (x, y, likelihood)
        Row 4+: frame data

    SLEAP CSV format:
        columns: frame_idx, track, node, x, y, score

    Set fmt='dlc' or fmt='sleap'. For DLC, set individual= to select an
    animal if multi-animal tracking was used.
    """

    def __init__(self, csv_path: str,
                 fmt: str = "dlc",
                 individual: Optional[str] = None,
                 min_likelihood: float = 0.1):
        self._path = Path(csv_path)
        self._fmt = fmt.lower()
        self._individual = individual
        self._min_likelihood = min_likelihood
        self._frames: dict[int, list[Keypoint2D]] = {}
        self._names: list[str] = []
        self._edges: list[tuple[str, str]] = []
        self._load()

    def _load(self):
        if self._fmt == "dlc":
            self._load_dlc()
        elif self._fmt == "sleap":
            self._load_sleap()
        else:
            raise ValueError(f"Unknown format: {self._fmt!r}. Use 'dlc' or 'sleap'.")

    def _load_dlc(self):
        with open(self._path, newline="") as f:
            rows = list(csv.reader(f))

        # Find header rows
        # Row structure: scorer / [individuals] / bodyparts / coords
        i = 0
        individuals_row = None
        while i < len(rows):
            if rows[i][0].lower() == "scorer":
                i += 1
                if rows[i][0].lower() == "individuals":
                    individuals_row = rows[i]
                    i += 1
                bodyparts_row = rows[i]
                i += 1
                coords_row = rows[i]
                i += 1
                break
            i += 1
        data_rows = rows[i:]

        # Identify column groups: each keypoint has 3 cols (x, y, likelihood)
        # Skip the first column (frame index / filename)
        groups = []
        col = 1
        while col < len(bodyparts_row) - 2:
            bp = bodyparts_row[col]
            ind = individuals_row[col] if individuals_row else None
            if self._individual and ind and ind != self._individual:
                col += 3
                continue
            name = f"{ind}_{bp}" if (ind and ind not in ("", "bodyparts")) else bp
            groups.append((name, col))
            col += 3

        self._names = [g[0] for g in groups]

        for row in data_rows:
            if not row or not row[0]:
                continue
            try:
                fidx = int(float(row[0]))
            except ValueError:
                # DLC sometimes puts the filename; use row index
                fidx = data_rows.index(row)
            kps = []
            for name, c in groups:
                try:
                    x = float(row[c])
                    y = float(row[c + 1])
                    lk = float(row[c + 2])
                except (IndexError, ValueError):
                    continue
                if lk >= self._min_likelihood:
                    kps.append(Keypoint2D(name, x, y, lk))
            self._frames[fidx] = kps

    def _load_sleap(self):
        with open(self._path, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        nodes = sorted(set(r["node"] for r in rows))
        self._names = nodes

        frame_data: dict[int, dict[str, Keypoint2D]] = {}
        for row in rows:
            fidx = int(row["frame_idx"])
            node = row["node"]
            track = row.get("track", "")
            if self._individual and track and track != self._individual:
                continue
            try:
                x, y = float(row["x"]), float(row["y"])
                score = float(row.get("score", 1.0))
            except ValueError:
                continue
            if score >= self._min_likelihood:
                if fidx not in frame_data:
                    frame_data[fidx] = {}
                frame_data[fidx][node] = Keypoint2D(node, x, y, score)

        for fidx, d in frame_data.items():
            self._frames[fidx] = list(d.values())

    @property
    def keypoint_names(self) -> list[str]:
        """Return landmark names as loaded from the CSV header."""
        """Return landmark names as loaded from the CSV header."""
        return self._names

    @property
    def skeleton_edges(self) -> list[tuple[str, str]]:
        """Return skeleton edges provided at construction time (default: empty)."""
        """Return skeleton edges provided at construction time (default: empty)."""
        return self._edges

    def set_edges(self, edges: list[tuple[str, str]]):
        """Optionally set anatomical connectivity for visualization."""
        self._edges = edges

    def detect(self, frame: np.ndarray, frame_idx: int) -> Pose2DResult:
        """Look up pre-computed keypoints for frame_idx from the loaded CSV."""
        kps = self._frames.get(frame_idx, [])
        return Pose2DResult(frame_idx, bool(kps), kps)
