"""
rpimocap.detection
==================
2D pose / keypoint detection backends.

All backends share the PoseDetector2D interface and return Pose2DResult
objects, keeping the triangulation stage detector-agnostic.

Backends
--------
MediaPipePoseDetector   Human pose (33 landmarks, requires mediapipe)
CentroidPoseDetector    Background-subtraction centroid + ellipse axis
CSVPoseDetector         Pre-computed DLC / SLEAP CSV keypoints
"""

from rpimocap.detection.detectors import (
    Keypoint2D,
    Pose2DResult,
    PoseDetector2D,
    MediaPipePoseDetector,
    CentroidPoseDetector,
    CSVPoseDetector,
)

__all__ = [
    "Keypoint2D",
    "Pose2DResult",
    "PoseDetector2D",
    "MediaPipePoseDetector",
    "CentroidPoseDetector",
    "CSVPoseDetector",
]
