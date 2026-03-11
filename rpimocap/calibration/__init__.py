"""
rpimocap.calibration
====================
Camera calibration sub-package.

Modules
-------
checkerboard    Checkerboard-based stereo calibration (OpenCV)
autocalib       Self-calibration from subject motion (Kruppa + arena metric)
"""

from rpimocap.calibration.checkerboard import (
    detect_corners_paired,
    calibrate_intrinsics,
    calibrate_stereo,
    stereo_rectify,
    validate_epipolar,
)

__all__ = [
    "detect_corners_paired",
    "calibrate_intrinsics",
    "calibrate_stereo",
    "stereo_rectify",
    "validate_epipolar",
]
