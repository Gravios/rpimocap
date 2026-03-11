"""
rpimocap-calibrate — Checkerboard stereo calibration
=====================================================
Entry point: rpimocap.cli.calibrate:main

Wraps rpimocap.calibration.checkerboard as a standalone CLI tool.

Usage
-----
    rpimocap-calibrate \\
        --cam0   data/calib_cam0.mp4 \\
        --cam1   data/calib_cam1.mp4 \\
        --pattern 9x6 \\
        --square  25.0 \\
        --out     calibration.npz
"""

# Delegate entirely to the library module's main()
from rpimocap.calibration.checkerboard import main

__all__ = ["main"]
