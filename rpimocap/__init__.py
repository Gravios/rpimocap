"""
rpimocap — Raspberry Pi multi-camera 3D motion capture
=======================================================
A Python package for stereo 3D reconstruction of a moving subject
within a constrained space from two synchronous video perspectives.

Sub-packages
------------
rpimocap.calibration        Stereo camera calibration (checkerboard + self-calibration)
rpimocap.detection          2D pose / keypoint detection backends
rpimocap.reconstruction     DLT triangulation, voxel carving, mesh extraction
rpimocap.io                 PLY, HDF5, and viewer JSON export
rpimocap.viewer             Bundled Three.js viewer assets

CLI entry points (after pip install)
-------------------------------------
rpimocap-calibrate          Checkerboard stereo calibration
rpimocap-autocalib          Self-calibration from subject motion
rpimocap-run                Full reconstruction pipeline
"""

__version__ = "0.1.0"
__author__  = "rpimocap contributors"
