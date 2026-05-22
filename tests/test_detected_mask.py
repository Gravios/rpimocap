"""
tests/test_detected_mask.py
============================
Unit tests for the /skeleton/<name>/detected boolean dataset.

Verifies:
- detected_masks kwarg writes the bool dataset alongside xyz
- shape validation raises on mismatch
- the fallback path derives the mask from confidence + kalman_outlier
- a frame marked kalman_outlier=True is detected=False
"""
from __future__ import annotations

import numpy as np
import pytest

from rpimocap.io.export import write_hdf5
from rpimocap.reconstruction.triangulate import Point3D


@pytest.fixture
def h5py():
    return pytest.importorskip("h5py")


def _frames(n=20):
    """20-frame trajectory with one keypoint. Frames 10–14 are missing."""
    frames = []
    for i in range(n):
        if 10 <= i < 15:
            frames.append([])
        else:
            frames.append([Point3D(
                name="animal",
                xyz=np.array([float(i), 0.0, 100.0]),
                confidence=1.0,
                reprojection_error=2.0)])
    return frames


class TestDetectedMask:

    def test_explicit_mask_passes_through(self, tmp_path, h5py):
        frames = _frames(20)
        mask = np.zeros(20, dtype=bool)
        mask[[0, 1, 2, 18, 19]] = True
        write_hdf5(
            tmp_path / "out.h5", frames,
            detected_masks={"animal": mask},
            fps=25.0)
        with h5py.File(tmp_path / "out.h5", "r") as f:
            det = f["skeleton"]["animal"]["detected"][:]
            assert f["skeleton"]["animal"]["detected"].dtype == np.bool_
        np.testing.assert_array_equal(det, mask)

    def test_default_mask_from_confidence(self, tmp_path, h5py):
        frames = _frames(20)
        write_hdf5(tmp_path / "out.h5", frames, fps=25.0)
        with h5py.File(tmp_path / "out.h5", "r") as f:
            det = f["skeleton"]["animal"]["detected"][:]
        expected = np.ones(20, dtype=bool)
        expected[10:15] = False
        np.testing.assert_array_equal(det, expected)

    def test_kalman_outlier_marked_not_detected(self, tmp_path, h5py):
        frames = _frames(20)
        frames[5][0].kalman_outlier = True
        write_hdf5(tmp_path / "out.h5", frames, fps=25.0)
        with h5py.File(tmp_path / "out.h5", "r") as f:
            det = f["skeleton"]["animal"]["detected"][:]
        assert det[5] == False
        assert det[4]; assert det[6]

    def test_shape_validation_raises(self, tmp_path):
        frames = _frames(20)
        bad_mask = np.ones(10, dtype=bool)
        with pytest.raises(ValueError, match="shape"):
            write_hdf5(
                tmp_path / "out.h5", frames,
                detected_masks={"animal": bad_mask},
                fps=25.0)

    def test_detected_alongside_other_datasets(self, tmp_path, h5py):
        frames = _frames(20)
        write_hdf5(tmp_path / "out.h5", frames, fps=25.0)
        with h5py.File(tmp_path / "out.h5", "r") as f:
            ds_names = set(f["skeleton"]["animal"].keys())
        assert ds_names == {
            "xyz", "confidence", "reprojection_error", "detected"}
