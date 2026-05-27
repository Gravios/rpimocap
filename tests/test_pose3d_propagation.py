"""
tests/test_pose3d_propagation.py
==================================
Regression for the silent bug that made every H5
/skeleton/<name>/reprojection_error column 0.00 for every frame.

Root cause: TrackResult.pose3d was constructing Point3D(name=k, xyz=v)
without passing reproj_err or confidence, so the dataclass defaults
(reprojection_error=0.0, confidence=1.0) shadowed the measured values
all the way through to the H5 writer.
"""
from __future__ import annotations

import numpy as np


def _track_result(xyz, reproj_err, conf=0.9):
    from rpimocap.detection.tracker import TrackResult
    from rpimocap.detection.segment import BodyRegion
    return TrackResult(
        frame_idx=0,
        regions_cam0=[BodyRegion(label="animal", cx=100.0, cy=200.0,
                                  area_px=500, confidence=conf, mask=None)],
        regions_cam1=[BodyRegion(label="animal", cx=110.0, cy=210.0,
                                  area_px=500, confidence=conf, mask=None)],
        xyz={"animal": np.asarray(xyz, dtype=np.float64)},
        reproj_err={"animal": reproj_err},
        detected=True,
    )


class TestPose3dPropagation:

    def test_reprojection_error_propagates(self):
        """When reproj_err = (3 px, 4 px), pose3d Point3D should hold
        the RMS = sqrt((9 + 16) / 2) ≈ 3.536 px."""
        tr = _track_result((10.0, 20.0, 30.0), (3.0, 4.0))
        pts = tr.pose3d
        assert len(pts) == 1
        expected = float(np.sqrt(0.5 * (9.0 + 16.0)))
        assert abs(pts[0].reprojection_error - expected) < 1e-6, \
            f"got {pts[0].reprojection_error}, expected {expected}"

    def test_reprojection_error_zero_when_symmetric(self):
        tr = _track_result((0.0, 0.0, 0.0), (0.0, 0.0))
        pts = tr.pose3d
        assert pts[0].reprojection_error == 0.0

    def test_confidence_propagates_from_cam0(self):
        tr = _track_result((1.0, 2.0, 3.0), (1.0, 1.0), conf=0.42)
        pts = tr.pose3d
        assert abs(pts[0].confidence - 0.42) < 1e-6

    def test_missing_reproj_err_defaults_to_zero(self):
        """If the matcher didn't compute an error (e.g. it raised),
        the Point3D should still construct with reprojection_error=0,
        NOT crash."""
        from rpimocap.detection.tracker import TrackResult
        from rpimocap.detection.segment import BodyRegion
        tr = TrackResult(
            frame_idx=0,
            regions_cam0=[BodyRegion(label="animal", cx=100.0, cy=200.0,
                                      area_px=500, confidence=1.0,
                                      mask=None)],
            regions_cam1=[BodyRegion(label="animal", cx=110.0, cy=210.0,
                                      area_px=500, confidence=1.0,
                                      mask=None)],
            xyz={"animal": np.asarray([1.0, 2.0, 3.0])},
            reproj_err={},   # ← empty: matcher failed/skipped
            detected=True,
        )
        pts = tr.pose3d
        assert pts[0].reprojection_error == 0.0

    def test_empty_xyz_returns_empty_list(self):
        from rpimocap.detection.tracker import TrackResult
        tr = TrackResult(frame_idx=0, regions_cam0=[], regions_cam1=[],
                         xyz={}, reproj_err={}, detected=False)
        assert tr.pose3d == []

    def test_h5_roundtrip_writes_nonzero_reprojection_error(self, tmp_path):
        """End-to-end: build TrackResult → results_to_skeleton_frames
        → write_hdf5 → read back. The reprojection_error column must
        contain the measured RMS, not zero."""
        import h5py
        from rpimocap.detection.tracker import SegmentTracker
        from rpimocap.io.export import write_hdf5

        # Two frames, one with measured error, one without
        tr0 = _track_result((10.0, 20.0, 30.0), (3.0, 4.0))
        tr1 = _track_result((11.0, 21.0, 31.0), (5.0, 12.0))
        tr1.frame_idx = 1
        skel = SegmentTracker.results_to_skeleton_frames([tr0, tr1])

        out = tmp_path / "test_reproj.h5"
        write_hdf5(path=str(out), skeleton_frames=skel, fps=25.0)

        with h5py.File(out) as f:
            err = f["skeleton"]["animal"]["reprojection_error"][:]

        # Frame 0: RMS of (3, 4) = sqrt(12.5) ≈ 3.536
        assert abs(err[0] - np.sqrt(12.5)) < 1e-4
        # Frame 1: RMS of (5, 12) = sqrt((25 + 144)/2) = sqrt(84.5) ≈ 9.192
        assert abs(err[1] - np.sqrt(84.5)) < 1e-4
        # Neither is 0.00 — the original bug
        assert err[0] > 0 and err[1] > 0
