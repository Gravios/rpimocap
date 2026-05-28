"""
tests/test_reprojection_error_signature.py
============================================
Regression for the silent TypeError that kept the H5
/reprojection_error column at 0.00 even after the pose3d
propagation fix (patch 0015).

Bug: EpipolarMatcher.reprojection_error() called the module-level
reprojection_error(P, X, pt) [3 args, one camera] as
reprojection_error(xyz, P0, P1, pt0, pt1) [5 args]. That raised
TypeError on every frame, which was swallowed by the try/except in
SegmentTracker._process_frame, leaving reproj_err empty.
"""
from __future__ import annotations

import numpy as np
import pytest


def _matcher():
    from rpimocap.detection.segment import EpipolarMatcher
    P0 = np.array([[800, 0, 512, 0],
                   [0, 800, 384, 0],
                   [0, 0, 1, 0]], dtype=float)
    P1 = np.array([[800, 0, 512, -800 * 60],
                   [0, 800, 384, 0],
                   [0, 0, 1, 0]], dtype=float)
    I3 = np.eye(3)
    return EpipolarMatcher(P0, P1, K0=I3, K1=I3,
                           dist0=np.zeros(5), dist1=np.zeros(5),
                           R=I3, T=np.zeros(3))


def _region(cx, cy):
    from rpimocap.detection.segment import BodyRegion
    return BodyRegion(label="animal", cx=cx, cy=cy,
                      area_px=100, confidence=1.0, mask=None)


def _proj(P, X):
    h = P @ np.append(X, 1.0)
    return h[:2] / h[2]


class TestReprojectionErrorSignature:

    def test_perfect_observation_gives_zero_error(self):
        m = _matcher()
        X = np.array([10.0, 20.0, 500.0])
        pt0 = _proj(m.P0, X)
        pt1 = _proj(m.P1, X)
        e0, e1 = m.reprojection_error(X, _region(*pt0), _region(*pt1))
        assert e0 < 1e-6
        assert e1 < 1e-6

    def test_perturbation_gives_expected_error(self):
        """A 5-px shift in cam1's observation must yield err1 ≈ 5 px."""
        m = _matcher()
        X = np.array([10.0, 20.0, 500.0])
        pt0 = _proj(m.P0, X)
        pt1 = _proj(m.P1, X)
        e0, e1 = m.reprojection_error(
            X, _region(*pt0), _region(pt1[0] + 5.0, pt1[1]))
        assert e0 < 1e-6
        assert abs(e1 - 5.0) < 1e-6

    def test_returns_tuple_of_two_floats(self):
        m = _matcher()
        X = np.array([0.0, 0.0, 500.0])
        out = m.reprojection_error(X, _region(512, 384), _region(512, 384))
        assert isinstance(out, tuple)
        assert len(out) == 2
        assert all(isinstance(v, float) for v in out)

    def test_does_not_raise_typeerror(self):
        """The original bug raised TypeError (5 args vs 3-arg sig).
        Guard against regressions of the call signature."""
        m = _matcher()
        X = np.array([10.0, 20.0, 500.0])
        try:
            m.reprojection_error(X, _region(500, 380), _region(500, 380))
        except TypeError as exc:
            pytest.fail(f"reprojection_error raised TypeError: {exc}")


class TestEndToEndReprojInH5:
    """Full path: TrackResult.reproj_err → pose3d → write_hdf5 →
    reread. The reprojection_error column must be non-zero when the
    matcher actually measured an error."""

    def test_reproj_err_survives_to_h5(self, tmp_path):
        import h5py
        from rpimocap.detection.tracker import SegmentTracker, TrackResult
        from rpimocap.detection.segment import BodyRegion
        from rpimocap.io.export import write_hdf5

        tr = TrackResult(
            frame_idx=0,
            regions_cam0=[BodyRegion(label="animal", cx=100.0, cy=200.0,
                                      area_px=500, confidence=0.9,
                                      mask=None)],
            regions_cam1=[BodyRegion(label="animal", cx=110.0, cy=210.0,
                                      area_px=500, confidence=0.9,
                                      mask=None)],
            xyz={"animal": np.array([10.0, 20.0, 30.0])},
            reproj_err={"animal": (2.0, 3.0)},   # measured by matcher
            detected=True,
        )
        skel = SegmentTracker.results_to_skeleton_frames([tr])
        out = tmp_path / "reproj.h5"
        write_hdf5(path=str(out), skeleton_frames=skel, fps=25.0)
        with h5py.File(out) as f:
            err = f["skeleton/animal/reprojection_error"][:]
        # RMS of (2, 3) = sqrt((4+9)/2) = sqrt(6.5) ≈ 2.55
        assert abs(err[0] - np.sqrt(6.5)) < 1e-4
        assert err[0] > 0   # the original bug made this 0.0
