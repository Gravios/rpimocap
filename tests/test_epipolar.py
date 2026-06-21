"""Tests for tightly-coupled two-view stereo selection."""
import numpy as np

from rpimocap.reconstruction import epipolar as ep
from rpimocap.model import synthetic_dataset as sd


def _make_P(cam_pos, look_at, f=1500, cx=1014, cy=540):
    cam_pos = np.asarray(cam_pos, float)
    look_at = np.asarray(look_at, float)
    fwd = look_at - cam_pos
    fwd /= np.linalg.norm(fwd)
    up = np.array([0, 0, 1.0])
    right = np.cross(fwd, up); right /= np.linalg.norm(right)
    up2 = np.cross(right, fwd)
    R = np.vstack([right, -up2, fwd])
    t = -R @ cam_pos
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
    return K @ np.hstack([R, t.reshape(3, 1)])


P0 = _make_P([-300, -400, 700], [0, 0, 194])
P1 = _make_P([300, -400, 700], [0, 0, 194])


def _proj(P, X):
    p = P @ np.append(X, 1.0)
    return (p[0] / p[2], p[1] / p[2])


class TestFundamental:

    def test_true_correspondence_zero_epipolar(self):
        F = ep.fundamental_from_projections(P0, P1)
        X = np.array([20.0, -50.0, 120.0])
        d = ep.epipolar_distance(F, _proj(P0, X), _proj(P1, X))
        assert d < 1e-3

    def test_off_line_point_large_epipolar(self):
        F = ep.fundamental_from_projections(P0, P1)
        X = np.array([0.0, 0.0, 150.0])
        p0 = _proj(P0, X)
        p1 = _proj(P1, X)
        p1_off = (p1[0] + 200, p1[1] - 150)
        assert ep.epipolar_distance(F, p0, p1_off) > 20.0


class TestMatching:

    def test_picks_consistent_pair(self):
        X = np.array([20.0, -50.0, 120.0])
        p0, p1 = _proj(P0, X), _proj(P1, X)
        false1 = (p1[0] + 200, p1[1] - 150)
        m = ep.best_stereo_point([p0], [false1, p1], P0, P1)
        assert m is not None
        assert m.i1 == 1                       # the true blob
        assert np.linalg.norm(m.point - X) < 1e-3

    def test_rejects_false_only_cam1(self):
        X = np.array([20.0, -50.0, 120.0])
        p0, p1 = _proj(P0, X), _proj(P1, X)
        false1 = (p1[0] + 200, p1[1] - 150)
        assert ep.best_stereo_point([p0], [false1], P0, P1) is None

    def test_rejects_out_of_arena(self):
        # a pair that triangulates outside the arena is rejected when
        # require_in_arena is on
        X_out = np.array([0.0, 0.0, 1500.0])   # way above the arena
        p0, p1 = _proj(P0, X_out), _proj(P1, X_out)
        m = ep.best_stereo_point([p0], [p1], P0, P1,
                                 require_in_arena=True)
        assert m is None
        # but accepted when the gate is off
        m2 = ep.best_stereo_point([p0], [p1], P0, P1,
                                  require_in_arena=False)
        assert m2 is not None

    def test_one_to_one_assignment(self):
        """Two real points → two matches, each index used once."""
        XA = np.array([-40.0, 60.0, 100.0])
        XB = np.array([50.0, -80.0, 200.0])
        cand0 = [_proj(P0, XA), _proj(P0, XB)]
        cand1 = [_proj(P1, XA), _proj(P1, XB)]
        matches = ep.match_stereo_candidates(cand0, cand1, P0, P1)
        assert len(matches) == 2
        i0s = {m.i0 for m in matches}
        i1s = {m.i1 for m in matches}
        assert i0s == {0, 1} and i1s == {0, 1}

    def test_empty_candidates(self):
        assert ep.match_stereo_candidates([], [], P0, P1) == []
        assert ep.best_stereo_point([], [(1, 2)], P0, P1) is None


class TestAgainstSyntheticGroundTruth:

    def test_recovers_truth_amid_false_blobs(self):
        cams = {0: P0, 1: P1}
        ds = sd.generate_dataset(10, cams, (2028, 1080), seed=3)
        rng = np.random.RandomState(0)
        n_ok = 0
        for s in ds.samples:
            c0 = tuple(s.keypoints2d[0].mean(0))
            c1 = tuple(s.keypoints2d[1].mean(0))
            false1 = (c1[0] + rng.uniform(150, 300),
                      c1[1] + rng.uniform(-200, 200))
            m = ep.best_stereo_point(
                [c0], [false1, c1], P0, P1, max_reproj_px=10)
            if m is not None and m.i1 == 1:
                n_ok += 1
        assert n_ok >= 9
