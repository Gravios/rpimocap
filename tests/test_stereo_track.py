"""Tests for stereo-gated detection selection (in-arena pairing)."""
import numpy as np

from rpimocap.reconstruction import stereo_track as st


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


RAT = np.array([20.0, -50.0, 30.0])           # in-arena floor
PATCH = np.array([60.0, 320.0, 0.0])          # external floor, beyond +y wall


class TestGatedDetection:

    def test_picks_in_arena_rat(self):
        cand0 = [_proj(P0, PATCH), _proj(P0, RAT)]    # patch first
        cand1 = [_proj(P1, PATCH), _proj(P1, RAT)]
        det = st.gated_stereo_detection(cand0, cand1, P0, P1)
        assert det is not None
        assert det.i0 == 1 and det.i1 == 1            # the rat
        assert np.linalg.norm(det.point - RAT) < 1.0

    def test_rejects_only_patch(self):
        det = st.gated_stereo_detection(
            [_proj(P0, PATCH)], [_proj(P1, PATCH)], P0, P1)
        assert det is None

    def test_empty_candidates(self):
        assert st.gated_stereo_detection([], [], P0, P1) is None
        assert st.gated_stereo_detection(
            [(1, 2)], [], P0, P1) is None

    def test_below_floor_rejected(self):
        below = np.array([20.0, -50.0, -90.0])        # reflection
        det = st.gated_stereo_detection(
            [_proj(P0, below)], [_proj(P1, below)], P0, P1)
        assert det is None


class TestGateTrajectory:

    def test_multi_frame(self):
        c0, c1 = {}, {}
        # frames 0,1: rat present (+ patch); frame 2: only patch
        for fi, Xs in [(0, [PATCH, RAT]), (1, [RAT, PATCH]),
                       (2, [PATCH])]:
            c0[fi] = [_proj(P0, X) for X in Xs]
            c1[fi] = [_proj(P1, X) for X in Xs]
        dets = st.gate_trajectory(c0, c1, P0, P1)
        assert set(dets.keys()) == {0, 1}             # frame 2 rejected
        for fi in (0, 1):
            assert np.linalg.norm(dets[fi].point - RAT) < 1.0

    def test_disjoint_frames(self):
        c0 = {0: [_proj(P0, RAT)]}
        c1 = {5: [_proj(P1, RAT)]}
        assert st.gate_trajectory(c0, c1, P0, P1) == {}
