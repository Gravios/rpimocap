"""The refinement must recompute dlt_P0/dlt_P1 from the refined geometry,
so downstream tools (stereo_gate, stereo_diagnose) that read those keys
pick up the refinement instead of the stale pre-refinement DLT."""
import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
pytest.importorskip("scipy")

from rpimocap.reconstruction.align import (
    refine_calibration_from_arena, AlignPoint)
from rpimocap.reconstruction.triangulate import triangulate_dlt


def _rig():
    B = (-140, 140, -215, 215, 0, 388)
    corners = np.array([
        [B[0], B[2], 0], [B[1], B[2], 0], [B[1], B[3], 0], [B[0], B[3], 0],
        [B[0], B[2], B[5]], [B[1], B[2], B[5]], [B[1], B[3], B[5]],
        [B[0], B[3], B[5]]], float)
    labels = ['BFL', 'BFR', 'BBR', 'BBL', 'TFL', 'TFR', 'TBR', 'TBL']
    K = np.array([[1500, 0, 1014], [0, 1500, 540], [0, 0, 1.]])

    def look(cp, la):
        cp = np.array(cp, float); la = np.array(la, float)
        f = la - cp; f /= np.linalg.norm(f)
        up = np.array([0, 0, 1.]); r = np.cross(f, up); r /= np.linalg.norm(r)
        u = np.cross(r, f)
        return np.vstack([r, -u, f]), (-np.vstack([r, -u, f]) @ cp).reshape(3, 1)

    R0, t0 = look([-300, -400, 700], [0, 0, 194])
    R1, t1 = look([300, -400, 700], [0, 0, 194])
    P0 = K @ np.hstack([R0, t0]); P1 = K @ np.hstack([R1, t1])

    def proj(P, X):
        p = P @ np.append(X, 1); return p[:2] / p[2]
    px0 = np.array([proj(P0, X) for X in corners])
    px1 = np.array([proj(P1, X) for X in corners])
    Rs = R1 @ R0.T; Ts = (t1 - Rs @ t0)
    return corners, labels, K, Rs, Ts, px0, px1


def test_refine_recomputes_dlt(tmp_path):
    corners, labels, K, Rs, Ts, px0, px1 = _rig()
    seed = str(tmp_path / "seed.npz")
    # deliberately WRONG seed dlt (scaled) to prove refine overwrites it
    np.savez(seed, K0=K, K1=K, R=Rs, T=Ts,
             dist0=np.zeros((1, 5)), dist1=np.zeros((1, 5)),
             dlt_P0=np.ones((3, 4)), dlt_P1=np.ones((3, 4)))
    pts = [AlignPoint(rec_xyz=corners[i], arena_xyz=corners[i],
                      label=labels[i], px0=px0[i], px1=px1[i])
           for i in range(8)]
    out = str(tmp_path / "refined.npz")
    refine_calibration_from_arena(pts, [], seed, out, verbose=False)

    c = np.load(out)
    # refined dlt matrices must triangulate the corners back to truth
    errs = [np.linalg.norm(
        triangulate_dlt(c["dlt_P0"], c["dlt_P1"],
                        tuple(px0[i]), tuple(px1[i]))[:3] - corners[i])
        for i in range(8)]
    assert max(errs) < 1.0
    # and they must NOT be the stale seed (all-ones)
    assert not np.allclose(c["dlt_P0"], 1.0)
