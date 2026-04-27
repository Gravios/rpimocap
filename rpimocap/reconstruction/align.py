"""
rpimocap.reconstruction.align
==============================
Rigid-body alignment, plumb-line distortion fitting, and bundle adjustment
using annotated arena corners and traced straight edges.
"""
from __future__ import annotations
import csv, math
from dataclasses import dataclass
from pathlib import Path
import numpy as np


# --------------------------------------------------------------------------- #
#  Data types                                                                  #
# --------------------------------------------------------------------------- #

@dataclass
class AlignPoint:
    rec_xyz:   np.ndarray
    arena_xyz: np.ndarray
    label:     str = ""
    px0:       np.ndarray = None   # type: ignore[assignment]
    px1:       np.ndarray = None   # type: ignore[assignment]


@dataclass
class AlignResult:
    R:        np.ndarray
    t:        np.ndarray
    rmse_mm:  float
    n_points: int

    def apply(self, xyz: np.ndarray) -> np.ndarray:
        xyz = np.asarray(xyz, dtype=float)
        scalar = xyz.ndim == 1
        if scalar: xyz = xyz[np.newaxis]
        out = np.full_like(xyz, np.nan)
        valid = ~np.isnan(xyz).any(axis=1)
        out[valid] = (self.R @ xyz[valid].T).T + self.t
        return out[0] if scalar else out


@dataclass
class TracedEdge:
    pts_px: np.ndarray
    camera: int
    label:  str = ""


@dataclass
class DistortionResult:
    dist0:    np.ndarray
    dist1:    np.ndarray
    rmse0:    float
    rmse1:    float
    n_edges0: int
    n_edges1: int
    converged: bool


# --------------------------------------------------------------------------- #
#  Kabsch alignment                                                            #
# --------------------------------------------------------------------------- #

def kabsch_align(points: list[AlignPoint]) -> AlignResult:
    if len(points) < 3:
        raise ValueError(f"Need >= 3 correspondences, got {len(points)}")
    A = np.stack([p.rec_xyz   for p in points])
    B = np.stack([p.arena_xyz for p in points])
    cA, cB = A.mean(0), B.mean(0)
    H  = (A - cA).T @ (B - cB)
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    t = cB - R @ cA
    rmse = math.sqrt(((( R @ A.T).T + t - B)**2).sum(1).mean())
    return AlignResult(R=R, t=t, rmse_mm=rmse, n_points=len(points))


# --------------------------------------------------------------------------- #
#  CSV I/O — corners                                                           #
# --------------------------------------------------------------------------- #

def save_align_csv(path, points: list[AlignPoint]) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["rec_x","rec_y","rec_z","arena_x","arena_y","arena_z",
                    "label","px0_x","px0_y","px1_x","px1_y"])
        for pt in points:
            px0 = pt.px0 if pt.px0 is not None else [float("nan"), float("nan")]
            px1 = pt.px1 if pt.px1 is not None else [float("nan"), float("nan")]
            w.writerow([f"{pt.rec_xyz[0]:.4f}", f"{pt.rec_xyz[1]:.4f}",
                        f"{pt.rec_xyz[2]:.4f}", f"{pt.arena_xyz[0]:.4f}",
                        f"{pt.arena_xyz[1]:.4f}", f"{pt.arena_xyz[2]:.4f}",
                        pt.label, f"{px0[0]:.3f}", f"{px0[1]:.3f}",
                        f"{px1[0]:.3f}", f"{px1[1]:.3f}"])


def load_align_csv(path) -> list[AlignPoint]:
    path = Path(path)
    if not path.exists(): raise FileNotFoundError(f"Not found: {path}")
    points = []
    with open(path, newline="") as fh:
        for i, row in enumerate(csv.DictReader(fh)):
            try:
                rec   = np.array([float(row["rec_x"]), float(row["rec_y"]),
                                   float(row["rec_z"])])
                arena = np.array([float(row["arena_x"]), float(row["arena_y"]),
                                   float(row["arena_z"])])
                label = row.get("label", "").strip()
                def _px(xk, yk):
                    try:
                        x, y = float(row[xk]), float(row[yk])
                        return None if (math.isnan(x) or math.isnan(y)) else np.array([x, y])
                    except (KeyError, ValueError):
                        return None
                points.append(AlignPoint(rec_xyz=rec, arena_xyz=arena, label=label,
                                         px0=_px("px0_x","px0_y"),
                                         px1=_px("px1_x","px1_y")))
            except (KeyError, ValueError) as e:
                raise ValueError(f"Bad row {i+2}: {e}") from e
    return points


# --------------------------------------------------------------------------- #
#  CSV I/O — edges                                                             #
# --------------------------------------------------------------------------- #

def save_edges_csv(path, edges: list[TracedEdge]) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["camera","label","x","y"])
        for edge in edges:
            for pt in edge.pts_px:
                w.writerow([edge.camera, edge.label, f"{pt[0]:.3f}", f"{pt[1]:.3f}"])
            w.writerow(["---","---","---","---"])


def load_edges_csv(path) -> list[TracedEdge]:
    path = Path(path)
    if not path.exists(): raise FileNotFoundError(f"Not found: {path}")
    edges, cur_pts, cur_cam, cur_lbl = [], [], None, ""
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            if row["camera"] == "---":
                if cur_pts and cur_cam is not None:
                    edges.append(TracedEdge(np.array(cur_pts, float), cur_cam, cur_lbl))
                cur_pts, cur_cam, cur_lbl = [], None, ""
            else:
                cur_cam = int(row["camera"]); cur_lbl = row["label"]
                cur_pts.append([float(row["x"]), float(row["y"])])
    if cur_pts and cur_cam is not None:
        edges.append(TracedEdge(np.array(cur_pts, float), cur_cam, cur_lbl))
    return edges


# --------------------------------------------------------------------------- #
#  Apply alignment to pipeline outputs                                         #
# --------------------------------------------------------------------------- #

def align_skeleton_frames(frames, result: AlignResult):
    import copy
    return [[setattr(pt := copy.copy(p), "xyz", result.apply(p.xyz)) or pt
             for p in frame] for frame in frames]


def align_voxel_frames(voxel_frames, result: AlignResult):
    return [result.apply(pc) if (pc is not None and len(pc)) else pc
            for pc in voxel_frames]


# --------------------------------------------------------------------------- #
#  Plumb-line distortion fitting                                               #
# --------------------------------------------------------------------------- #

def _undistort_radial(pts, cx, cy, f, k1, k2, k3, n=8):
    xd, yd = (pts[:,0]-cx)/f, (pts[:,1]-cy)/f
    xu, yu = xd.copy(), yd.copy()
    for _ in range(n):
        r2 = xu**2 + yu**2
        D  = 1 + k1*r2 + k2*r2**2 + k3*r2**3
        xu, yu = xd/D, yd/D
    return np.stack([xu*f+cx, yu*f+cy], 1)


def _edge_line_rmse(pts):
    if len(pts) < 2: return 0.0
    c = pts.mean(0)
    _, _, Vt = np.linalg.svd(pts - c, full_matrices=False)
    return float(math.sqrt(((( pts - c) @ Vt[1])**2).mean()))


def _edge_cost(params, edges, cx, cy, f):
    k1, k2, k3 = params
    return sum(_edge_line_rmse(_undistort_radial(e.pts_px, cx, cy, f, k1, k2, k3))**2
               for e in edges) / max(len(edges), 1)


def fit_distortion_plumb_line(edges, K0, K1, *, k_init=(0,0,0),
                               max_iter=2000, tol=1e-7) -> DistortionResult:
    from scipy.optimize import minimize

    def _fit(cam_edges, K):
        if len(cam_edges) < 3:
            raise ValueError(f"Need >= 3 edges per camera (got {len(cam_edges)})")
        cx, cy = float(K[0,2]), float(K[1,2])
        f = float((K[0,0]+K[1,1])/2)
        res = minimize(_edge_cost, np.array(k_init, float),
                       args=(cam_edges, cx, cy, f), method="Nelder-Mead",
                       options={"maxiter":max_iter,"xatol":tol,"fatol":tol,"adaptive":True})
        k1, k2, k3 = res.x
        rmse = float(np.mean([_edge_line_rmse(
            _undistort_radial(e.pts_px, cx, cy, f, k1, k2, k3)) for e in cam_edges]))
        return np.array([k1,k2,0,0,k3]), rmse, res.success

    e0 = [e for e in edges if e.camera==0]
    e1 = [e for e in edges if e.camera==1]
    d0, r0, ok0 = _fit(e0, K0)
    d1, r1, ok1 = _fit(e1, K1)
    return DistortionResult(dist0=d0, dist1=d1, rmse0=r0, rmse1=r1,
                             n_edges0=len(e0), n_edges1=len(e1),
                             converged=ok0 and ok1)


def patch_calibration_distortion(calib_path, dist_result, out_path):
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    cal = dict(np.load(calib_path))
    cal["dist0"] = dist_result.dist0.reshape(1,5)
    cal["dist1"] = dist_result.dist1.reshape(1,5)
    np.savez(out_path, **cal)


# --------------------------------------------------------------------------- #
#  Bundle adjustment                                                           #
# --------------------------------------------------------------------------- #

def _project_with_dist(X, rvec, tvec, fx, fy, cx, cy, k1, k2, k3):
    import cv2
    R, _ = cv2.Rodrigues(rvec.astype(np.float64))
    Xc = (R @ X.T).T + tvec
    xn, yn = Xc[:,0]/Xc[:,2], Xc[:,1]/Xc[:,2]
    r2 = xn**2 + yn**2
    D  = 1 + k1*r2 + k2*r2**2 + k3*r2**3
    return np.stack([fx*xn*D+cx, fy*yn*D+cy], 1)


def _ba_residuals(params, corner_points, edges, ew):
    import cv2
    (fx0,fy0,cx0,cy0, k1_0,k2_0,k3_0,
     fx1,fy1,cx1,cy1, k1_1,k2_1,k3_1) = params[:14]
    rvec_s, tvec_s = params[14:17], params[17:20]
    rvec_p, tvec_p = params[20:23], params[23:26]
    R_s, _ = cv2.Rodrigues(rvec_s.astype(np.float64))
    res = []
    for pt in corner_points:
        if pt.px0 is None or pt.px1 is None: continue
        X = pt.arena_xyz[np.newaxis]
        res.extend(_project_with_dist(X, rvec_p, tvec_p, fx0,fy0,cx0,cy0, k1_0,k2_0,k3_0)[0] - pt.px0)
        Xc0 = (cv2.Rodrigues(rvec_p.astype(np.float64))[0] @ X.T).T + tvec_p
        Xc1 = (R_s @ Xc0.T).T + tvec_s
        res.extend(_project_with_dist(Xc1, np.zeros(3), np.zeros(3), fx1,fy1,cx1,cy1, k1_1,k2_1,k3_1)[0] - pt.px1)
    for edge in edges:
        if edge.camera==0: fx,fy,cx,cy,k1,k2,k3 = fx0,fy0,cx0,cy0,k1_0,k2_0,k3_0
        else:              fx,fy,cx,cy,k1,k2,k3 = fx1,fy1,cx1,cy1,k1_1,k2_1,k3_1
        f=(fx+fy)/2; u=_undistort_radial(edge.pts_px, cx,cy,f, k1,k2,k3)
        c=u.mean(0); _,_,Vt=np.linalg.svd(u-c,full_matrices=False)
        res.extend(((u-c)@Vt[1]*ew).tolist())
    return np.array(res, dtype=np.float64)


def refine_calibration_from_arena(corner_points, edges, calib_path, out_path,
                                   *, edge_weight=0.3, fix_principal=False,
                                   verbose=True) -> dict:
    import cv2
    from scipy.optimize import least_squares

    usable = [p for p in corner_points if p.px0 is not None and p.px1 is not None]
    if len(usable) < 4:
        raise ValueError(f"Need >= 4 corners with pixel clicks (have {len(usable)}). "
                         "Re-annotate corners in the GUI.")

    cal = np.load(calib_path)
    K0, K1 = cal["K0"].astype(np.float64), cal["K1"].astype(np.float64)
    d0 = np.ravel(cal.get("dist0", np.zeros(5))).astype(np.float64)
    d1 = np.ravel(cal.get("dist1", np.zeros(5))).astype(np.float64)
    if len(d0)<5: d0=np.pad(d0,(0,5-len(d0)))
    if len(d1)<5: d1=np.pad(d1,(0,5-len(d1)))
    R_i = cal["R"].astype(np.float64); T_i = cal["T"].ravel().astype(np.float64)
    rvec_s, _ = cv2.Rodrigues(R_i); rvec_s = rvec_s.ravel()

    obj = np.stack([p.arena_xyz for p in usable]).astype(np.float64)
    img = np.stack([p.px0       for p in usable]).astype(np.float64)
    ok, rvec_p, tvec_p = cv2.solvePnP(obj, img, K0, d0.reshape(1,5),
                                        flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok: raise RuntimeError("solvePnP failed on initial pose.")
    rvec_p, tvec_p = rvec_p.ravel(), tvec_p.ravel()

    x0 = np.array([K0[0,0],K0[1,1],K0[0,2],K0[1,2], d0[0],d0[1],d0[4],
                   K1[0,0],K1[1,1],K1[0,2],K1[1,2], d1[0],d1[1],d1[4],
                   *rvec_s, *T_i, *rvec_p, *tvec_p], dtype=np.float64)
    lo = np.full_like(x0,-np.inf); hi = np.full_like(x0,+np.inf)
    if fix_principal:
        for i in (2,3,10,11): lo[i]=x0[i]-1e-6; hi[i]=x0[i]+1e-6

    r0 = _ba_residuals(x0, usable, edges, edge_weight)
    cost0 = float(np.sqrt((r0**2).mean()))
    if verbose: print(f"  BA initial RMSE: {cost0:.3f} px")

    result = least_squares(_ba_residuals, x0,
                           args=(usable, edges, edge_weight),
                           bounds=(lo, hi), method="trf",
                           ftol=1e-8, xtol=1e-8, gtol=1e-8, max_nfev=5000,
                           verbose=2 if verbose else 0)
    xf = result.x
    r1 = _ba_residuals(xf, usable, edges, edge_weight)
    cost1 = float(np.sqrt((r1**2).mean()))
    if verbose: print(f"  BA final   RMSE: {cost1:.3f} px  "
                      f"({'converged' if result.success else 'NOT converged'})")

    fx0f,fy0f,cx0f,cy0f,k1_0f,k2_0f,k3_0f = xf[0],xf[1],xf[2],xf[3],xf[4],xf[5],xf[6]
    fx1f,fy1f,cx1f,cy1f,k1_1f,k2_1f,k3_1f = xf[7],xf[8],xf[9],xf[10],xf[11],xf[12],xf[13]
    K0f = np.array([[fx0f,0,cx0f],[0,fy0f,cy0f],[0,0,1]])
    K1f = np.array([[fx1f,0,cx1f],[0,fy1f,cy1f],[0,0,1]])
    d0f = np.array([k1_0f,k2_0f,0,0,k3_0f]); d1f = np.array([k1_1f,k2_1f,0,0,k3_1f])
    Rf, _ = cv2.Rodrigues(xf[14:17].astype(np.float64)); Tf = xf[17:20].reshape(3,1)

    rc = _ba_residuals(xf, usable, [], 0); rmse_c = float(np.sqrt((rc**2).mean())) if len(rc) else 0
    re = _ba_residuals(xf, [], edges, 1); rmse_e = float(np.sqrt((re**2).mean())) if edges and len(re) else 0

    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    cal_out = dict(np.load(calib_path))
    cal_out.update({"K0":K0f,"K1":K1f,"dist0":d0f.reshape(1,5),"dist1":d1f.reshape(1,5),
                    "R":Rf,"T":Tf,
                    "P0":K0f@np.hstack([np.eye(3),np.zeros((3,1))]),
                    "P1":K1f@np.hstack([Rf,Tf])})
    np.savez(out_path, **cal_out)

    return {"cost_before":cost0,"cost_after":cost1,"converged":result.success,
            "rmse_corners_px":rmse_c,"rmse_edges_px":rmse_e,
            "K0":K0f,"K1":K1f,"dist0":d0f,"dist1":d1f,"R":Rf,"T":Tf}
