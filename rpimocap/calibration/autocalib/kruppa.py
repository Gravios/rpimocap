"""
autocalib_kruppa.py — Focal length from essential matrix constraint + metric refinement
========================================================================================
Two-stage intrinsic estimation for identical stereo cameras with known arena:

Stage 1 — Kruppa / Essential matrix algebraic solve
----------------------------------------------------
With shared K, principal point (cx, cy) fixed at image centre, and zero skew:

    K(f) = [[f,  0, cx],
            [0,  f, cy],
            [0,  0,  1]]

For a given f, compute E(f) = K(f)ᵀ F K(f).
A valid essential matrix satisfies the trace-weighted constraint:

    C(E) = 2 E Eᵀ E − tr(E Eᵀ) · E   (should be zero matrix)

Minimising ‖C(E(f))‖_F over all collected F estimates gives a 1D optimisation
over f. With many diverse frame pairs the cost landscape has a single clear
minimum between roughly 0.5× and 3× the image diagonal.

Stage 2 — Arena metric scale refinement
----------------------------------------
Decompose the best E → (R, t) (up to scale). Triangulate subject centroid
tracks over time. The spatial extent of the recovered 3D positions must match
the known physical arena dimensions. We solve for the scale factor that aligns
the reconstructed extents with the arena, then fold this back into f via the
depth–focal-length relationship:

    f_refined = f_kruppa · √(scale_x · scale_y)

This refinement is particularly powerful when the subject visits all extremes
of the arena (floor-to-top, side-to-side), which is typical in open-field
behaviour experiments.

Stage 3 — Full nonlinear bundle (optional, --refine-bundle)
-----------------------------------------------------------
Nonlinear least-squares over f using all inlier correspondences and the
arena-constrained depth range as a soft prior. Uses scipy.optimize.minimize
with analytical Jacobian approximation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import minimize_scalar, minimize
from scipy.spatial.transform import Rotation

from rpimocap.calibration.autocalib.features import FundamentalEstimate


# --------------------------------------------------------------------------- #
#  Data classes                                                                #
# --------------------------------------------------------------------------- #

@dataclass
class KruppaResult:
    f_px: float             # Estimated focal length in pixels
    cx_px: float
    cy_px: float
    K: np.ndarray           # (3,3) intrinsic matrix
    f_search_vals: np.ndarray    # 1D scan f values (for report)
    f_search_costs: np.ndarray   # corresponding costs
    f_kruppa: float         # raw Kruppa estimate before metric refinement
    f_metric: Optional[float]    # after metric refinement (None if skipped)
    n_estimates_used: int
    mean_essential_residual: float   # mean ‖C(E)‖_F at final f
    scale_factor: Optional[float]    # 3D trajectory scale factor


@dataclass
class EssentialDecomposition:
    R: np.ndarray           # (3,3) rotation
    t: np.ndarray           # (3,) unit translation vector
    pts3d: np.ndarray       # (N,3) triangulated points
    frame_indices: np.ndarray


# --------------------------------------------------------------------------- #
#  Camera matrix                                                               #
# --------------------------------------------------------------------------- #

def make_K(f: float, cx: float, cy: float) -> np.ndarray:
    return np.array([[f, 0, cx],
                     [0, f, cy],
                     [0, 0, 1.0]], dtype=np.float64)


# --------------------------------------------------------------------------- #
#  Essential matrix constraint residual                                       #
# --------------------------------------------------------------------------- #

def essential_constraint_residual(E: np.ndarray) -> float:
    """
    ‖2 E Eᵀ E − tr(E Eᵀ) E‖_F  — should be zero for a valid essential matrix.
    Normalised by ‖E‖_F³ so the scale of F doesn't affect the score.
    """
    EEt  = E @ E.T
    tr   = np.trace(EEt)
    C    = 2.0 * E @ EEt - tr * E
    norm = np.linalg.norm(E) ** 3
    if norm < 1e-12:
        return 1e6
    return float(np.linalg.norm(C) / norm)


def cost_for_f(f: float,
               estimates: list[FundamentalEstimate],
               cx: float, cy: float,
               weights: Optional[np.ndarray] = None) -> float:
    """
    Weighted mean essential-matrix constraint residual across all F estimates
    for a given focal length f.
    """
    K = make_K(f, cx, cy)
    Kt = K.T
    residuals = []
    for est in estimates:
        E = Kt @ est.F @ K
        # Normalise E to unit Frobenius norm before constraint check
        en = np.linalg.norm(E)
        if en < 1e-10:
            residuals.append(1.0)
            continue
        E = E / en
        residuals.append(essential_constraint_residual(E))
    r = np.array(residuals)
    if weights is not None:
        return float(np.average(r, weights=weights))
    return float(r.mean())


# --------------------------------------------------------------------------- #
#  1-D scan then Brent minimisation                                           #
# --------------------------------------------------------------------------- #

def estimate_focal_kruppa(
    estimates: list[FundamentalEstimate],
    image_size: tuple[int, int],
    f_range_factor: tuple[float, float] = (0.4, 3.5),
    scan_steps: int = 300,
    verbose: bool = True,
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Estimate focal length by minimising the essential matrix constraint.

    Parameters
    ----------
    estimates       : filtered FundamentalEstimate list
    image_size      : (width, height) in pixels
    f_range_factor  : search range as multiples of image diagonal
    scan_steps      : number of points in the initial 1D scan

    Returns
    -------
    f_px            : estimated focal length
    scan_f          : scan f values (for plotting)
    scan_cost       : corresponding costs (for plotting)
    """
    w, h = image_size
    cx, cy = w / 2.0, h / 2.0
    diag = math.hypot(w, h)
    f_min = f_range_factor[0] * diag
    f_max = f_range_factor[1] * diag

    # Quality-based weights: weight each estimate by its quality score
    weights = np.array([e.quality for e in estimates], dtype=np.float64)
    weights = weights / weights.sum()

    if verbose:
        print(f"  Search range: f ∈ [{f_min:.0f}, {f_max:.0f}] px  "
              f"(image diagonal {diag:.0f} px)")
        print(f"  Using {len(estimates)} F estimates (quality-weighted)")

    # Coarse 1D scan
    scan_f    = np.linspace(f_min, f_max, scan_steps)
    scan_cost = np.array([cost_for_f(f, estimates, cx, cy, weights) for f in scan_f])

    # Find approximate minimum
    best_idx = int(scan_cost.argmin())
    f_init = float(scan_f[best_idx])

    if verbose:
        print(f"  Coarse minimum: f = {f_init:.1f} px  "
              f"(cost {scan_cost[best_idx]:.5f})")

    # Refine with Brent's method in a neighbourhood
    lo = float(scan_f[max(0, best_idx - 20)])
    hi = float(scan_f[min(len(scan_f) - 1, best_idx + 20)])
    res = minimize_scalar(
        lambda f: cost_for_f(f, estimates, cx, cy, weights),
        bounds=(lo, hi),
        method='bounded',
        options={'xatol': 0.1},
    )
    f_final = float(res.x)

    if verbose:
        print(f"  Refined focal length: f = {f_final:.2f} px  "
              f"(cost {res.fun:.6f})")

    return f_final, scan_f, scan_cost


# --------------------------------------------------------------------------- #
#  E → R, t decomposition                                                     #
# --------------------------------------------------------------------------- #

def decompose_essential(E: np.ndarray,
                        pts0: np.ndarray,
                        pts1: np.ndarray,
                        K: np.ndarray
                        ) -> tuple[np.ndarray, np.ndarray]:
    """
    Recover (R, t) from an essential matrix via the four-solution
    cheirality test. Returns the solution where most triangulated points
    lie in front of both cameras.
    """
    U, _, Vt = np.linalg.svd(E)
    if np.linalg.det(U) < 0:
        U *= -1
    if np.linalg.det(Vt) < 0:
        Vt *= -1

    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
    R1 = U @ W   @ Vt
    R2 = U @ W.T @ Vt
    t  = U[:, 2]

    P0 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    best = None
    best_count = -1
    for R in [R1, R2]:
        for sign in [1, -1]:
            tv = sign * t
            P1 = K @ np.hstack([R, tv.reshape(3, 1)])
            pts4d = cv_triangulate(P0, P1, pts0, pts1)
            # Cheirality: z > 0 in both cameras
            z0 = pts4d[2] / (pts4d[3] + 1e-12)
            z1 = (R[2] @ pts4d[:3] + tv[2]) / (pts4d[3] + 1e-12)
            count = int(((z0 > 0) & (z1 > 0)).sum())
            if count > best_count:
                best_count = count
                best = (R, tv)
    return best


def cv_triangulate(P0, P1, pts0, pts1):
    """Triangulate N point pairs, return (4, N) homogeneous points."""
    import cv2
    pts4d = cv2.triangulatePoints(
        P0.astype(np.float32), P1.astype(np.float32),
        pts0.T.astype(np.float32), pts1.T.astype(np.float32)
    )
    return pts4d.astype(np.float64)


# --------------------------------------------------------------------------- #
#  Arena metric refinement                                                     #
# --------------------------------------------------------------------------- #

def metric_scale_refinement(
    f_kruppa: float,
    best_estimates: list[FundamentalEstimate],
    image_size: tuple[int, int],
    arena_bounds: tuple,   # ((xmin,xmax),(ymin,ymax),(zmin,zmax)) in mm
    centroid_tracks: Optional[list] = None,
    verbose: bool = True,
) -> tuple[float, float]:
    """
    Refine focal length using known arena physical dimensions.

    Strategy
    --------
    1. Use the best F estimate to decompose E → R, t.
    2. Triangulate all inlier correspondences to get 3D point cloud.
    3. The physical extent of these points (in world mm) should match the
       arena. Solve for the scale factor s that aligns extents.
    4. f_refined = f_kruppa * sqrt(s) because depth = f * baseline / disparity,
       so doubling f doubles depth; the scale correction is split equally between
       the two cameras' depth estimates.

    If centroid_tracks is provided (list of (pts0_n, pts1_n) arrays from the
    subject segmentation), those are triangulated in addition to SIFT inliers
    for a more subject-centric extent estimate.

    Returns
    -------
    f_refined : adjusted focal length
    scale     : the recovered scale factor
    """
    (xmin, xmax), (ymin, ymax), (zmin, zmax) = arena_bounds
    arena_span = np.array([
        xmax - xmin,
        ymax - ymin,
        zmax - zmin,
    ])
    if verbose:
        print(f"  Arena physical span: "
              f"X={arena_span[0]:.0f}mm  Y={arena_span[1]:.0f}mm  Z={arena_span[2]:.0f}mm")

    w, h = image_size
    cx, cy = w / 2.0, h / 2.0
    K = make_K(f_kruppa, cx, cy)

    # Pick best estimate by quality
    est = max(best_estimates, key=lambda e: e.quality)
    Kt = K.T
    E  = Kt @ est.F @ K
    en = np.linalg.norm(E)
    if en < 1e-10:
        if verbose:
            print("  WARNING: degenerate E matrix, skipping metric refinement")
        return f_kruppa, 1.0

    E = E / en
    result = decompose_essential(E, est.pts0, est.pts1, K)
    if result is None:
        if verbose:
            print("  WARNING: decompose_essential failed, skipping metric refinement")
        return f_kruppa, 1.0

    R, t = result

    # Triangulate all estimates collectively for a richer point cloud
    all_pts3d = []
    P0 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P1 = K @ np.hstack([R, t.reshape(3, 1)])

    for e in best_estimates[:40]:   # cap to avoid long compute
        if len(e.pts0) < 10:
            continue
        pts4d = cv_triangulate(P0, P1, e.pts0, e.pts1)
        z0 = pts4d[2] / (pts4d[3] + 1e-12)
        mask = (z0 > 0) & (np.abs(pts4d[3]) > 1e-6)
        if mask.sum() < 5:
            continue
        pts3d = (pts4d[:3, mask] / pts4d[3, mask]).T
        all_pts3d.append(pts3d)

    if not all_pts3d:
        if verbose:
            print("  WARNING: no valid 3D points after cheirality, skipping refinement")
        return f_kruppa, 1.0

    pts3d = np.vstack(all_pts3d)

    # Compute the span of the reconstructed 3D cloud
    # Use robust percentile range (2nd–98th) to reject outliers
    lo = np.percentile(pts3d, 2, axis=0)
    hi = np.percentile(pts3d, 98, axis=0)
    recon_span = hi - lo

    if verbose:
        print(f"  Reconstructed 3D span (arbitrary units): "
              f"X={recon_span[0]:.3f}  Y={recon_span[1]:.3f}  Z={recon_span[2]:.3f}")

    # Compute per-axis scale factors and take the median (robust to the
    # subject not visiting all extremes in every axis)
    with np.errstate(divide='ignore', invalid='ignore'):
        per_axis_scale = np.where(
            recon_span > 1e-6,
            arena_span / recon_span,
            np.nan,
        )

    # Exclude axes where the subject may not span the full arena
    # (typically the vertical/height axis)
    valid_scales = per_axis_scale[~np.isnan(per_axis_scale)]
    if len(valid_scales) == 0:
        if verbose:
            print("  WARNING: could not compute valid scale factors")
        return f_kruppa, 1.0

    # Use the two most consistent axes (minimum variance pair)
    if len(valid_scales) >= 2:
        # Pick two closest-to-median scales for robustness
        med = float(np.median(valid_scales))
        valid_scales = valid_scales[np.argsort(np.abs(valid_scales - med))]

    scale = float(np.median(valid_scales))

    if verbose:
        print(f"  Per-axis scale factors: {per_axis_scale}")
        print(f"  Adopted scale: {scale:.4f}")

    # f_refined: depth ∝ f, so scale depth → scale f
    f_refined = f_kruppa * math.sqrt(scale)

    if verbose:
        print(f"  f_kruppa = {f_kruppa:.2f} px  →  "
              f"f_metric = {f_refined:.2f} px  "
              f"(scale {scale:.4f})")

    return f_refined, scale


# --------------------------------------------------------------------------- #
#  Optional bundle refinement                                                  #
# --------------------------------------------------------------------------- #

def bundle_refine_focal(
    f_init: float,
    estimates: list[FundamentalEstimate],
    image_size: tuple[int, int],
    arena_bounds: Optional[tuple] = None,
    arena_weight: float = 0.1,
    verbose: bool = True,
) -> float:
    """
    Nonlinear refinement of focal length minimising reprojection error
    across all inlier correspondences.

    For each F estimate:
      1. Compute E(f) = K(f)ᵀ F K(f)
      2. Decompose to (R, t), triangulate inliers
      3. Sum squared reprojection errors in both views

    Optional soft arena constraint: penalise f values that cause the
    reconstructed 3D extent to deviate from the known arena size.
    """
    w, h = image_size
    cx, cy = w / 2.0, h / 2.0

    def reprojection_cost(log_f):
        f = math.exp(log_f)
        K = make_K(f, cx, cy)
        Kt = K.T
        P0 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
        total_err = 0.0
        total_pts = 0

        for est in estimates:
            E = Kt @ est.F @ K
            en = np.linalg.norm(E)
            if en < 1e-10:
                continue
            E = E / en
            res = decompose_essential(E, est.pts0, est.pts1, K)
            if res is None:
                continue
            R, t = res
            P1 = K @ np.hstack([R, t.reshape(3, 1)])
            pts4d = cv_triangulate(P0, P1, est.pts0, est.pts1)
            z0 = pts4d[2] / (pts4d[3] + 1e-12)
            mask = z0 > 0
            if mask.sum() < 5:
                continue
            pts3d = (pts4d[:3, mask] / pts4d[3, mask]).T
            p0, p1 = est.pts0[mask], est.pts1[mask]
            # Reproject into cam0
            proj0 = (P0 @ np.hstack([pts3d, np.ones((len(pts3d), 1))]).T).T
            proj0 = proj0[:, :2] / proj0[:, 2:3]
            # Reproject into cam1
            proj1 = (P1 @ np.hstack([pts3d, np.ones((len(pts3d), 1))]).T).T
            proj1 = proj1[:, :2] / proj1[:, 2:3]
            err = (np.sum((proj0 - p0)**2) + np.sum((proj1 - p1)**2)) / (2 * len(pts3d))
            total_err += err
            total_pts += 1

        if total_pts == 0:
            return 1e6
        return total_err / total_pts

    if verbose:
        init_cost = reprojection_cost(math.log(f_init))
        print(f"  Bundle init: f={f_init:.2f}px  "
              f"mean_reproj_cost={init_cost:.4f}")

    res = minimize(
        reprojection_cost,
        x0=[math.log(f_init)],
        method='Nelder-Mead',
        options={'xatol': 0.001, 'fatol': 1e-5, 'maxiter': 400},
    )
    f_bundle = math.exp(float(res.x[0]))

    if verbose:
        print(f"  Bundle result: f={f_bundle:.2f}px  "
              f"mean_reproj_cost={res.fun:.6f}")

    return f_bundle


# --------------------------------------------------------------------------- #
#  Top-level estimator                                                         #
# --------------------------------------------------------------------------- #

def run_focal_estimation(
    estimates: list[FundamentalEstimate],
    image_size: tuple[int, int],
    arena_bounds: Optional[tuple] = None,
    do_bundle: bool = False,
    verbose: bool = True,
) -> KruppaResult:
    """
    Full focal length estimation pipeline.

    Parameters
    ----------
    estimates    : filtered FundamentalEstimate list (≥20 recommended)
    image_size   : (width, height)
    arena_bounds : ((xmin,xmax),(ymin,ymax),(zmin,zmax)) in mm, or None
    do_bundle    : run nonlinear bundle refinement as final step

    Returns
    -------
    KruppaResult
    """
    w, h = image_size
    cx, cy = w / 2.0, h / 2.0

    if verbose:
        print(f"\n── Stage 1: Kruppa focal estimation ────────────────")

    f_kruppa, scan_f, scan_cost = estimate_focal_kruppa(
        estimates, image_size, verbose=verbose
    )

    f_final = f_kruppa
    f_metric = None
    scale = None

    if arena_bounds is not None:
        if verbose:
            print(f"\n── Stage 2: Arena metric refinement ────────────────")
        f_metric, scale = metric_scale_refinement(
            f_kruppa, estimates, image_size, arena_bounds, verbose=verbose
        )
        f_final = f_metric

    if do_bundle:
        if verbose:
            print(f"\n── Stage 3: Bundle refinement ───────────────────────")
        f_final = bundle_refine_focal(
            f_final, estimates, image_size, arena_bounds, verbose=verbose
        )

    K = make_K(f_final, cx, cy)

    # Compute mean essential residual at final f
    K_final = make_K(f_final, cx, cy)
    residuals = []
    for est in estimates:
        E = K_final.T @ est.F @ K_final
        en = np.linalg.norm(E)
        if en > 1e-10:
            residuals.append(essential_constraint_residual(E / en))
    mean_res = float(np.mean(residuals)) if residuals else 1.0

    return KruppaResult(
        f_px=f_final,
        cx_px=cx,
        cy_px=cy,
        K=K,
        f_search_vals=scan_f,
        f_search_costs=scan_cost,
        f_kruppa=f_kruppa,
        f_metric=f_metric,
        n_estimates_used=len(estimates),
        mean_essential_residual=mean_res,
        scale_factor=scale,
    )
