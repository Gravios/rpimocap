"""
voxel.py — Silhouette extraction and voxel carving
===================================================
Given two calibrated camera views and binary foreground masks per frame,
this module recovers the Visual Hull of the subject by carving a 3D voxel
grid: any voxel whose projection falls outside EITHER camera's silhouette
is removed from the candidate occupancy.

Two cameras provide the absolute minimum for voxel carving; accuracy improves
with additional views. With only two cameras, the hull will be an over-estimate
(concavities visible in neither view are filled in), but it reliably captures
the subject's overall volume and extent.

Silhouette extraction uses OpenCV's MOG2 background subtractor, which is
well-suited to static-camera setups with gradual illumination changes.

Usage (standalone silhouette extraction):
    python voxel.py --cam0 video0.mp4 --cam1 video1.mp4 \\
                    --calib calibration.npz --out silhouettes/
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


# --------------------------------------------------------------------------- #
#  Data class                                                                  #
# --------------------------------------------------------------------------- #

@dataclass
class VoxelGrid:
    """3D binary occupancy grid in world coordinates."""
    origin: np.ndarray     # (3,) lower-left-front corner (world units = mm if calib in mm)
    voxel_size: float      # edge length of one voxel (same units as origin)
    shape: tuple           # (nx, ny, nz)
    occupancy: np.ndarray  # bool, shape == self.shape

    @property
    def n_occupied(self) -> int:
        return int(self.occupancy.sum())

    @property
    def bounds_mm(self) -> tuple:
        nx, ny, nz = self.shape
        hi = self.origin + np.array([nx, ny, nz]) * self.voxel_size
        return (
            (self.origin[0], hi[0]),
            (self.origin[1], hi[1]),
            (self.origin[2], hi[2]),
        )


# --------------------------------------------------------------------------- #
#  Grid construction                                                           #
# --------------------------------------------------------------------------- #

def build_voxel_grid(
    bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    voxel_size: float,
) -> VoxelGrid:
    """
    Create a fully-occupied voxel grid within the specified world bounds.

    Parameters
    ----------
    bounds     : ((xmin, xmax), (ymin, ymax), (zmin, zmax)) in world units
    voxel_size : edge length per voxel (same units)

    Returns
    -------
    VoxelGrid with occupancy=True everywhere (carving starts from full)
    """
    (xmin, xmax), (ymin, ymax), (zmin, zmax) = bounds
    nx = max(1, int(np.ceil((xmax - xmin) / voxel_size)))
    ny = max(1, int(np.ceil((ymax - ymin) / voxel_size)))
    nz = max(1, int(np.ceil((zmax - zmin) / voxel_size)))
    occupancy = np.ones((nx, ny, nz), dtype=bool)
    origin = np.array([xmin, ymin, zmin], dtype=np.float64)
    print(f"  Voxel grid  : {nx}×{ny}×{nz} = {nx*ny*nz:,} voxels  "
          f"({voxel_size:.1f} units/voxel)")
    return VoxelGrid(origin, float(voxel_size), (nx, ny, nz), occupancy)


def voxel_centers(grid: VoxelGrid) -> np.ndarray:
    """
    Return the world coordinates of all voxel centres.
    Shape: (nx*ny*nz, 3)
    """
    nx, ny, nz = grid.shape
    xi = np.arange(nx, dtype=np.float32)
    yi = np.arange(ny, dtype=np.float32)
    zi = np.arange(nz, dtype=np.float32)
    xx, yy, zz = np.meshgrid(xi, yi, zi, indexing='ij')
    centers = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)
    centers = centers * grid.voxel_size + grid.origin + grid.voxel_size * 0.5
    return centers


# --------------------------------------------------------------------------- #
#  Projection                                                                  #
# --------------------------------------------------------------------------- #

def project_points_batch(P: np.ndarray, pts3d: np.ndarray) -> np.ndarray:
    """
    Project N world points through a (3, 4) projection matrix.

    Parameters
    ----------
    P     : (3, 4)
    pts3d : (N, 3)

    Returns
    -------
    (N, 2) pixel coordinates (float)
    """
    ones = np.ones((pts3d.shape[0], 1), dtype=np.float64)
    pts_h = np.concatenate([pts3d, ones], axis=1)   # (N, 4)
    proj = (P @ pts_h.T).T                            # (N, 3)
    with np.errstate(divide='ignore', invalid='ignore'):
        px = proj[:, :2] / proj[:, 2:3]
    # voxels behind the camera → send to -inf
    behind = proj[:, 2] <= 0
    px[behind] = -1e9
    return px


# --------------------------------------------------------------------------- #
#  Silhouette extraction                                                       #
# --------------------------------------------------------------------------- #

def make_bg_subtractor(
    history: int = 300,
    var_threshold: float = 40.0,
    detect_shadows: bool = False,
) -> cv2.BackgroundSubtractorMOG2:
    return cv2.createBackgroundSubtractorMOG2(
        history=history,
        varThreshold=var_threshold,
        detectShadows=detect_shadows,
    )


def extract_silhouette(
    frame: np.ndarray,
    bg_subtractor: cv2.BackgroundSubtractorMOG2,
    morph_ksize: int = 7,
    min_area_px: int = 500,
    dilate_px: int = 3,
) -> np.ndarray:
    """
    Extract a binary foreground mask from a single BGR frame.

    Steps:
      1. MOG2 background subtraction
      2. Morphological close → fill small holes
      3. Morphological open  → remove isolated noise
      4. Keep only contours above min_area_px (largest blob heuristic)
      5. Optional dilation to account for silhouette boundary uncertainty

    Parameters
    ----------
    frame         : BGR image
    bg_subtractor : fitted MOG2 subtractor (shared across frames for learning)
    morph_ksize   : kernel size for morphological ops
    min_area_px   : ignore foreground blobs below this pixel area
    dilate_px     : pixels to dilate final mask (0 = no dilation)

    Returns
    -------
    uint8 mask: 255 = foreground, 0 = background
    """
    fg = bg_subtractor.apply(frame)
    k = morph_ksize
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel)

    cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(fg)
    for c in cnts:
        if cv2.contourArea(c) >= min_area_px:
            cv2.drawContours(mask, [c], -1, 255, cv2.FILLED)

    if dilate_px > 0:
        dk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1,) * 2)
        mask = cv2.dilate(mask, dk)

    return mask


# --------------------------------------------------------------------------- #
#  Voxel carving                                                               #
# --------------------------------------------------------------------------- #

def carve_frame(
    grid: VoxelGrid,
    P0: np.ndarray,
    P1: np.ndarray,
    mask0: np.ndarray,
    mask1: np.ndarray,
    image_size: tuple[int, int],
    chunk_size: int = 100_000,
) -> np.ndarray:
    """
    Carve the occupancy grid using two silhouette masks.

    Voxels whose projection falls outside EITHER mask are marked unoccupied.
    Processing is chunked to control peak memory usage.

    Parameters
    ----------
    grid       : VoxelGrid (occupancy is read but NOT mutated)
    P0, P1     : (3, 4) projection matrices for cameras 0 and 1
    mask0/1    : uint8 binary masks (255 = foreground)
    image_size : (width, height)
    chunk_size : number of voxels to project at once

    Returns
    -------
    bool array of shape grid.shape — True = still occupied after carving
    """
    w, h = image_size
    centers = voxel_centers(grid)           # (N, 3)
    N = centers.shape[0]
    keep = np.ones(N, dtype=bool)

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        chunk = centers[start:end]

        px0 = project_points_batch(P0, chunk)
        px1 = project_points_batch(P1, chunk)

        def in_mask(px, mask):
            xi = np.clip(np.round(px[:, 0]).astype(np.int32), 0, mask.shape[1] - 1)
            yi = np.clip(np.round(px[:, 1]).astype(np.int32), 0, mask.shape[0] - 1)
            in_bounds = (
                (px[:, 0] >= 0) & (px[:, 0] < w) &
                (px[:, 1] >= 0) & (px[:, 1] < h)
            )
            inside = np.zeros(len(px), dtype=bool)
            inside[in_bounds] = mask[yi[in_bounds], xi[in_bounds]] > 0
            return inside

        keep[start:end] = in_mask(px0, mask0) & in_mask(px1, mask1)

    return keep.reshape(grid.shape)


def apply_carving(grid: VoxelGrid, new_occupancy: np.ndarray) -> VoxelGrid:
    """Return a new VoxelGrid with occupancy updated by carving."""
    return VoxelGrid(
        origin=grid.origin.copy(),
        voxel_size=grid.voxel_size,
        shape=grid.shape,
        occupancy=new_occupancy,
    )


# --------------------------------------------------------------------------- #
#  Point cloud extraction                                                      #
# --------------------------------------------------------------------------- #

def occupied_centers(grid: VoxelGrid) -> np.ndarray:
    """Return (N, 3) world coordinates of occupied voxel centres."""
    centers = voxel_centers(grid)
    return centers[grid.occupancy.ravel()]


def surface_centers(grid: VoxelGrid) -> np.ndarray:
    """
    Return centres of surface voxels only (occupied voxels with at least
    one unoccupied face-neighbour).  Useful for thinner point clouds.
    """
    from scipy.ndimage import binary_erosion
    interior = binary_erosion(grid.occupancy, structure=np.ones((3, 3, 3)))
    surface = grid.occupancy & ~interior
    centers = voxel_centers(grid)
    return centers[surface.ravel()]


# --------------------------------------------------------------------------- #
#  Mesh extraction via Marching Cubes                                          #
# --------------------------------------------------------------------------- #

def grid_to_mesh(
    grid: VoxelGrid,
    level: float = 0.5,
    smooth_iterations: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run Marching Cubes on the occupancy volume to extract a surface mesh.

    Parameters
    ----------
    grid              : VoxelGrid
    level             : iso-value for Marching Cubes (0.5 for binary volume)
    smooth_iterations : Laplacian smoothing passes (0 = no smoothing)

    Returns
    -------
    vertices : (V, 3) float32 world coordinates
    faces    : (F, 3) int32 vertex indices
    """
    try:
        from skimage.measure import marching_cubes
    except ImportError:
        raise ImportError("pip install scikit-image")

    occ = grid.occupancy.astype(np.float32)
    if occ.max() == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.int32)

    verts_idx, faces, _, _ = marching_cubes(occ, level=level, method='lewiner')
    verts_world = (verts_idx * grid.voxel_size + grid.origin).astype(np.float32)

    if smooth_iterations > 0:
        verts_world = _laplacian_smooth(verts_world, faces, smooth_iterations)

    return verts_world, faces.astype(np.int32)


def _laplacian_smooth(verts: np.ndarray, faces: np.ndarray, n: int) -> np.ndarray:
    """Simple uniform Laplacian mesh smoothing."""
    from collections import defaultdict
    adj: dict = defaultdict(set)
    for f in faces:
        for i, j in [(0, 1), (1, 2), (2, 0)]:
            adj[f[i]].add(f[j])
            adj[f[j]].add(f[i])
    v = verts.copy()
    for _ in range(n):
        new_v = v.copy()
        for i, nbrs in adj.items():
            if nbrs:
                new_v[i] = v[list(nbrs)].mean(axis=0)
        v = new_v
    return v


# --------------------------------------------------------------------------- #
#  Silhouette-only CLI                                                         #
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(
        description="Preview silhouette extraction from two video sources"
    )
    ap.add_argument("--cam0", required=True)
    ap.add_argument("--cam1", required=True)
    ap.add_argument("--history", type=int, default=300)
    ap.add_argument("--var-threshold", type=float, default=40.0)
    ap.add_argument("--morph-ksize", type=int, default=7)
    ap.add_argument("--min-area", type=int, default=500)
    ap.add_argument("--out", default=None,
                    help="Optional directory to save mask images")
    args = ap.parse_args()

    bg0 = make_bg_subtractor(args.history, args.var_threshold)
    bg1 = make_bg_subtractor(args.history, args.var_threshold)

    cap0 = cv2.VideoCapture(args.cam0)
    cap1 = cv2.VideoCapture(args.cam1)
    out_dir = Path(args.out) if args.out else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    frame_idx = 0
    while True:
        r0, f0 = cap0.read()
        r1, f1 = cap1.read()
        if not r0 or not r1:
            break

        m0 = extract_silhouette(f0, bg0, args.morph_ksize, args.min_area)
        m1 = extract_silhouette(f1, bg1, args.morph_ksize, args.min_area)

        vis = np.hstack([
            cv2.cvtColor(m0, cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(m1, cv2.COLOR_GRAY2BGR),
        ])
        cv2.putText(vis, f"Frame {frame_idx}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Silhouettes (cam0 | cam1)", vis)

        if out_dir:
            cv2.imwrite(str(out_dir / f"mask0_{frame_idx:06d}.png"), m0)
            cv2.imwrite(str(out_dir / f"mask1_{frame_idx:06d}.png"), m1)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        frame_idx += 1

    cap0.release()
    cap1.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
