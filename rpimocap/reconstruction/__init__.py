"""
rpimocap.reconstruction
========================
3D reconstruction from calibrated stereo views.

Modules
-------
triangulate     DLT triangulation, trajectory smoothing, gap interpolation
voxel           MOG2 silhouette extraction, voxel carving, mesh export
"""

from rpimocap.reconstruction.triangulate import (
    Point3D,
    triangulate_dlt,
    reprojection_error,
    triangulate_keypoints,
    build_trajectory_dict,
    smooth_trajectory,
    fill_trajectory_gaps,
    trajectory_stats,
)
from rpimocap.reconstruction.voxel import (
    VoxelGrid,
    build_voxel_grid,
    voxel_centers,
    project_points_batch,
    make_bg_subtractor,
    extract_silhouette,
    carve_frame,
    apply_carving,
    occupied_centers,
    surface_centers,
    grid_to_mesh,
)

__all__ = [
    # triangulate
    "Point3D",
    "triangulate_dlt",
    "reprojection_error",
    "triangulate_keypoints",
    "build_trajectory_dict",
    "smooth_trajectory",
    "fill_trajectory_gaps",
    "trajectory_stats",
    # voxel
    "VoxelGrid",
    "build_voxel_grid",
    "voxel_centers",
    "project_points_batch",
    "make_bg_subtractor",
    "extract_silhouette",
    "carve_frame",
    "apply_carving",
    "occupied_centers",
    "surface_centers",
    "grid_to_mesh",
]
