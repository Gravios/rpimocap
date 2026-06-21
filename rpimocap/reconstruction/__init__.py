from rpimocap.reconstruction.align import (
    AlignPoint, AlignResult, AlignPoint as _AP,
    kabsch_align, save_align_csv, load_align_csv,
    TracedEdge, DistortionResult,
    fit_distortion_plumb_line, patch_calibration_distortion,
    save_edges_csv, load_edges_csv,
    align_skeleton_frames, align_voxel_frames,
    refine_calibration_from_arena,
    _edge_line_rmse, _undistort_radial,
)
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
from rpimocap.reconstruction.refraction import (
    RefractivePlane,
    ArenaRefractionModel,
    build_box_arena,
    snell_refract,
    refract_through_wall,
    pixel_to_world_ray,
    closest_point_two_lines,
    triangulate_refracted,
    save_arena_config,
    load_arena_config,
)
from rpimocap.reconstruction.kalman import (
    KalmanInfo,
    KalmanTracker3D,
    smooth_trajectory_kalman,
    smooth_trajectory_dict_kalman,
)
from rpimocap.reconstruction.rearing import (
    PostureState,
    RearingClassifier,
    trace_postures,
)
from rpimocap.reconstruction.epipolar import (
    StereoMatch,
    fundamental_from_projections,
    epipolar_distance,
    match_stereo_candidates,
    best_stereo_point,
)
from rpimocap.reconstruction.arena_gate import (
    StaticDepthGate,
    in_arena_volume,
    above_floor,
    accept_point,
    build_static_depth_gate,
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
    # epipolar (tightly-coupled two-view selection)
    "StereoMatch",
    "fundamental_from_projections",
    "epipolar_distance",
    "match_stereo_candidates",
    "best_stereo_point",
    # arena gate (static-scene geometric rejection)
    "StaticDepthGate",
    "in_arena_volume",
    "above_floor",
    "accept_point",
    "build_static_depth_gate",
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
    # refraction
    "RefractivePlane",
    "ArenaRefractionModel",
    "build_box_arena",
    "snell_refract",
    "refract_through_wall",
    "pixel_to_world_ray",
    "closest_point_two_lines",
    "triangulate_refracted",
    "save_arena_config",
    "load_arena_config",
    # kalman
    "KalmanInfo",
    "KalmanTracker3D",
    "smooth_trajectory_kalman",
    "smooth_trajectory_dict_kalman",
    # rearing
    "PostureState",
    "RearingClassifier",
    "trace_postures",
]
