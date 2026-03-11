"""
rpimocap.io
===========
Serialisation and export utilities.

Formats
-------
PLY         Per-frame point clouds and triangle meshes (MeshLab, Blender, CloudCompare)
HDF5        Full trajectory archive with optional voxel frames (h5py)
JSON        Compact viewer JSON consumed by the bundled Three.js viewer
CSV         Detection statistics
"""

from rpimocap.io.export import (
    write_ply_pointcloud,
    write_ply_mesh,
    write_ply_skeleton_frame,
    write_hdf5,
    write_viewer_json,
    write_stats_csv,
)

__all__ = [
    "write_ply_pointcloud",
    "write_ply_mesh",
    "write_ply_skeleton_frame",
    "write_hdf5",
    "write_viewer_json",
    "write_stats_csv",
]
