"""Rat skeletal model, body surfaces, synthetic pose generation, multi-view
silhouette pose fitting, and a physics-based pose prior.

- rat_skeleton    : rat23 skeleton, forward kinematics (+ per-joint transforms), joint limits.
- body_model      : tapered-capsule body surface + per-camera silhouette.
- mesh_model      : smooth skinned triangle mesh (marching cubes + linear blend skinning).
- physics         : gravity/ground-contact/uprightness pose prior + settle_pose.
- fit             : fit a pose to multi-view silhouettes (analysis by synthesis).
- synthetic_dataset: labeled synthetic frames from valid poses.
"""
