"""Rat skeletal model, capsule body surface, synthetic pose generation,
and multi-view silhouette pose fitting.

- rat_skeleton    : rat23 skeleton, forward kinematics, joint limits.
- body_model      : tapered-capsule body surface + per-camera silhouette.
- fit             : fit a pose to multi-view silhouettes (analysis by synthesis).
- synthetic_dataset: labeled synthetic frames from valid poses.
"""
