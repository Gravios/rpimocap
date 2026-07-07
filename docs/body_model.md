# Body Model & Pose Fitting

A **visually comparable 3D body surface** wrapped on the rat23 skeleton, and a
fitter that optimizes the pose so the body's projected silhouette matches the
observed rat masks across the calibrated cameras — *analysis by synthesis*.
This is the shape prior the skeleton's docstring anticipates, made concrete.

- `rpimocap.model.body_model` — the surface + per-camera silhouette rendering.
- `rpimocap.model.fit` — the multi-view pose fitter.

It closes the loop with detection: `detection.topo_detect` gives the observed
rat silhouette and a triangulated body centroid; this fits a full articulated
body to those, so you can *see* the model on the frames and score how well a
pose explains the images.

---

## The body surface

Each bone (parent → child) is a **tapered capsule** — a truncated cone with
spherical end caps — with a radius at each end. The union of the capsules is a
rat-shaped volume that follows any articulated pose. A capsule is convex, so
its image projection is the convex hull of its projected surface samples, and
the body silhouette is the union of the per-bone hulls.

```python
from rpimocap.model.body_model import DEFAULT_RADII, scale_radii
```

`DEFAULT_RADII` is a plausible adult-rat shape — a plump trunk (~40 mm across),
tapering head and tail, thin limbs. Scale it per individual with
`scale_radii(DEFAULT_RADII, factor)`, or override any bone's `(parent_r,
child_r)` entry. It is reference geometry, not measured anatomy.

## Rendering a silhouette

```python
from rpimocap.model.body_model import render_silhouette, render_pose_silhouette
from rpimocap.model.rat_skeleton import forward_kinematics

kp  = forward_kinematics(pose)                       # (23, 3) mm
sil = render_silhouette(kp, dlt_P0, image_shape=(1080, 2028))   # uint8 mask
# or straight from a pose:
sil = render_pose_silhouette(pose, dlt_P0, image_shape=(1080, 2028))
```

`P` is the arena-DLT matrix for that camera (`dlt_P0`/`dlt_P1`). Points behind
the camera are dropped. `silhouette_iou(a, b)` scores the overlap of two masks.

## Fitting a pose

```python
from rpimocap.model.fit import (fit_pose, fit_pose_multistart,
                                fit_pose_staged, curled_pose, multiview_iou)
```

The objective is **mean silhouette IoU** across cameras, maximized with a
derivative-free optimizer (Powell). The pose is low-dimensional — a 6-DOF root
(position + orientation) plus a body-scale, optionally a few joint angles.
Because the body's yaw is ambiguous from a blob, `fit_pose_multistart` sweeps
several initial headings and keeps the best.

```python
# masks: [cam0_mask, cam1_mask]  (e.g. detect_stereo(...).det{0,1}.mask)
# seed the root from the triangulated centroid (detect_stereo(...).point)
pose, iou = fit_pose_multistart(masks, [dlt_P0, dlt_P1], root_pos=seed_xyz,
                                headings=6, downscale=4)

# refine specific joints too (each adds 3 Euler angles, clamped to limits):
pose, iou = fit_pose(masks, [dlt_P0, dlt_P1], pose,
                     joints=["SpineF", "SpineL"], downscale=4)
```

`downscale` renders and compares at reduced resolution for speed. Fitting only
the root + scale places, orients, and sizes the body; add `joints` to bend the
spine or tuck limbs. Fitted joint angles are **clamped to the skeleton's
`JOINT_LIMITS`** (`clamp=True`), so the optimizer can't produce an
anatomically impossible pose.

### Staged fit for a curled rat

Fitting all joints at once is high-dimensional and gets stuck. `fit_pose_staged`
does it coarse-to-fine: root + scale (multistart), then **tuck the limbs**
(`TUCKED_ANGLES` / `curled_pose` — forelimb and hindlimb hinges folded up into
the compact shape of a resting animal), then add joint groups progressively
(spine, then limb hinges), each a clamped `fit_pose` refinement:

```python
pose, iou = fit_pose_staged(masks, [dlt_P0, dlt_P1], root_pos=seed_xyz,
                            headings=4, tucked=True,
                            stages=(("SpineF", "SpineL"),
                                    ("ElbowL", "ElbowR", "KneeL", "KneeR")))
```

---

## End-to-end example

```python
import numpy as np, cv2
from rpimocap.detection.topo_detect import detect_stereo, build_floor_mask
from rpimocap.model.fit import fit_pose_multistart
from rpimocap.model.body_model import render_pose_silhouette

cal = np.load("calib_from_corners.npz"); P0, P1 = cal["dlt_P0"], cal["dlt_P1"]
arena = np.array([[-140,-215,0],[140,-215,0],[140,215,0],[-140,215,0],
                  [-140,-215,388],[140,-215,388],[140,215,388],[-140,215,388]], float)
fl0 = build_floor_mask(P0, arena, g0.shape, mode="floor")
fl1 = build_floor_mask(P1, arena, g1.shape, mode="floor")

R = detect_stereo(g0, g1, fl0, fl1, P0, P1)          # masks + 3D centroid
seed = [R.point[0], R.point[1], max(R.point[2], 45.0)]
pose, iou = fit_pose_multistart([R.det0.mask, R.det1.mask], [P0, P1],
                                root_pos=seed, headings=4, downscale=5)
sil0 = render_pose_silhouette(pose, P0, image_shape=g0.shape)   # overlay on cam0
```

---

## Validation & known limitations

- **Synthetic recovery:** rendering a known pose and fitting it back recovers
  IoU ≈ 0.96 with the heading, scale, and position close to truth — the fitter
  and renderer are correct.
- **Real frames:** on a grooming (curled) rat, the root + scale fit places,
  orients, and sizes the body but reaches only moderate IoU (~0.46), because
  the rest-pose limbs don't match a tucked animal. `fit_pose_staged` with the
  tucked init and spine + limb-hinge stages raises this to ~0.52; the ceiling
  is then set by **inconsistency between the two views' detector masks** (one
  over-, one under-segmented), which no single 3D pose can satisfy — more
  consistent masks (e.g. `--seg-barrier fur` on the under-segmenting view)
  raise it further, as would adding more joints.
- **Speed:** the fit is derivative-free and renders per evaluation — seconds
  per frame per heading, not real-time. It is a **refinement/analysis** tool,
  not a per-frame tracker. For speed, warm-start each frame from the previous
  pose (a single heading), coarsen `downscale`, or move to a differentiable
  silhouette renderer for gradient-based fitting.
- **Shape, not appearance:** the objective is silhouette overlap only — no
  shading or texture. It constrains pose and size, not surface detail.
- **Joint limits** are enforced by clamping fitted angles to `JOINT_LIMITS`
  (`clamp=True`); the fit cannot leave the valid range, though it uses
  projection (clamping) rather than a smooth barrier, so angles can pin to a
  limit.
