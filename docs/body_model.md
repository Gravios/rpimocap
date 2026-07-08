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

## Skinned mesh surface (higher fidelity)

`rpimocap.model.mesh_model` provides a smoother, more rat-shaped alternative to
the capsule union, with proper **joint deformation** (linear blend skinning).
Grounded in rat side-profile references — a rounded body tapering to a pointed
snout, small ears, thin tucked limbs, a thick-to-thin tail.

- Each bone contributes a tapered-capsule signed distance; a smooth-minimum
  blends them (metaball style) into one organic surface, and marching cubes
  extracts a watertight triangle mesh in the rest pose.
- Each vertex is bound to its nearest bones (inverse-square falloff), so
  vertices near a joint blend the two bones' frames — smooth bending, no rigid
  crease. `skin_mesh` uses the per-joint world transforms from
  `forward_kinematics_transforms`.

```python
from rpimocap.model.mesh_model import (build_rat_mesh, skin_mesh,
                                        render_mesh_pose_silhouette)

mesh  = build_rat_mesh()                      # once — marching cubes + weights
verts = skin_mesh(mesh, pose)                 # (V, 3) posed vertices (LBS)
sil   = render_mesh_pose_silhouette(mesh, pose, dlt_P0, image_shape=(1080, 2028))
```

Build the mesh once (marching cubes + weights are the cost, ~1 s); skinning
(~7 ms) and rendering (~25 ms) per pose are fast. To fit the mesh instead of
the capsules, pass it as the fitter's `render_fn`:

```python
render_fn = lambda pose, P, shape: render_mesh_pose_silhouette(mesh, pose, P, shape)
pose, iou = fit_pose_staged(masks, [dlt_P0, dlt_P1], root_pos=seed,
                            render_fn=render_fn)
```

On real frame_002716 the mesh's root+scale fit reaches IoU ≈ 0.65 versus the
capsule model's 0.46 — the rounded body matches a curled rat far better than a
splayed capsule union.

### Loading an external artist mesh

`load_obj_mesh(obj_path)` adapts a real artist OBJ rat model for the pipeline:
it parses the mesh, rotates it into the rat convention (+x forward, +y left,
+z up), scales it to the skeleton, aligns the nose to the snout and the feet
to the floor, and binds every vertex to the nearest bones — returning a
`RatMesh` that skins, renders, and fits exactly like the built-in one.

```python
from rpimocap.model.mesh_model import load_obj_mesh, render_mesh_pose_silhouette
mesh = load_obj_mesh("RAT MODEL.obj", trim_tail=True, decimate=0.85)
render_fn = lambda pose, P, shape: render_mesh_pose_silhouette(mesh, pose, P, shape)
pose, iou = fit_pose_staged(masks, [dlt_P0, dlt_P1], root_pos=seed,
                            render_fn=render_fn, physics_weight=2.0)
```

- **`decimate`** (fraction of faces to remove, e.g. `0.85`) needs
  `fast-simplification`; a 50 k-face model renders in ~110 ms, a decimated
  ~7 k-face one in ~16 ms, which is what makes fitting practical.
- **`trim_tail`** drops the thin tail behind the tail base (the detector masks
  don't include it), which lifts silhouette IoU.
- **Orientation** defaults suit a Y-up, +Z-forward model (head at +Z); adjust
  `forward_axis` / `up_axis` / `forward_sign` and the alignment offsets
  (`scale_mult`, `nose_dx`, `feet_dz`) for a differently-posed model.

Caveats: the trunk and hindquarters bind well, but a **sculpted forelimb pose**
that differs from the skeleton's straight-down rest binds only approximately —
use root/scale (+ spine) fitting rather than relying on forelimb articulation.
And an artist model is typically a **standing** rat: on a curled grooming frame
its detailed legs and raised head fit a tucked ball slightly worse than the
smooth procedural blob (IoU ~0.47 vs ~0.56 with physics), but it is the more
faithful surface for a standing or moving animal.

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

## Physical plausibility (gravity + ground contact)

The silhouette objective is orientation-ambiguous — a sideways or floating rat
can project to a similar blob — so the fit can return physically impossible
poses. `rpimocap.model.physics` adds the consequences of gravity as a
lightweight pose prior (not a rigid-body dynamics engine):

- **uprightness** — the body's up-axis should point to world +z,
- **ground contact** — the lowest hind foot should touch the floor (z = 0),
- **non-penetration** — nothing below the floor,
- **support** — the trunk's centre of mass over the grounded feet (static
  stability under gravity).

Add it to any fit via `physics_weight`:

```python
pose, iou = fit_pose_staged(masks, [dlt_P0, dlt_P1], root_pos=seed,
                            render_fn=render_fn, physics_weight=2.0)
```

Or project a finished pose to a resting one — righting the body and dropping it
until the hind feet contact the floor, as if it had fallen under gravity:

```python
from rpimocap.model.physics import settle_pose
resting = settle_pose(pose)          # upright, hind feet at z = 0, heading kept
```

On real frame_002716 the plain mesh fit returns the body tilted ~44° off
vertical and sunk 15 mm into the floor; with `physics_weight=2.0` it is upright
with the hind feet on the ground, at a small IoU cost (0.65 → 0.56) — the
physically correct pose rather than the silhouette-optimal-but-impossible one.

## Manual fitting & frame-to-frame tracking

For frames the automatic fit struggles with (ambiguous silhouettes, unusual
poses), `tools/pose_gui.py` is a Qt GUI to set the pose by hand and drive the
fitter. Sliders control position, orientation, scale, spine bend, and limb
tuck; the model (green) overlays both camera views alongside the detector mask
(orange), with the live IoU shown.

```
python tools/pose_gui.py --frames-dir SESSION/raw --calib calib_from_corners.npz \
    --model procedural --poses keyframes.json
```

`--model` is `procedural` (built-in mesh), `capsule` (fastest), or an artist
OBJ path. **Save keyframe** stores the current pose for the frame into the
`--poses` JSON; **Fit (detect)** fits freely from the detection; **Fit (local)**
refines within a neighbourhood of the current pose; **Prev/Next** carry the
pose forward as a warm start. The GUI logic lives in Qt-free
`rpimocap.gui.pose_state.PoseFitterState` (scriptable, unit-tested).

### Restricting the search on neighbouring frames

Once a frame is posed well — by hand or a good auto-fit — `fit_pose_local`
restricts the optimizer to nearby poses, so a moving rat can be tracked
frame-to-frame without the fit jumping to a distant pose:

```python
from rpimocap.model.fit import fit_pose_local
pose, iou = fit_pose_local(masks, [P0, P1], prev_pose,
                           pos_tol=25, ang_tol=0.35, scale_tol=0.15,
                           render_fn=render_fn, physics_weight=2.0)
```

It bounds root position to ±`pos_tol` mm, every angle to ±`ang_tol` rad, and
scale to ±`scale_tol` (fractional), warm-started from `prev_pose`. Carry each
frame's result to the next as the seed and set a fresh keyframe by hand
whenever the pose drifts enough to need it.

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
