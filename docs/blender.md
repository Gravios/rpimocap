# Manual pose fitting in Blender

A workflow for placing and posing the rat by hand in a real 3D viewport, with
the two calibrated camera views shown as backgrounds so that — looking through
each camera — the 3D scene lines up with the frame. You position and articulate
the rat (with inverse kinematics on the limbs) until it matches both views, then
export the pose back to the pipeline as 3D keypoints in arena millimetres.

This complements the automatic silhouette fitter: use Blender for hard frames,
ground-truth annotation, or seeding the fitter.

## Overview

Two steps, two environments:

1. **In the pipeline** (`rpimocap.gui.blender_export`) — decompose the DLT
   calibration into Blender cameras and write a JSON *scene spec* (plus an
   aligned copy of the rat mesh). This is where the camera math lives, and it
   is exact.
2. **In Blender** (`tools/blender_build_scene.py`) — read the spec and build the
   scene: cameras + backgrounds, arena wireframe, rat armature with IK, and the
   skinned mesh.

Only numpy and json are used inside Blender, so Blender needs none of rpimocap's
dependencies.

## 1. Write the scene spec

```python
from rpimocap.gui.blender_export import build_scene_spec
build_scene_spec(
    "calib_from_corners.npz",          # dlt_P0 / dlt_P1
    "frame_002716_cam0.png",           # cam0 background
    "frame_002716_cam1.png",           # cam1 background
    "scene_spec.json",                 # output spec
    obj_path="RAT MODEL.obj",          # optional: your rat OBJ
)
```

This writes `scene_spec.json` and, if an OBJ is given, `aligned_rat.obj` next to
it (the mesh rotated/scaled/placed onto the rat23 rest skeleton so Blender
imports it already positioned).

### How the cameras are derived

Each DLT matrix `P` (arena-mm → pixel) is decomposed into intrinsics `K`,
rotation `R`, and camera centre `C` (`decompose_dlt`), which reproduces the DLT
projection to ~1e-11 px. Those map to a Blender camera:

- **location** = `C` (the camera centre — for this rig, ~780 mm above the
  arena, slightly left/right: an overhead stereo pair);
- **orientation** = `Rᵀ` with the CV→Blender axis flip (Blender cameras look
  down −Z with +Y up);
- **lens** = `fx · sensor_width / image_width` (sensor fit HORIZONTAL);
- **shift_x / shift_y** = the principal-point offset from the image centre;
- **pixel_aspect_y** = `fx/fy` for the DLT's slightly non-square pixels.

## 2. Build the scene in Blender

```
blender --python tools/blender_build_scene.py -- scene_spec.json
```

You get: two cameras (`cam0`, `cam1`) each with its frame as a background;
the **arena** as a wireframe box; a **RatArmature** with IK targets
(`IK_HandL`, `IK_HandR`, `IK_FootL`, `IK_FootR`); and the rat mesh skinned to
the armature.

### Verify the camera alignment

Look through **cam0** (Numpad 0). The arena **wireframe corners** should sit on
the arena corners in the background frame. To check the other camera, make
`cam1` the active camera (select it, Ctrl+Numpad 0) — its background follows the
active camera.

This is the same check you can run in the pipeline: projecting the arena
wireframe through the DLT lands the corners within the calibration's ~7 px on
the real arena.

## 3. Pose the rat

- **Placement (gross):** select the **RatArmature object** and move/rotate it
  (G / R) in Object mode until the body sits at the rat's position in both
  views. The IK targets and mesh follow.
- **Articulation:** enter **Pose mode**. Bend the spine bones (SpineF, SpineL,
  …) and grab the **IK_\*** empties (G) to place the hands and feet; the limb
  chains (3 bones each) solve automatically. Check both camera views as you go.

## 4. Export the pose

With the pose set, from Blender:

```
blender --background your_scene.blend --python tools/blender_export_pose.py -- \
    pose_2716.json frame_002716
```

(or run it from Blender's text editor). It writes the 23 joint world positions
in arena mm:

```json
{"frame": "frame_002716", "keypoints": {"Snout": [x, y, z], "SpineM": [...], ...}}
```

That is a hand-annotated 3D pose. Project it with the DLT to check it against
the frames, use it as ground truth, or seed `fit_pose_local`.

## Caveats (read before trusting alignment)

- **Lens distortion.** The DLT is a *pinhole* model, but the wide-angle frames
  have visible barrel distortion (the arena edges bow). The DLT places the
  corners correctly (~7 px), but a straight wireframe edge won't follow a curved
  image edge at the periphery. The rat is near the image centre where distortion
  is smallest, so posing there is accurate. For exact edge alignment, undistort
  the frames first (if you have distortion coefficients) and re-fit the DLT to
  the undistorted images.
- **Principal-point shift signs.** The geometry (position, orientation, lens) is
  exact; the sensor shift is small (~10–36 px here). If the background is
  slightly offset from the wireframe, nudge the camera's `shift_x` / `shift_y`
  (or flip a sign) in the camera data — a <1 % adjustment. `shift_y`'s sign in
  particular depends on Blender's vertical convention.
- **Pixel aspect is global.** Blender's pixel aspect is a scene setting, so both
  cameras share one value (they differ by <0.5 % here). It's taken from cam0.
- **Forelimb skin weights.** The artist mesh is sculpted in a standing pose
  whose forelimbs differ from the skeleton's straight-down rest, so bone-heat
  automatic weights on the shoulders/forelimbs may need touch-up in weight
  paint. The trunk and hindquarters bind cleanly.
