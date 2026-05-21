# Planar refraction correction

> **OFF by default.** The standard `rpimocap` rig has no plexiglass
> enclosure between the cameras and the animal — straight-ray DLT is
> correct. This module is wired in for future setups where an acrylic
> wall is interposed; opt in with `--enable-refraction` **and**
> `--refraction-config`.

`rpimocap.reconstruction.refraction` corrects the systematic
apparent-position bias that arises when a camera observes a subject
through transparent acrylic (PMMA) arena walls. With a refractive index
of n ≈ 1.49 and typical wall thickness of 6 mm, a ray entering the
arena at an oblique angle is laterally shifted by ~1–3 mm relative to
its in-air extension — enough to dominate the triangulation budget for
sub-mm whisker tracking.

This module models each wall as a **parallel-faced glass slab** and
refracts every camera ray with Snell's law before intersection. The
calibration itself is **not** modified: the arena corners used to
calibrate sit on the outer face of the wall, so calibration rays do
not traverse the acrylic.

---

## Physics

A camera ray with direction `d_in` hits the outer face of a wall whose
outward unit normal is `n`. At the outer face, Snell's law in vector
form gives the refracted direction inside the acrylic:

```
cos θ_i  = -d_in · n          (n flipped so cos θ_i > 0)
sin² θ_t = (n_air/n_glass)² · (1 − cos²θ_i)
d_glass  = (n_air/n_glass) · d_in + (n_air/n_glass · cos θ_i − cos θ_t) · n
```

The ray traverses the slab in a straight line to the inner face, where
Snell's law applied a second time bends the ray back. **Key invariant:
for parallel faces the emerging in-arena direction equals the original
camera direction**. The only effect is a lateral shift of the ray's
origin from the camera centre to a point on the inner face.

Concretely, `refract_through_wall(O, d, plane)` returns
`(B, d_out)` where `B` is the intersection of the refracted ray with
the inner face of the slab and `d_out ≈ d`.

## Triangulation algorithm

Given pixel observations in two views:

1. Build each camera's world-space ray:
   `O_i, d_i = pixel_to_world_ray(K_i, R_i, T_i, uv_i, dist_i)`.
2. For each ray, find the wall it crosses inside the arena:
   `plane_i = arena.find_traversed_plane(O_i, d_i)`.
3. Refract each ray through its wall, producing a shifted ray
   `(B_i, d_i)` originating on the inner face.
4. Intersect the two shifted rays by the closest-point-of-two-lines
   method (midpoint of the common perpendicular).

Because the in-arena ray is parallel to the original camera ray, the
wall traversed is determined entirely by the input ray and does not
depend on the unknown 3D point — so the algorithm is a single-shot
calculation rather than an iterative one. (`triangulate_refracted`
still loops as a safety net for future non-parallel-slab extensions.)

---

## Module API

```python
from rpimocap.reconstruction.refraction import (
    RefractivePlane, ArenaRefractionModel, build_box_arena,
    snell_refract, refract_through_wall,
    pixel_to_world_ray, closest_point_two_lines,
    triangulate_refracted,
    save_arena_config, load_arena_config,
)
```

### `RefractivePlane`
A single wall, parameterised by a point on its outer face, an outward
unit normal, slab thickness, and refractive indices. Optional
`half_extent` and `in_plane_axes` clip the wall to a finite rectangle so
that rays passing past its edge are not refracted.

### `ArenaRefractionModel`
A bag of `RefractivePlane`s. The key method is
`find_traversed_plane(O, d)`, which returns the first front-facing wall
the ray hits, or `(None, None)` if the ray misses all walls.

### `build_box_arena(xmin, xmax, ymin, ymax, zmin, zmax, *, thickness=6.0, n_glass=1.49, include_walls=("+x","-x","+y","-y"), include_ceiling=False, include_floor=False)`
Convenience constructor for an axis-aligned rectangular enclosure. Wall
points sit at the centre of each face and `half_extent` is set
symmetrically so that the four walls form a closed box. The default
includes only the four side walls — the top is usually open and the
floor is opaque flooring (not a refracting surface).

### `triangulate_refracted(O0, d0, O1, d1, arena, *, initial_xyz=None, max_iter=8, tol=1e-4)`
Returns `(xyz, gap, n_iter)`:
- `xyz` — recovered 3D point (mm)
- `gap` — distance between the two refracted rays at the closest
  approach (mm); a quality indicator analogous to reprojection error
- `n_iter` — number of iterations performed

### Config I/O
`save_arena_config(path, arena)` writes a human-readable JSON file
listing every wall's point, normal, thickness, refractive indices,
optional half-extent, and an optional label (`"+y wall"`, etc.).
`load_arena_config(path)` parses it back.

---

## Pipeline integration

The main `rpimocap-run` pipeline (`rpimocap.cli.pipeline`) accepts a
new flag:

```
--refraction-config JSON     Arena wall model produced by save_arena_config
```

When supplied, the pipeline loads the model and routes every
triangulation through `triangulate_refracted` instead of the
straight-ray DLT. The public `triangulate_keypoints` entrypoint takes
an `arena_model` kwarg together with the per-camera
`K_i, dist_i, R_i, T_i` so that any caller can opt in to refractive
triangulation without touching the lower-level solver.

If the required intrinsics/extrinsics are not supplied alongside
`arena_model`, the function silently falls back to straight-ray DLT.

---

## Example: build and persist a default rodent arena

```python
from rpimocap.reconstruction.refraction import build_box_arena, save_arena_config

# Arena bounds in mm: ±140 mm in X, ±215 mm in Y, 0–388 mm in Z
arena = build_box_arena(
    xmin=-140, xmax=140,
    ymin=-215, ymax=215,
    zmin=0,    zmax=388,
    thickness=6.0,
    n_glass=1.49,                       # PMMA
    include_walls=("+x", "-x", "+y", "-y"),
    include_ceiling=False,              # top is open
    include_floor=False,                # opaque base, no refraction
)
save_arena_config("arena.json", arena)
```

Then run the pipeline with refraction enabled:

```bash
rpimocap-run \
    --calib calibration.npz \
    --enable-refraction \
    --refraction-config arena.json \
    --videos cam0.tif cam1.tif \
    --out results/
```

---

## Practical notes

- **Calibration corners must lie on the OUTER wall face.** Otherwise
  calibration absorbs an unmodelled refraction shift and adding the
  refraction model later double-corrects.
- The +y, −y, +x, −x walls are independent; if the enclosure has only
  three glass walls (a common configuration with one opaque side),
  pass an explicit `include_walls=("+x","-x","-y")` and rays exiting
  toward the open side will be left unrefracted.
- Rays passing very close to a wall edge can produce numerically
  unstable refraction. The `half_extent` clipping inside
  `find_traversed_plane` is conservative and rejects edge cases by
  returning `(None, None)`, which causes the solver to fall back to
  the straight-ray geometry for that ray.
- For acrylic, n ≈ 1.49 across the visible spectrum to within 0.5%.
  Chromatic dispersion is well below the calibration noise floor.
