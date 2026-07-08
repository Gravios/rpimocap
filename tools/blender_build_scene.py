#!/usr/bin/env python3
"""
tools/blender_build_scene.py — build an rpimocap manual-fitting scene in Blender.

Run INSIDE Blender against a scene spec written by
``rpimocap.gui.blender_export.build_scene_spec``:

    blender --python tools/blender_build_scene.py -- scene_spec.json

It creates:
  * two cameras placed/oriented/lensed from the DLT calibration, each showing
    its frame as a camera background (look through a camera → the 3D world
    lines up with the image);
  * the arena as a wireframe box;
  * the rat23 skeleton as an armature with IK targets on the four limbs;
  * the (pre-aligned) rat mesh imported and skinned to the armature.

Then pose the rat by hand — grab the IK target empties (IK_HandL/…/IK_FootL)
and the root/bones — checking both camera views (Numpad 0 cycles the active
camera; set each camera active and toggle its background). Export the pose back
to the pipeline with tools/blender_export_pose.py.

Only numpy + json are needed inside Blender (both bundled).
"""
import json
import sys

import numpy as np

try:
    import bpy
    from mathutils import Matrix, Vector
except ImportError:
    sys.exit("This script must be run inside Blender:  blender --python "
             "tools/blender_build_scene.py -- scene_spec.json")


def _argv_spec():
    argv = sys.argv
    if "--" not in argv:
        sys.exit("Pass the scene spec after --:  ... -- scene_spec.json")
    rest = argv[argv.index("--") + 1:]
    if not rest:
        sys.exit("No scene spec path given after --")
    return rest[0]


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    for coll in (bpy.data.meshes, bpy.data.cameras, bpy.data.armatures):
        for block in list(coll):
            if block.users == 0:
                coll.remove(block)


def build_cameras(spec, scene):
    w, h = spec["resolution"]
    scene.render.resolution_x, scene.render.resolution_y = w, h
    # global pixel aspect from the cameras (they are near-identical)
    scene.render.pixel_aspect_x = 1.0
    scene.render.pixel_aspect_y = spec["cameras"][0]["pixel_aspect_y"]
    cams = []
    for cs in spec["cameras"]:
        cam = bpy.data.cameras.new(cs["name"])
        cam.lens = cs["lens"]
        cam.sensor_width = cs["sensor_width"]
        cam.sensor_fit = "HORIZONTAL"
        cam.shift_x = cs["shift_x"]
        cam.shift_y = cs["shift_y"]
        cam.clip_start = 1.0
        cam.clip_end = 10000.0
        obj = bpy.data.objects.new(cs["name"], cam)
        scene.collection.objects.link(obj)
        M = Matrix([r + [0.0] for r in cs["rotation_c2w"]]).to_4x4()
        M.translation = Vector(cs["location"])
        obj.matrix_world = M
        try:
            img = bpy.data.images.load(cs["image"])
            cam.show_background_images = True
            bg = cam.background_images.new()
            bg.image = img
            bg.display_depth = "FRONT"
            bg.frame_method = "CROP"
            bg.alpha = 0.85
        except Exception as e:
            print(f"[warn] could not load background {cs['image']}: {e}")
        cams.append(obj)
    scene.camera = cams[0]
    return cams


def build_arena(spec, scene):
    mesh = bpy.data.meshes.new("Arena")
    mesh.from_pydata([tuple(c) for c in spec["arena"]["corners"]],
                     [tuple(e) for e in spec["arena"]["edges"]], [])
    mesh.update()
    obj = bpy.data.objects.new("Arena", mesh)
    scene.collection.objects.link(obj)
    obj.display_type = "WIRE"
    obj.show_in_front = True
    obj.hide_select = True                # don't grab it while posing
    return obj


def build_armature(spec, scene):
    sk = spec["skeleton"]
    rest = {k: Vector(v) for k, v in sk["rest"].items()}
    arm_data = bpy.data.armatures.new("RatArmature")
    arm = bpy.data.objects.new("RatArmature", arm_data)
    scene.collection.objects.link(arm)
    bpy.context.view_layer.objects.active = arm
    arm.show_in_front = True

    bpy.ops.object.mode_set(mode="EDIT")
    eb = {}
    for parent, child in sk["bones"]:      # each bone named by its child joint
        b = arm_data.edit_bones.new(child)
        b.head = rest[parent]
        b.tail = rest[child]
        eb[child] = (b, parent)
    for child, (b, parent) in eb.items():
        if parent in eb:                   # parent bone = the one ending at `parent`
            b.parent = eb[parent][0]
    bpy.ops.object.mode_set(mode="OBJECT")

    # IK targets + constraints
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="POSE")
    for chain in sk["ik_chains"]:
        tip, tgt_name = chain["tip"], chain["target"]
        tgt = bpy.data.objects.new(tgt_name, None)
        tgt.empty_display_type = "PLAIN_AXES"
        tgt.empty_display_size = 14.0
        tgt.location = rest[tip]           # end effector = tail of the tip bone
        scene.collection.objects.link(tgt)
        tgt.parent = arm                   # follow gross placement of the rat
        pb = arm.pose.bones[tip]
        con = pb.constraints.new("IK")
        con.target = tgt
        con.chain_count = int(chain["length"])
    bpy.ops.object.mode_set(mode="OBJECT")
    return arm


def import_and_skin(spec, arm):
    if not spec.get("obj"):
        print("[info] no mesh in spec — armature only")
        return None
    try:
        bpy.ops.wm.obj_import(filepath=spec["obj"])          # Blender 4.x
    except AttributeError:
        bpy.ops.import_scene.obj(filepath=spec["obj"])       # Blender 3.x
    mesh_obj = bpy.context.selected_objects[0]
    bpy.ops.object.select_all(action="DESELECT")
    mesh_obj.select_set(True)
    arm.select_set(True)
    bpy.context.view_layer.objects.active = arm
    try:                                    # bone-heat automatic weights
        bpy.ops.object.parent_set(type="ARMATURE_AUTO")
    except RuntimeError as e:
        print(f"[warn] auto weights failed ({e}); falling back to envelopes")
        bpy.ops.object.parent_set(type="ARMATURE_ENVELOPE")
    return mesh_obj


def try_camera_view():
    for area in bpy.context.screen.areas:
        if area.type == "VIEW_3D":
            for space in area.spaces:
                if space.type == "VIEW_3D":
                    space.region_3d.view_perspective = "CAMERA"
                    space.clip_end = 10000.0
            break


def main():
    spec = json.load(open(_argv_spec()))
    scene = bpy.context.scene
    clear_scene()
    build_arena(spec, scene)
    cams = build_cameras(spec, scene)
    arm = build_armature(spec, scene)
    import_and_skin(spec, arm)
    try:
        try_camera_view()
    except Exception:
        pass
    print(f"Scene built: {len(cams)} cameras, arena wireframe, "
          f"rat armature with {len(spec['skeleton']['ik_chains'])} IK chains"
          + (", mesh skinned." if spec.get("obj") else "."))
    print("Look through cam0 (Numpad 0). Grab IK_* empties to pose the limbs.")


if __name__ == "__main__":
    main()
