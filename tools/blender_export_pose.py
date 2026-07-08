#!/usr/bin/env python3
"""
tools/blender_export_pose.py — export a hand-posed rat back to the pipeline.

Run INSIDE Blender after posing the RatArmature:

    blender --background scene.blend --python tools/blender_export_pose.py -- \
        pose_2716.json frame_002716

Writes the 23 rat23 joint world positions (arena mm — the same frame the
calibration and triangulation use) as JSON:

    {"frame": "...", "keypoints": {"Snout": [x,y,z], ...}}

That is a hand-annotated 3D pose for the frame: project it with the DLT to
check it against the images, use it as ground truth, or seed the fitter.
"""
import json
import sys

try:
    import bpy
except ImportError:
    sys.exit("Run inside Blender: blender --python tools/blender_export_pose.py "
             "-- out.json [frame_name]")


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    out = argv[0] if argv else "pose.json"
    frame_name = argv[1] if len(argv) > 1 else "frame"

    arm = bpy.data.objects.get("RatArmature")
    if arm is None:
        sys.exit("No 'RatArmature' in the scene — build it first.")
    Mw = arm.matrix_world

    # each joint is the TAIL of the bone named after it; the root SpineM is the
    # HEAD of a bone emanating from it (SpineF: SpineM→SpineF).
    kp = {pb.name: list(Mw @ pb.tail) for pb in arm.pose.bones}
    if "SpineF" in arm.pose.bones:
        kp["SpineM"] = list(Mw @ arm.pose.bones["SpineF"].head)

    data = {"frame": frame_name,
            "keypoints": {k: [round(float(c), 3) for c in v]
                          for k, v in kp.items()}}
    with open(out, "w") as fh:
        json.dump(data, fh, indent=2)
    print(f"wrote {len(kp)} keypoints (arena mm) to {out}")


if __name__ == "__main__":
    main()
