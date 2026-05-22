"""
pipeline.py — Main 3D reconstruction pipeline
==============================================
Orchestrates the full reconstruction from two synchronised video streams:

  1. Load stereo calibration (from calibrate.py output)
  2. Optionally rectify frames
  3. Per frame:
     a. Extract silhouette masks (voxel carving path)
     b. Run 2D pose detector on both views
     c. Triangulate keypoints → 3D skeleton
     d. Carve voxel grid → visual hull point cloud
  4. Temporal smoothing and gap-filling of skeleton trajectories
  5. Export:
     - PLY per-frame point clouds and meshes (output/ply/)
     - HDF5 archive (output/reconstruction.h5)
     - Viewer JSON (output/viewer_data.json)
     - Detection statistics CSV

Usage (minimal)
---------------
    python pipeline.py \\
        --cam0   data/video_cam0.mp4 \\
        --cam1   data/video_cam1.mp4 \\
        --calib  calibration.npz \\
        --bounds "-300,300,-200,200,0,400" \\
        --out    output/

Usage (with pre-computed DLC keypoints)
---------------------------------------
    python pipeline.py \\
        --cam0   data/video_cam0.mp4 \\
        --cam1   data/video_cam1.mp4 \\
        --calib  calibration.npz \\
        --detector dlc \\
        --dlc0   data/cam0_DLC.csv \\
        --dlc1   data/cam1_DLC.csv \\
        --bounds "-300,300,-200,200,0,400" \\
        --out    output/

Bounds format
-------------
    "xmin,xmax,ymin,ymax,zmin,zmax"  in the same units as calibration (mm).
    This defines the constrained physical space the subject moves within.
    Larger bounds → more voxels → slower carving. Start coarse (voxel_size=10)
    and refine once you have a working pipeline.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# --------------------------------------------------------------------------- #
#  Helpers                                                                     #
# --------------------------------------------------------------------------- #

def load_calibration(npz_path: str):
    """Load calibration .npz produced by calibrate.py."""
    d = np.load(npz_path)
    return d


def parse_bounds(s: str):
    """Parse "xmin,xmax,ymin,ymax,zmin,zmax" string."""
    vals = [float(x) for x in s.split(",")]
    if len(vals) != 6:
        raise ValueError("--bounds expects 6 comma-separated values: xmin,xmax,ymin,ymax,zmin,zmax")
    xmin, xmax, ymin, ymax, zmin, zmax = vals
    return ((xmin, xmax), (ymin, ymax), (zmin, zmax))


def open_video(path: str, bayer_pattern: str = "RGGB"):
    """Open a video file, routing .tif/.tiff through TiffCapture for 50GB+ support.

    Parameters
    ----------
    path          : video file or multi-frame TIFF stack
    bayer_pattern : Bayer CFA pattern for single-channel (raw) TIFFs.
                    IMX477 default is RGGB.  Ignored for non-TIFF files
                    and for TIFFs that already contain RGB data.
                    Valid values: RGGB, BGGR, GRBG, GBRG
    """
    if Path(path).suffix.lower() in (".tif", ".tiff"):
        from rpimocap.io.export import TiffCapture
        cap = TiffCapture(path, bayer_pattern=bayer_pattern)
    else:
        cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {path}")
    return cap


def remap_frame(frame: np.ndarray, mapx: np.ndarray, mapy: np.ndarray) -> np.ndarray:
    return cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)


# --------------------------------------------------------------------------- #
#  Main pipeline                                                               #
# --------------------------------------------------------------------------- #

def build_detector(args):
    """Instantiate the 2D pose detector based on CLI flags."""
    from rpimocap.detection.detectors import (
        CentroidPoseDetector,
        CSVPoseDetector,
        MediaPipePoseDetector,
    )

    det = args.detector.lower()
    if det == "mediapipe":
        print("  Detector: MediaPipe Pose (human, 33 landmarks)")
        d0 = MediaPipePoseDetector(
            model_complexity=args.model_complexity,
            min_detection_confidence=args.min_detection_conf,
            min_tracking_confidence=args.min_tracking_conf,
        )
        d1 = MediaPipePoseDetector(
            model_complexity=args.model_complexity,
            min_detection_confidence=args.min_detection_conf,
            min_tracking_confidence=args.min_tracking_conf,
        )
    elif det == "centroid":
        print("  Detector: Centroid (background subtraction, 3 pseudo-landmarks)")
        d0 = CentroidPoseDetector(
            var_threshold=args.var_threshold,
            morph_ksize=args.morph_ksize,
            min_area_px=args.min_area,
        )
        d1 = CentroidPoseDetector(
            var_threshold=args.var_threshold,
            morph_ksize=args.morph_ksize,
            min_area_px=args.min_area,
        )
    elif det == "dlc":
        print(f"  Detector: DeepLabCut CSV  cam0={args.dlc0}  cam1={args.dlc1}")
        if not args.dlc0 or not args.dlc1:
            print("ERROR: --dlc0 and --dlc1 required for --detector dlc")
            sys.exit(1)
        d0 = CSVPoseDetector(args.dlc0, fmt="dlc", min_likelihood=args.min_detection_conf)
        d1 = CSVPoseDetector(args.dlc1, fmt="dlc", min_likelihood=args.min_detection_conf)
    elif det == "sleap":
        print(f"  Detector: SLEAP CSV  cam0={args.sleap0}  cam1={args.sleap1}")
        if not args.sleap0 or not args.sleap1:
            print("ERROR: --sleap0 and --sleap1 required for --detector sleap")
            sys.exit(1)
        d0 = CSVPoseDetector(args.sleap0, fmt="sleap", min_likelihood=args.min_detection_conf)
        d1 = CSVPoseDetector(args.sleap1, fmt="sleap", min_likelihood=args.min_detection_conf)
    else:
        print(f"ERROR: unknown detector {det!r}")
        sys.exit(1)

    return d0, d1


def run(args):
    from rpimocap.io.export import (
        write_hdf5,
        write_ply_mesh,
        write_ply_pointcloud,
        write_ply_skeleton_frame,
        write_stats_csv,
        write_viewer_json,
    )
    from rpimocap.reconstruction.triangulate import (
        fill_trajectory_gaps,
        smooth_trajectory,
        trajectory_stats,
        triangulate_keypoints,
    )
    from rpimocap.reconstruction.voxel import (
        apply_carving,
        build_voxel_grid,
        carve_frame,
        extract_silhouette,
        grid_to_mesh,
        make_bg_subtractor,
        occupied_centers,
    )

    out_dir = Path(args.out)
    ply_dir = out_dir / "ply"
    ply_skeleton_dir = ply_dir / "skeleton"
    ply_volume_dir = ply_dir / "volume"
    for d in [out_dir, ply_skeleton_dir, ply_volume_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # ── Calibration ────────────────────────────────────────────────────────
    print("\n── Loading calibration ──────────────────────────────")
    cal = load_calibration(args.calib)
    P0 = cal["P0"]
    P1 = cal["P1"]
    image_size = tuple(int(x) for x in cal["image_size"])  # (w, h)
    do_rectify = args.rectify
    if do_rectify:
        map0x, map0y = cal["map0x"], cal["map0y"]
        map1x, map1y = cal["map1x"], cal["map1y"]
        print(f"  Rectification: ON  (image size {image_size})")
    else:
        # Use raw cameras with non-rectified projection matrices
        # For raw (unrectified) cameras, P = K [I | 0] for cam0
        # and P = K [R | t] for cam1; reconstruct from stored params
        K0, dist0 = cal["K0"], cal["dist0"]
        K1, dist1 = cal["K1"], cal["dist1"]
        R, T = cal["R"], cal["T"]
        P0 = K0 @ np.hstack([np.eye(3), np.zeros((3, 1))])
        P1 = K1 @ np.hstack([R, T.reshape(3, 1)])
        print("  Rectification: OFF  (using raw projection matrices)")

    # ── Refraction model ──────────────────────────────────────────────────
    # Refraction correction is OFF by default. The current arena has no
    # plexiglass enclosure between the cameras and the animal, so straight-
    # ray DLT is correct. The module is wired in for future setups where
    # an acrylic wall is interposed; opt in by passing both
    # --enable-refraction and --refraction-config.
    arena_model = None
    refraction_kwargs = {}
    if args.refraction_config and not args.enable_refraction:
        print("\n── Arena refraction model ───────────────────────────")
        print(f"  --refraction-config={args.refraction_config} supplied "
              "but --enable-refraction not set; refraction OFF.")
    elif args.enable_refraction and not args.refraction_config:
        print("\n── Arena refraction model ───────────────────────────")
        print("  --enable-refraction set without --refraction-config; "
              "refraction OFF (no wall model to apply).")
    elif args.enable_refraction and args.refraction_config:
        from rpimocap.reconstruction.refraction import load_arena_config
        arena_model = load_arena_config(args.refraction_config)
        print("\n── Arena refraction model ───────────────────────────")
        print(f"  Loaded {len(arena_model.planes)} walls from "
              f"{args.refraction_config}")
        if do_rectify:
            print("  WARNING: --refraction-config requires raw (un-rectified) "
                  "intrinsics/extrinsics. Disabling rectification path for "
                  "refractive triangulation.")
            # Pull raw K/dist/R/T even when rectification was requested
            K0_raw, dist0_raw = cal["K0"], cal["dist0"]
            K1_raw, dist1_raw = cal["K1"], cal["dist1"]
            R_raw, T_raw = cal["R"], cal["T"]
        else:
            K0_raw, dist0_raw = K0, dist0
            K1_raw, dist1_raw = K1, dist1
            R_raw, T_raw = R, T
        refraction_kwargs = dict(
            arena_model=arena_model,
            K0=K0_raw, dist0=dist0_raw,
            R0=np.eye(3), T0=np.zeros(3),
            K1=K1_raw, dist1=dist1_raw,
            R1=R_raw, T1=T_raw.reshape(3),
        )
    else:
        # Default path — no refraction. Print a brief confirmation so users
        # see in the log that this is the straight-ray DLT path.
        print("\n── Refraction correction: OFF (default) ─────────────")

    # ── Detectors ──────────────────────────────────────────────────────────
    print("\n── Pose detectors ───────────────────────────────────")
    det0, det1 = build_detector(args)
    kp_names = det0.keypoint_names
    skel_edges = det0.skeleton_edges
    print(f"  Landmarks: {len(kp_names)}")

    # ── Voxel grid ─────────────────────────────────────────────────────────
    print("\n── Voxel grid ───────────────────────────────────────")
    bounds = parse_bounds(args.bounds)
    grid_template = build_voxel_grid(bounds, args.voxel_size)
    print(f"  World bounds: X{bounds[0]} Y{bounds[1]} Z{bounds[2]}")

    # Shared-per-frame voxel occupancy (we reset each frame to the template)
    # Background subtractors for voxel carving (separate from detector ones)
    bg0_vox = make_bg_subtractor(args.bg_history, args.var_threshold)
    bg1_vox = make_bg_subtractor(args.bg_history, args.var_threshold)

    # ── Video capture ──────────────────────────────────────────────────────
    print("\n── Opening video streams ────────────────────────────")
    cap0 = open_video(args.cam0, bayer_pattern=args.bayer_pattern)
    cap1 = open_video(args.cam1, bayer_pattern=args.bayer_pattern)
    fps = cap0.get(cv2.CAP_PROP_FPS) or args.fps
    total_frames = int(cap0.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  FPS: {fps:.2f}  |  Total frames: {total_frames}")

    # Seek to start frame
    if args.start_frame > 0:
        cap0.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)
        cap1.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)

    # ── Frame loop ─────────────────────────────────────────────────────────
    print("\n── Processing ───────────────────────────────────────")
    skeleton_frames: list[list] = []
    voxel_pcs: list = []  # point clouds per frame

    frame_idx = 0
    n_target = args.n_frames if args.n_frames > 0 else total_frames
    t_start = time.time()

    while frame_idx < n_target:
        ret0, f0 = cap0.read()
        ret1, f1 = cap1.read()
        if not ret0 or not ret1:
            break

        # Rectify if requested
        if do_rectify:
            f0 = remap_frame(f0, map0x, map0y)
            f1 = remap_frame(f1, map1x, map1y)

        # ── Silhouette masks (always computed, used by voxel carving) ──────
        mask0 = extract_silhouette(
            f0, bg0_vox,
            morph_ksize=args.morph_ksize,
            min_area_px=args.min_area,
        )
        mask1 = extract_silhouette(
            f1, bg1_vox,
            morph_ksize=args.morph_ksize,
            min_area_px=args.min_area,
        )

        # ── Voxel carving ──────────────────────────────────────────────────
        if args.voxel:
            occ = carve_frame(
                grid_template, P0, P1,
                mask0, mask1, image_size,
                chunk_size=args.chunk_size,
            )
            carved_grid = apply_carving(grid_template, occ)
            pc = occupied_centers(carved_grid) if carved_grid.n_occupied else np.zeros((0, 3))
            voxel_pcs.append(pc)

            if args.save_ply_volume and frame_idx % args.ply_stride == 0:
                ply_path = ply_volume_dir / f"volume_{frame_idx:06d}.ply"
                if args.save_mesh and len(pc) > 0:
                    verts, faces = grid_to_mesh(carved_grid, smooth_iterations=1)
                    if len(verts):
                        write_ply_mesh(ply_path, verts, faces)
                else:
                    write_ply_pointcloud(ply_path, pc)
        else:
            voxel_pcs.append(None)

        # ── Pose detection ─────────────────────────────────────────────────
        abs_idx = args.start_frame + frame_idx
        r0 = det0.detect(f0, abs_idx)
        r1 = det1.detect(f1, abs_idx)

        # ── Triangulation ──────────────────────────────────────────────────
        pts3d = triangulate_keypoints(
            P0, P1, r0, r1,
            min_confidence=args.min_detection_conf,
            max_reprojection_px=args.max_repr_error,
            **refraction_kwargs,
        )
        skeleton_frames.append(pts3d)

        if args.save_ply_skeleton and frame_idx % args.ply_stride == 0:
            write_ply_skeleton_frame(
                ply_skeleton_dir / f"skeleton_{frame_idx:06d}.ply",
                pts3d,
            )

        # ── Progress ───────────────────────────────────────────────────────
        frame_idx += 1
        if frame_idx % 25 == 0:
            elapsed = time.time() - t_start
            fps_proc = frame_idx / elapsed
            eta = (n_target - frame_idx) / fps_proc if fps_proc > 0 else 0
            det_rate = sum(1 for f in skeleton_frames if f) / len(skeleton_frames)
            print(f"  frame {frame_idx:5d}/{n_target}  "
                  f"{fps_proc:.1f} fps  ETA {eta:.0f}s  "
                  f"det {det_rate:.0%}  "
                  f"{'vox ' + str(carved_grid.n_occupied) + ' voxels' if args.voxel else ''}")

    cap0.release()
    cap1.release()
    det0.close()
    det1.close()

    elapsed = time.time() - t_start
    print(f"\n  Processed {frame_idx} frames in {elapsed:.1f}s "
          f"({frame_idx/elapsed:.1f} fps)")

    # ── Temporal smoothing ─────────────────────────────────────────────────
    if args.smooth_sigma > 0 and skeleton_frames:
        print("\n── Smoothing trajectories ───────────────────────────")
        skeleton_frames = smooth_trajectory(skeleton_frames, sigma=args.smooth_sigma)

    if args.fill_gaps > 0 and skeleton_frames:
        print(f"── Filling gaps (max {args.fill_gaps} frames) ────────────")
        skeleton_frames = fill_trajectory_gaps(skeleton_frames, max_gap=args.fill_gaps)

    # ── Arena alignment ───────────────────────────────────────────────────────
    align_result = None
    if args.align_points:
        from rpimocap.reconstruction.align import (
            align_skeleton_frames,
            align_voxel_frames,
            kabsch_align,
            kabsch_align_from_pixels,
            load_align_csv,
        )
        print(f"\n── Arena alignment ({args.align_points}) ──────────────────────")
        try:
            align_pts = load_align_csv(args.align_points)
            _has_px   = any(pt.px0 is not None for pt in align_pts)
            if _has_px:
                align_result = kabsch_align_from_pixels(
                    align_pts, cal.get('P0', K0 @ np.hstack([np.eye(3), np.zeros((3,1))])), cal.get('P1', K1 @ np.hstack([R, T.reshape(3,1)])))
                print(f"  {align_result.n_points} pts  "
                      f"RMSE={align_result.rmse_mm:.2f}mm  "
                      "(re-triangulated from pixel clicks)")
            else:
                align_result = kabsch_align(align_pts)
                print(f"  {align_result.n_points} pts  "
                      f"RMSE={align_result.rmse_mm:.2f}mm  "
                      "(WARNING: no px stored -- re-annotate)")
            skeleton_frames = align_skeleton_frames(skeleton_frames, align_result)
            if args.voxel:
                voxel_pcs = align_voxel_frames(voxel_pcs, align_result)
        except Exception as exc:
            print(f"  WARNING: arena alignment failed — {exc}")
            print("  Exporting in reconstructed (uncorrected) world frame.")

        # ── Export ─────────────────────────────────────────────────────────────
    print("\n── Exporting ────────────────────────────────────────")

    # Statistics
    stats = trajectory_stats(skeleton_frames)
    write_stats_csv(out_dir / "detection_stats.csv", stats)
    print("\n  Detection rates:")
    for name, s in stats.items():
        bar = "█" * int(s["detection_rate"] * 20)
        print(f"    {name:25s}  {bar:<20s}  "
              f"{s['detection_rate']:.0%}  "
              f"repr_err={s['mean_repr_err']:.2f}px")

    # HDF5
    write_hdf5(
        out_dir / "reconstruction.h5",
        skeleton_frames,
        voxel_frames=voxel_pcs if args.voxel else None,
        fps=fps,
        metadata={
            "cam0": args.cam0,
            "cam1": args.cam1,
            "calib": args.calib,
            "detector": args.detector,
            "voxel_size": args.voxel_size,
        },
    )

    # Viewer JSON
    write_viewer_json(
        out_dir / "viewer_data.json",
        skeleton_frames,
        keypoint_names=kp_names,
        skeleton_edges=skel_edges,
        fps=fps,
        voxel_frames=voxel_pcs if args.voxel else None,
        voxel_downsample=args.voxel_downsample,
    )

    print(f"\n✓ Done.  Output in: {out_dir}/")
    print("  Open the viewer:  viewer/index.html")
    print(f"  Copy {out_dir}/viewer_data.json → viewer/data/viewer_data.json")


# --------------------------------------------------------------------------- #
#  CLI                                                                         #
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Inputs
    io = ap.add_argument_group("Input / Output")
    io.add_argument("--bayer-pattern", default="RGGB",
                    choices=["RGGB","BGGR","GRBG","GBRG"],
                    help="Bayer CFA pattern for raw single-channel TIFF stacks "
                         "(IMX477 default: RGGB). Ignored for RGB TIFFs and video files.")
    io.add_argument("--cam0", required=True, help="Camera 0 video file")
    io.add_argument("--cam1", required=True, help="Camera 1 video file")
    io.add_argument("--calib", required=True, help="calibration.npz from calibrate.py")
    io.add_argument("--out", default="output", help="Output directory")
    io.add_argument("--start-frame", type=int, default=0)
    io.add_argument("--n-frames", type=int, default=0,
                    help="Process N frames (0 = all)")
    io.add_argument("--fps", type=float, default=30.0,
                    help="Fallback FPS if video metadata missing")
    io.add_argument("--rectify", action="store_true",
                    help="Apply stereo rectification to frames before processing")

    # Detector
    det = ap.add_argument_group("Pose detector")
    det.add_argument("--detector", default="centroid",
                     choices=["mediapipe", "centroid", "dlc", "sleap"],
                     help="2D keypoint detector backend")
    det.add_argument("--model-complexity", type=int, default=2,
                     help="MediaPipe model complexity (0-2)")
    det.add_argument("--min-detection-conf", type=float, default=0.4,
                     help="Minimum keypoint confidence for triangulation")
    det.add_argument("--min-tracking-conf", type=float, default=0.4)
    det.add_argument("--dlc0", default=None, help="DLC CSV for camera 0")
    det.add_argument("--dlc1", default=None, help="DLC CSV for camera 1")
    det.add_argument("--sleap0", default=None, help="SLEAP CSV for camera 0")
    det.add_argument("--sleap1", default=None, help="SLEAP CSV for camera 1")

    # Voxel
    vox = ap.add_argument_group("Voxel carving")
    vox.add_argument("--bounds", required=True,
                     help="World bounds: xmin,xmax,ymin,ymax,zmin,zmax (mm)")
    vox.add_argument("--voxel-size", type=float, default=8.0,
                     help="Voxel edge length in mm (default: 8)")
    vox.add_argument("--no-voxel", dest="voxel", action="store_false",
                     help="Disable voxel carving (skeleton only)")
    vox.add_argument("--chunk-size", type=int, default=200_000,
                     help="Voxels per projection batch (tune for RAM)")
    vox.add_argument("--bg-history", type=int, default=300)
    vox.add_argument("--var-threshold", type=float, default=40.0)
    vox.add_argument("--morph-ksize", type=int, default=7)
    vox.add_argument("--min-area", type=int, default=500,
                     help="Min foreground blob area in pixels")
    vox.set_defaults(voxel=True)

    # Smoothing
    sm = ap.add_argument_group("Temporal smoothing")
    sm.add_argument("--smooth-sigma", type=float, default=1.5,
                    help="Gaussian temporal smoothing σ in frames (0 = off)")
    sm.add_argument("--fill-gaps", type=int, default=10,
                    help="Max gap length to interpolate (0 = off)")
    sm.add_argument("--max-repr-error", type=float, default=20.0,
                    help="Discard triangulations above this reprojection error (px)")

    # Arena alignment
    al = ap.add_argument_group("Arena alignment")
    al.add_argument("--align-points", default=None, metavar="CSV",
                    help="CSV of rec<->arena correspondences from rpimocap-align. "
                         "Applies a Kabsch rigid transform to all exported "
                         "coordinates so they are expressed in arena space.")
    al.add_argument("--refraction-config", default=None, metavar="JSON",
                    help="JSON file describing the arena walls (planes, "
                         "thickness, refractive index). Used only when "
                         "--enable-refraction is also passed; otherwise "
                         "ignored. Refraction is OFF by default because "
                         "the standard rpimocap rig has no plexiglass "
                         "between the cameras and the animal.")
    al.add_argument("--enable-refraction", action="store_true",
                    help="Opt in to refractive triangulation. Requires "
                         "--refraction-config to also be supplied. Off by "
                         "default; only relevant for arenas with acrylic "
                         "walls between the cameras and the subject.")

    # Export
    ex = ap.add_argument_group("Export")
    ex.add_argument("--save-ply-skeleton", action="store_true",
                    help="Save per-frame skeleton PLY files")
    ex.add_argument("--save-ply-volume", action="store_true",
                    help="Save per-frame volume PLY files")
    ex.add_argument("--save-mesh", action="store_true",
                    help="Save Marching Cubes mesh instead of point cloud")
    ex.add_argument("--ply-stride", type=int, default=1,
                    help="Save every Nth frame as PLY (1 = all)")
    ex.add_argument("--voxel-downsample", type=int, default=8,
                    help="Voxel downsample factor for viewer JSON (reduce file size)")

    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
