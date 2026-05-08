"""
rpimocap-segment — texture-based animal segmentation and 3D tracking
=====================================================================
Detects and tracks body regions (nose, ears, back, tail etc.) through
a stereo TIFF recording without manual annotation per frame.

Pipeline
--------
1. Background model  — median over N evenly-spaced frames
2. Foreground mask   — per-frame background subtraction + morphology
3. Body part labels  — geometric spine-axis analysis (no ML required)
                       or SAM/SAM2 if weights are supplied
4. Stereo matching   — epipolar-constrained region matching
5. Triangulation     — DLT stereo triangulation → 3D coordinates (mm)
6. Temporal output   — HDF5 + viewer JSON in the same format as
                       rpimocap-run, directly compatible with downstream tools

Output
------
    output/
    ├── reconstruction.h5      ← 3D trajectories per body part
    ├── viewer_data.json       ← HTML 3D viewer
    ├── detection_stats.csv    ← per-frame detection rates
    └── background/
        ├── bg_cam0.png        ← background image (for inspection)
        └── bg_cam1.png

Usage
-----
    # Basic (no SAM, optical flow tracking)
    rpimocap-segment \\
        --cam0  cam0_raw.tif \\
        --cam1  cam1_raw.tif \\
        --calib autocalib_refined.npz \\
        --out   output/

    # With arena alignment
    rpimocap-segment \\
        --cam0  cam0_raw.tif --cam1 cam1_raw.tif \\
        --calib autocalib_refined.npz \\
        --align-points align_points.csv \\
        --bounds="-140,140,-215,215,0,388" \\
        --out   output/

    # With SAM2 (better part segmentation)
    rpimocap-segment \\
        --cam0  cam0_raw.tif --cam1 cam1_raw.tif \\
        --calib autocalib_refined.npz \\
        --sam2-checkpoint /path/to/sam2_hiera_large.pt \\
        --out   output/

    # Reuse a saved background model
    rpimocap-segment ... --background-model output/background/bg.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


def _parse_bounds(s: str) -> list[float]:
    return [float(v) for v in s.split(",")]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    # ── Input ────────────────────────────────────────────────────────────────
    io = ap.add_argument_group("Input / Output")
    io.add_argument("--cam0",   required=True,
                    help="Camera 0 video or TIFF stack")
    io.add_argument("--cam1",   required=True,
                    help="Camera 1 video or TIFF stack")
    io.add_argument("--calib",  required=True,
                    help="Calibration .npz (from rpimocap-calibrate or "
                         "rpimocap-autocalib)")
    io.add_argument("--out",    required=True,
                    help="Output directory")
    io.add_argument("--bayer-pattern", default="RGGB",
                    choices=["RGGB","BGGR","GRBG","GBRG"],
                    help="Bayer CFA pattern for raw TIFF stacks (default: RGGB)")

    # ── Background ───────────────────────────────────────────────────────────
    bg = ap.add_argument_group("Background model")
    bg.add_argument("--background-frames", type=int, default=200,
                    metavar="N",
                    help="Number of frames for background estimation "
                         "(default: 200)")
    bg.add_argument("--background-start", type=int, default=0,
                    help="First frame index to use for background "
                         "(default: 0 — use start of video)")
    bg.add_argument("--background-model", default=None, metavar="NPZ",
                    help="Load a pre-computed background model .npz instead "
                         "of computing from scratch")
    bg.add_argument("--background-method", default="median",
                    choices=["median","mean"],
                    help="Background estimation method (default: median)")
    bg.add_argument("--background-extra-cam0", nargs="+", default=[],
                    metavar="TIFF",
                    help="Additional cam0 TIFF files for background estimation. "
                         "Combining multiple sessions makes the median far more "
                         "robust when the animal is present throughout each recording.")
    bg.add_argument("--background-extra-cam1", nargs="+", default=[],
                    metavar="TIFF",
                    help="Additional cam1 TIFF files (must match --background-extra-cam0)")

    # ── Detection ────────────────────────────────────────────────────────────
    det = ap.add_argument_group("Detection")
    det.add_argument("--threshold", type=float, default=25.0,
                     help="Foreground detection threshold 0-255 "
                          "(default: 25)")
    det.add_argument("--min-area", type=int, default=500,
                     help="Minimum blob area in pixels (default: 500)")
    det.add_argument("--morph-k", type=int, default=7,
                     help="Morphological kernel size (default: 7)")
    det.add_argument("--max-epipolar-px", type=float, default=8.0,
                     help="Maximum epipolar line distance for stereo "
                          "matching (default: 8 px)")

    # ── Contrast enhancement ─────────────────────────────────────────────────
    con = ap.add_argument_group("Contrast enhancement (NIR footage)")
    con.add_argument("--clahe", action="store_true",
                     help="Apply CLAHE (adaptive histogram equalisation) before "
                          "background subtraction.  Strongly recommended for NIR "
                          "footage where animal fur blends with bedding.")
    con.add_argument("--clahe-clip", type=float, default=2.0,
                     help="CLAHE clip limit — higher = more contrast enhancement "
                          "but also more noise amplification (default: 2.0)")
    con.add_argument("--clahe-tile", type=int, default=8,
                     help="CLAHE tile grid size — smaller tiles = more local "
                          "enhancement (default: 8)")
    con.add_argument("--green-channel", action="store_true",
                     help="Use the green Bayer channel instead of luminance. "
                          "Green carries ~2x NIR signal on IMX477.  "
                          "Recommended with --bayer-pattern RGGB.")
    con.add_argument("--bilateral", action="store_true",
                     help="Apply bilateral filter instead of Gaussian blur. "
                          "Preserves fur/bedding edges while reducing sensor noise.")
    con.add_argument("--bilateral-d", type=int, default=9,
                     help="Bilateral filter neighbourhood diameter (default: 9)")
    con.add_argument("--bilateral-sigma", type=float, default=50.0,
                     help="Bilateral filter sigma for colour and spatial "
                          "domains (default: 50.0)")

    # ── SAM ──────────────────────────────────────────────────────────────────
    sam = ap.add_argument_group("SAM (optional)")
    sam.add_argument("--sam2-checkpoint", default=None, metavar="PATH",
                     help="Path to SAM2 model weights (.pt). If supplied, "
                          "SAM2 is used for body part segmentation.")
    sam.add_argument("--sam2-config", default="sam2_hiera_large.yaml",
                     help="SAM2 config yaml (default: sam2_hiera_large.yaml)")
    sam.add_argument("--device", default="cuda",
                     help="PyTorch device for SAM2 (default: cuda)")

    # ── Sequence ─────────────────────────────────────────────────────────────
    seq = ap.add_argument_group("Sequence")
    seq.add_argument("--start-frame", type=int, default=0,
                     help="First frame to process (default: 0)")
    seq.add_argument("--end-frame", type=int, default=None,
                     help="Last frame to process (default: end of video)")
    seq.add_argument("--sample-every", type=int, default=1,
                     help="Process every Nth frame (default: 1 = all frames)")
    seq.add_argument("--redetect-every", type=int, default=60,
                     help="Force re-detection every N frames for optical "
                          "flow tracker (default: 60)")

    # ── Arena alignment ───────────────────────────────────────────────────────
    al = ap.add_argument_group("Arena alignment")
    al.add_argument("--align-points", default=None, metavar="CSV",
                    help="Alignment CSV from rpimocap-align "
                         "(expresses output in arena mm frame)")
    al.add_argument("--bounds", default=None,
                    help="Reconstruction bounds xmin,xmax,ymin,ymax,zmin,zmax "
                         "in mm (used for viewer only)")

    # ── Smoothing ─────────────────────────────────────────────────────────────
    sm = ap.add_argument_group("Smoothing")
    sm.add_argument("--smooth-sigma", type=float, default=1.5,
                    help="Gaussian smoothing sigma in frames (default: 1.5, "
                         "0 = off)")
    sm.add_argument("--fill-gaps", type=int, default=5,
                    help="Interpolate gaps up to N frames (default: 5, 0=off)")

    # ── Diagnostics ──────────────────────────────────────────────────────────
    dg = ap.add_argument_group("Diagnostics")
    dg.add_argument("--diagnostics", default="/tmp/rpimocap_diag",
                    metavar="DIR",
                    help="Write diagnostic images (background, enhanced frames, "
                         "diff maps, mask overlays) to this directory. "
                         "Default: /tmp/rpimocap_diag  (set to '' to disable)")
    dg.add_argument("--diag-frames", type=int, default=6,
                    help="Number of sample frames to save for diagnostics "
                         "(default: 6, evenly spaced through the video)")

    args = ap.parse_args()

    # ── Imports ───────────────────────────────────────────────────────────────
    from rpimocap.cli.pipeline import open_video
    from rpimocap.detection.segment import (
        BackgroundModel, ForegroundDetector,
        GeometricLabeller, EpipolarMatcher,
    )
    from rpimocap.detection.tracker import SegmentTracker
    from rpimocap.reconstruction.triangulate import (
        smooth_trajectory, fill_trajectory_gaps, trajectory_stats)
    from rpimocap.io.export import write_hdf5, write_viewer_json, write_stats_csv

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    bg_dir  = out_dir / "background"
    bg_dir.mkdir(exist_ok=True)

    print("rpimocap-segment")
    print(f"  cam0   : {args.cam0}")
    print(f"  cam1   : {args.cam1}")
    print(f"  calib  : {args.calib}")
    print(f"  out    : {args.out}")

    # ── Open videos ──────────────────────────────────────────────────────────
    cap0 = open_video(args.cam0, bayer_pattern=args.bayer_pattern)
    cap1 = open_video(args.cam1, bayer_pattern=args.bayer_pattern)
    n_frames = int(min(cap0.get(cv2.CAP_PROP_FRAME_COUNT),
                       cap1.get(cv2.CAP_PROP_FRAME_COUNT)))
    vid_w = int(cap0.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap0.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps   = cap0.get(cv2.CAP_PROP_FPS) or 25.0
    print(f"  resolution : {vid_w}x{vid_h}  "
          f"({n_frames} frames @ {fps:.1f} fps)")

    # ── Calibration ──────────────────────────────────────────────────────────
    cal = np.load(args.calib)

    # ── Background model ─────────────────────────────────────────────────────
    print("\n── Background model ─────────────────────────────────────────────")
    bg_npz = bg_dir / "bg.npz"
    if args.background_model and Path(args.background_model).exists():
        print(f"  Loading: {args.background_model}")
        bg = BackgroundModel.from_npz(args.background_model)
    elif bg_npz.exists() and not (args.background_extra_cam0):
        print(f"  Reusing: {bg_npz}")
        bg = BackgroundModel.from_npz(bg_npz)
    else:
        extra0 = args.background_extra_cam0
        extra1 = args.background_extra_cam1
        if extra0 and not extra1:
            # If only cam0 extras given, use the same files for cam1
            extra1 = extra0
        if len(extra1) != len(extra0):
            print("ERROR: --background-extra-cam0 and --background-extra-cam1 "
                  "must have the same number of files")
            sys.exit(1)
        if extra0:
            from rpimocap.io.export import TiffCapture as _TC
            all_caps0 = [cap0] + [_TC(f, bayer_pattern=args.bayer_pattern)
                                   for f in extra0]
            all_caps1 = [cap1] + [_TC(f, bayer_pattern=args.bayer_pattern)
                                   for f in extra1]
            print(f"  Building from {len(all_caps0)} sessions "
                  f"x {args.background_frames} frames each ...")
            bg = BackgroundModel.from_multiple_captures(
                all_caps0, all_caps1,
                n_frames_each=args.background_frames,
                method=args.background_method,
                start_frame=args.background_start,
                verbose=True)
            for c in all_caps0[1:] + all_caps1[1:]:
                c.release()
        else:
            bg = BackgroundModel.from_captures(
                cap0, cap1,
                n_frames=args.background_frames,
                method=args.background_method,
                start_frame=args.background_start,
                verbose=True)
        bg.save(bg_npz)
        print(f"  Saved: {bg_npz}")

    # Save background images for inspection
    for img, name in [(bg.bg0, "bg_cam0.png"), (bg.bg1, "bg_cam1.png")]:
        cv2.imwrite(str(bg_dir / name),
                    np.clip(img, 0, 255).astype(np.uint8))

    # ── Epipolar matcher ──────────────────────────────────────────────────────
    matcher = EpipolarMatcher.from_calibration(
        cal, max_epipolar_px=args.max_epipolar_px)

    # ── Arena alignment ───────────────────────────────────────────────────────
    align_result = None
    if args.align_points:
        from rpimocap.reconstruction.align import (
                load_align_csv, kabsch_align,
                kabsch_align_from_pixels)
        print(f"\n── Arena alignment ({args.align_points}) ─────────────────────")
        try:
            align_pts = load_align_csv(args.align_points)
            _has_px   = any(pt.px0 is not None for pt in align_pts)
            if _has_px:
                align_result = kabsch_align_from_pixels(
                    align_pts, matcher.P0, matcher.P1)
                print(f"  {align_result.n_points} pts  "
                      f"RMSE={align_result.rmse_mm:.2f}mm  "
                      "(re-triangulated from pixel clicks)")
            else:
                align_result = kabsch_align(align_pts)
                print(f"  {align_result.n_points} pts  "
                      f"RMSE={align_result.rmse_mm:.2f}mm  "
                      "(WARNING: no px stored -- re-annotate)")
        except Exception as e:
            print(f"  WARNING: alignment failed — {e}")

    # ── Tracker ──────────────────────────────────────────────────────────────
    print("\n── Tracking ────────────────────────────────────────────────────")
    tracker = SegmentTracker(
        background=bg,
        matcher=matcher,
        sam2_checkpoint=args.sam2_checkpoint,
        sam2_config=args.sam2_config,
        device=args.device,
        threshold=args.threshold,
        min_area_px=args.min_area,
        morph_k=args.morph_k,
        redetect_every=args.redetect_every,
        clahe=args.clahe,
        clahe_clip=args.clahe_clip,
        clahe_tile=args.clahe_tile,
        use_green_channel=args.green_channel,
        bilateral=args.bilateral,
        bilateral_d=args.bilateral_d,
        bilateral_sigma=args.bilateral_sigma,
        verbose=True)

    # ── Diagnostics ──────────────────────────────────────────────────────────
    if args.diagnostics:
        from rpimocap.detection.segment import (
            save_diagnostics, GeometricLabeller)
        diag_header = f"\n── Diagnostics -> {args.diagnostics} ──────────────────────"
        print(diag_header)
        # Reset captures to start for diagnostic sampling
        for cap in (cap0, cap1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        save_diagnostics(
            cap0, cap1,
            detector=tracker._det,
            labeller=GeometricLabeller(),
            out_dir=args.diagnostics,
            n_frames=args.diag_frames,
        )
        # Reset again for tracking
        for cap in (cap0, cap1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    results = tracker.track_sequence(
        cap0, cap1,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        sample_every=args.sample_every,
        align_result=align_result)

    cap0.release()
    cap1.release()

    # ── Convert to skeleton_frames ────────────────────────────────────────────
    skeleton_frames = SegmentTracker.results_to_skeleton_frames(results)

    # ── Smoothing / gap fill ─────────────────────────────────────────────────
    print("\n── Post-processing ─────────────────────────────────────────────")
    if args.smooth_sigma > 0 and skeleton_frames:
        skeleton_frames = smooth_trajectory(
            skeleton_frames, sigma=args.smooth_sigma)
        print(f"  Smoothed (sigma={args.smooth_sigma})")
    if args.fill_gaps > 0 and skeleton_frames:
        skeleton_frames = fill_trajectory_gaps(
            skeleton_frames, max_gap=args.fill_gaps)
        print(f"  Gap-filled (max_gap={args.fill_gaps})")

    # ── Stats ────────────────────────────────────────────────────────────────
    stats = trajectory_stats(skeleton_frames)

    # ── Bounds ───────────────────────────────────────────────────────────────
    bounds = None
    if args.bounds:
        bounds = _parse_bounds(args.bounds)

    # ── Export ───────────────────────────────────────────────────────────────
    print("\n── Exporting ───────────────────────────────────────────────────")

    write_stats_csv(out_dir / "detection_stats.csv", stats)
    print(f"  detection_stats.csv")

    write_hdf5(
        out_dir / "reconstruction.h5",
        skeleton_frames,
        voxel_frames=None,
        fps=fps,
        metadata={
            "cam0":             args.cam0,
            "cam1":             args.cam1,
            "calib":            args.calib,
            "detector":         "segment",
            "bayer_pattern":    args.bayer_pattern,
            "align_points":     args.align_points,
            "align_rmse_mm":    (align_result.rmse_mm
                                 if align_result else None),
            "sam2_checkpoint":  args.sam2_checkpoint,
        })
    print(f"  reconstruction.h5")

    # Derive keypoint names and skeleton edges from the tracking results
    kp_names = sorted({pt.name for frame in skeleton_frames for pt in frame})

    # Skeleton connectivity along the spine + ears
    _SPINE = ["nose","head","neck","back","rump","tail_base","tail_tip"]
    _EAR   = [("head","left_ear"), ("head","right_ear")]
    edges  = ([(a, b) for a, b in zip(_SPINE, _SPINE[1:])
                if a in kp_names and b in kp_names]
              + [(a, b) for a, b in _EAR
                 if a in kp_names and b in kp_names])

    write_viewer_json(
        out_dir / "viewer_data.json",
        skeleton_frames,
        keypoint_names=kp_names,
        skeleton_edges=edges,
        bounds=bounds,
        fps=fps,
        voxel_frames=None)
    print(f"  viewer_data.json")

    # ── Summary ───────────────────────────────────────────────────────────────
    n_det  = sum(1 for r in results if r.detected)
    n_tot  = len(results)
    parts  = set(k for r in results for k in r.xyz)
    print(f"\n── Done ────────────────────────────────────────────────────────")
    print(f"  Processed : {n_tot} frames")
    print(f"  Detected  : {n_det} ({100*n_det/max(n_tot,1):.1f}%)")
    print(f"  Body parts: {sorted(parts)}")
    print(f"  Output    : {out_dir}/")


if __name__ == "__main__":
    main()
