"""
rpimocap-preview — render tracked skeleton overlaid on the original video
=========================================================================
Reads reconstruction.h5, reprojects each 3D keypoint back into both
camera views, and writes side-by-side MP4 with the skeleton drawn on top.

Usage
-----
    rpimocap-preview \\
        --cam0          cam0_raw.tif \\
        --cam1          cam1_raw.tif \\
        --calib         autocalib_refined.npz \\
        --h5            segment-output/reconstruction.h5 \\
        --bayer-pattern RGGB \\
        --out           <session>/tracking/preview.mp4   (H264 via ffmpeg if available)

Options
-------
--side-by-side    (default) cam0 left, cam1 right in one video
--cam0-only       write only cam0
--cam1-only       write only cam1
--scale 0.5       resize output (default: 0.5 of original resolution)
--fps             override playback fps (default: from h5 metadata)
--start-frame     first frame to render
--end-frame       last frame to render
--dot-radius      keypoint dot radius in pixels (default: 6)
--line-width      skeleton edge line width (default: 2)
--alpha           overlay opacity 0-1 (default: 0.85)
--codec           fourcc codec string (default: mp4v)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


# ── Colour palette per body part ────────────────────────────────────────────
_PART_COLOURS: dict[str, tuple[int, int, int]] = {
    "nose":      (255, 100,  50),   # orange
    "head":      (255, 180,  50),   # amber
    "left_ear":  ( 80, 200, 255),   # cyan
    "right_ear": ( 80, 200, 255),
    "neck":      (180, 255,  80),   # lime
    "back":      ( 80, 255, 120),   # green
    "rump":      ( 80, 180, 255),   # sky blue
    "tail_base": (140,  80, 255),   # violet
    "tail_tip":  (220,  80, 255),   # magenta
    "animal":    (255, 255,   0),   # bright yellow — centroid-only mode
}
_DEFAULT_COLOUR = (200, 200, 200)

_SPINE = ["nose", "head", "neck", "back", "rump", "tail_base", "tail_tip"]
_SPINE_EDGES = list(zip(_SPINE, _SPINE[1:]))
_EAR_EDGES   = [("head", "left_ear"), ("head", "right_ear")]
_ALL_EDGES   = _SPINE_EDGES + _EAR_EDGES


def _project(P: np.ndarray, xyz: np.ndarray) -> tuple[int, int]:
    """Project a 3-D world point through projection matrix P → (u, v) pixels."""
    Xh   = np.append(xyz, 1.0)
    proj = P @ Xh
    u    = proj[0] / proj[2]
    v    = proj[1] / proj[2]
    return int(round(u)), int(round(v))


def _in_frame(u: int, v: int, w: int, h: int) -> bool:
    return 0 <= u < w and 0 <= v < h


def _colour(name: str) -> tuple[int, int, int]:
    return _PART_COLOURS.get(name, _DEFAULT_COLOUR)


def _draw_skeleton(
    canvas:     np.ndarray,
    kp_px:      dict[str, tuple[int, int]],
    dot_r:      int   = 12,
    line_w:     int   = 2,
    alpha:      float = 0.85,
) -> tuple[np.ndarray, int, int]:
    """Draw skeleton dots and connecting lines onto canvas.

    Returns
    -------
    (drawn_canvas, n_drawn, n_dropped)
      n_drawn   : keypoints that landed inside the canvas and were drawn
      n_dropped : keypoints whose projected (u, v) was outside the canvas
                  bounds (silent before; counted now so callers can warn
                  the user when projection is mis-configured).
    """
    h, w = canvas.shape[:2]
    overlay = canvas.copy()
    n_drawn   = 0
    n_dropped = 0

    # Edges first (drawn under dots)
    for a, b in _ALL_EDGES:
        if a not in kp_px or b not in kp_px:
            continue
        ua, va = kp_px[a]
        ub, vb = kp_px[b]
        if not (_in_frame(ua, va, w, h) or _in_frame(ub, vb, w, h)):
            continue
        col = tuple(int((c1 + c2) / 2)
                    for c1, c2 in zip(_colour(a), _colour(b)))
        cv2.line(overlay, (ua, va), (ub, vb), col, line_w,
                 cv2.LINE_AA)

    # Dots
    for name, (u, v) in kp_px.items():
        if not _in_frame(u, v, w, h):
            n_dropped += 1
            continue
        col = _colour(name)
        cv2.circle(overlay, (u, v), dot_r,     col, -1, cv2.LINE_AA)
        cv2.circle(overlay, (u, v), dot_r + 1, (0, 0, 0), 1, cv2.LINE_AA)
        n_drawn += 1

    return cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0), \
           n_drawn, n_dropped


def _load_h5(path: str) -> tuple[dict[str, np.ndarray], float, int]:
    """Load reconstruction.h5 → (xyz_dict, fps, n_frames).

    Returns
    -------
    xyz_dict  : {keypoint_name: (n_frames, 3) float32, NaN = missing}
    fps       : recording frame rate
    n_frames  : total frame count in h5
    """
    import h5py
    with h5py.File(path, "r") as f:
        # fps and n_frames are stored at root level by write_hdf5
        fps      = float(f.attrs.get("fps", 25.0))
        skel     = f["skeleton"]
        xyz_dict = {}
        for name in skel.keys():
            arr = skel[name]["xyz"][:]        # (n_frames, 3)
            xyz_dict[name] = arr
        # derive n_frames from the data — attrs location varies
        n_frames = (int(f.attrs["n_frames"])
                    if "n_frames" in f.attrs
                    else int(skel.attrs.get("n_frames", 0))
                    if "n_frames" in skel.attrs
                    else max((v.shape[0] for v in xyz_dict.values()), default=0))
    return xyz_dict, fps, n_frames


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    io = ap.add_argument_group("Input / Output")
    io.add_argument("--cam0",   required=True)
    io.add_argument("--cam1",   required=True)
    io.add_argument("--calib",  required=True)
    io.add_argument("--h5",     required=True,
                    help="reconstruction.h5 from rpimocap-segment or rpimocap-run")
    io.add_argument("--out",    required=True,
                    metavar="MP4",
                    help="Output video path (e.g. <session>/tracking/preview.mp4)")
    io.add_argument("--bayer-pattern", default="RGGB",
                    choices=["RGGB","BGGR","GRBG","GBRG"])
    io.add_argument("--align-points", default=None, metavar="CSV",
                    help="Alignment CSV used when generating the h5. Required "
                         "when coordinates are in arena space -- the inverse "
                         "transform is applied before reprojection into pixels.")

    view = ap.add_argument_group("Layout")
    grp = view.add_mutually_exclusive_group()
    grp.add_argument("--side-by-side", dest="layout",
                     action="store_const", const="sbs", default="sbs",
                     help="cam0 left + cam1 right (default)")
    grp.add_argument("--cam0-only", dest="layout",
                     action="store_const", const="cam0")
    grp.add_argument("--cam1-only", dest="layout",
                     action="store_const", const="cam1")

    vis = ap.add_argument_group("Visualisation")
    vis.add_argument("--scale",      type=float, default=0.5,
                     help="Output scale factor (default: 0.5)")
    vis.add_argument("--fps",        type=float, default=None,
                     help="Playback fps (default: from h5)")
    vis.add_argument("--dot-radius", type=int,   default=6)
    vis.add_argument("--line-width", type=int,   default=2)
    vis.add_argument("--alpha",      type=float, default=0.85,
                     help="Overlay opacity 0-1 (default: 0.85)")
    vis.add_argument("--codec",      default="XVID",
                     help="FourCC codec (default: mp4v)")
    vis.add_argument("--undistort",  action="store_true", default=False,
                     help="Apply K+dist undistortion to frames before "
                          "drawing. OFF by default because "
                          "rpimocap-calibrate-from-corners fits the DLT "
                          "projection matrices on the raw (distorted) "
                          "corner-annotation pixels — projecting those Ps "
                          "onto undistorted frames puts dots in the wrong "
                          "place, often far off-screen for fisheye lenses. "
                          "Enable only with a K-based projection chain.")

    seq = ap.add_argument_group("Sequence")
    vis.add_argument("--no-ffmpeg", action="store_true",
                     help="Force MJPG AVI output even if ffmpeg is available")
    seq.add_argument("--start-frame", type=int, default=0)
    seq.add_argument("--end-frame",   type=int, default=None)

    args = ap.parse_args()

    # ── Load calibration ───────────────────────────────────────────────────
    cal  = np.load(args.calib)
    K0   = cal["K0"]
    K1   = cal["K1"]
    d0   = np.ravel(cal.get("dist0", np.zeros(5)))
    d1   = np.ravel(cal.get("dist1", np.zeros(5)))
    R    = cal["R"]
    T    = cal["T"].ravel()
    # P matrix priority matches EpipolarMatcher.from_calibration:
    #   1. dlt_P0/dlt_P1  → DLT fit on annotated arena corners
    #      (this is what produced the H5 xyz, so it's the only set
    #      that will project them back to the same pixels)
    #   2. P0/P1          → some other source (e.g. an older calib
    #      pre-DLT). Will produce a coordinate-frame mismatch with
    #      DLT-based triangulation; warn the user.
    #   3. K + R + T      → constructed from scratch; almost certainly
    #      mis-aligned with the DLT H5; warn loudly.
    if "dlt_P0" in cal.files and "dlt_P1" in cal.files:
        P0 = cal["dlt_P0"]
        P1 = cal["dlt_P1"]
        _p_source = "dlt_P0/dlt_P1 (matches segmentation triangulation)"
    elif "P0" in cal.files and "P1" in cal.files:
        P0 = cal["P0"]
        P1 = cal["P1"]
        _p_source = ("P0/P1 (autocalib-style — WARN if H5 came from a "
                     "DLT-based run, dots will be mis-aligned)")
    else:
        P0 = K0 @ np.hstack([np.eye(3),  np.zeros((3, 1))])
        P1 = K1 @ np.hstack([R, T.reshape(3, 1)])
        _p_source = ("K + R + T composition (no P matrices in calib npz "
                     "— ALMOST CERTAINLY MIS-ALIGNED with DLT-based H5)")
    print(f"  P matrices source: {_p_source}")

    # ── Inverse arena alignment (arena mm → calibration world frame) ──────
    # Coordinates stored in h5 are in arena space if --align-points was used
    # during rpimocap-segment/run.  We must invert the Kabsch transform to get
    # back to calibration world frame (camera 0 optical centre as origin)
    # before projecting through P0/P1.
    inv_R = np.eye(3)
    inv_t = np.zeros(3)
    if args.align_points:
        from rpimocap.reconstruction.align import load_align_csv, kabsch_align
        try:
            align_pts    = load_align_csv(args.align_points)
            align_result = kabsch_align(align_pts)
            # Inverse of R @ x + t  is  R.T @ (x - t)
            inv_R = align_result.R.T
            inv_t = -align_result.R.T @ align_result.t
            print(f"  Arena alignment loaded: {args.align_points}")
            print(f"  RMSE = {align_result.rmse_mm:.2f} mm  "
                  f"(inverse applied before reprojection)")
        except Exception as e:
            print(f"  WARNING: could not load alignment ({e}) — "
                  f"reprojection may be offset")

    # ── Load h5 ────────────────────────────────────────────────────────────
    print(f"Loading {args.h5} ...")
    xyz_dict, h5_fps, n_h5_frames = _load_h5(args.h5)
    fps  = args.fps or h5_fps
    parts = sorted(xyz_dict.keys())
    print(f"  {n_h5_frames} frames  {len(parts)} keypoints: {parts}")

    # ── Open captures ──────────────────────────────────────────────────────
    from rpimocap.cli.pipeline import open_video
    cap0 = open_video(args.cam0, bayer_pattern=args.bayer_pattern)
    cap1 = open_video(args.cam1, bayer_pattern=args.bayer_pattern)
    n_vid = int(min(cap0.get(cv2.CAP_PROP_FRAME_COUNT),
                    cap1.get(cv2.CAP_PROP_FRAME_COUNT)))
    vid_w = int(cap0.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap0.get(cv2.CAP_PROP_FRAME_HEIGHT))

    start = args.start_frame
    end   = min(args.end_frame or n_h5_frames,
                n_h5_frames, n_vid)
    n_out = end - start

    out_w  = int(vid_w  * args.scale)
    out_h  = int(vid_h  * args.scale)
    # Dot / line widths are specified in OUTPUT pixels. Drawing happens
    # at the native frame resolution, then the whole frame is cv2.resize'd
    # by args.scale. So we must PRE-multiply by (1/scale) here so the
    # final on-screen size matches what the user asked for. The previous
    # 'args.X * args.scale' was backwards: with --dot-radius 6 --scale 0.5,
    # users saw 1.5-px dots in the output instead of 6-px ones — often
    # invisible against textured bedding.
    inv_s  = 1.0 / max(args.scale, 1e-6)
    dot_r  = max(2, int(args.dot_radius * inv_s))
    line_w = max(1, int(args.line_width * inv_s))

    # ── Output writer ──────────────────────────────────────────────────────
    import shutil, subprocess as _sp
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas_w = out_w * 2 if args.layout == "sbs" else out_w

    use_ffmpeg = shutil.which("ffmpeg") is not None and not args.no_ffmpeg
    if use_ffmpeg:
        # Pipe raw BGR frames to ffmpeg → H264 MP4 (no container issues)
        out_path = out_path.with_suffix(".mp4")
        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{canvas_w}x{out_h}",
            "-r", str(fps),
            "-i", "pipe:0",
            "-vcodec", "libx264",
            "-preset", "fast",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(out_path),
        ]
        writer_proc = _sp.Popen(cmd, stdin=_sp.PIPE,
                                 stderr=_sp.DEVNULL)
        writer = None
        print(f"  Using ffmpeg pipe → {out_path}")
    else:
        # Fallback: MJPG AVI via OpenCV
        out_path = out_path.with_suffix(".avi")
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps,
                                  (canvas_w, out_h))
        writer_proc = None
        if not writer.isOpened():
            print(f"ERROR: could not open VideoWriter for {out_path}")
            sys.exit(1)
        print(f"  Using MJPG AVI → {out_path}")

    # ── No undistortion ────────────────────────────────────────────────────
    # The DLT projection matrices (dlt_P0, dlt_P1) are fit in
    # rpimocap-calibrate-from-corners on the RAW corner-annotation pixel
    # coordinates — i.e., on distorted frames. If we undistort the frames
    # here but project through the same P, dots land at distorted pixel
    # positions on an undistorted canvas → visible offset of tens to
    # hundreds of pixels depending on radial distortion magnitude.
    # Fisheye lenses can put projected points entirely off-screen at
    # the periphery, which looks like "the preview isn't drawing dots
    # at all". Fix: draw on the original distorted frame; project
    # through P; everything stays in the same coord system.
    #
    # The cv2.initUndistortRectifyMap maps below remain available
    # behind --undistort for users who want the rectified view AND
    # are using a K-based projection chain (kept off by default).
    if args.undistort:
        map0x, map0y = cv2.initUndistortRectifyMap(
            K0, d0.reshape(1,-1), None, K0, (vid_w, vid_h), cv2.CV_32FC1)
        map1x, map1y = cv2.initUndistortRectifyMap(
            K1, d1.reshape(1,-1), None, K1, (vid_w, vid_h), cv2.CV_32FC1)
        print("  --undistort: WARNING — dots will be offset because DLT "
              "was fit on distorted corners. Use only for K-projection.")
    else:
        map0x = map0y = map1x = map1y = None

    print(f"Rendering {n_out} frames → {out_path}  "
          f"({canvas_w}×{out_h} @ {fps:.1f} fps) ...")

    cap0.set(cv2.CAP_PROP_POS_FRAMES, start)
    cap1.set(cv2.CAP_PROP_POS_FRAMES, start)

    # Accumulators for end-of-run sanity report
    sanity_printed   = False
    total_drawn_0    = 0
    total_drawn_1    = 0
    total_dropped_0  = 0
    total_dropped_1  = 0

    for fi, frame_idx in enumerate(range(start, end)):
        ret0, f0 = cap0.read()
        ret1, f1 = cap1.read()
        if not ret0 or not ret1:
            break

        # Undistort frames only if explicitly opted in (default off:
        # DLT projection expects distorted pixel coords)
        if map0x is not None:
            f0 = cv2.remap(f0, map0x, map0y, cv2.INTER_LINEAR)
            f1 = cv2.remap(f1, map1x, map1y, cv2.INTER_LINEAR)

        # Build per-camera projected keypoints
        kp0: dict[str, tuple[int, int]] = {}
        kp1: dict[str, tuple[int, int]] = {}

        for name, xyz_seq in xyz_dict.items():
            if frame_idx >= len(xyz_seq):
                continue
            xyz = xyz_seq[frame_idx]
            if np.any(np.isnan(xyz)):
                continue
            # Invert arena alignment → calibration world frame
            xyz_cal = inv_R @ xyz + inv_t
            u0, v0 = _project(P0, xyz_cal)
            u1, v1 = _project(P1, xyz_cal)
            # Scale for undistorted → same image coords
            kp0[name] = (u0, v0)
            kp1[name] = (u1, v1)

        # First-frame sanity check: on the first frame where any
        # keypoint has valid xyz, print the projected pixel coords and
        # the canvas size. If projected u/v is wildly outside [0, W]
        # × [0, H], dots will be silently dropped and the preview will
        # look 'empty' — but now the user gets a clear diagnostic line
        # telling them exactly that.
        if not sanity_printed and (kp0 or kp1):
            sanity_printed = True
            print(f"\n── Projection sanity check (frame {frame_idx}) ──")
            print(f"  cam0 frame size : {f0.shape[1]} × {f0.shape[0]}")
            for name in sorted(set(list(kp0) + list(kp1))):
                # Re-fetch the source xyz (and the post-inv-alignment one)
                # so the user can compare H5 values against the projection.
                xyz_src = xyz_dict[name][frame_idx]
                xyz_cal = inv_R @ xyz_src + inv_t
                print(f"  '{name}'  H5 xyz       : "
                      f"({xyz_src[0]:+8.1f}, {xyz_src[1]:+8.1f}, "
                      f"{xyz_src[2]:+8.1f})  mm  (calibration-world frame)")
                if not np.allclose(xyz_cal, xyz_src):
                    print(f"  '{name}'  post-inv-align: "
                          f"({xyz_cal[0]:+8.1f}, {xyz_cal[1]:+8.1f}, "
                          f"{xyz_cal[2]:+8.1f})  mm")
                if name in kp0:
                    u, v = kp0[name]
                    inside = 0 <= u < f0.shape[1] and 0 <= v < f0.shape[0]
                    tag    = "in-frame" if inside else "OFF-FRAME — bad calib?"
                    print(f"  '{name}'  cam0 pixel   : "
                          f"({u}, {v})  [{tag}]")
                if name in kp1:
                    u, v = kp1[name]
                    inside = 0 <= u < f1.shape[1] and 0 <= v < f1.shape[0]
                    tag    = "in-frame" if inside else "OFF-FRAME — bad calib?"
                    print(f"  '{name}'  cam1 pixel   : "
                          f"({u}, {v})  [{tag}]")
            # And the matrices we used (one row each — distinctive enough
            # to tell DLT-fit-on-corners from autocalib-from-K-R-T)
            print(f"  P0 row 0       : "
                  f"[{P0[0,0]:+.2f}, {P0[0,1]:+.2f}, {P0[0,2]:+.2f}, {P0[0,3]:+.2f}]")
            print(f"  P1 row 0       : "
                  f"[{P1[0,0]:+.2f}, {P1[0,1]:+.2f}, {P1[0,2]:+.2f}, {P1[0,3]:+.2f}]")
            print()

        # Draw overlays
        f0, d0, dr0 = _draw_skeleton(f0, kp0, dot_r, line_w, args.alpha)
        f1, d1, dr1 = _draw_skeleton(f1, kp1, dot_r, line_w, args.alpha)
        total_drawn_0   += d0
        total_drawn_1   += d1
        total_dropped_0 += dr0
        total_dropped_1 += dr1

        # Add frame counter
        for frame, label in [(f0, "CAM0"), (f1, "CAM1")]:
            cv2.putText(frame, f"{label}  fr {frame_idx}",
                        (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 1, cv2.LINE_AA)

        # Resize to output scale
        f0s = cv2.resize(f0, (out_w, out_h), interpolation=cv2.INTER_AREA)
        f1s = cv2.resize(f1, (out_w, out_h), interpolation=cv2.INTER_AREA)

        # Compose output canvas
        if args.layout == "sbs":
            canvas = np.concatenate([f0s, f1s], axis=1)
        elif args.layout == "cam0":
            canvas = f0s
        else:
            canvas = f1s

        if use_ffmpeg:
            writer_proc.stdin.write(canvas.tobytes())
        else:
            writer.write(canvas)

        if (fi + 1) % 500 == 0:
            print(f"  {fi + 1}/{n_out}  ({100*(fi+1)/n_out:.0f}%)")

    if use_ffmpeg:
        writer_proc.stdin.close()
        writer_proc.wait()
    else:
        writer.release()
    cap0.release()
    cap1.release()
    # End-of-run draw count. If dropped >> drawn, the user knows the
    # projection is mis-aligned (most dots fell outside the frame).
    print(f"\n── Skeleton draw summary ──")
    print(f"  cam0 : drawn={total_drawn_0}  dropped(out-of-frame)={total_dropped_0}")
    print(f"  cam1 : drawn={total_drawn_1}  dropped(out-of-frame)={total_dropped_1}")
    if total_drawn_0 == 0 and total_drawn_1 == 0 and (
            total_dropped_0 > 0 or total_dropped_1 > 0):
        print("  WARNING: every keypoint projected OUTSIDE the frame. "
              "Possible causes: wrong --calib, --align-points missing, "
              "or H5 coords in a different frame than DLT P matrices.")
    elif total_drawn_0 == 0 and total_drawn_1 == 0:
        print("  WARNING: no keypoints drawn at all. The H5 may contain "
              "only NaN xyz (no detections), or the H5 frame indices "
              "don't overlap with the requested --start-frame/--end-frame.")
    print(f"Done → {out_path}")


if __name__ == "__main__":
    main()
