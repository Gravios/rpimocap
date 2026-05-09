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
        --out           segment-output/preview.mp4   (H264 via ffmpeg if available)

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
) -> np.ndarray:
    """Draw skeleton dots and connecting lines onto canvas (in-place overlay).

    Uses an alpha-blended overlay so the original image shows through.
    """
    h, w = canvas.shape[:2]
    overlay = canvas.copy()

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
            continue
        col = _colour(name)
        cv2.circle(overlay, (u, v), dot_r,     col, -1, cv2.LINE_AA)
        cv2.circle(overlay, (u, v), dot_r + 1, (0, 0, 0), 1, cv2.LINE_AA)

    return cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0)


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
                    help="Output .mp4 path")
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
    P0   = cal.get("P0", K0 @ np.hstack([np.eye(3),  np.zeros((3, 1))]))
    P1   = cal.get("P1", K1 @ np.hstack([R, T.reshape(3, 1)]))

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
    dot_r  = max(1, int(args.dot_radius * args.scale))
    line_w = max(1, int(args.line_width * args.scale))

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

    # ── Distortion maps (precomputed for speed) ────────────────────────────
    map0x, map0y = cv2.initUndistortRectifyMap(
        K0, d0.reshape(1,-1), None, K0, (vid_w, vid_h), cv2.CV_32FC1)
    map1x, map1y = cv2.initUndistortRectifyMap(
        K1, d1.reshape(1,-1), None, K1, (vid_w, vid_h), cv2.CV_32FC1)

    print(f"Rendering {n_out} frames → {out_path}  "
          f"({canvas_w}×{out_h} @ {fps:.1f} fps) ...")

    cap0.set(cv2.CAP_PROP_POS_FRAMES, start)
    cap1.set(cv2.CAP_PROP_POS_FRAMES, start)

    for fi, frame_idx in enumerate(range(start, end)):
        ret0, f0 = cap0.read()
        ret1, f1 = cap1.read()
        if not ret0 or not ret1:
            break

        # Undistort frames
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

        # Draw overlays
        f0 = _draw_skeleton(f0, kp0, dot_r, line_w, args.alpha)
        f1 = _draw_skeleton(f1, kp1, dot_r, line_w, args.alpha)

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
    print(f"Done → {out_path}")


if __name__ == "__main__":
    main()
