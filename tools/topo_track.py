#!/usr/bin/env python3
"""Run the topological rat detector over a stereo session → 3D track CSV.

Loads the two camera TIFFs and a DLT calibration (dlt_P0/dlt_P1), builds the
arena floor ROI for each view, and runs the median-bandpass grain-density
detector (rpimocap.detection.topo_detect) frame by frame. Each frame's
per-view centroids are triangulated and gated (in-arena / above-floor, which
rejects the floor reflection that triangulates below z=0), writing a track CSV
with the per-view detections and the 3D point.

Example
-------
  python tools/topo_track.py \\
      --cam0 session/raw/cam0.tif --cam1 session/raw/cam1.tif \\
      --calib calib_from_corners.npz --out track3d.csv --stride 1

  # or via the wrapper:
  tools/topo_track.sh cam0.tif cam1.tif calib_from_corners.npz track3d.csv
"""
import argparse
import sys
import time

import cv2
import numpy as np

# Arena corners in mm (floor z=0 first four, ceiling z=388 last four) —
# matches tools/texture_distance_probe.py _ARENA_CORNERS.
_ARENA_CORNERS = np.array([
    [-140, -215,   0], [140, -215,   0],
    [140,  215,   0], [-140,  215,   0],
    [-140, -215, 388], [140, -215, 388],
    [140,  215, 388], [-140,  215, 388],
], dtype=np.float64)


def _green(frame):
    """Green channel of a demosaiced frame (matches the probe)."""
    if frame is None:
        return None
    return frame[:, :, 1] if frame.ndim == 3 else frame


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cam0", required=True, help="cam0 TIFF")
    ap.add_argument("--cam1", required=True, help="cam1 TIFF")
    ap.add_argument("--calib", required=True,
                    help="calibration .npz with dlt_P0/dlt_P1 (or P0/P1)")
    ap.add_argument("--out", required=True, help="output track CSV path")
    ap.add_argument("--bayer-pattern", default="RGGB")
    ap.add_argument("--roi-mode", default="volume",
                    choices=["box", "floor", "volume"],
                    help="arena ROI mode (default volume; see 0076)")
    ap.add_argument("--roi-max-height-mm", type=float, default=260.0,
                    help="volume ROI height band (default 260, see 0076)")
    ap.add_argument("--patch", type=int, default=112,
                    help="grain-count window px (detection scale)")
    ap.add_argument("--blob-sigma", type=float, default=80.0,
                    help="body-scale -LoG sigma for the centroid")
    ap.add_argument("--barrier-pct", type=float, default=45.0,
                    help="grain barrier percentile for segmentation "
                         "(raise for the lower-contrast view)")
    ap.add_argument("--detect-pct", type=float, default=90.0)
    ap.add_argument("--min-area", type=int, default=1500)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max-frames", type=int, default=0, help="0 = all")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    from rpimocap.detection.topo_detect import (build_floor_mask,
                                                detect_stereo)
    from rpimocap.io.export import TiffCapture

    cal = np.load(args.calib)
    P0 = cal["dlt_P0"] if "dlt_P0" in cal else cal.get("P0", None)
    P1 = cal["dlt_P1"] if "dlt_P1" in cal else cal.get("P1", None)
    if P0 is None or P1 is None:
        sys.exit(f"error: {args.calib} has no dlt_P0/dlt_P1 (or P0/P1)")

    cap0 = TiffCapture(args.cam0, bayer_pattern=args.bayer_pattern)
    cap1 = TiffCapture(args.cam1, bayer_pattern=args.bayer_pattern)
    n = min(int(cap0.get(cv2.CAP_PROP_FRAME_COUNT)),
            int(cap1.get(cv2.CAP_PROP_FRAME_COUNT)))

    cap0.set(cv2.CAP_PROP_POS_FRAMES, args.start)
    ok, f0 = cap0.read()
    if not ok or f0 is None:
        sys.exit("error: cannot read cam0 first frame")
    shape = _green(f0).shape
    fl0 = build_floor_mask(P0, _ARENA_CORNERS, shape, mode=args.roi_mode,
                           max_height_mm=args.roi_max_height_mm)
    fl1 = build_floor_mask(P1, _ARENA_CORNERS, shape, mode=args.roi_mode,
                           max_height_mm=args.roi_max_height_mm)
    print(f"frames={n}  ROI={args.roi_mode}"
          + (f"({args.roi_max_height_mm:.0f}mm)"
             if args.roi_mode == "volume" else "")
          + f"  patch={args.patch}px  blob_sigma={args.blob_sigma:.0f}")

    rng = np.random.default_rng(args.seed)
    kw = dict(patch=args.patch, blob_sigma=args.blob_sigma,
              barrier_pct=args.barrier_pct, detect_pct=args.detect_pct,
              min_area=args.min_area)

    last = (min(args.start + args.max_frames * args.stride, n)
            if args.max_frames > 0 else n)
    idxs = range(args.start, last, args.stride)

    t0 = time.time()
    n_found = n_accept = n_rows = 0
    with open(args.out, "w") as fh:
        fh.write("frame,found,cam0_cx,cam0_cy,cam1_cx,cam1_cy,"
                 "X_mm,Y_mm,Z_mm,accepted,sep0,sep1\n")
        for idx in idxs:
            cap0.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok0, fr0 = cap0.read()
            cap1.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok1, fr1 = cap1.read()
            if not (ok0 and ok1):
                continue
            n_rows += 1
            X, acc, d0, d1 = detect_stereo(_green(fr0), _green(fr1),
                                           fl0, fl1, P0, P1, rng=rng, **kw)
            found = d0.found and d1.found
            n_found += int(found)
            n_accept += int(acc)
            if found and X is not None:
                fh.write(f"{idx},1,{d0.centroid[0]:.1f},{d0.centroid[1]:.1f},"
                         f"{d1.centroid[0]:.1f},{d1.centroid[1]:.1f},"
                         f"{X[0]:.1f},{X[1]:.1f},{X[2]:.1f},{int(acc)},"
                         f"{d0.separation:.2f},{d1.separation:.2f}\n")
            else:
                c0 = d0.centroid if d0.found else (float("nan"),) * 2
                c1 = d1.centroid if d1.found else (float("nan"),) * 2
                fh.write(f"{idx},0,{c0[0]:.1f},{c0[1]:.1f},{c1[0]:.1f},"
                         f"{c1[1]:.1f},nan,nan,nan,0,"
                         f"{d0.separation:.2f},{d1.separation:.2f}\n")
            if n_rows % 200 == 0:
                fps = n_rows / (time.time() - t0)
                print(f"  {n_rows} frames  ({n_found} detected, "
                      f"{n_accept} accepted)  {fps:.1f} fps")

    dt = time.time() - t0
    pct = (100 * n_found / n_rows) if n_rows else 0.0
    print(f"\n{n_rows} frames in {dt:.1f}s ({n_rows / max(dt, 1e-6):.1f} fps): "
          f"{n_found} detected ({pct:.0f}%), {n_accept} accepted "
          f"→ {args.out}")


if __name__ == "__main__":
    main()
