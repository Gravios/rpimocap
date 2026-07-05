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
import os
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


def _overlay(g0, g1, R, idx, out_w=1000):
    """Side-by-side cam0|cam1 detection overlay for one frame (BGR uint8).

    green = segmented mask, cyan = centroid, orange = candidate blobs, plus a
    banner with the triangulated point / accepted / reprojection error — so a
    whole session run can be eyeballed frame by frame.
    """
    def draw(g, det, matched):
        v = g.astype(np.float32)
        a, b = np.percentile(v, 1), np.percentile(v, 99)
        v = np.clip((v - a) / (b - a + 1e-6), 0, 1)
        img = cv2.cvtColor((v * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        if det.found:
            cnts, _ = cv2.findContours(det.mask.astype(np.uint8),
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, cnts, -1, (60, 220, 60), 3)     # green mask
            for cx, cy in det.candidates:
                cv2.circle(img, (int(cx), int(cy)), 8, (0, 140, 255), -1)  # orange
            # the MATCHED centroid (the one that produced the 3D point) in
            # cyan; if this view had no accepted match, show its primary grey.
            pt = matched if matched is not None else det.centroid
            col = (255, 255, 0) if matched is not None else (180, 180, 180)
            cv2.circle(img, (int(pt[0]), int(pt[1])), 12, col, -1)
        s = out_w / img.shape[1]
        return cv2.resize(img, (out_w, int(round(img.shape[0] * s))))

    v0, v1 = draw(g0, R.det0, R.pt0), draw(g1, R.det1, R.pt1)
    h = max(v0.shape[0], v1.shape[0])
    combo = np.zeros((h + 40, v0.shape[1] + v1.shape[1], 3), np.uint8)
    combo[40:40 + v0.shape[0], :v0.shape[1]] = v0
    combo[40:40 + v1.shape[0], v0.shape[1]:v0.shape[1] + v1.shape[1]] = v1
    if R.point is not None:
        txt = (f"frame {idx}  X=({R.point[0]:.0f},{R.point[1]:.0f},"
               f"{R.point[2]:.0f})mm  accepted={R.accepted}  "
               f"reproj={R.reproj_err:.1f}px")
    else:
        txt = f"frame {idx}  no consistent stereo pair"
    cv2.putText(combo, txt, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 255, 255), 2, cv2.LINE_AA)
    return combo


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
    ap.add_argument("--barrier-pct", type=float, default=55.0,
                    help="grain barrier percentile for segmentation "
                         "(raise for the lower-contrast view)")
    ap.add_argument("--detect-pct", type=float, default=90.0)
    ap.add_argument("--min-area", type=int, default=1500)
    ap.add_argument("--seg-barrier", default="grain",
                    choices=["grain", "laplacian", "both", "fur", "grain+fur"],
                    help="segmentation barrier: grain-count (default), "
                         "Laplacian energy, both, fur (cable-suppressed "
                         "bright+smooth body), or grain+fur")
    ap.add_argument("--barrier-sigma", type=float, default=3.0,
                    help="Gaussian sigma for the Laplacian-energy barrier")
    ap.add_argument("--cable-suppress", action="store_true",
                    help="use the cable-suppressed (invert-mix) centroid, "
                         "which folds the tether into bedding so it stops "
                         "dragging the centroid")
    ap.add_argument("--max-epipolar-px", type=float, default=60.0,
                    help="max symmetric epipolar distance for a stereo match")
    ap.add_argument("--max-reproj-px", type=float, default=60.0,
                    help="max reprojection error for a stereo match")
    ap.add_argument("--overlay-dir", default=None,
                    help="if set, write a per-frame detection overlay PNG "
                         "(cam0|cam1, mask+centroid+candidates+3D) here")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max-frames", type=int, default=0, help="0 = all")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    from rpimocap.detection.topo_detect import (build_floor_mask,
                                                detect_stereo)
    from rpimocap.io.export import TiffCapture

    cal = np.load(args.calib)
    if "dlt_P0" in cal and "dlt_P1" in cal:
        P0, P1 = cal["dlt_P0"], cal["dlt_P1"]
    elif "P0" in cal and "P1" in cal:
        P0, P1 = cal["P0"], cal["P1"]
        print("WARNING: calib has no dlt_P0/dlt_P1 — falling back to P0/P1.\n"
              "  topo_track needs the ARENA-registered DLT matrices\n"
              "  (arena mm -> pixel), e.g. calib_from_corners.npz. Standard\n"
              "  projection matrices from autocalib.npz use a different frame\n"
              "  and will make every triangulation wrong. Re-run with the\n"
              "  corner-calibrated .npz.", file=sys.stderr)
    else:
        sys.exit(f"error: {args.calib} has no dlt_P0/dlt_P1 (or P0/P1). "
                 f"Use the arena-registered calibration (calib_from_corners.npz).")

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
    if args.overlay_dir:
        os.makedirs(args.overlay_dir, exist_ok=True)
    kw = dict(patch=args.patch, blob_sigma=args.blob_sigma,
              barrier_pct=args.barrier_pct, detect_pct=args.detect_pct,
              min_area=args.min_area, seg_barrier=args.seg_barrier,
              barrier_sigma=args.barrier_sigma,
              cable_suppress=args.cable_suppress)

    last = (min(args.start + args.max_frames * args.stride, n)
            if args.max_frames > 0 else n)
    idxs = range(args.start, last, args.stride)

    t0 = time.time()
    n_found = n_accept = n_rows = 0
    with open(args.out, "w") as fh:
        fh.write("frame,found,cam0_cx,cam0_cy,cam1_cx,cam1_cy,"
                 "X_mm,Y_mm,Z_mm,accepted,reproj_px,sep0,sep1\n")
        for idx in idxs:
            cap0.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok0, fr0 = cap0.read()
            cap1.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok1, fr1 = cap1.read()
            if not (ok0 and ok1):
                continue
            n_rows += 1
            R = detect_stereo(_green(fr0), _green(fr1), fl0, fl1, P0, P1,
                              max_epipolar_px=args.max_epipolar_px,
                              max_reproj_px=args.max_reproj_px, rng=rng, **kw)
            d0, d1, X, acc = R.det0, R.det1, R.point, R.accepted
            found = d0.found and d1.found
            n_found += int(found)
            n_accept += int(acc)
            if args.overlay_dir:
                cv2.imwrite(os.path.join(args.overlay_dir,
                                         f"frame_{idx:06d}.png"),
                            _overlay(_green(fr0), _green(fr1), R, idx))
            if acc and X is not None:
                fh.write(f"{idx},1,{R.pt0[0]:.1f},{R.pt0[1]:.1f},"
                         f"{R.pt1[0]:.1f},{R.pt1[1]:.1f},"
                         f"{X[0]:.1f},{X[1]:.1f},{X[2]:.1f},1,"
                         f"{R.reproj_err:.1f},"
                         f"{d0.separation:.2f},{d1.separation:.2f}\n")
            else:
                c0 = d0.centroid if d0.found else (float("nan"),) * 2
                c1 = d1.centroid if d1.found else (float("nan"),) * 2
                fh.write(f"{idx},{int(found)},{c0[0]:.1f},{c0[1]:.1f},"
                         f"{c1[0]:.1f},{c1[1]:.1f},nan,nan,nan,0,nan,"
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
