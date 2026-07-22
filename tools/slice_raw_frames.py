#!/usr/bin/env python3
"""
slice_raw_frames.py
====================
Extract a short sequence of raw frames from a multi-page TIFF
recording for sharing/diagnostic purposes.

Writes:
  * <out>_cam0_slice.tif      — multi-page TIFF, raw values preserved
  * <out>_cam1_slice.tif      — multi-page TIFF, raw values preserved
  * <out>_cam0_preview/       — PNG snapshots (every Nth frame) for easy viewing
  * <out>_cam1_preview/
  * <out>_slice_manifest.txt  — what was sliced (start, end, frame indices)

Two modes:

  1. Contiguous range  (default; passes --start + --count)
        python slice_raw_frames.py \\
            --cam0 path/to/cam0.tif --cam1 path/to/cam1.tif \\
            --start 2700 --count 40 \\
            --out /tmp/strohA_slice

  2. Random sample across the recording (--random + --count)
        python slice_raw_frames.py \\
            --cam0 ... --cam1 ... \\
            --random --count 40 --seed 42 \\
            --out /tmp/strohA_slice
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import tifffile
import cv2


def _read_raw_frame(tf: tifffile.TiffFile, idx: int) -> np.ndarray:
    """Read page `idx` as a raw numpy array (no demosaic, no
    color conversion). Returns the underlying dtype/shape as
    stored in the file."""
    return tf.pages[idx].asarray()


def _bayer_to_preview_bgr(raw: np.ndarray,
                            bayer: str = "RGGB") -> np.ndarray:
    """Convert raw Bayer frame to a viewable BGR for PNG preview
    snapshots. Does NOT modify the saved raw TIFF — only for
    human preview.

    Delegates to :func:`rpimocap.io.export.bayer_to_bgr` so previews use the
    SAME demosaic as TiffCapture. Two bugs are fixed by doing so:

    * the local pattern map was the INVERSE of TiffCapture's (it mapped
      ``RGGB -> COLOR_BayerRG2BGR``), so the same ``--bayer`` name produced
      R/B-swapped output depending on which module you went through. OpenCV
      names its constants from the second row/second-and-third columns, so
      RGGB genuinely needs ``COLOR_BayerBG2BGR``;
    * ``(raw >> 2).astype(np.uint8)`` assumed 10-bit-in-uint16 and WRAPPED for
      wider data — on these sessions (raw range ~4k-65k) it corrupted 100% of
      pixels, giving previews that correlate 0.03 with a correct demosaic.
    """
    from rpimocap.io.export import bayer_to_bgr
    return bayer_to_bgr(raw, bayer)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cam0", required=True,
                    help="cam0 raw TIFF (multi-page)")
    ap.add_argument("--cam1", required=True,
                    help="cam1 raw TIFF (multi-page)")
    ap.add_argument("--out", required=True,
                    help="Output basename. Files <out>_cam{0,1}_slice.tif "
                         "+ preview directories will be created.")
    ap.add_argument("--count", type=int, default=40,
                    help="Number of frames to extract. Default 40.")
    ap.add_argument("--start", type=int, default=0,
                    help="Starting frame index for contiguous slice. "
                         "Default 0. Ignored in --random mode.")
    ap.add_argument("--random", action="store_true",
                    help="Sample --count frames at random instead of "
                         "contiguous range from --start.")
    ap.add_argument("--seed", type=int, default=42,
                    help="RNG seed for --random mode.")
    ap.add_argument("--bayer", default="RGGB",
                    choices=["RGGB", "BGGR", "GRBG", "GBRG"],
                    help="Bayer pattern for preview PNGs (does NOT "
                         "affect the saved raw TIFF). Default RGGB.")
    ap.add_argument("--preview-every", type=int, default=4,
                    help="Save a preview PNG every Nth frame. Default 4.")
    ap.add_argument("--compress", action="store_true",
                    help="Use LZW compression on output TIFFs "
                         "(smaller files, may not be necessary).")
    args = ap.parse_args()

    cam0_path = Path(args.cam0)
    cam1_path = Path(args.cam1)
    if not cam0_path.exists():
        print(f"ERROR: cam0 not found: {cam0_path}")
        sys.exit(2)
    if not cam1_path.exists():
        print(f"ERROR: cam1 not found: {cam1_path}")
        sys.exit(2)

    out_base = Path(args.out)
    out_base.parent.mkdir(parents=True, exist_ok=True)

    # Open both TIFFs and verify they have enough pages
    tf0 = tifffile.TiffFile(str(cam0_path))
    tf1 = tifffile.TiffFile(str(cam1_path))
    try:
        n0 = len(tf0.pages)
        n1 = len(tf1.pages)
        n_total = min(n0, n1)
        if args.count <= 0:
            print(f"ERROR: --count must be positive, got {args.count}")
            sys.exit(2)
        if args.count > n_total:
            print(f"WARNING: --count {args.count} > available "
                  f"frames {n_total}; clamping.")
            args.count = n_total

        # Pick frame indices
        if args.random:
            rng = np.random.RandomState(args.seed)
            idxs = sorted(rng.choice(n_total,
                                       size=args.count,
                                       replace=False).tolist())
            label = f"random (seed={args.seed})"
        else:
            start = max(0, min(args.start, n_total - args.count))
            idxs = list(range(start, start + args.count))
            label = f"contiguous from {start}"

        print(f"── Slicing {args.count} frames ({label})")
        print(f"  cam0 source : {cam0_path.name}  ({n0} frames)")
        print(f"  cam1 source : {cam1_path.name}  ({n1} frames)")
        print(f"  output base : {out_base}")
        print(f"  frame range : {idxs[0]} .. {idxs[-1]}")

        # Read frames
        cam0_frames = []
        cam1_frames = []
        for idx in idxs:
            cam0_frames.append(_read_raw_frame(tf0, idx))
            cam1_frames.append(_read_raw_frame(tf1, idx))

        # Verify shape/dtype consistency
        s0 = cam0_frames[0].shape
        d0 = cam0_frames[0].dtype
        s1 = cam1_frames[0].shape
        d1 = cam1_frames[0].dtype
        print(f"  cam0 shape  : {s0}  dtype {d0}")
        print(f"  cam1 shape  : {s1}  dtype {d1}")

        # Write multi-page TIFFs (raw values preserved)
        cam0_out = out_base.with_name(out_base.name + "_cam0_slice.tif")
        cam1_out = out_base.with_name(out_base.name + "_cam1_slice.tif")
        compress_kw = {"compression": "lzw"} if args.compress else {}
        print("\n  Writing TIFF slices ...")
        with tifffile.TiffWriter(str(cam0_out), bigtiff=True) as tw:
            for f in cam0_frames:
                tw.write(f, contiguous=False, **compress_kw)
        with tifffile.TiffWriter(str(cam1_out), bigtiff=True) as tw:
            for f in cam1_frames:
                tw.write(f, contiguous=False, **compress_kw)
        print(f"    {cam0_out.name}  ({cam0_out.stat().st_size / 1e6:.1f} MB)")
        print(f"    {cam1_out.name}  ({cam1_out.stat().st_size / 1e6:.1f} MB)")

        # Write preview PNGs
        prev0_dir = out_base.with_name(out_base.name + "_cam0_preview")
        prev1_dir = out_base.with_name(out_base.name + "_cam1_preview")
        prev0_dir.mkdir(exist_ok=True)
        prev1_dir.mkdir(exist_ok=True)
        n_preview = 0
        for i, src_idx in enumerate(idxs):
            if (i % args.preview_every) != 0:
                continue
            bgr0 = _bayer_to_preview_bgr(cam0_frames[i], args.bayer)
            bgr1 = _bayer_to_preview_bgr(cam1_frames[i], args.bayer)
            cv2.imwrite(str(prev0_dir / f"frame_{src_idx:06d}_cam0.png"),
                         bgr0)
            cv2.imwrite(str(prev1_dir / f"frame_{src_idx:06d}_cam1.png"),
                         bgr1)
            n_preview += 1
        print(f"\n  Preview PNGs: {n_preview} per camera")
        print(f"    {prev0_dir.name}/")
        print(f"    {prev1_dir.name}/")

        # Manifest
        manifest = out_base.with_name(out_base.name + "_slice_manifest.txt")
        with open(manifest, "w") as f:
            f.write(f"# rpimocap slice manifest\n")
            f.write(f"cam0_source = {cam0_path}\n")
            f.write(f"cam1_source = {cam1_path}\n")
            f.write(f"mode = {label}\n")
            f.write(f"count = {args.count}\n")
            f.write(f"original_resolution = {s0}\n")
            f.write(f"dtype = {d0}\n")
            f.write(f"bayer_pattern = {args.bayer}\n")
            f.write(f"# original frame indices:\n")
            for k in idxs:
                f.write(f"{k}\n")
        print(f"\n  Manifest: {manifest.name}")
        print("\nDone.")
    finally:
        tf0.close()
        tf1.close()


if __name__ == "__main__":
    main()
