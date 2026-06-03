#!/usr/bin/env python3
"""
build_bg_with_std.py
====================
Rebuild background/bg.npz with per-pixel σ fields populated, as
required by patch 0020 (Mahalanobis-style bg-subtraction).

Reads the same cam0/cam1 raw TIFFs as your tracking run, samples
N frames uniformly, computes per-pixel μ AND σ from the same
sample stack, and writes the result to background/bg.npz.

Usage (from the project root /data/source/video/strohA-al/strohA-al-RPICAM):

    python build_bg_with_std.py \
        --cam0 strohA-al-RPICAM-20260214/raw/cam0_20260214_021722_raw.tif \
        --cam1 strohA-al-RPICAM-20260214/raw/cam1_20260214_021722_raw.tif \
        --bayer RGGB \
        --n-frames 200 \
        --method median \
        --out background/bg.npz

If you have multiple session TIFFs and want a more robust bg (rat
appears at different positions across sessions), pass them all:

    python build_bg_with_std.py \
        --cam0 sessionA/raw/cam0.tif sessionB/raw/cam0.tif sessionC/raw/cam0.tif \
        --cam1 sessionA/raw/cam1.tif sessionB/raw/cam1.tif sessionC/raw/cam1.tif \
        --bayer RGGB \
        --n-frames 100 \
        --method median \
        --out background/bg.npz

Output: bg.npz with fields bg0, bg1, std0, std1, method. The std
stats printed by BackgroundModel are also useful as diagnostics —
if 95th percentile σ is huge (>40) the sample is bg-noisy and you
may want a longer/cleaner sample window.
"""
import argparse
import sys
from pathlib import Path

from rpimocap.detection.segment import BackgroundModel
from rpimocap.io.export import TiffCapture


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cam0", nargs="+", required=True, metavar="TIFF",
                    help="One or more cam0 raw TIFFs.")
    ap.add_argument("--cam1", nargs="+", required=True, metavar="TIFF",
                    help="One or more cam1 raw TIFFs (must match cam0 count).")
    ap.add_argument("--bayer", default="RGGB",
                    help="Bayer pattern (default RGGB)")
    ap.add_argument("--n-frames", type=int, default=200, metavar="N",
                    help="Frames per session to sample (default 200). "
                         "With multiple sessions: per-session count.")
    ap.add_argument("--method", default="median", choices=["median", "mean"],
                    help="Background estimation method (default median).")
    ap.add_argument("--start-frame", type=int, default=0,
                    help="First frame index per session (default 0).")
    ap.add_argument("--out", required=True, metavar="NPZ",
                    help="Output bg.npz path.")
    ap.add_argument("--with-gabor", action="store_true",
                    help="Also compute and cache bg Gabor energy maps "
                         "(needed if you previously used --texture-suppress; "
                         "harmless otherwise, ~5-10 s extra at build time).")
    args = ap.parse_args()

    if len(args.cam0) != len(args.cam1):
        print(f"ERROR: --cam0 and --cam1 must have the same number of files "
              f"({len(args.cam0)} vs {len(args.cam1)})")
        sys.exit(2)

    # Validate inputs exist
    for p in args.cam0 + args.cam1:
        if not Path(p).exists():
            print(f"ERROR: file not found: {p}")
            sys.exit(2)

    print(f"── Building background model ─────────────────────────────")
    print(f"  Sessions   : {len(args.cam0)}")
    print(f"  Per-session: {args.n_frames} frames ({args.method})")
    print(f"  Start frame: {args.start_frame}")
    print()

    caps0 = [TiffCapture(p, bayer_pattern=args.bayer) for p in args.cam0]
    caps1 = [TiffCapture(p, bayer_pattern=args.bayer) for p in args.cam1]

    try:
        if len(caps0) == 1:
            bg = BackgroundModel.from_captures(
                caps0[0], caps1[0],
                n_frames=args.n_frames,
                method=args.method,
                start_frame=args.start_frame,
                verbose=True,
            )
        else:
            bg = BackgroundModel.from_multiple_captures(
                caps0, caps1,
                n_frames_each=args.n_frames,
                method=args.method,
                start_frame=args.start_frame,
                verbose=True,
            )

        if args.with_gabor:
            print()
            print("  Computing Gabor bedding-energy maps ...")
            bg.compute_gabor(lambdas=(8.0, 12.0, 16.0), n_orientations=4)

        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        bg.save(out_path)

        print()
        print(f"── Saved bg.npz ──────────────────────────────────────────")
        print(f"  Path  : {out_path.resolve()}")
        print(f"  Fields: bg0, bg1, std0, std1, method"
              + (", bg_gabor0, bg_gabor1" if bg.bg_gabor0 is not None else ""))

        # Quick post-build sanity check
        loaded = BackgroundModel.from_npz(out_path)
        assert loaded.std0 is not None, "std0 missing after save/load!"
        assert loaded.std1 is not None, "std1 missing after save/load!"
        print(f"  std0  : present, shape {loaded.std0.shape}, "
              f"dtype {loaded.std0.dtype}")
        print(f"  std1  : present, shape {loaded.std1.shape}, "
              f"dtype {loaded.std1.dtype}")

    finally:
        for c in caps0 + caps1:
            try:
                c.release()
            except Exception:
                pass


if __name__ == "__main__":
    main()
