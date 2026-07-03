#!/usr/bin/env python3
"""
session_stats.py
================
Inventory and texture statistics across ALL recording sessions in a
capture dataset (the "stats from all the videos" step — ROADMAP Phase 0).

Two modes:

  inventory (always): discover every stereo session under --raw-dir and
    report each camera's frame count, whether the pair is complete, and
    whether the two cameras' frame counts match (the stereo-validity
    precondition — this surfaces the cam1-truncated sessions at a
    glance). Writes a CSV.

  --texture-stats (optional, parallel over --n-workers cores): for each
    session, build the persistent background texture model over a sample
    of frames and accumulate the per-descriptor-channel background
    moments (mean / std / min / max). Aggregate across all sessions into
    one global background-texture statistics file (.npz + .json). This
    pooled background distribution is what the future KDE/GMM region-
    density detector is trained on.

Example
-------
  python tools/session_stats.py \
      --raw-dir strohA-al-RPICAM-20260214/raw \
      --texture-stats --bg-frames 60 --bg-stride 40 \
      --n-orientations 8 --n-workers 8 \
      --out strohA-al-RPICAM-20260214/session_stats
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rpimocap.io.sessions import discover_sessions, inventory


def _texture_stats_for_session(args_tuple):
    """Worker: pooled background descriptor moments for one camera file.
    Returns (count, sum, sumsq, vmin, vmax) per descriptor channel, or
    None if unreadable."""
    (path, bayer, green, scales, n_orient, smooth_k,
     bg_frames, bg_start, bg_stride, intensity, illum_sigma) = args_tuple
    try:
        import cv2
        from rpimocap.io.export import TiffCapture
        from rpimocap.detection.texture_distance import (
            dense_gabor_descriptor, illumination_intensity)
        from rpimocap.detection.rat_texture import build_gabor_kernels

        kernels = build_gabor_kernels(
            [i * np.pi / n_orient for i in range(n_orient)], scales)
        cap = TiffCapture(path, bayer_pattern=bayer)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        count = None
        idx = bg_start
        collected = 0
        while collected < bg_frames and idx < n:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            idx += bg_stride
            if not ok or frame is None:
                continue
            if green and frame.ndim == 3:
                gray = frame[:, :, 1]
            elif frame.ndim == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame
            desc = dense_gabor_descriptor(
                gray, kernels, n_orient, len(scales), smooth_k=smooth_k)
            if intensity:
                # Prepend the illumination-flattened intensity channel:
                # the cue that actually separates the (bright) rat, since
                # coarse Gabor alone rings the boundary. Channel 0 = intensity.
                ich = illumination_intensity(
                    gray, illum_sigma=illum_sigma, smooth_k=smooth_k)
                desc = np.concatenate([ich[None], desc], axis=0)
            D = desc.shape[0]
            flat = desc.reshape(D, -1)
            if count is None:
                count = 0
                ssum = np.zeros(D, np.float64)
                ssq = np.zeros(D, np.float64)
                vmin = np.full(D, np.inf)
                vmax = np.full(D, -np.inf)
            count += flat.shape[1]
            ssum += flat.sum(axis=1)
            ssq += (flat.astype(np.float64) ** 2).sum(axis=1)
            vmin = np.minimum(vmin, flat.min(axis=1))
            vmax = np.maximum(vmax, flat.max(axis=1))
            collected += 1
        if count is None:
            return None
        return (count, ssum, ssq, vmin, vmax)
    except Exception as e:                       # noqa: BLE001
        sys.stderr.write(f"  texture-stats failed for {path}: {e}\n")
        return None


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", required=True,
                    help="Directory of cam{0,1}_<ts>_raw.tif recordings.")
    ap.add_argument("--bayer-pattern", default="RGGB")
    ap.add_argument("--green-channel", action="store_true")
    ap.add_argument("--texture-stats", action="store_true",
                    help="Also pool background descriptor stats across "
                         "all sessions (heavier; parallel).")
    ap.add_argument("--scales", type=int, nargs="+", default=[5, 9, 13],
                    help="Gabor kernel sizes (px). Fine [5 9 13] do NOT "
                         "separate the rat; use body-scale (e.g. 65 129) "
                         "so the rat is a genuine texture outlier.")
    ap.add_argument("--n-orientations", type=int, default=8)
    ap.add_argument("--smooth-k", type=int, default=7)
    ap.add_argument("--intensity", action="store_true",
                    help="Prepend an illumination-flattened intensity "
                         "channel (channel 0) to the pooled descriptor. "
                         "This is the cue that actually separates the "
                         "bright rat; pair with body-scale --scales for a "
                         "background model the distance detector can "
                         "threshold on. Output npz format is unchanged "
                         "(mean/std/vmin/vmax gain one leading channel).")
    ap.add_argument("--illum-sigma", type=float, default=151.0,
                    help="Gaussian sigma (px) of the per-frame "
                         "illumination field for --intensity. Large keeps "
                         "the rat's brightness intact while flattening the "
                         "IR falloff.")
    ap.add_argument("--bg-frames", type=int, default=60)
    ap.add_argument("--bg-start", type=int, default=0)
    ap.add_argument("--bg-stride", type=int, default=40)
    ap.add_argument("--n-workers", type=int, default=1)
    ap.add_argument("--out", required=True,
                    help="Output prefix (writes <out>_inventory.csv and, "
                         "with --texture-stats, <out>_texture.npz/json).")
    args = ap.parse_args(argv)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".",
                exist_ok=True)

    # ── Inventory (always) ────────────────────────────────────────
    rows = inventory(args.raw_dir, bayer_pattern=args.bayer_pattern)
    inv_path = f"{args.out}_inventory.csv"
    with open(inv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())
                           if rows else ["timestamp"])
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} sessions → {inv_path}")
    n_mismatch = sum(1 for r in rows if not r["frames_match"])
    n_total_overlap = sum(max(r["stereo_overlap"], 0) for r in rows)
    for r in rows:
        flag = "" if r["frames_match"] else "   <-- frame-count MISMATCH"
        print(f"  {r['timestamp']}: cam0={r['cam0_frames']} "
              f"cam1={r['cam1_frames']} overlap={r['stereo_overlap']}"
              f"{flag}")
    print(f"  {n_mismatch} session(s) with mismatched frame counts; "
          f"{n_total_overlap} total stereo-valid frames across dataset")

    if not args.texture_stats:
        print("(pass --texture-stats to also pool background descriptor "
              "statistics across all sessions)")
        return

    # ── Cross-session background texture stats (parallel) ─────────
    sessions = [s for s in discover_sessions(args.raw_dir) if s.complete]
    tasks = []
    for s in sessions:
        for path in (s.cam0_path, s.cam1_path):
            tasks.append((path, args.bayer_pattern, args.green_channel,
                          args.scales, args.n_orientations, args.smooth_k,
                          args.bg_frames, args.bg_start, args.bg_stride,
                          args.intensity, args.illum_sigma))
    print(f"\nPooling background descriptor stats over {len(tasks)} "
          f"camera files ({args.n_workers} workers)…")

    if args.n_workers > 1:
        from multiprocessing import Pool
        with Pool(args.n_workers) as pool:
            results = pool.map(_texture_stats_for_session, tasks)
    else:
        results = [_texture_stats_for_session(t) for t in tasks]

    # aggregate moments
    count = None
    for r in results:
        if r is None:
            continue
        c, ssum, ssq, vmin, vmax = r
        if count is None:
            count, gsum, gsq = 0, np.zeros_like(ssum), np.zeros_like(ssq)
            gmin = np.full_like(vmin, np.inf)
            gmax = np.full_like(vmax, -np.inf)
        count += c
        gsum += ssum
        gsq += ssq
        gmin = np.minimum(gmin, vmin)
        gmax = np.maximum(gmax, vmax)
    if count is None or count == 0:
        print("  no descriptor stats collected.")
        return

    mean = gsum / count
    var = np.maximum(gsq / count - mean ** 2, 0.0)
    std = np.sqrt(var)
    npz_path = f"{args.out}_texture.npz"
    np.savez_compressed(npz_path, mean=mean, std=std, vmin=gmin,
                        vmax=gmax, n_pixels=count)
    meta = {
        "n_sessions": len(sessions),
        "n_camera_files": len(tasks),
        "n_pixels_pooled": int(count),
        "n_descriptor_channels": int(mean.shape[0]),
        "scales": args.scales,
        "n_orientations": args.n_orientations,
        "intensity_channel": bool(args.intensity),
        "illum_sigma": float(args.illum_sigma) if args.intensity else None,
        "channel_layout": (["intensity"] if args.intensity else [])
        + [f"gabor_s{s}" for s in args.scales for _ in range(
            (int(mean.shape[0]) - (1 if args.intensity else 0))
            // max(len(args.scales), 1))],
        "channel_mean": mean.tolist(),
        "channel_std": std.tolist(),
    }
    with open(f"{args.out}_texture.json", "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"  pooled {count:,} background descriptor samples → "
          f"{npz_path}")
    print(f"  per-channel mean range "
          f"[{mean.min():.3f}, {mean.max():.3f}], "
          f"std range [{std.min():.3f}, {std.max():.3f}]")


if __name__ == "__main__":
    main()
