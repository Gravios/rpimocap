# rpimocap Processing Workflow
_Working directory: `strohA-al-RPICAM/`_

> **Shell compatibility.** Every fenced `bash` block in this doc is
> written to copy-paste cleanly into **both `bash` and `zsh`** at the
> project root. The array idioms use the portable `set --`
> positional-parameter pattern instead of `${arr[0]}` because zsh
> arrays are 1-indexed and `[0]` returns empty there. Glob-on-no-match
> errors in zsh are suppressed at the top of each loop block with
> `setopt NULL_GLOB 2>/dev/null; shopt -s nullglob 2>/dev/null` — the
> first works in zsh and is a no-op in bash; the second is the reverse.

---

## Directory Layout

```
strohA-al-RPICAM/                              ← CWD (cd here first)
│
├── calib/                                     ← ONE calibration for all sessions
│   ├── autocalib.npz
│   ├── calib_from_corners.npz                 ← USE THIS for all sessions
│   ├── align.csv
│   └── align_edges.csv
│
├── background/
│   └── bg.npz                                 ← built once from ALL sessions
│
├── strohA-al-RPICAM-20260214/
│   ├── raw/
│   │   ├── cam0_20260214_021722_raw.tif       ← immutable
│   │   └── cam1_20260214_021722_raw.tif
│   └── 20260214-021722/
│       └── tracking/
│           ├── reconstruction.h5
│           ├── detection_stats.csv
│           ├── viewer_data.json
│           └── preview.mp4
│
├── strohA-al-RPICAM-20260215/
│   ├── raw/
│   │   ├── cam0_20260215_134500_raw.tif
│   │   └── cam1_20260215_134500_raw.tif
│   └── 20260215-134500/
│       └── tracking/
│           └── ...
└── ...
```

---

## Step 1 — Set Variables

```bash
# Edit these for the animal and recording dates
SUBJECT=strohA-al
RPICAM_DIR=$(pwd)                              # assumes you are already here

# One representative session for corner annotation
REF_DATE=20260214
REF_TIME=021722
REF_RAW=$RPICAM_DIR/strohA-al-RPICAM-${REF_DATE}/raw
```

---

## Step 2 — Autocalibration  _(run once, shared across sessions)_

```bash
mkdir -p calib

rpimocap-autocalib \
    --cam0          $REF_RAW/cam0_${REF_DATE}_${REF_TIME}_raw.tif \
    --cam1          $REF_RAW/cam1_${REF_DATE}_${REF_TIME}_raw.tif \
    --bayer-pattern RGGB \
    --out           calib/autocalib.npz
```

- [ ] Runs without error
- [ ] `fx` discrepancy between cam0/cam1 < 10%

---

## Step 3 — Corner Annotation  _(run once, shared across sessions)_

```bash
rpimocap-align \
    --cam0          $REF_RAW/cam0_${REF_DATE}_${REF_TIME}_raw.tif \
    --cam1          $REF_RAW/cam1_${REF_DATE}_${REF_TIME}_raw.tif \
    --calib         calib/autocalib.npz \
    --bayer-pattern RGGB \
    --out           calib/align.csv
```

Click all 8 corners (BFL BFR BBR BBL TFL TFR TBR TBL) in both cameras. Arena dims: X=±140mm, Y=±215mm, Z=388mm.

- [ ] All 8 corners annotated
- [ ] CSV saved with px0/px1 columns

---

## Step 4 — DLT Calibration from Corners  _(run once)_

```bash
rpimocap-calibrate-from-corners \
    --align         calib/align.csv \
    --calib         calib/autocalib.npz \
    --out           calib/calib_from_corners.npz
```

- [ ] cam0 reprojection < 10 px
- [ ] cam1 reprojection < 10 px
- [ ] Kabsch RMSE < 10 mm

---

## Step 5 — Background Model  _(built once from ALL sessions)_

```bash
# Tolerate globs that don't match (zsh errors by default; bash needs nullglob)
setopt NULL_GLOB 2>/dev/null; shopt -s nullglob 2>/dev/null

mkdir -p background

# Collect all cam0 and cam1 TIFFs across every session automatically.
CAM0_ALL=()
CAM1_ALL=()
for session_dir in strohA-al-RPICAM-*/; do
    [ -d "$session_dir" ] || continue
    raw_dir="${session_dir}raw"
    for f in "$raw_dir"/cam0_*_raw.tif; do
        [ -f "$f" ] && CAM0_ALL+=("$f")
    done
    for f in "$raw_dir"/cam1_*_raw.tif; do
        [ -f "$f" ] && CAM1_ALL+=("$f")
    done
done

echo "Found ${#CAM0_ALL[@]} cam0 sessions:"
printf '  %s\n' "${CAM0_ALL[@]}"

# Split off the first session for --cam0/--cam1; pass the rest as
# --background-extra-*.  Done via positional parameters so this works
# identically in bash (0-indexed arrays) and zsh (1-indexed arrays).
set -- "${CAM0_ALL[@]}";  CAM0_FIRST="$1"; shift; CAM0_REST=("$@")
set -- "${CAM1_ALL[@]}";  CAM1_FIRST="$1"; shift; CAM1_REST=("$@")

rpimocap-segment \
    --cam0                    "$CAM0_FIRST" \
    --cam1                    "$CAM1_FIRST" \
    --calib                   calib/calib_from_corners.npz \
    --bayer-pattern           RGGB \
    --background-extra-cam0   "${CAM0_REST[@]}" \
    --background-extra-cam1   "${CAM1_REST[@]}" \
    --background-frames       100 \
    --bounds="-140,140,-215,215,0,388" \
    --threshold               20 \
    --green-channel --bilateral \
    --centroid-only \
    --texture-suppress \
    --end-frame               0 \
    --out                     .
# --end-frame 0    → build the background model then exit without tracking
# --texture-suppress → also cache the Gabor energy model in bg.npz
#                      (required by --gabor-refine; harmless if unused)
```

- [ ] `background/bg.npz` created
- [ ] `background/bg_cam0.png` and `bg_cam1.png` look clean (empty arena)

> The command above includes `--texture-suppress` by default. That
> bakes a Gabor energy model into `bg.npz` alongside the intensity
> background and is required by `--gabor-refine` at tracking time.
> It's safe to keep on even if you don't plan to use `--gabor-refine`
> — costs about 30 s on a typical session and is otherwise inert.

---

## Step 6 — Track Each Session

### Single session (paste once per recording)

For a single recording, fill in the four variables at the top of the
block and paste:

```bash
# Tolerate globs that don't match
setopt NULL_GLOB 2>/dev/null; shopt -s nullglob 2>/dev/null

# ── EDIT THESE ──────────────────────────────────────────────────────────
DATE_PART=20260214
TIME_PART=021722
SUBJECT_DIR=strohA-al-RPICAM-${DATE_PART}
# ────────────────────────────────────────────────────────────────────────

raw_dir="${SUBJECT_DIR}/raw"
cam0_tif="${raw_dir}/cam0_${DATE_PART}_${TIME_PART}_raw.tif"
cam1_tif="${raw_dir}/cam1_${DATE_PART}_${TIME_PART}_raw.tif"
session="${SUBJECT_DIR}/${DATE_PART}-${TIME_PART}"

rpimocap-segment \
    --cam0             "$cam0_tif" \
    --cam1             "$cam1_tif" \
    --calib            calib/calib_from_corners.npz \
    --bayer-pattern    RGGB \
    --background-model background/bg.npz \
    --bounds="-140,140,-215,215,0,388" \
    --threshold        20 \
    --green-channel --bilateral \
    --centroid-only \
    --out              "$session"

rpimocap-preview \
    --cam0          "$cam0_tif" \
    --cam1          "$cam1_tif" \
    --calib         calib/calib_from_corners.npz \
    --h5            "$session/tracking/reconstruction.h5" \
    --bayer-pattern RGGB \
    --out           "$session/tracking/preview.mp4" \
    --end-frame     500
```

### Batch (loop over every session under the working directory)

```bash
setopt NULL_GLOB 2>/dev/null; shopt -s nullglob 2>/dev/null

for session_dir in strohA-al-RPICAM-*/; do
    [ -d "$session_dir" ] || continue

    # Extract DATE_PART from the directory name (strohA-al-RPICAM-20260214)
    DATE_PART="${session_dir#strohA-al-RPICAM-}"
    DATE_PART="${DATE_PART%/}"

    raw_dir="${session_dir}raw"

    for cam0_tif in "$raw_dir"/cam0_*_raw.tif; do
        [ -f "$cam0_tif" ] || continue

        # Extract TIME_PART from the filename via parameter expansion
        # (avoids grep -oP, which is GNU-only and missing on macOS).
        fname="${cam0_tif##*/}"             # cam0_20260214_021722_raw.tif
        tmp="${fname#cam0_}"                # 20260214_021722_raw.tif
        tmp="${tmp%_raw.tif}"               # 20260214_021722
        TIME_PART="${tmp#*_}"               # 021722

        cam1_tif="${raw_dir}/cam1_${DATE_PART}_${TIME_PART}_raw.tif"
        session="${session_dir}${DATE_PART}-${TIME_PART}"

        echo ""
        echo "═══ $DATE_PART-$TIME_PART ══════════════════════════════════"

        # ── Validate on 50 frames first ─────────────────────────────────
        rpimocap-segment \
            --cam0             "$cam0_tif" \
            --cam1             "$cam1_tif" \
            --calib            calib/calib_from_corners.npz \
            --bayer-pattern    RGGB \
            --background-model background/bg.npz \
            --bounds="-140,140,-215,215,0,388" \
            --threshold        20 \
            --green-channel --bilateral \
            --centroid-only \
            --end-frame        50 \
            --out              "$session"

        # ── Quick validity check ─────────────────────────────────────────
        python3 - "$session/tracking/reconstruction.h5" << 'PYEOF'
import sys
import h5py, numpy as np
h5path = sys.argv[1]
with h5py.File(h5path) as f:
    skel = f.get("skeleton") or {}
    if "animal" not in skel:
        print("  WARN: no animal key — check background/threshold")
        sys.exit(1)
    g = skel["animal"]
    # Prefer /detected (v0.5.0+): captured pre-post-processing, so
    # gap-filled frames are correctly False even when xyz is non-NaN.
    if "detected" in g:
        det = g["detected"][:]
        n_real, n_total = int(det.sum()), int(len(det))
    else:
        xyz = g["xyz"][:]
        v = ~np.isnan(xyz).any(axis=1)
        n_real, n_total = int(v.sum()), int(len(xyz))
pct = 100*n_real/max(n_total, 1)
print(f"  Validation: {n_real}/{n_total} real frames ({pct:.0f}%)")
if pct < 60:
    print("  WARN: < 60% detection — see Troubleshooting "
          "(try --gabor-refine --kalman --kalman-online)")
    sys.exit(1)
PYEOF
        if [ $? -ne 0 ]; then
            echo "  SKIP full run (validation failed)"
            continue
        fi

        # ── Full run ─────────────────────────────────────────────────────
        rpimocap-segment \
            --cam0             "$cam0_tif" \
            --cam1             "$cam1_tif" \
            --calib            calib/calib_from_corners.npz \
            --bayer-pattern    RGGB \
            --background-model background/bg.npz \
            --bounds="-140,140,-215,215,0,388" \
            --threshold        20 \
            --green-channel --bilateral \
            --centroid-only \
            --out              "$session"

        # ── Preview (first 500 frames) ───────────────────────────────────
        rpimocap-preview \
            --cam0          "$cam0_tif" \
            --cam1          "$cam1_tif" \
            --calib         calib/calib_from_corners.npz \
            --h5            "$session/tracking/reconstruction.h5" \
            --bayer-pattern RGGB \
            --out           "$session/tracking/preview.mp4" \
            --end-frame     500
    done
done
```

---

## Step 6b — Robust-pipeline invocation (v0.5.0 features)

The bare invocation in Step 6 is the minimum that works. For the
bedding-disturbance / cable-contamination / wall-reflection edge
cases the v0.5.0 release added a layered defence-in-depth toolkit.
Enable as many of the following as suit the recording. Each flag is
independent and off by default; the cost of enabling them all is
roughly 30 % of bare runtime.

### Pre-flight: build a `--texture-suppress` background

The Gabor-edge body contour (`--gabor-refine`) needs a cached Gabor
model in `background/bg.npz`. Step 5 builds it by default (the
command there already passes `--texture-suppress`). If you built
`bg.npz` earlier without it, re-run Step 5 to rebake.

### Full robust run

This command is **self-contained for a single session** — fill in
the three variables at the top, then paste:

```bash
# ── EDIT THESE ──────────────────────────────────────────────────────────
DATE_PART=20260214
TIME_PART=021722
SUBJECT_DIR=strohA-al-RPICAM-${DATE_PART}
# ────────────────────────────────────────────────────────────────────────

raw_dir="${SUBJECT_DIR}/raw"
cam0_tif="${raw_dir}/cam0_${DATE_PART}_${TIME_PART}_raw.tif"
cam1_tif="${raw_dir}/cam1_${DATE_PART}_${TIME_PART}_raw.tif"
session="${SUBJECT_DIR}/${DATE_PART}-${TIME_PART}"

rpimocap-segment \
    --cam0             "$cam0_tif" \
    --cam1             "$cam1_tif" \
    --calib            calib/calib_from_corners.npz \
    --bayer-pattern    RGGB \
    --background-model background/bg.npz \
    --bounds="-140,140,-215,215,0,388" \
    --threshold        20 \
    --green-channel --bilateral \
    --centroid-only \
    --cable-erosion       12 \
    \
    `# ── Centroid refinement (steps 3b + 5 of hull_centroid) ──` \
    --gabor-refine \
    --canny-low           30   --canny-high   90 \
    --body-length        180   --body-width   70   --body-z   0 \
    \
    `# ── Per-frame selection robustness ──` \
    --trajectory-prior \
    --trajectory-prior-lambda 0.05 \
    --bg-adapt-alpha     0.995 \
    --bg-adapt-dilate-px 25 \
    \
    `# ── Online tracking (per-frame Kalman + rearing) ──` \
    --kalman-online \
    --rearing-detection \
    --rearing-z-enter   100 --rearing-z-exit 70 \
    \
    `# ── Offline post-processing (trajectory-level Kalman/RTS) ──` \
    --kalman \
    --kalman-fps         25 \
    --kalman-max-speed   1000 \
    --kalman-max-accel   2000 \
    --kalman-noise       8 \
    --kalman-outlier-sigma 4.0 \
    \
    --out              "$session"
```

To batch this across every session, wrap it in the Step 6 batch
loop's outer `for session_dir in strohA-al-RPICAM-*/` shell (which
provides `$cam0_tif` / `$cam1_tif` / `$session` per iteration) and
drop the EDIT-THESE block above.

### What each flag fixes

| Flag(s) | Problem it addresses |
|---|---|
| `--cable-erosion 12` | Headstage cable biases the unweighted centroid toward the cable end |
| `--gabor-refine` (+ `--canny-low/--canny-high`) | Disturbed bedding makes the intensity boundary noisy; texture-space contour is clean |
| `--body-length/--body-width/--body-z` | Anatomical Gaussian prior pulls centroid toward the expected body shape, suppressing cable-tip and bedding-edge outliers |
| `--trajectory-prior` | Wall reflections with low epipolar distance no longer outrank the actual animal blob |
| `--bg-adapt-alpha` | Bedding moved earlier in the recording gradually becomes part of the background |
| `--kalman-online` | Velocity-aware blob prior keeps producing predictions through gaps |
| `--rearing-detection` | Vertical body posture swaps in `90 × 45 mm` body dims so the anatomical prior doesn't pull the centroid toward a horizontal body that isn't there |
| `--kalman` (offline) | Mahalanobis-gate outlier rejection + RTS backward smoother on the final trajectory |
| `--sam2-video-checkpoint` | Replace bg-subtraction entirely with SAM2 video propagation. Best for sessions with severe bedding disturbance or sustained occlusion; requires the `sam2` package + checkpoint and a one-time pre-pass (cached, reused on re-runs) |

### Optional: flat-field correction for NIR vignette

Capture a flat-field reference frame at rig assembly (uniform NIR-lit
target with no animal). To use it, **add these flags** to the
`rpimocap-segment` invocation above:

    --flat-field-cam0 calib/flat_cam0.png \
    --flat-field-cam1 calib/flat_cam1.png \

If no calibration capture exists, synthesise one from the
background — **add this flag** instead:

    --synthesize-flat-field \

### Optional: SAM2 video propagation for foreground masks

When bedding disturbance is severe enough that even Gabor refinement
struggles, replacing bg-subtraction entirely with SAM2 video
propagation gives the cleanest masks. Requires the `sam2` package
installed and a downloaded checkpoint.

**Add these flags** to the `rpimocap-segment` invocation above:

    --sam2-video-checkpoint /path/to/sam2_hiera_large.pt \
    --sam2-video-config     sam2.1_hiera_l.yaml \
    --sam2-video-prompt-frame 0 \

How it works:

1. **Pre-pass.** Before tracking starts, `rpimocap-segment` reads frame
   `--sam2-video-prompt-frame` (default 0), runs bg-subtraction to find
   the rat blob's centroid in both cameras, and uses those as point
   prompts to seed SAM2.
2. **Propagation.** SAM2's video predictor propagates the mask through
   the whole session using its appearance-and-motion model.
3. **Caching.** Per-frame masks are written to
   `<session>/tracking/sam2_masks/cam{0,1}/NNNNNN.png` (or to
   `--sam2-video-cache-dir DIR` if set). A second run on the same
   session reuses the cache and skips the pre-pass.
4. **Per-frame consumption.** Inside `track_sequence`, the cached mask
   is loaded instead of running bg-subtraction. Everything downstream
   (labelling, epipolar matching, hull_centroid with all the v0.5.0
   refinements, triangulation, post-processing) is unchanged.

When SAM2 loses track on a particular frame the cache returns `None`
and the pipeline falls back to bg-subtraction for that frame, so a
partial propagation is still useful.

### Reading the new `detected` HDF5 field

Every `reconstruction.h5` written by v0.5.0+ carries
`/skeleton/<name>/detected` (bool, `n_frames`) alongside `xyz`. True
= genuine triangulated detection in that frame; False = gap-filled
(Kalman prediction or linear interpolation).

```python
import h5py, numpy as np

with h5py.File("reconstruction.h5") as f:
    det = f["skeleton"]["animal"]["detected"][:]
    xyz = f["skeleton"]["animal"]["xyz"][:]

# Per-frame detection rate
print(f"{det.sum()}/{len(det)} = {100*det.sum()/len(det):.1f}% real detections")

# Only use frames with genuine detections for downstream analysis
xyz_real = xyz[det]
```

---

## Step 7 — Batch Summary

```bash
setopt NULL_GLOB 2>/dev/null; shopt -s nullglob 2>/dev/null

echo ""
echo "Session                        valid/total   X mean±std    Y mean±std    Z mean±std"
echo "─────────────────────────────────────────────────────────────────────────────────"

for h5 in strohA-al-RPICAM-*/*/tracking/reconstruction.h5; do
    [ -f "$h5" ] || continue
    python3 - "$h5" << 'PYEOF'
import sys
import h5py, numpy as np
h5path = sys.argv[1]
# strohA-al-RPICAM-YYYYMMDD/YYYYMMDD-HHMMSS
label = "/".join(h5path.split("/")[:-2])
with h5py.File(h5path) as f:
    skel = f.get("skeleton") or {}
    if "animal" not in skel:
        print(f"  {label:<30s}  no animal key")
        sys.exit(0)
    g   = skel["animal"]
    xyz = g["xyz"][:]
    # Use /detected (v0.5.0+) when present — accurate post-Kalman.
    if "detected" in g:
        v = g["detected"][:]
    else:
        v = ~np.isnan(xyz).any(axis=1)
x, y, z = xyz[v, 0], xyz[v, 1], xyz[v, 2]
print(f"  {label:<30s}  {int(v.sum()):4d}/{len(xyz):<5d}"
      f"  X={x.mean():+6.0f}±{x.std():4.0f}"
      f"  Y={y.mean():+6.0f}±{y.std():4.0f}"
      f"  Z={z.mean():+6.0f}±{z.std():4.0f}")
PYEOF
done
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| Kabsch RMSE > 20 mm | Bad autocalib focal lengths | `calib_from_corners` is the fix — check DLT reprojection |
| < 60% valid frames, plain pipeline | Threshold too low or BG mismatch | Raise `--threshold` (try 25–35), or enable the v0.5.0 robust toolkit: `--gabor-refine --kalman --kalman-online` |
| Z values > 400 mm or negative | Epipolar mismatch | Check bounds filter; re-verify DLT reprojection < 10 px |
| Animal near wall → missing frames | Normal occlusion | First try `--kalman` (predictive gap-fill respecting `--kalman-max-speed`); falls back to `--fill-gaps 15` if Kalman is off |
| Centroid jumps to wall reflections | Epipolar selector tied | `--trajectory-prior` (last-frame prior) and/or `--kalman-online` (velocity-aware prior); raise `--trajectory-prior-lambda` if reflections still win |
| Centroid drags toward headstage cable | Cable pixels included in centroid | `--cable-erosion 10–15` to disconnect the cable, then `--body-length 180 --body-width 70` for the anatomical Gaussian prior |
| Bedding has been disturbed | Intensity-based bg-sub captures bedding edges | `--gabor-refine` (requires `--texture-suppress` background); for severe disturbance, `--sam2-video-checkpoint` |
| Centroid wrong during rearing | Horizontal body prior applied vertically | `--rearing-detection` (needs `--kalman-online`); swaps in vertical body dims when reared |
| Lighting non-uniform across image | NIR vignette | `--synthesize-flat-field`, or `--flat-field-cam0/--flat-field-cam1` with captured references |
| One session much worse than others | Lighting change | Build session-specific BG: add `--background-extra-*` for just that session |
| Hand visible in background | BG built while experimenter present | Use only empty-arena sessions for `--background-extra-*` |
| Gap-filled frames look like real detections | xyz NaN check is post-Kalman | Read `/skeleton/<name>/detected` from the HDF5 — `True` iff a real detection pre-post-processing |

