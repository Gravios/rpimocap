# rpimocap Processing Workflow
_Working directory: `strohA-al-RPICAM/`_

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
mkdir -p background

# Collect all cam0 and cam1 TIFFs across every session automatically
CAM0_ALL=()
CAM1_ALL=()
for session_dir in strohA-al-RPICAM-*/; do
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

# Use the first session as --cam0/--cam1; pass the rest as --background-extra-*
rpimocap-segment \
    --cam0                    "${CAM0_ALL[0]}" \
    --cam1                    "${CAM1_ALL[0]}" \
    --calib                   calib/calib_from_corners.npz \
    --bayer-pattern           RGGB \
    --background-extra-cam0   "${CAM0_ALL[@]:1}" \
    --background-extra-cam1   "${CAM1_ALL[@]:1}" \
    --background-frames       100 \
    --bounds="-140,140,-215,215,0,388" \
    --threshold               20 \
    --green-channel --bilateral \
    --centroid-only \
    --end-frame               0 \
    --out                     .
# --end-frame 0 builds the background model then exits without tracking

mv background/bg.npz background/bg.npz   # already in place
```

- [ ] `background/bg.npz` created
- [ ] `background/bg_cam0.png` and `bg_cam1.png` look clean (empty arena)

---

## Step 6 — Track Each Session

```bash
# Loop over every session/recording automatically
for session_dir in strohA-al-RPICAM-*/; do
    DATE=$(echo "$session_dir" | grep -oP '\d{8}(?=-RPICAM)')
    # Wait — the dir is strohA-al-RPICAM-20260214, so:
    DATE_PART="${session_dir#strohA-al-RPICAM-}"
    DATE_PART="${DATE_PART%/}"   # e.g. 20260214

    raw_dir="${session_dir}raw"

    for cam0_tif in "$raw_dir"/cam0_*_raw.tif; do
        [ -f "$cam0_tif" ] || continue
        fname=$(basename "$cam0_tif")                 # cam0_20260214_021722_raw.tif
        TIME=$(echo "$fname" | grep -oP '(?<=_)\d{6}(?=_raw)')
        cam1_tif="${raw_dir}/cam1_${DATE_PART}_${TIME}_raw.tif"
        session="${session_dir}${DATE_PART}-${TIME}"

        echo ""
        echo "═══ $DATE_PART-$TIME ══════════════════════════════════"

        # ── Validate on 50 frames first ──────────────────────────────
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

        # ── Quick validity check ──────────────────────────────────────
        python3 - << PYEOF
import h5py, numpy as np, sys
h5 = "$session/tracking/reconstruction.h5"
with h5py.File(h5) as f:
    if "animal" not in f.get("skeleton", {}):
        print("  WARN: no animal key — check background/threshold"); sys.exit(1)
    xyz = f["skeleton"]["animal"]["xyz"][:]
v = ~np.isnan(xyz).any(axis=1)
pct = 100*v.sum()/max(len(xyz),1)
print(f"  Validation: {v.sum()}/50 frames ({pct:.0f}%)")
if pct < 60:
    print("  WARN: < 60% detection — check threshold or background")
    sys.exit(1)
PYEOF
        [ $? -ne 0 ] && { echo "  SKIP full run (validation failed)"; continue; }

        # ── Full run ──────────────────────────────────────────────────
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

        # ── Preview (first 500 frames) ────────────────────────────────
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

## Step 7 — Batch Summary

```bash
# Print detection stats for every completed session
echo ""
echo "Session                        valid/total   X mean±std    Y mean±std    Z mean±std"
echo "─────────────────────────────────────────────────────────────────────────────────"

for h5 in strohA-al-RPICAM-*/*/tracking/reconstruction.h5; do
    python3 - << PYEOF
import h5py, numpy as np
h5 = "$h5"
label = "/".join(h5.split("/")[:-2])   # strohA-al-RPICAM-YYYYMMDD/YYYYMMDD-HHMMSS
with h5py.File(h5) as f:
    if "animal" not in f.get("skeleton", {}):
        print(f"  {label:<30s}  no animal key"); exit()
    xyz = f["skeleton"]["animal"]["xyz"][:]
v = ~np.isnan(xyz).any(axis=1)
x,y,z = xyz[v,0], xyz[v,1], xyz[v,2]
print(f"  {label:<30s}  {v.sum():4d}/{len(xyz):<5d}"
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
| < 60% valid frames | Threshold too low or BG mismatch | Raise `--threshold` (try 25–35) |
| Z values > 400 mm or negative | Epipolar mismatch | Check bounds filter; re-verify DLT reprojection < 10 px |
| Animal near wall → missing frames | Normal occlusion | Increase `--fill-gaps 15` |
| One session much worse than others | Lighting change | Build session-specific BG: add `--background-extra-*` for just that session |
| Hand visible in background | BG built while experimenter present | Use only empty-arena sessions for `--background-extra-*` |
