#!/usr/bin/env bash
#
# topo_track.sh — convenience wrapper for the topological rat detector CLI
# (tools/topo_track.py). Runs the median-bandpass grain-density detector over
# a stereo session and writes a gated 3D track CSV.
#
# Usage:
#   tools/topo_track.sh CAM0.tif CAM1.tif CALIB.npz [OUT.csv] [extra flags...]
#
# Examples:
#   tools/topo_track.sh raw/cam0.tif raw/cam1.tif calib_from_corners.npz
#   tools/topo_track.sh raw/cam0.tif raw/cam1.tif calib.npz track3d.csv \
#       --stride 2 --max-frames 500 --barrier-pct 50
#
# Any flags after the (optional) output path are passed straight through to
# tools/topo_track.py (run it with --help to see them all).
#
set -euo pipefail

usage() {
    sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit "${1:-0}"
}

[[ $# -lt 3 || "${1:-}" == "-h" || "${1:-}" == "--help" ]] && usage 0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CAM0="$1"; CAM1="$2"; CALIB="$3"
shift 3

for f in "$CAM0" "$CAM1" "$CALIB"; do
    [[ -f "$f" ]] || { echo "error: file not found: $f" >&2; exit 1; }
done

# Optional 4th positional = output CSV (anything starting with '-' is a flag).
OUT="topo_track3d.csv"
if [[ $# -gt 0 && "$1" != -* ]]; then
    OUT="$1"; shift
fi

PY="${PYTHON:-python3}"

# Resolve rpimocap without a pip install (harmless if installed editable).
# Do NOT cd into the repo: that would resolve the caller's relative paths
# (TIFFs / calib / output) against the repo dir. Run the CLI by absolute
# path and stay in the caller's working directory.
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

echo "topo_track: ${CAM0} + ${CAM1}  (calib ${CALIB})  →  ${OUT}"
exec "${PY}" "${SCRIPT_DIR}/topo_track.py" \
    --cam0 "${CAM0}" --cam1 "${CAM1}" --calib "${CALIB}" --out "${OUT}" "$@"
