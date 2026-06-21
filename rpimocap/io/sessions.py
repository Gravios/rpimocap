"""
rpimocap.io.sessions
====================
Discover and inventory the recording sessions in a capture dataset.

A capture date directory holds several timestamped recordings, each a
stereo pair plus a metadata CSV, e.g.

    raw/
      cam0_20260214_021058_raw.tif   cam1_20260214_021058_raw.tif
      20260214_021058_metadata.csv
      cam0_20260214_021255_raw.tif   cam1_20260214_021255_raw.tif
      20260214_021255_metadata.csv
      ...

This module groups those files into Session records (keyed by
timestamp), and reports each camera's frame count so mismatches — a
dropped XVS sync or an early stop that truncates one camera, like the
cam1-stops-at-950 case — are visible across the whole dataset at once.
The texture-statistics aggregation that consumes these sessions lives in
tools/session_stats.py.
"""
from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass
from typing import Optional


_CAM_RE = re.compile(r"cam(\d)_(\d{8}_\d{6})_raw\.tif$")


@dataclass
class Session:
    """One stereo recording: a timestamp + the two camera TIFFs (+ the
    metadata CSV if present)."""
    timestamp: str
    cam0_path: Optional[str]
    cam1_path: Optional[str]
    metadata_path: Optional[str] = None

    @property
    def complete(self) -> bool:
        """True iff both camera files are present."""
        return self.cam0_path is not None and self.cam1_path is not None


def discover_sessions(raw_dir: str) -> list[Session]:
    """Find all recording sessions under raw_dir, grouped by timestamp
    and sorted chronologically. A session is included if at least one
    camera file is present; use `.complete` to filter to stereo pairs."""
    by_ts: dict[str, dict] = {}
    for path in sorted(glob.glob(os.path.join(raw_dir, "cam*_raw.tif"))):
        m = _CAM_RE.search(os.path.basename(path))
        if not m:
            continue
        cam, ts = m.group(1), m.group(2)
        slot = by_ts.setdefault(ts, {})
        slot[f"cam{cam}"] = path

    sessions = []
    for ts in sorted(by_ts):
        meta = os.path.join(raw_dir, f"{ts}_metadata.csv")
        sessions.append(Session(
            timestamp=ts,
            cam0_path=by_ts[ts].get("cam0"),
            cam1_path=by_ts[ts].get("cam1"),
            metadata_path=meta if os.path.exists(meta) else None))
    return sessions


def session_frame_counts(session: Session,
                         bayer_pattern: str = "RGGB") -> tuple:
    """Return (n_cam0, n_cam1) frame counts (−1 where a file is missing
    or unreadable). Reads only the TIFF page count — cheap, no decode."""
    from rpimocap.io.export import TiffCapture
    import cv2

    def _count(path):
        if not path:
            return -1
        try:
            cap = TiffCapture(path, bayer_pattern=bayer_pattern)
            return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        except Exception:
            return -1

    return _count(session.cam0_path), _count(session.cam1_path)


def inventory(raw_dir: str, bayer_pattern: str = "RGGB") -> list[dict]:
    """Per-session inventory rows: timestamp, frame counts, whether the
    stereo pair is complete, and whether the two cameras' frame counts
    match (the stereo-validity precondition)."""
    rows = []
    for s in discover_sessions(raw_dir):
        n0, n1 = session_frame_counts(s, bayer_pattern)
        overlap = min(n0, n1) if (n0 >= 0 and n1 >= 0) else -1
        rows.append({
            "timestamp": s.timestamp,
            "cam0_frames": n0,
            "cam1_frames": n1,
            "complete": s.complete,
            "frames_match": (n0 == n1) if (n0 >= 0 and n1 >= 0) else False,
            "stereo_overlap": overlap,
            "has_metadata": s.metadata_path is not None,
        })
    return rows
