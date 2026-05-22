"""
rpimocap.detection.sam2_mask_cache
====================================
Disk-backed cache mapping frame index → per-camera SAM2 mask.

The SAM2 video predictor is whole-clip oriented: it takes an initial
prompt on one frame and propagates the mask through the entire clip
using an appearance-and-motion model. The existing SegmentTracker is
per-frame: it consumes one stereo pair at a time. Bridging the two
requires either materialising all frames in memory (~46 GB for a
12000-frame 1080p session) or staging through disk.

This module implements the disk-staged path. Per-frame masks are
written as compressed PNGs (typically 30-100 KB each, so ~700 MB - 2 GB
for a full session — fits on disk comfortably).

Usage
-----

1. Pre-pass: precompute masks for the whole session::

    from rpimocap.detection.tracker import SAM2VideoTracker
    from rpimocap.detection.sam2_mask_cache import SAM2MaskCache

    sam2_video = SAM2VideoTracker(
        checkpoint="/path/to/sam2_hiera_large.pt",
        config="sam2.1_hiera_l.yaml",
    )
    cache = SAM2MaskCache.precompute(
        sam2_video,
        frames_cam0_iter, frames_cam1_iter,
        prompt0_xy=(640, 360),
        prompt1_xy=(660, 360),
        cache_dir="/tmp/session_xxx_sam2_masks",
    )

2. Per-frame consumption (inside SegmentTracker._process_frame)::

    mask0, mask1 = cache[frame_idx]
    if mask0 is not None and mask1 is not None:
        fg0 = foreground_result_from_mask(mask0, f0)
        fg1 = foreground_result_from_mask(mask1, f1)

Cache layout::

    <cache_dir>/
        cam0/
            000000.png
            000001.png
            ...
        cam1/
            000000.png
            000001.png
            ...

Missing PNGs (e.g., SAM2 lost track) return None for that camera/frame;
the caller falls back to bg-subtraction.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np


def foreground_result_from_mask(
        mask:     np.ndarray,
        frame:    np.ndarray,
        min_area_px: int = 50,
):
    """Build a synthetic ForegroundResult from a binary mask.

    Used to feed SAM2-produced masks into the SegmentTracker re-hull
    pipeline as if they had come from bg-subtraction.

    Parameters
    ----------
    mask         : uint8 (H, W), 0 = background, >0 = foreground
    frame        : the original BGR (or grayscale) frame, for the
                   frame_gray field of the result
    min_area_px  : drop components smaller than this (default 50)

    Returns
    -------
    ForegroundResult with mask, label_map, frame_gray, n_blobs populated.
    The gabor_energy field is left as None; the blobs list is empty
    (downstream code uses label_map for re-hulling, not the BodyRegion
    list, so this is fine for the SAM2 path).
    """
    # Import lazily to avoid a circular dep (segment imports vignette etc.)
    from rpimocap.detection.segment import ForegroundResult

    mask = np.asarray(mask, dtype=np.uint8)
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2D; got shape {mask.shape}")
    binary = (mask > 0).astype(np.uint8) * 255

    # Connected components + area filter
    n_lbl, label_map, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8)
    if n_lbl > 1:
        keep = np.zeros(n_lbl, dtype=bool)
        keep[0] = True   # always keep background label 0
        keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area_px
        if not keep.all():
            # Remap small components to 0
            remap = np.zeros(n_lbl, dtype=np.int32)
            new_idx = 0
            for old in range(n_lbl):
                if keep[old]:
                    remap[old] = new_idx
                    new_idx += 1
            label_map = remap[label_map]
            binary = ((label_map > 0).astype(np.uint8)) * 255

    n_blobs = int(label_map.max())

    # Grayscale conversion for frame_gray
    if frame.ndim == 3:
        gray = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    else:
        gray = frame.astype(np.uint8)

    return ForegroundResult(
        mask=binary,
        blobs=[],
        frame_gray=gray,
        n_blobs=n_blobs,
        label_map=label_map.astype(np.int32),
        gabor_energy=None)


class SAM2MaskCache:
    """Disk-backed per-frame mask cache for SAM2 video propagation."""

    def __init__(self, cache_dir):
        self.cache_dir = Path(cache_dir)
        self._cam0_dir = self.cache_dir / "cam0"
        self._cam1_dir = self.cache_dir / "cam1"

    @property
    def exists(self) -> bool:
        """True if either camera's directory has masks staged on disk."""
        return self._cam0_dir.is_dir() or self._cam1_dir.is_dir()

    def _mask_path(self, cam: int, frame_idx: int) -> Path:
        d = self._cam0_dir if cam == 0 else self._cam1_dir
        return d / f"{frame_idx:06d}.png"

    def _load_one(self, cam: int, frame_idx: int) -> Optional[np.ndarray]:
        p = self._mask_path(cam, frame_idx)
        if not p.exists():
            return None
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        return img    # may be None if cv2 fails to decode

    def __getitem__(
            self, frame_idx: int
    ) -> "tuple[Optional[np.ndarray], Optional[np.ndarray]]":
        """Return (mask_cam0, mask_cam1) for this frame.

        Either element is None if the SAM2 propagation didn't produce
        a mask for that camera/frame (lost track, before-prompt, etc.).
        Callers should fall back to bg-subtraction when either is None.
        """
        m0 = self._load_one(0, frame_idx)
        m1 = self._load_one(1, frame_idx)
        return m0, m1

    @classmethod
    def precompute(
            cls,
            sam2_video_tracker,                       # SAM2VideoTracker
            frames_cam0:        "Iterable[np.ndarray]",
            frames_cam1:        "Iterable[np.ndarray]",
            prompt0_xy:         "tuple[float, float]",
            prompt1_xy:         "tuple[float, float]",
            cache_dir:          "str | Path",
            prompt_frame_idx:   int = 0,
            label:              str = "animal",
    ) -> "SAM2MaskCache":
        """Run SAM2 video propagation on both cameras; cache masks to disk.

        Parameters
        ----------
        sam2_video_tracker  : SAM2VideoTracker instance with .available=True
        frames_cam0/cam1    : iterables of HxWxC (or HxW) uint8 frames
        prompt0_xy/prompt1_xy : (x, y) seed centroid in each camera at
                                ``prompt_frame_idx``. Typically from
                                bg-subtraction on the prompt frame.
        cache_dir           : where to write masks; will be created
        prompt_frame_idx    : which frame the seed pixels refer to
                              (default 0)
        label               : pose label name (cosmetic; saved into
                              the per-camera SAM2 state)

        Returns
        -------
        SAM2MaskCache pointing at cache_dir. Masks are written as 8-bit
        single-channel PNGs (0 = background, 255 = foreground).

        Notes
        -----
        - If sam2_video_tracker.available is False (sam2 not installed),
          the method raises ImportError early — there is nothing to
          propagate, so falling through silently would just mean an
          empty cache and silent loss of intent.
        - Each camera is propagated independently. The masks themselves
          are not cross-validated across cameras; epipolar consistency
          is enforced downstream by EpipolarMatcher.match() as usual.
        - This is a single-pass precompute; for chunked / streaming
          propagation across very long sessions, call precompute()
          per chunk on disjoint frame ranges.
        """
        if not getattr(sam2_video_tracker, "available", False):
            raise ImportError(
                "SAM2VideoTracker is not available (sam2 package not "
                "installed). Cannot precompute masks.")

        cache_dir = Path(cache_dir)
        (cache_dir / "cam0").mkdir(parents=True, exist_ok=True)
        (cache_dir / "cam1").mkdir(parents=True, exist_ok=True)

        # Materialise frames as JPEGs in a tmp dir per cam (SAM2's
        # init_state takes a directory of JPEGs).
        import tempfile
        for cam, frames, prompt in (
            (0, frames_cam0, prompt0_xy),
            (1, frames_cam1, prompt1_xy),
        ):
            with tempfile.TemporaryDirectory(prefix=f"sam2_cam{cam}_") as td:
                td_path = Path(td)
                # Stage frames as JPEGs
                n = 0
                for i, frame in enumerate(frames):
                    if frame is None:
                        continue
                    img = frame
                    if img.ndim == 2:
                        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                    cv2.imwrite(str(td_path / f"{i:06d}.jpg"), img)
                    n += 1
                if n == 0:
                    continue

                # Init SAM2 state on this dir, add the prompt, propagate
                state = sam2_video_tracker.init_state(str(td_path))
                sam2_video_tracker.add_point_prompt(
                    state,
                    frame_idx=prompt_frame_idx,
                    x=float(prompt[0]), y=float(prompt[1]),
                    label=label)
                cam_out_dir = (cache_dir / f"cam{cam}")
                for frame_idx, mask in sam2_video_tracker.propagate(state):
                    if mask is None:
                        continue
                    m = (mask > 0.5).astype(np.uint8) * 255
                    cv2.imwrite(
                        str(cam_out_dir / f"{frame_idx:06d}.png"),
                        m,
                        [cv2.IMWRITE_PNG_COMPRESSION, 5])

        return cls(cache_dir)
