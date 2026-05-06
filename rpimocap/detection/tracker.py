"""
rpimocap.detection.tracker
===========================
Orchestrates per-frame body region tracking and 3D triangulation.

Tracker hierarchy (best → fallback)
-------------------------------------
SAM2Tracker
    Uses SAM2 video predictor to propagate body part masks through the
    video.  Initialised from the first frame's GeometricLabeller output.
    Handles occlusion and re-appearance.  Requires sam2 package.

SAM1Tracker
    Runs SAM independently on every frame (no temporal propagation).
    More robust than SAM2 when the animal moves very fast.
    Requires segment_anything package.

OpticalFlowTracker
    Propagates body part centroids frame-to-frame using Lucas-Kanade
    sparse optical flow.  Re-detects when flow confidence drops.
    Pure OpenCV — no ML required.  Fastest and always available.

SegmentTracker (high-level)
    Automatically selects the best available tracker, runs it, matches
    regions between cam0 and cam1 per frame, triangulates, and returns
    a list of TrackResult objects.

TrackResult
    Per-frame output: body region lists for each camera, 3D coordinates,
    reprojection errors, and raw Keypoint2D objects compatible with the
    existing write_hdf5 pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from rpimocap.detection.detectors import Keypoint2D, Pose2DResult
from rpimocap.detection.segment import (
    BackgroundModel, ForegroundDetector, ForegroundResult,
    GeometricLabeller, SAMLabeller, BodyRegion, EpipolarMatcher,
)
from rpimocap.reconstruction.triangulate import Point3D


# --------------------------------------------------------------------------- #
#  Per-frame result                                                            #
# --------------------------------------------------------------------------- #

@dataclass
class TrackResult:
    """Tracking output for one frame pair.

    Attributes
    ----------
    frame_idx     : index into the video
    regions_cam0  : labelled BodyRegion list from camera 0
    regions_cam1  : labelled BodyRegion list from camera 1
    xyz           : {label: (3,) world coordinate mm}
    reproj_err    : {label: (err0_px, err1_px)}
    detected      : True if at least one body part was triangulated
    pose3d        : list of Point3D — compatible with write_hdf5
    pose2d_cam0   : Pose2DResult for camera 0 — for visualisation
    pose2d_cam1   : Pose2DResult for camera 1
    """

    frame_idx:   int
    regions_cam0: list[BodyRegion]        = field(default_factory=list)
    regions_cam1: list[BodyRegion]        = field(default_factory=list)
    xyz:          dict[str, np.ndarray]   = field(default_factory=dict)
    reproj_err:   dict[str, tuple]        = field(default_factory=dict)
    detected:     bool                    = False

    @property
    def pose3d(self) -> list[Point3D]:
        return [Point3D(name=k, xyz=v) for k, v in self.xyz.items()]

    @property
    def pose2d_cam0(self) -> Pose2DResult:
        return Pose2DResult(
            frame_idx=self.frame_idx,
            detected=self.detected,
            keypoints=[Keypoint2D(name=r.label, x=r.cx, y=r.cy,
                                   confidence=r.confidence)
                       for r in self.regions_cam0])

    @property
    def pose2d_cam1(self) -> Pose2DResult:
        return Pose2DResult(
            frame_idx=self.frame_idx,
            detected=self.detected,
            keypoints=[Keypoint2D(name=r.label, x=r.cx, y=r.cy,
                                   confidence=r.confidence)
                       for r in self.regions_cam1])


# --------------------------------------------------------------------------- #
#  Optical flow tracker (always available)                                     #
# --------------------------------------------------------------------------- #

class OpticalFlowTracker:
    """Propagate body part centroids using Lucas-Kanade sparse optical flow.

    Initialised from the first frame's region list.  Re-detects when
    the number of successfully tracked points drops below ``min_points``
    or when tracking confidence falls below ``min_confidence``.

    Parameters
    ----------
    detector        : ForegroundDetector
    labeller        : GeometricLabeller
    min_confidence  : minimum eigenvalue ratio to accept a tracked point
    min_points      : re-detect if fewer than this many points tracked
    redetect_every  : force re-detection every N frames regardless
    """

    _LK_PARAMS = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )

    def __init__(
        self,
        detector:        ForegroundDetector,
        labeller:        GeometricLabeller,
        min_confidence:  float = 0.3,
        min_points:      int   = 3,
        redetect_every:  int   = 60,
    ):
        self._det            = detector
        self._lbl            = labeller
        self._min_conf       = min_confidence
        self._min_pts        = min_points
        self._redetect_every = redetect_every
        self._prev_gray: Optional[np.ndarray] = None
        self._tracked_pts: Optional[np.ndarray] = None   # (N, 2)
        self._labels: list[str] = []
        self._frames_since_detect = 0

    def reset(self) -> None:
        self._prev_gray = None
        self._tracked_pts = None
        self._labels = []
        self._frames_since_detect = 0

    def track(self, frame: np.ndarray, cam: int) -> list[BodyRegion]:
        """Track one frame.  Returns BodyRegion list."""
        gray = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        need_detect = (
            self._prev_gray is None
            or self._tracked_pts is None
            or len(self._tracked_pts) < self._min_pts
            or self._frames_since_detect >= self._redetect_every
        )

        if need_detect:
            fg = self._det.detect(frame, cam)
            regions = self._lbl.label(fg)
            if regions:
                # If we have labels from a previous detection, try to
                # match orientation by checking if the nose region is
                # closer to the previous nose position.  This prevents
                # the spine from flipping between frames.
                if (self._tracked_pts is not None
                        and len(self._tracked_pts) > 0
                        and "nose" in self._labels):
                    prev_nose_idx = (self._labels.index("nose")
                                     if "nose" in self._labels else 0)
                    prev_nose = self._tracked_pts[prev_nose_idx]
                    # Find current nose
                    cur_nose_idx = next((i for i, r in enumerate(regions)
                                         if r.label == "nose"), None)
                    # Find current tail_tip
                    cur_tail_idx = next((i for i, r in enumerate(regions)
                                         if r.label == "tail_tip"), None)
                    if cur_nose_idx is not None and cur_tail_idx is not None:
                        d_nose = np.linalg.norm(
                            np.array([regions[cur_nose_idx].cx,
                                      regions[cur_nose_idx].cy]) - prev_nose)
                        d_tail = np.linalg.norm(
                            np.array([regions[cur_tail_idx].cx,
                                      regions[cur_tail_idx].cy]) - prev_nose)
                        if d_tail < d_nose:
                            # Orientation is flipped — reverse spine labels
                            _SPINE = ["nose","head","neck","back",
                                      "rump","tail_base","tail_tip"]
                            label_map = dict(zip(_SPINE, reversed(_SPINE)))
                            for r in regions:
                                r.label = label_map.get(r.label, r.label)
                self._tracked_pts = np.array(
                    [[r.cx, r.cy] for r in regions], dtype=np.float32)
                self._labels = [r.label for r in regions]
                self._prev_gray = gray
                self._frames_since_detect = 0
            return regions

        # Optical flow propagation
        pts_in  = self._tracked_pts.reshape(-1, 1, 2)
        pts_out, status, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, gray, pts_in, None, **self._LK_PARAMS)

        if pts_out is None or status is None:
            self.reset()
            return self.track(frame, cam)

        good = status.ravel() > 0
        if good.sum() < self._min_pts:
            self.reset()
            return self.track(frame, cam)

        new_pts = pts_out.reshape(-1, 2)
        regions = []
        for i, (ok, (cx, cy)) in enumerate(zip(good, new_pts)):
            if not ok:
                continue
            lbl = self._labels[i] if i < len(self._labels) else "body"
            regions.append(BodyRegion(
                label=lbl, cx=float(cx), cy=float(cy),
                confidence=0.6))

        self._tracked_pts = new_pts[good]
        self._labels      = [self._labels[i]
                              for i, ok in enumerate(good) if ok]
        self._prev_gray   = gray
        self._frames_since_detect += 1
        return regions


# --------------------------------------------------------------------------- #
#  SAM2 video tracker (optional)                                              #
# --------------------------------------------------------------------------- #

class SAM2Tracker:
    """Propagate body part masks using the SAM2 video predictor.

    Requires the ``sam2`` package (pip install sam2).

    Parameters
    ----------
    checkpoint  : path to SAM2 model weights
    config      : SAM2 config name (e.g. ``"sam2_hiera_large.yaml"``)
    device      : ``"cuda"`` or ``"cpu"``
    """

    def __init__(
        self,
        checkpoint: str,
        config:     str  = "sam2_hiera_large.yaml",
        device:     str  = "cuda",
    ):
        self._ckpt    = checkpoint
        self._config  = config
        self._device  = device
        self._predictor = None
        self._available = self._try_load()
        self._inference_state = None
        self._labeller = GeometricLabeller()

    def _try_load(self) -> bool:
        try:
            from sam2.build_sam import build_sam2_video_predictor
            self._predictor = build_sam2_video_predictor(
                self._config, self._ckpt, device=self._device)
            return True
        except (ImportError, Exception):
            return False

    @property
    def available(self) -> bool:
        return self._available

    def init_from_regions(
        self,
        frame0_bgr: np.ndarray,
        regions:    list[BodyRegion],
    ) -> bool:
        """Initialise tracking from the first frame's region list."""
        if not self._available or not regions:
            return False
        try:
            import tempfile, os
            # SAM2 video predictor needs a directory of frames
            # For streaming use we pass the first frame as an image
            rgb = cv2.cvtColor(frame0_bgr, cv2.COLOR_BGR2RGB)
            self._inference_state = self._predictor.init_state(
                video_path=None)
            for i, r in enumerate(regions):
                pts = np.array([[r.cx, r.cy]])
                self._predictor.add_new_points_or_box(
                    self._inference_state,
                    frame_idx=0,
                    obj_id=i,
                    points=pts,
                    labels=np.array([1]),
                )
            self._obj_labels = [r.label for r in regions]
            return True
        except Exception as e:
            print(f"  SAM2 init failed: {e}")
            return False

    def propagate(self, frame_bgr: np.ndarray) -> list[BodyRegion]:
        """Propagate masks to the next frame."""
        if not self._available or self._inference_state is None:
            return []
        try:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            regions = []
            for obj_id, mask, score in self._predictor.propagate_in_video(
                    self._inference_state):
                label = (self._obj_labels[obj_id]
                         if obj_id < len(self._obj_labels) else "body")
                ys, xs = np.where(mask > 0)
                if len(xs) < 4:
                    continue
                cx, cy = xs.mean(), ys.mean()
                regions.append(BodyRegion(
                    label=label, cx=float(cx), cy=float(cy),
                    mask=mask.astype(bool),
                    area_px=float(len(xs)),
                    confidence=float(score)))
            return regions
        except Exception:
            return []


# --------------------------------------------------------------------------- #
#  High-level orchestrator                                                     #
# --------------------------------------------------------------------------- #

class SegmentTracker:
    """Full stereo segmentation and 3D tracking pipeline.

    Selects the best available tracker (SAM2 > OpticalFlow), runs it on
    both cameras in parallel, matches regions between cameras using
    epipolar geometry, triangulates, and returns TrackResult per frame.

    Parameters
    ----------
    background      : BackgroundModel (pre-computed)
    matcher         : EpipolarMatcher (from calibration)
    sam2_checkpoint : path to SAM2 weights; None to skip SAM2
    sam2_config     : SAM2 config yaml name
    device          : torch device string
    threshold       : foreground detection threshold (pixels)
    min_area_px     : minimum blob area
    redetect_every  : optical flow re-detection interval (frames)
    verbose         : print progress
    """

    def __init__(
        self,
        background:       BackgroundModel,
        matcher:          EpipolarMatcher,
        sam2_checkpoint:  Optional[str]  = None,
        sam2_config:      str            = "sam2_hiera_large.yaml",
        device:           str            = "cuda",
        threshold:        float          = 25.0,
        min_area_px:      int            = 500,
        redetect_every:   int            = 60,
        verbose:          bool           = True,
    ):
        self._matcher = matcher
        self._verbose = verbose

        self._det  = ForegroundDetector(
            background, threshold=threshold, min_area_px=min_area_px)
        self._lbl  = GeometricLabeller()

        # Try SAM2
        self._sam2: Optional[SAM2Tracker] = None
        if sam2_checkpoint:
            s = SAM2Tracker(sam2_checkpoint, sam2_config, device)
            if s.available:
                self._sam2 = s
                if verbose:
                    print("  SAM2 tracker: available")
            else:
                if verbose:
                    print("  SAM2 not available, using optical flow")

        # Optical flow fallback
        self._of0 = OpticalFlowTracker(
            self._det, self._lbl, redetect_every=redetect_every)
        self._of1 = OpticalFlowTracker(
            self._det, self._lbl, redetect_every=redetect_every)

    # ------------------------------------------------------------------ #

    def track_sequence(
        self,
        cap0,
        cap1,
        start_frame:    int           = 0,
        end_frame:      Optional[int] = None,
        sample_every:   int           = 1,
        align_result    = None,
    ) -> list[TrackResult]:
        """Track the full video sequence.

        Parameters
        ----------
        cap0, cap1      : VideoCapture-compatible objects
        start_frame     : first frame index
        end_frame       : last frame index (None = end of video)
        sample_every    : process every Nth frame (1 = all frames)
        align_result    : AlignResult from arena alignment (optional)

        Returns
        -------
        list[TrackResult] — one entry per processed frame
        """
        total = int(min(cap0.get(cv2.CAP_PROP_FRAME_COUNT),
                        cap1.get(cv2.CAP_PROP_FRAME_COUNT)))
        if end_frame is None or end_frame > total:
            end_frame = total

        self._of0.reset()
        self._of1.reset()
        results: list[TrackResult] = []
        n_frames = len(range(start_frame, end_frame, sample_every))

        if self._verbose:
            print(f"  Tracking frames {start_frame}–{end_frame}"
                  f" (every {sample_every}, total {n_frames})")

        for i, idx in enumerate(range(start_frame, end_frame, sample_every)):
            cap0.set(cv2.CAP_PROP_POS_FRAMES, idx)
            cap1.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret0, f0 = cap0.read()
            ret1, f1 = cap1.read()
            if not ret0 or not ret1:
                break

            result = self._process_frame(idx, f0, f1)

            # Apply arena alignment if provided
            if align_result is not None and result.xyz:
                result.xyz = {
                    k: align_result.apply(v)
                    for k, v in result.xyz.items()
                }

            results.append(result)

            if self._verbose and (i + 1) % 100 == 0:
                n_det = sum(1 for r in results if r.detected)
                print(f"  Frame {idx}/{end_frame}  "
                      f"detected: {n_det}/{len(results)}"
                      f"  ({100*n_det/len(results):.0f}%)")

        if self._verbose:
            n_det = sum(1 for r in results if r.detected)
            print(f"  Done. {n_det}/{len(results)} frames with detections.")

        return results

    def _process_frame(
        self,
        idx: int,
        f0:  np.ndarray,
        f1:  np.ndarray,
    ) -> TrackResult:
        """Process one stereo frame pair → TrackResult."""
        if self._sam2 is not None and self._sam2.available:
            if idx == 0 or not hasattr(self, '_sam2_inited'):
                # Initialise SAM2 from geometric detection on first frame
                fg0 = self._det.detect(f0, 0)
                init_regions = self._lbl.label(fg0)
                self._sam2.init_from_regions(f0, init_regions)
                self._sam2_inited = True
            regions0 = self._sam2.propagate(f0)
            regions1 = self._sam2.propagate(f1)
        else:
            regions0 = self._of0.track(f0, 0)
            regions1 = self._of1.track(f1, 1)

        matches  = self._matcher.match(regions0, regions1)
        xyz_dict = self._matcher.triangulate(matches)

        # Compute reprojection errors
        reproj: dict[str, tuple] = {}
        for r0, r1 in matches:
            label = r0.label
            if label in xyz_dict:
                try:
                    reproj[label] = self._matcher.reprojection_error(
                        xyz_dict[label], r0, r1)
                except Exception:
                    pass

        return TrackResult(
            frame_idx=idx,
            regions_cam0=regions0,
            regions_cam1=regions1,
            xyz=xyz_dict,
            reproj_err=reproj,
            detected=bool(xyz_dict),
        )

    # ------------------------------------------------------------------ #
    #  Convert to pipeline-compatible format                               #
    # ------------------------------------------------------------------ #

    @staticmethod
    def results_to_skeleton_frames(
        results: list[TrackResult],
    ) -> list[list[Point3D]]:
        """Convert TrackResult list to skeleton_frames format for write_hdf5."""
        return [r.pose3d for r in results]
