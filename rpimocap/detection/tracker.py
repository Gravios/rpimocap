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
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from rpimocap.detection.detectors import Keypoint2D, Pose2DResult
from rpimocap.detection.segment import (
    BackgroundModel,
    BodyRegion,
    EpipolarMatcher,
    ForegroundDetector,
    GeometricLabeller,
)
from rpimocap.reconstruction.triangulate import Point3D


def _project_xyz_to_pixel(
        P: np.ndarray, xyz: np.ndarray
) -> "tuple[float, float] | None":
    """Project a world XYZ point to pixel coordinates via a 3×4 P matrix.

    Returns None if the point is behind the camera or on the principal
    plane (degenerate projection). Used by the online-Kalman wiring to
    seed the next frame's epipolar prior from the predicted XYZ.
    """
    P = np.asarray(P, dtype=np.float64)
    xyz = np.asarray(xyz, dtype=np.float64).reshape(3)
    h = P @ np.array([xyz[0], xyz[1], xyz[2], 1.0])
    if not np.isfinite(h[2]) or abs(h[2]) < 1e-9 or h[2] < 0:
        return None
    return float(h[0] / h[2]), float(h[1] / h[2])



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
        self._last_fg        = None   # ForegroundResult from last detect()
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
            self._last_fg = fg          # expose for post-match re-hulling
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

class SAM2VideoTracker:
    """SAM2 video-propagation tracker.

    Unlike ``SAM2Tracker`` (which runs the SAM2 *image* predictor on
    every frame), this class uses the SAM2 *video* predictor: a single
    initial annotation (typically the optical-flow-seeded centroid in
    frame 0) is propagated forward by SAM2's learned appearance-and-
    motion model. The result is a per-frame mask that is dramatically
    more robust to lighting changes, cable occlusion, and bedding
    texture than a Gabor + morphology pipeline.

    Performance: at 1080p on a 5070 Ti, propagation runs at ~15 fps —
    fast enough for a 25 fps recording to finish in roughly real time
    on the GPU portion alone. For an 11995-frame session that's about
    a 13-minute pass.

    Workflow
    --------
    1. Build the predictor:

           tracker = SAM2VideoTracker(checkpoint, config, device)
           if not tracker.available:
               # fall back to OpticalFlowTracker / SAM2Tracker

    2. Initialise from a frame index and a list of point prompts (one
       (x, y, label) per body part):

           tracker.init_state(frames_iter, prompts={
               "animal": [(cx, cy)],         # positive prompt for body
           })

    3. Iterate per-frame masks:

           for frame_idx, masks in tracker.propagate():
               # masks: dict {label: (H, W) bool array}
               ...

    Until SAM2 video is actually wired up (the import surface in the
    sam2 package is moving rapidly between versions and is not always
    available in headless setups), this class probes for the API and
    cleanly reports ``available=False`` if the symbols are missing,
    falling back to the SAM2 image predictor path.
    """

    def __init__(
        self,
        checkpoint: str,
        config:     str  = "sam2.1_hiera_l.yaml",
        device:     str  = "cuda",
        chunk_size: int  = 256,
    ):
        self._ckpt      = checkpoint
        self._config    = config
        self._device    = device
        self._chunk     = int(chunk_size)
        self._predictor = None
        self._state     = None
        self._prompts: dict = {}
        self._available = self._try_load()

    # ------------------------------------------------------------------ #

    def _try_load(self) -> bool:
        """Try to import the SAM2 video predictor.

        Several sam2 releases expose different builder symbols
        (``build_sam2_video_predictor`` in mainline, ``SAM2VideoPredictor``
        in older forks). We try the canonical name first and fall back.
        """
        try:
            try:
                from sam2.build_sam import build_sam2_video_predictor as build
            except ImportError:
                # Older API
                from sam2.sam2_video_predictor import SAM2VideoPredictor as build  # type: ignore
            self._predictor = build(self._config, self._ckpt,
                                    device=self._device)
            print(f"  SAM2-video loaded: {Path(self._ckpt).name}  "
                  f"({self._config})  device={self._device}")
            return True
        except ImportError:
            print("  SAM2 video predictor not available "
                  "(install/upgrade sam2 to enable propagation)")
            return False
        except FileNotFoundError:
            print(f"  SAM2 checkpoint not found: {self._ckpt}")
            return False
        except Exception as e:
            print(f"  SAM2-video load failed: {e}")
            return False

    @property
    def available(self) -> bool:
        return self._available

    # ------------------------------------------------------------------ #
    #  Initialisation                                                      #
    # ------------------------------------------------------------------ #

    def init_state(
        self,
        frames: "list[np.ndarray] | str",
        prompts: dict,
    ) -> None:
        """Initialise the video predictor on a clip and seed prompts.

        Parameters
        ----------
        frames  : either a directory path containing JPEG frames (the
                  format SAM2-video expects natively) or an in-memory
                  list of BGR frames that this method will materialise
                  to a temp directory.
        prompts : dict mapping label → list of (x, y) positive prompts
                  in frame 0. Each label becomes a tracked object.
        """
        if not self._available:
            raise RuntimeError("SAM2VideoTracker is not available")
        # Materialise in-memory frames to disk if needed
        if isinstance(frames, list):
            import tempfile
            tmp = Path(tempfile.mkdtemp(prefix="sam2_video_"))
            for i, f in enumerate(frames):
                cv2.imwrite(str(tmp / f"{i:06d}.jpg"), f)
            frames_dir = str(tmp)
        else:
            frames_dir = str(frames)
        self._state = self._predictor.init_state(video_path=frames_dir)
        self._prompts = dict(prompts)
        for obj_id, (label, pts) in enumerate(self._prompts.items()):
            pts_arr = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
            labels  = np.ones(len(pts_arr), dtype=np.int32)
            self._predictor.add_new_points_or_box(
                inference_state=self._state,
                frame_idx=0, obj_id=obj_id,
                points=pts_arr, labels=labels)

    # ------------------------------------------------------------------ #
    #  Propagation                                                         #
    # ------------------------------------------------------------------ #

    def propagate(self):
        """Yield (frame_idx, {label: mask}) for each propagated frame."""
        if not self._available:
            raise RuntimeError("SAM2VideoTracker is not available")
        if self._state is None:
            raise RuntimeError("Call init_state() before propagate()")
        obj_id_to_label = {i: lbl for i, lbl in enumerate(self._prompts)}
        for fidx, obj_ids, mask_logits in self._predictor.propagate_in_video(
                self._state):
            masks: dict = {}
            for i, obj_id in enumerate(obj_ids):
                label = obj_id_to_label.get(int(obj_id), str(obj_id))
                masks[label] = (mask_logits[i] > 0.0).cpu().numpy().astype(bool)
            yield int(fidx), masks


class SAM2Tracker:
    """Per-frame body part segmentation using SAM2 image predictor.

    Uses the SAM2 image predictor (not the video predictor) so it works
    directly on streaming TIFF frames without dumping to disk.  Optical
    flow provides approximate centroids as prompts; SAM2 produces
    high-quality binary masks for each body part.

    Requires the ``sam2`` package::

        pip install sam2
        rpimocap-download-models          # downloads weights

    Parameters
    ----------
    checkpoint  : path to SAM2 .pt weights file
    config      : SAM2 yaml config name (e.g. ``"sam2.1_hiera_l.yaml"``)
    device      : ``"cuda"`` or ``"cpu"`` (cuda strongly recommended)
    multimask   : return multiple mask candidates per prompt and pick best
    """

    def __init__(
        self,
        checkpoint: str,
        config:     str  = "sam2.1_hiera_l.yaml",
        device:     str  = "cuda",
        multimask:  bool = True,
    ):
        self._ckpt      = checkpoint
        self._config    = config
        self._device    = device
        self._multimask = multimask
        self._predictor = None
        self._available = self._try_load()
        self._labeller  = GeometricLabeller()

    def _try_load(self) -> bool:
        """Try to import and load the SAM2 image predictor."""
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            model = build_sam2(self._config, self._ckpt,
                               device=self._device)
            self._predictor = SAM2ImagePredictor(model)
            print(f"  SAM2 loaded: {Path(self._ckpt).name}  "
                  f"({self._config})  device={self._device}")
            return True
        except ImportError:
            print("  SAM2 not installed — run: pip install sam2")
            return False
        except FileNotFoundError:
            print(f"  SAM2 checkpoint not found: {self._ckpt}")
            print("  Run: rpimocap-download-models")
            return False
        except Exception as e:
            print(f"  SAM2 load failed: {e}")
            return False

    @property
    def available(self) -> bool:
        return self._available

    def segment(
        self,
        frame_bgr:   np.ndarray,
        hint_regions: list[BodyRegion],
    ) -> list[BodyRegion]:
        """Segment body parts in one frame using hint centroids as prompts.

        Parameters
        ----------
        frame_bgr    : BGR uint8 video frame
        hint_regions : approximate body part locations (from optical flow
                       or geometric labeller) used as SAM2 point prompts

        Returns
        -------
        list[BodyRegion] with SAM2 mask quality, same labels as hints
        """
        if not self._available or not hint_regions:
            return hint_regions

        try:
            import torch
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            with torch.inference_mode():
                self._predictor.set_image(rgb)

                results = []
                for hint in hint_regions:
                    pts    = np.array([[hint.cx, hint.cy]], dtype=np.float32)
                    labels = np.array([1], dtype=np.int32)

                    masks, scores, _ = self._predictor.predict(
                        point_coords=pts,
                        point_labels=labels,
                        multimask_output=self._multimask,
                    )

                    # Pick the mask with the highest score
                    best  = int(np.argmax(scores))
                    mask  = masks[best].astype(bool)
                    score = float(scores[best])

                    ys, xs = np.where(mask)
                    if len(xs) < 4:
                        # SAM2 found nothing — keep the hint centroid
                        results.append(hint)
                        continue

                    results.append(BodyRegion(
                        label=hint.label,
                        cx=float(xs.mean()),
                        cy=float(ys.mean()),
                        mask=mask,
                        bbox=(int(xs.min()), int(ys.min()),
                              int(xs.max()-xs.min()), int(ys.max()-ys.min())),
                        area_px=float(len(xs)),
                        confidence=score,
                        orientation=hint.orientation))

            return results

        except Exception:
            # Any SAM2 error → return hints unchanged
            return hint_regions


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
        background:        BackgroundModel,
        matcher:           EpipolarMatcher,
        sam2_checkpoint:   Optional[str]  = None,
        sam2_config:       str            = "sam2.1_hiera_l.yaml",
        device:            str            = "cuda",
        threshold:         float          = 25.0,
        min_area_px:       int            = 500,
        max_area_px:       "Optional[int]" = None,
        min_solidity:      float = 0.0,
        morph_k:           int            = 7,
        redetect_every:    int            = 60,
        clahe:             bool           = False,
        clahe_clip:        float          = 2.0,
        clahe_tile:        int            = 8,
        use_green_channel: bool           = False,
        bilateral:         bool           = False,
        bilateral_d:       int            = 9,
        bilateral_sigma:   float          = 50.0,
        centroid_only:     bool           = False,
        verbose:           bool           = True,
        roi_mask:          "Optional[np.ndarray]" = None,
        wall_weight:       "Optional[np.ndarray]" = None,
        cable_erosion_px:  int = 0,
        texture_suppress:  bool  = False,
        texture_lambdas:   tuple = (8, 12, 16),
        texture_alpha:     float = 0.7,
        texture_n_orient:  int   = 4,
        polarity:          str   = "either",
        bg_adapt_alpha:    "Optional[float]" = None,
        bg_adapt_dilate_px: int  = 25,
        use_trajectory_prior: bool = False,
        trajectory_prior_lambda: float = 0.05,
        flat_field_cam0:  "Optional[np.ndarray]" = None,
        flat_field_cam1:  "Optional[np.ndarray]" = None,
        body_length_mm:   float = 0.0,
        body_width_mm:    float = 70.0,
        body_z_mm:        float = 0.0,
        P0:               "Optional[np.ndarray]" = None,
        P1:               "Optional[np.ndarray]" = None,
        gabor_refine:     bool  = False,
        canny_low:        float = 30.0,
        canny_high:       float = 90.0,
        kalman_online:        "Optional[object]" = None,
        rearing_classifier:   "Optional[object]" = None,
        rearing_track_name:   str = "animal",
        fps:                  float = 25.0,
        sam2_mask_cache:      "Optional[object]" = None,
    ):
        self._matcher        = matcher
        self._verbose        = verbose
        self._cable_erosion  = cable_erosion_px
        self._background     = background
        self._bg_adapt_alpha = bg_adapt_alpha
        self._bg_adapt_dilate_px = int(bg_adapt_dilate_px)
        # Trajectory-constrained selection: remember the last confirmed
        # blob centroid in each camera so the global epipolar selector
        # can prefer candidates near the previous detection.
        self._prior_cx0:    "tuple[float, float] | None" = None
        self._prior_cx1:    "tuple[float, float] | None" = None
        self._prior_lambda: float = float(trajectory_prior_lambda)
        self._use_prior:    bool  = bool(use_trajectory_prior)
        self._flat0 = flat_field_cam0
        self._flat1 = flat_field_cam1
        self._body_length_mm = float(body_length_mm)
        self._body_width_mm  = float(body_width_mm)
        self._body_z_mm      = float(body_z_mm)
        self._P0 = P0
        self._P1 = P1
        self._gabor_refine = bool(gabor_refine)
        self._canny_low    = float(canny_low)
        self._canny_high   = float(canny_high)
        # Online Kalman filter: feeds next-frame trajectory prior and
        # drives the rearing classifier. The triangulated XYZ is stepped
        # in after every successful match; the predicted XYZ for the
        # NEXT frame is back-projected to pixel coords in both cameras
        # and used to seed the EpipolarMatcher prior.
        self._kalman_online      = kalman_online
        self._rearing_classifier = rearing_classifier
        self._rearing_track_name = str(rearing_track_name)
        # Current posture (None until first Kalman update); when reared,
        # overrides body_length_mm / body_width_mm on the next frame.
        self._current_posture = None
        self._fps             = float(fps)
        # SAM2 video propagation cache. When present, _process_frame
        # prefers cached masks over bg-subtraction. The cache is built
        # by a pre-pass (see SAM2MaskCache.precompute) and indexed by
        # the frame counter maintained in track_sequence.
        self._sam2_mask_cache = sam2_mask_cache
        self._frame_idx       = 0    # set by track_sequence per iteration
        # Pipeline-step counters populated in _process_frame and inside
        # hull_centroid (which mutates the dict in place). Counters are
        # per-camera-refinement, not per-frame: cable_erosion_attempted
        # increments by 2 per frame when both cameras have valid blobs.
        # Reset at the start of every track_sequence call.
        self._step_stats: dict = {}

        self._det  = ForegroundDetector(
            background, threshold=threshold,
            min_area_px=min_area_px, max_area_px=max_area_px,
            min_solidity=min_solidity,
            morph_k=morph_k,
            clahe=clahe, clahe_clip=clahe_clip, clahe_tile=clahe_tile,
            use_green_channel=use_green_channel,
            bilateral=bilateral, bilateral_d=bilateral_d,
            bilateral_sigma=bilateral_sigma,
            roi_mask=roi_mask,
            wall_weight=wall_weight,
            texture_suppress=texture_suppress,
            texture_lambdas=texture_lambdas,
            texture_alpha=texture_alpha,
            texture_n_orient=texture_n_orient,
            polarity=polarity)
        self._lbl  = GeometricLabeller(centroid_only=centroid_only)

        # Try SAM2 (used as mask refiner on top of optical flow)
        self._sam2: Optional[SAM2Tracker] = None
        if sam2_checkpoint:
            s = SAM2Tracker(sam2_checkpoint, sam2_config, device)
            if s.available:
                self._sam2 = s
            # SAM2 logs its own status in _try_load

        # Optical flow fallback
        self._of0 = OpticalFlowTracker(
            self._det, self._lbl, redetect_every=redetect_every)
        self._of1 = OpticalFlowTracker(
            self._det, self._lbl, redetect_every=redetect_every)

    # ------------------------------------------------------------------ #

    @property
    def step_stats(self) -> dict:
        """Per-step diagnostic counters from the most recent track_sequence.

        Counter keys are stable; absent keys are treated as 0. Per-camera
        counters (cable_erosion_attempted, gabor_refine_attempted,
        anatomical_prior_attempted, etc.) sum to ~2 × frame-count when
        both cameras have valid blobs. Per-frame counters
        (frames_with_match, kalman_with_measurement, kalman_gap,
        rearing_frames, sam2_mask_hits, bg_adapt_updates) sum to ≤
        frame-count.
        """
        return dict(self._step_stats)

    # ------------------------------------------------------------------ #

    def track_sequence(
        self,
        cap0,
        cap1,
        start_frame:    int           = 0,
        end_frame:      Optional[int] = None,
        sample_every:   int           = 1,
        align_result    = None,
        bounds:         "np.ndarray | None" = None,
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
        self._prior_cx0 = None
        self._prior_cx1 = None
        # Reset online Kalman + posture so each sequence starts fresh.
        # Critical: a stale converged covariance from a previous sequence
        # would make the filter under-weight the first frames of the new
        # one. KalmanTracker3D.reset() restores x, P, and the
        # initialised flag to their construction-time state.
        if self._kalman_online is not None:
            self._kalman_online.reset()
        if self._rearing_classifier is not None:
            self._rearing_classifier.reset()
        self._current_posture = None
        self._step_stats = {}            # fresh counters per sequence
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

            result = self._process_frame(idx, f0, f1, bounds=bounds)

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
        idx:    int,
        f0:     np.ndarray,
        f1:     np.ndarray,
        bounds: "np.ndarray | None" = None,
    ) -> TrackResult:
        """Process one stereo frame pair → TrackResult.

        Pipeline with SAM2:
          1. Optical flow tracker gives approximate centroids (fast)
          2. SAM2 image predictor refines each centroid into a quality mask
          3. Mask centroid is used for triangulation

        Without SAM2:
          Optical flow only.
        """
        # ── Flat-field (NIR vignette) correction ────────────────────────
        # If a flat-field was supplied, divide-and-rescale each frame
        # before any further processing. The background was already
        # flat-fielded at construction time so background subtraction
        # remains symmetric.
        if self._flat0 is not None or self._flat1 is not None:
            from rpimocap.detection.vignette import apply_flat_field
            if self._flat0 is not None:
                f0 = apply_flat_field(f0, self._flat0, clip=True)
            if self._flat1 is not None:
                f1 = apply_flat_field(f1, self._flat1, clip=True)

        # ── SAM2 video-propagation mask (if cache available) ──────────
        # When a SAM2MaskCache is supplied, the per-frame mask comes
        # from SAM2 video propagation rather than from background
        # subtraction. Building a synthetic ForegroundResult from the
        # SAM2 mask and stuffing it into both OpticalFlowTrackers'
        # _last_fg slot keeps the downstream re-hull / labelling path
        # unchanged. We still call self._of{0,1}.track() so optical-
        # flow point tracking stays warm; .track() will re-detect from
        # bg-sub when its points die and that's fine — by then either
        # the SAM2 mask is still available (and overwrites _last_fg
        # again) or it isn't (in which case bg-sub is the desired
        # fallback).
        sam2_fg0 = sam2_fg1 = None
        if self._sam2_mask_cache is not None:
            from rpimocap.detection.sam2_mask_cache import foreground_result_from_mask
            m0, m1 = self._sam2_mask_cache[idx]
            if m0 is not None:
                sam2_fg0 = foreground_result_from_mask(m0, f0)
            if m1 is not None:
                sam2_fg1 = foreground_result_from_mask(m1, f1)

        # Step 1: optical flow → approximate centroids in both cameras
        regions0 = self._of0.track(f0, 0)
        regions1 = self._of1.track(f1, 1)

        # If SAM2 video masks are available for this frame, OVERRIDE the
        # optical-flow / bg-sub regions with regions derived from those
        # masks. The labeller produces BodyRegion list from the synthetic
        # ForegroundResult, and we stuff that result into the optical-
        # flow tracker's _last_fg slot so the downstream re-hull step
        # (which reads _last_fg) sees the SAM2 mask, not the stale bg-
        # sub one.
        if sam2_fg0 is not None:
            try:
                regions0 = self._lbl.label(sam2_fg0)
                self._of0._last_fg = sam2_fg0
            except Exception:
                pass    # fall back to optical-flow regions
        if sam2_fg1 is not None:
            try:
                regions1 = self._lbl.label(sam2_fg1)
                self._of1._last_fg = sam2_fg1
            except Exception:
                pass

        # Step 2: SAM2 mask refinement (if available)
        if self._sam2 is not None and self._sam2.available:
            if regions0:
                regions0 = self._sam2.segment(f0, regions0)
            if regions1:
                regions1 = self._sam2.segment(f1, regions1)

        if self._use_prior:
            matches = self._matcher.match(
                regions0, regions1,
                prior0=self._prior_cx0, prior1=self._prior_cx1,
                prior_lambda=self._prior_lambda)
        else:
            matches = self._matcher.match(regions0, regions1)

        # Re-hull: for each matched pair, replace the raw connected-
        # component centroid with the convex-hull centroid of the
        # selected blob.  The hull centroid is more stable because it
        # is not pulled toward bedding pixels at the blob boundary.
        fg0 = getattr(self._of0, '_last_fg', None)
        fg1 = getattr(self._of1, '_last_fg', None)
        refined = []
        # Posture-adapted body dimensions for hull_centroid. If the
        # rearing classifier flagged a rear, swap the horizontal body
        # ellipse for the vertical-posture prior so step 5 doesn't pull
        # the centroid toward a 180-mm-long horizontal body that isn't
        # there.
        _body_L = self._body_length_mm
        _body_W = self._body_width_mm
        if (self._current_posture is not None
                and getattr(self._current_posture, "reared", False)):
            _body_L = float(self._current_posture.body_length_mm)
            _body_W = float(self._current_posture.body_width_mm)

        # Frame-visible counters: things _process_frame can see without
        # peering into hull_centroid's internals.
        if sam2_fg0 is not None or sam2_fg1 is not None:
            self._step_stats["sam2_mask_hits"] = (
                self._step_stats.get("sam2_mask_hits", 0) + 1)
        if matches:
            self._step_stats["frames_with_match"] = (
                self._step_stats.get("frames_with_match", 0) + 1)

        for r0, r1 in matches:
            if fg0 is not None:
                hx0, hy0 = self._det.hull_centroid(
                    fg0, r0.cx, r0.cy,
                    cable_erosion_px=self._cable_erosion,
                    P=self._P0,
                    body_length_mm=_body_L,
                    body_width_mm=_body_W,
                    body_z_mm=self._body_z_mm,
                    gabor_refine=self._gabor_refine,
                    canny_low=self._canny_low,
                    canny_high=self._canny_high,
                    stats=self._step_stats)
                r0 = r0.__class__(
                    label=r0.label, cx=hx0, cy=hy0,
                    area_px=r0.area_px, confidence=r0.confidence,
                    mask=r0.mask)
            if fg1 is not None:
                hx1, hy1 = self._det.hull_centroid(
                    fg1, r1.cx, r1.cy,
                    cable_erosion_px=self._cable_erosion,
                    P=self._P1,
                    body_length_mm=_body_L,
                    body_width_mm=_body_W,
                    body_z_mm=self._body_z_mm,
                    gabor_refine=self._gabor_refine,
                    canny_low=self._canny_low,
                    canny_high=self._canny_high,
                    stats=self._step_stats)
                r1 = r1.__class__(
                    label=r1.label, cx=hx1, cy=hy1,
                    area_px=r1.area_px, confidence=r1.confidence,
                    mask=r1.mask)
            refined.append((r0, r1))
        matches = refined

        xyz_dict = self._matcher.triangulate(matches, bounds=bounds)

        # ── Online Kalman + rearing classification ───────────────────────
        # When configured, step the online Kalman with the triangulated XYZ
        # for our tracked label, then classify the resulting state for
        # posture. The NEXT frame's hull_centroid will see the posture-
        # adapted body dimensions, and the next epipolar match will get a
        # pixel prior derived from the Kalman prediction.
        kalman_pred_xyz = None
        if self._kalman_online is not None:
            z = xyz_dict.get(self._rearing_track_name) if xyz_dict else None
            if z is not None and not np.any(np.isnan(z)):
                self._kalman_online.step(np.asarray(z, dtype=np.float64))
                self._step_stats["kalman_with_measurement"] = (
                    self._step_stats.get("kalman_with_measurement", 0) + 1)
            else:
                self._kalman_online.step(None)
                self._step_stats["kalman_gap"] = (
                    self._step_stats.get("kalman_gap", 0) + 1)
            if getattr(self._kalman_online, "initialised", False):
                # x = [x, y, z, vx, vy, vz]
                kalman_pred_xyz = self._kalman_online.x[:3].copy()
                if self._rearing_classifier is not None:
                    self._current_posture = self._rearing_classifier.classify(
                        self._kalman_online.x)
                    if getattr(self._current_posture, "reared", False):
                        self._step_stats["rearing_frames"] = (
                            self._step_stats.get("rearing_frames", 0) + 1)

        # Update trajectory prior centroids from the (possibly refined)
        # matched pair OR from the back-projected Kalman prediction.
        # The Kalman path is preferred when available because (a) it
        # accounts for velocity and (b) keeps producing predictions
        # even during gap frames where no match exists.
        if self._use_prior:
            used_kalman_prior = False
            if (kalman_pred_xyz is not None
                    and self._P0 is not None and self._P1 is not None):
                p0 = _project_xyz_to_pixel(self._P0, kalman_pred_xyz)
                p1 = _project_xyz_to_pixel(self._P1, kalman_pred_xyz)
                if p0 is not None and p1 is not None:
                    self._prior_cx0 = p0
                    self._prior_cx1 = p1
                    used_kalman_prior = True
            if not used_kalman_prior and matches:
                r0_last, r1_last = matches[0]
                self._prior_cx0 = (r0_last.cx, r0_last.cy)
                self._prior_cx1 = (r1_last.cx, r1_last.cy)

        # ── Temporal background adaptation ──────────────────────────────
        # Only adapt on frames with a confirmed (epipolar-validated) match,
        # so that one spurious cam0 blob can't bake itself into the model.
        # The mask is the foreground mask dilated by `bg_adapt_dilate_px`
        # so that we don't bake the animal's own shadow / fur halo into
        # the new background.
        if (self._bg_adapt_alpha is not None
                and xyz_dict
                and fg0 is not None and fg1 is not None
                and getattr(fg0, "mask", None) is not None
                and getattr(fg1, "mask", None) is not None):
            try:
                import cv2 as _cv2
                k = max(1, 2 * self._bg_adapt_dilate_px + 1)
                kern = _cv2.getStructuringElement(_cv2.MORPH_ELLIPSE, (k, k))
                # fg0/fg1 are ForegroundResult objects (cached on the
                # OpticalFlowTracker as _last_fg); their .mask attribute
                # is the uint8 0/255 foreground mask. The previous code
                # called .astype() on the ForegroundResult itself, which
                # silently raised AttributeError every frame and was
                # swallowed by the broad except below — bg-adapt has
                # been a no-op since the feature shipped.
                m0 = _cv2.dilate(
                    fg0.mask.astype(np.uint8), kern).astype(bool)
                m1 = _cv2.dilate(
                    fg1.mask.astype(np.uint8), kern).astype(bool)
                self._background.update(
                    f0, f1, mask0=m0, mask1=m1,
                    alpha=self._bg_adapt_alpha)
                self._step_stats["bg_adapt_updates"] = (
                    self._step_stats.get("bg_adapt_updates", 0) + 1)
            except Exception:
                # Adaptation is best-effort; never let it abort tracking.
                pass

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
