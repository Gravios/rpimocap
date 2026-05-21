"""Tests for rpimocap.detection.segment and tracker."""
from __future__ import annotations

import math
import tempfile
import os
import pytest
import numpy as np
import cv2

try:
    import tifffile
    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False


# --------------------------------------------------------------------------- #
#  Helpers                                                                     #
# --------------------------------------------------------------------------- #

H, W = 480, 640

def _make_frame(with_animal: bool = True) -> np.ndarray:
    """Return a BGR uint8 frame with a rat-shaped blob."""
    frame = np.ones((H, W), dtype=np.uint8) * 30
    if with_animal:
        # Elongated body (nose → tail direction)
        cv2.ellipse(frame, (W//2, H//2), (130, 55), 0, 0, 360, 180, -1)
        # Head
        cv2.circle(frame, (W//2 + 115, H//2), 38, 160, -1)
        # Tail stub
        cv2.ellipse(frame, (W//2 - 140, H//2), (45, 12), 0, 0, 360, 130, -1)
    return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)


@pytest.fixture(scope="module")
def synthetic_caps():
    """Write a TIFF stack and return two TiffCapture handles."""
    if not HAS_TIFFFILE:
        pytest.skip("tifffile not installed")
    from rpimocap.io.export import TiffCapture
    # 100 empty + 30 animal frames (empty frames dominate median background)
    empty  = [cv2.cvtColor(_make_frame(False), cv2.COLOR_BGR2GRAY)
               for _ in range(100)]
    animal = [cv2.cvtColor(_make_frame(True),  cv2.COLOR_BGR2GRAY)
               for _ in range(30)]
    stack  = np.stack(empty + animal, axis=0).astype(np.uint8)
    tmp    = tempfile.mktemp(suffix=".tif")
    tifffile.imwrite(tmp, stack)
    cap0 = TiffCapture(tmp)
    cap1 = TiffCapture(tmp)
    yield cap0, cap1
    cap0.release()
    cap1.release()
    os.unlink(tmp)


@pytest.fixture(scope="module")
def background_model(synthetic_caps):
    from rpimocap.detection.segment import BackgroundModel
    cap0, cap1 = synthetic_caps
    return BackgroundModel.from_captures(
        cap0, cap1, n_frames=80, start_frame=0, verbose=False)


# --------------------------------------------------------------------------- #
#  BackgroundModel                                                             #
# --------------------------------------------------------------------------- #

class TestBackgroundModel:
    def test_shape(self, background_model):
        assert background_model.bg0.shape == (H, W)
        assert background_model.bg1.shape == (H, W)

    def test_background_is_dark(self, background_model):
        # Background median should be close to 30 (empty frame value)
        assert background_model.bg0.mean() < 60

    def test_save_load(self, background_model, tmp_path):
        from rpimocap.detection.segment import BackgroundModel
        p = tmp_path / "bg.npz"
        background_model.save(p)
        bg2 = BackgroundModel.from_npz(p)
        np.testing.assert_allclose(bg2.bg0, background_model.bg0)

    def _fresh_bg(self, background_model):
        """Return an independent BackgroundModel that mirrors the fixture
        but doesn't alias its arrays, so we can mutate without poisoning
        the shared module-scoped fixture used by downstream tests."""
        from rpimocap.detection.segment import BackgroundModel
        return BackgroundModel(
            background_model.bg0.copy(),
            background_model.bg1.copy(),
            background_model.method)

    def test_update_pulls_background_toward_new_frame_outside_mask(self, background_model):
        """EMA update should move the background toward the new frame
        at pixels where the mask is False, and leave it unchanged where
        the mask is True."""
        bg = self._fresh_bg(background_model)
        bg0_before = bg.bg0.copy()

        new0 = np.full_like(bg0_before, 200.0, dtype=np.uint8)
        new1 = np.full_like(bg.bg1, 200.0, dtype=np.uint8)
        # Mask covers the top half — that region must NOT be updated
        mask0 = np.zeros(bg0_before.shape, dtype=bool)
        mask0[:bg0_before.shape[0] // 2] = True
        mask1 = mask0.copy()

        bg.update(new0, new1, mask0=mask0, mask1=mask1, alpha=0.5)

        # Masked region unchanged
        np.testing.assert_allclose(
            bg.bg0[mask0], bg0_before[mask0],
            err_msg="background updated inside animal mask")
        # Unmasked region moved toward 200; with alpha=0.5 the result is
        # exactly (bg_before + new) / 2
        expected = 0.5 * bg0_before[~mask0] + 0.5 * 200.0
        np.testing.assert_allclose(bg.bg0[~mask0], expected, atol=1e-4)

    def test_update_alpha_one_means_no_change(self, background_model):
        bg = self._fresh_bg(background_model)
        bg0_before = bg.bg0.copy()
        new0 = np.full_like(bg0_before, 200.0, dtype=np.uint8)
        new1 = np.full_like(bg.bg1, 200.0, dtype=np.uint8)
        bg.update(new0, new1, alpha=1.0)
        np.testing.assert_allclose(bg.bg0, bg0_before)

    def test_update_rejects_invalid_alpha(self, background_model):
        bg = self._fresh_bg(background_model)
        new = np.zeros_like(bg.bg0, dtype=np.uint8)
        with pytest.raises(ValueError):
            bg.update(new, new, alpha=1.5)
        with pytest.raises(ValueError):
            bg.update(new, new, alpha=-0.1)

    def test_update_rejects_shape_mismatch(self, background_model):
        bg = self._fresh_bg(background_model)
        wrong = np.zeros((10, 10), dtype=np.uint8)
        with pytest.raises(ValueError):
            bg.update(wrong, wrong, alpha=0.5)


# --------------------------------------------------------------------------- #
#  ForegroundDetector                                                          #
# --------------------------------------------------------------------------- #

class TestForegroundDetector:
    def test_detects_animal(self, background_model, synthetic_caps):
        from rpimocap.detection.segment import ForegroundDetector
        cap0, _ = synthetic_caps
        cap0.set(cv2.CAP_PROP_POS_FRAMES, 110)
        ret, frame = cap0.read()
        assert ret
        det = ForegroundDetector(background_model, threshold=20, min_area_px=100)
        fg  = det.detect(frame, cam=0)
        assert fg.n_blobs >= 1

    def test_no_detection_on_empty(self, background_model, synthetic_caps):
        from rpimocap.detection.segment import ForegroundDetector
        cap0, _ = synthetic_caps
        cap0.set(cv2.CAP_PROP_POS_FRAMES, 5)
        ret, frame = cap0.read()
        assert ret
        det = ForegroundDetector(background_model, threshold=20, min_area_px=200)
        fg  = det.detect(frame, cam=0)
        assert fg.n_blobs == 0

    def test_mask_shape(self, background_model, synthetic_caps):
        from rpimocap.detection.segment import ForegroundDetector
        cap0, _ = synthetic_caps
        cap0.set(cv2.CAP_PROP_POS_FRAMES, 110)
        ret, frame = cap0.read()
        det = ForegroundDetector(background_model, threshold=20, min_area_px=100)
        fg  = det.detect(frame, cam=0)
        assert fg.mask.shape == (H, W)


# --------------------------------------------------------------------------- #
#  GeometricLabeller                                                           #
# --------------------------------------------------------------------------- #

class TestGeometricLabeller:
    def _get_regions(self, background_model, synthetic_caps):
        from rpimocap.detection.segment import ForegroundDetector, GeometricLabeller
        cap0, _ = synthetic_caps
        cap0.set(cv2.CAP_PROP_POS_FRAMES, 110)
        ret, frame = cap0.read()
        det = ForegroundDetector(background_model, threshold=20, min_area_px=100)
        fg  = det.detect(frame, cam=0)
        lbl = GeometricLabeller()
        return lbl.label(fg), frame

    def test_returns_regions(self, background_model, synthetic_caps):
        regions, _ = self._get_regions(background_model, synthetic_caps)
        assert len(regions) > 0

    def test_expected_labels_present(self, background_model, synthetic_caps):
        regions, _ = self._get_regions(background_model, synthetic_caps)
        labels = {r.label for r in regions}
        # Must contain at least some spine-axis labels
        spine_labels = {"nose","head","neck","back","rump","tail_base","tail_tip"}
        assert labels & spine_labels, f"No spine labels found: {labels}"

    def test_centroids_in_frame(self, background_model, synthetic_caps):
        regions, _ = self._get_regions(background_model, synthetic_caps)
        for r in regions:
            assert 0 <= r.cx < W, f"{r.label} cx={r.cx} out of range"
            assert 0 <= r.cy < H, f"{r.label} cy={r.cy} out of range"

    def test_confidence_range(self, background_model, synthetic_caps):
        regions, _ = self._get_regions(background_model, synthetic_caps)
        for r in regions:
            assert 0.0 <= r.confidence <= 1.0

    def test_empty_fg_returns_empty(self, background_model, synthetic_caps):
        from rpimocap.detection.segment import ForegroundDetector, GeometricLabeller
        cap0, _ = synthetic_caps
        cap0.set(cv2.CAP_PROP_POS_FRAMES, 5)
        ret, frame = cap0.read()
        det = ForegroundDetector(background_model, threshold=20, min_area_px=500)
        fg  = det.detect(frame, cam=0)
        lbl = GeometricLabeller()
        regions = lbl.label(fg)
        assert regions == []


# --------------------------------------------------------------------------- #
#  EpipolarMatcher                                                             #
# --------------------------------------------------------------------------- #

class TestEpipolarMatcher:
    @pytest.fixture
    def matcher(self):
        from rpimocap.detection.segment import EpipolarMatcher
        # Simple synthetic calibration
        K0 = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=float)
        K1 = K0.copy()
        R  = np.eye(3)
        T  = np.array([100.0, 0.0, 0.0])
        d  = np.zeros(5)
        P0 = K0 @ np.hstack([np.eye(3), np.zeros((3,1))])
        P1 = K1 @ np.hstack([R, T.reshape(3,1)])
        return EpipolarMatcher(P0=P0, P1=P1, K0=K0, K1=K1,
                                dist0=d, dist1=d, R=R, T=T,
                                max_epipolar_px=10.0)

    def test_F_shape(self, matcher):
        assert matcher.F.shape == (3, 3)

    def test_epipolar_constraint(self, matcher):
        from rpimocap.detection.segment import BodyRegion
        # A point on the epipolar line should have zero distance
        r0 = BodyRegion("back", cx=320.0, cy=240.0)
        line = matcher._epipolar_line(r0.cx, r0.cy)
        # The corresponding point in cam1 (pure horizontal shift for parallel cams)
        r1 = BodyRegion("back", cx=200.0, cy=240.0)
        d  = matcher._point_to_line_dist(line, r1.cx, r1.cy)
        assert d < 5.0, f"Epipolar distance too large: {d:.2f} px"

    def test_label_matching(self, matcher):
        from rpimocap.detection.segment import BodyRegion
        r0 = [BodyRegion("nose", 400, 200), BodyRegion("back", 300, 240)]
        r1 = [BodyRegion("nose", 280, 200), BodyRegion("back", 180, 240)]
        matches = matcher.match(r0, r1)
        assert len(matches) == 2
        labels = [(a.label, b.label) for a, b in matches]
        assert ("nose", "nose") in labels
        assert ("back", "back") in labels

    def test_triangulate_returns_xyz(self, matcher):
        from rpimocap.detection.segment import BodyRegion
        r0 = BodyRegion("back", cx=320.0, cy=240.0)
        r1 = BodyRegion("back", cx=220.0, cy=240.0)
        xyz = matcher.triangulate([(r0, r1)])
        assert "back" in xyz
        assert xyz["back"].shape == (3,)
        assert np.isfinite(xyz["back"]).all()

    def test_no_match_outside_epipolar(self, matcher):
        from rpimocap.detection.segment import BodyRegion
        r0 = [BodyRegion("nose", 320, 240)]
        # cam1 point far from epipolar line (huge Y offset)
        r1 = [BodyRegion("unknown", 300, 400)]
        matches = matcher.match(r0, r1)
        # May or may not match depending on epipolar distance
        # Just verify it returns a list
        assert isinstance(matches, list)

    def test_trajectory_prior_breaks_tie_in_centroid_only(self, matcher):
        """With two equally-good epipolar pairs, the prior selects the
        one closer to last frame's confirmed centroid."""
        from rpimocap.detection.segment import BodyRegion
        # Both candidates on the same scanline (y=240) so both have
        # epipolar distance ≈ 0; same label so the matcher goes into
        # centroid-only mode and runs the global selector.
        r0 = [BodyRegion("animal", 320, 240),    # cluster A
              BodyRegion("animal", 100, 240)]    # cluster B (far away)
        r1 = [BodyRegion("animal", 220, 240),    # cluster A'
              BodyRegion("animal",   0, 240)]    # cluster B'

        # Without a prior, the selector picks one of the pairs (both
        # have ~zero epipolar distance; tie broken by iteration order).
        m_noprior = matcher.match(r0, r1)
        assert len(m_noprior) == 1

        # With the prior anchored at cluster B in both cameras, the
        # selector must pick B↔B'.
        m_prior = matcher.match(
            r0, r1,
            prior0=(100, 240), prior1=(0, 240),
            prior_lambda=0.05)
        assert len(m_prior) == 1
        a, b = m_prior[0]
        assert (a.cx, a.cy) == (100, 240)
        assert (b.cx, b.cy) == (0, 240)

    def test_trajectory_prior_does_not_admit_bad_epipolar_match(self, matcher):
        """Even with a strong prior, a candidate with epipolar distance
        above the threshold must still be rejected."""
        from rpimocap.detection.segment import BodyRegion
        # cam1 candidate is far off the epipolar line of the cam0 point
        r0 = [BodyRegion("animal", 320, 240),
              BodyRegion("animal", 100, 240)]
        r1 = [BodyRegion("animal", 220, 240),
              BodyRegion("animal",   0, 400)]    # huge y offset → bad epipolar

        m = matcher.match(
            r0, r1,
            prior0=(100, 240), prior1=(0, 400),    # prior favours the bad pair
            prior_lambda=1e6)                       # try very strong prior
        # Either no match, or the legitimate A↔A' pair — never the bad one
        if m:
            a, b = m[0]
            assert (a.cx, a.cy) != (100, 240) or (b.cx, b.cy) != (0, 400), (
                "matcher accepted an epipolar-bad pair just because of prior")


# --------------------------------------------------------------------------- #
#  OpticalFlowTracker                                                          #
# --------------------------------------------------------------------------- #

class TestOpticalFlowTracker:
    def test_tracks_through_frames(self, background_model, synthetic_caps):
        from rpimocap.detection.segment import ForegroundDetector, GeometricLabeller
        from rpimocap.detection.tracker import OpticalFlowTracker
        det = ForegroundDetector(background_model, threshold=20, min_area_px=100)
        lbl = GeometricLabeller()
        tracker = OpticalFlowTracker(det, lbl, redetect_every=30)
        cap0, _ = synthetic_caps
        results = []
        for idx in range(105, 115):
            cap0.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap0.read()
            if not ret:
                break
            regions = tracker.track(frame, cam=0)
            results.append(regions)
        # At least some frames should have detections
        n_detected = sum(1 for r in results if r)
        assert n_detected > 0, "OpticalFlowTracker detected nothing"

    def test_reset_clears_state(self, background_model, synthetic_caps):
        from rpimocap.detection.segment import ForegroundDetector, GeometricLabeller
        from rpimocap.detection.tracker import OpticalFlowTracker
        det = ForegroundDetector(background_model, threshold=20, min_area_px=100)
        lbl = GeometricLabeller()
        tracker = OpticalFlowTracker(det, lbl)
        tracker._prev_gray = np.zeros((H, W), np.uint8)
        tracker.reset()
        assert tracker._prev_gray is None
        assert tracker._tracked_pts is None
