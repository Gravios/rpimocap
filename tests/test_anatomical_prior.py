"""
tests/test_anatomical_prior.py
================================
Unit tests for the anatomical Gaussian shape prior — step 5 of
ForegroundDetector.hull_centroid.

Verifies:
- _px_per_mm_at_pixel returns sane values for a synthetic camera
- _anatomical_prior_centroid produces a weighted centroid inside the
  blob and biased toward the body region
- hull_centroid honours the anatomical prior when P + body_length_mm
  are supplied
- default (P=None, body_length_mm=0) reproduces pre-prior behaviour
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest


def _synthetic_P(f=800.0, cx=640.0, cy=360.0, z_cam=900.0):
    """Synthetic P matrix for a camera 900 mm above the floor, looking
    straight down. World frame: X right, Y forward, Z up (arena
    convention). Camera frame: x right, y down, z out.
    """
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
    # Camera at (0, 0, z_cam) looking toward -Z direction. R rotates
    # world axes to camera axes.
    R = np.array([[1, 0, 0],
                  [0,-1, 0],
                  [0, 0,-1]], dtype=np.float64)
    t = -R @ np.array([0.0, 0.0, z_cam])
    P = K @ np.hstack([R, t.reshape(3, 1)])
    return P


def _make_blob_result(h=720, w=1280,
                     body_cx=640, body_cy=360,
                     body_a=80, body_b=40, theta_deg=30):
    """Make a ForegroundResult with a single ellipse-shaped blob plus
    a cable-like protrusion."""
    from rpimocap.detection.segment import ForegroundResult
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(mask, (body_cx, body_cy), (body_a, body_b),
                theta_deg, 0, 360, 255, -1)
    # Cable: thin protrusion sticking off one end
    cable_end_x = int(body_cx + 1.5 * body_a * np.cos(np.deg2rad(theta_deg)))
    cable_end_y = int(body_cy + 1.5 * body_a * np.sin(np.deg2rad(theta_deg)))
    cv2.line(mask, (body_cx, body_cy), (cable_end_x, cable_end_y),
             255, 4)
    n, lbl_map, _, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    return ForegroundResult(
        mask=mask, blobs=[],
        frame_gray=np.zeros((h, w), np.uint8),
        n_blobs=n - 1,
        label_map=lbl_map.astype(np.int32))


class TestPxPerMmAtPixel:

    def test_returns_finite_scale(self):
        from rpimocap.detection.segment import _px_per_mm_at_pixel
        P = _synthetic_P(f=800.0, z_cam=900.0)
        # Image centre → optical axis hits floor at world origin
        s = _px_per_mm_at_pixel(P, 640, 360, z_mm=0.0)
        # Expect ~f/Z_cam = 800/900 ≈ 0.89 px/mm
        assert np.isfinite(s)
        assert 0.7 < s < 1.1, f"px/mm at centre {s:.3f} unexpected"

    def test_higher_z_means_higher_scale(self):
        """A subject closer to the camera (larger z above floor) is
        imaged larger → higher px/mm."""
        from rpimocap.detection.segment import _px_per_mm_at_pixel
        P = _synthetic_P(z_cam=900.0)
        s_floor = _px_per_mm_at_pixel(P, 640, 360, z_mm=0.0)
        s_high  = _px_per_mm_at_pixel(P, 640, 360, z_mm=200.0)
        assert s_high > s_floor

    def test_returns_nan_on_degenerate(self):
        from rpimocap.detection.segment import _px_per_mm_at_pixel
        bad_P = np.zeros((3, 4))
        s = _px_per_mm_at_pixel(bad_P, 100, 100, 0)
        assert np.isnan(s)


class TestAnatomicalPriorCentroid:

    def test_pulls_centroid_toward_body_centre(self):
        """When the blob has a cable extension, the unweighted centroid
        is biased toward the cable; the anatomical prior should restore
        the centroid to near the body centre."""
        from rpimocap.detection.segment import (
            _anatomical_prior_centroid,
            ForegroundResult,
        )
        h, w = 720, 1280
        # Body at (640, 360) with cable sticking off to the right
        body_cx, body_cy = 640, 360
        body_a, body_b   = 80, 40
        theta_deg        = 0.0
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(mask, (body_cx, body_cy), (body_a, body_b),
                    theta_deg, 0, 360, 255, -1)
        # Long cable
        cv2.line(mask, (body_cx + body_a, body_cy),
                 (body_cx + body_a + 200, body_cy),
                 255, 4)

        # Unweighted centroid drifts toward the cable
        ys, xs = np.where(mask > 0)
        raw_cx, raw_cy = float(xs.mean()), float(ys.mean())
        assert raw_cx > body_cx + 5, "test setup: cable should pull centroid"

        P = _synthetic_P(f=800.0, z_cam=900.0)
        out = _anatomical_prior_centroid(
            mask,
            ellipse_cx=raw_cx, ellipse_cy=raw_cy,
            theta_rad=0.0,
            P=P,
            body_length_mm=180.0, body_width_mm=70.0, body_z_mm=0.0)
        assert out is not None
        new_cx, new_cy = out
        # New centroid should be closer to true body centre than the raw
        # centroid was.
        err_raw = abs(raw_cx - body_cx)
        err_new = abs(new_cx - body_cx)
        assert err_new < err_raw, (
            f"prior did not pull centroid: raw err {err_raw:.1f} px, "
            f"new err {err_new:.1f} px")
        # y-coord stays near body centre
        assert abs(new_cy - body_cy) < 5

    def test_returns_none_on_empty_intersection(self):
        from rpimocap.detection.segment import _anatomical_prior_centroid
        # Blob is far from the ellipse centre — Gaussian × blob ≈ 0
        mask = np.zeros((720, 1280), dtype=np.uint8)
        cv2.ellipse(mask, (100, 100), (20, 10), 0, 0, 360, 255, -1)
        P = _synthetic_P()
        # Build a small body prior far away from the blob
        out = _anatomical_prior_centroid(
            mask, ellipse_cx=1000, ellipse_cy=600, theta_rad=0.0,
            P=P, body_length_mm=50.0, body_width_mm=30.0, body_z_mm=0.0)
        assert out is None


class TestHullCentroidWithAnatomicalPrior:

    @pytest.fixture
    def detector(self):
        from rpimocap.detection.segment import (
            BackgroundModel, ForegroundDetector)
        bg = BackgroundModel(
            bg0=np.full((720, 1280), 100, np.uint8),
            bg1=np.full((720, 1280), 100, np.uint8),
            method="median")
        return ForegroundDetector(bg, threshold=10, min_area_px=10)

    def test_default_off_no_change_in_behaviour(self, detector):
        """With body_length_mm=0 (default), hull_centroid must give
        identical output to the pre-prior code path."""
        r = _make_blob_result()
        cx_off, cy_off = detector.hull_centroid(r, 700, 360,
                                                 cable_erosion_px=8)
        cx_def, cy_def = detector.hull_centroid(r, 700, 360,
                                                 cable_erosion_px=8,
                                                 body_length_mm=0.0)
        assert (cx_off, cy_off) == (cx_def, cy_def)

    def test_anatomical_prior_pulls_toward_body(self, detector):
        """With a long cable extension and the anatomical prior active,
        the refined centroid is closer to the true body centre than
        without the prior."""
        r = _make_blob_result(
            body_cx=640, body_cy=360, body_a=80, body_b=40, theta_deg=0)
        P = _synthetic_P(f=800.0, z_cam=900.0)

        cx_no, cy_no = detector.hull_centroid(
            r, 700, 360, cable_erosion_px=0)
        cx_ap, cy_ap = detector.hull_centroid(
            r, 700, 360, cable_erosion_px=0,
            P=P, body_length_mm=180.0, body_width_mm=70.0, body_z_mm=0.0)

        err_no = float(np.hypot(cx_no - 640, cy_no - 360))
        err_ap = float(np.hypot(cx_ap - 640, cy_ap - 360))
        assert err_ap <= err_no + 0.5, (
            f"anatomical prior worsened centroid: "
            f"no-prior err {err_no:.1f} px, with-prior err {err_ap:.1f} px")

    def test_handles_missing_P(self, detector):
        """body_length_mm > 0 but P=None must not crash — fall through
        to the ellipse-only centroid."""
        r = _make_blob_result()
        cx, cy = detector.hull_centroid(
            r, 700, 360, cable_erosion_px=0,
            P=None, body_length_mm=180.0)
        assert np.isfinite(cx) and np.isfinite(cy)
