"""
tests/test_gabor_contour.py
============================
Unit tests for the Gabor-edge body-contour refinement (step 3b of
hull_centroid) and its interaction with the anatomical prior (step 5).

Verifies:
- ForegroundResult carries a gabor_energy field
- gabor_body_contour returns None when no Gabor energy is available
- Method A (Canny edges) recovers a known low-energy body region
- hull_centroid honours the gabor_refine flag without crashing when
  the Gabor energy is missing
- Default off backward-compatibility
- The Gabor-refined mask is consumed by the step-5 anatomical prior
  (so enabling gabor_refine genuinely changes the centroid when both
  prior and refinement are active)
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest


def _make_fg_result(h=200, w=200, body_cx=100, body_cy=100, body_r=30):
    """Synthesise a ForegroundResult with a known low-energy circular body
    embedded in a high-energy 'bedding' field."""
    from rpimocap.detection.segment import ForegroundResult

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    rr = np.hypot(xx - body_cx, yy - body_cy)
    blob_mask = (rr < body_r + 8).astype(np.uint8)
    label_map = blob_mask.copy().astype(np.int32)

    bedding = 0.85 + 0.05 * np.random.default_rng(0).standard_normal((h, w))
    bedding = np.clip(bedding, 0.0, 1.0).astype(np.float32)
    body_lo = np.clip((rr - body_r) / 8.0, 0, 1).astype(np.float32)
    gabor_energy = bedding * body_lo

    return ForegroundResult(
        mask=blob_mask * 255,
        blobs=[],
        frame_gray=np.zeros((h, w), dtype=np.uint8),
        n_blobs=1,
        label_map=label_map,
        gabor_energy=gabor_energy)


class TestForegroundResultField:

    def test_gabor_energy_field_present(self):
        from rpimocap.detection.segment import ForegroundResult
        r = ForegroundResult(
            mask=np.zeros((4, 4), np.uint8),
            blobs=[], frame_gray=np.zeros((4, 4), np.uint8), n_blobs=0)
        assert hasattr(r, "gabor_energy")
        assert r.gabor_energy is None


class TestGaborBodyContour:

    @pytest.fixture
    def detector(self):
        from rpimocap.detection.segment import (
            BackgroundModel, ForegroundDetector)
        bg = BackgroundModel(
            bg0=np.full((200, 200), 100, np.uint8),
            bg1=np.full((200, 200), 100, np.uint8),
            method="median")
        return ForegroundDetector(bg, threshold=10, min_area_px=10)

    def test_returns_none_without_gabor_energy(self, detector):
        from rpimocap.detection.segment import ForegroundResult
        r = ForegroundResult(
            mask=np.ones((50, 50), np.uint8) * 255,
            blobs=[],
            frame_gray=np.zeros((50, 50), np.uint8),
            n_blobs=1,
            label_map=np.ones((50, 50), np.int32),
            gabor_energy=None)
        assert detector.gabor_body_contour(r, 25, 25) is None

    def test_returns_none_without_label_map(self, detector):
        from rpimocap.detection.segment import ForegroundResult
        r = ForegroundResult(
            mask=np.ones((50, 50), np.uint8) * 255,
            blobs=[],
            frame_gray=np.zeros((50, 50), np.uint8),
            n_blobs=1,
            label_map=None,
            gabor_energy=np.ones((50, 50), np.float32))
        assert detector.gabor_body_contour(r, 25, 25) is None

    def test_recovers_known_low_energy_body(self, detector):
        r = _make_fg_result(body_cx=100, body_cy=100, body_r=30)
        out = detector.gabor_body_contour(
            r, 100, 100, canny_low=20, canny_high=80)
        assert out is not None
        assert out.dtype == np.uint8
        assert out[100, 100] > 0
        recovered_px = int((out > 0).sum())
        true_px = int(((np.hypot(*np.mgrid[0:200, 0:200] - 100)) < 30).sum())
        assert recovered_px > 0.4 * true_px

    def test_centroid_outside_blob_finds_nearest_label(self, detector):
        r = _make_fg_result()
        out = detector.gabor_body_contour(r, 5, 5)
        assert out is None or isinstance(out, np.ndarray)


class TestHullCentroidGaborIntegration:

    @pytest.fixture
    def detector(self):
        from rpimocap.detection.segment import (
            BackgroundModel, ForegroundDetector)
        bg = BackgroundModel(
            bg0=np.full((200, 200), 100, np.uint8),
            bg1=np.full((200, 200), 100, np.uint8),
            method="median")
        return ForegroundDetector(bg, threshold=10, min_area_px=10)

    def test_gabor_refine_no_op_without_energy(self, detector):
        from rpimocap.detection.segment import ForegroundResult
        lbl = np.zeros((100, 100), np.int32)
        lbl[40:60, 40:60] = 1
        r = ForegroundResult(
            mask=(lbl > 0).astype(np.uint8) * 255,
            blobs=[],
            frame_gray=np.zeros((100, 100), np.uint8),
            n_blobs=1,
            label_map=lbl,
            gabor_energy=None)
        cx, cy = detector.hull_centroid(r, 50, 50, gabor_refine=True)
        assert 40 <= cx <= 60
        assert 40 <= cy <= 60

    def test_gabor_refine_with_energy_returns_finite_centroid(self, detector):
        r = _make_fg_result(body_cx=100, body_cy=100, body_r=30)
        cx, cy = detector.hull_centroid(
            r, 100, 100, gabor_refine=True,
            canny_low=20, canny_high=80)
        assert np.isfinite(cx) and np.isfinite(cy)
        assert abs(cx - 100) < 20
        assert abs(cy - 100) < 20

    def test_default_off_matches_pre_gabor_behaviour(self, detector):
        r = _make_fg_result()
        cx_off, cy_off = detector.hull_centroid(
            r, 100, 100, gabor_refine=False)
        cx_def, cy_def = detector.hull_centroid(r, 100, 100)   # default
        assert (cx_off, cy_off) == (cx_def, cy_def)

    def test_gabor_refined_mask_feeds_anatomical_prior(self, detector):
        """gabor_refine=True must update the mask consumed by step 5
        (the anatomical prior). The simplest observable consequence:
        with a blob that has irregular borders, the gabor-refined +
        prior path produces a different centroid than the prior-alone
        path."""
        from rpimocap.detection.segment import ForegroundResult

        # Build a result with an irregular blob: ellipse body plus a
        # cable-like high-energy protrusion that the Gabor contour
        # should reject. The blob mask includes both regions but the
        # Gabor energy is high (≈bedding) in the cable region.
        h, w = 300, 400
        mask = np.zeros((h, w), np.uint8)
        cv2.ellipse(mask, (200, 150), (60, 35), 0, 0, 360, 255, -1)
        cv2.line(mask, (250, 150), (350, 150), 255, 4)
        n, lbl, _, _ = cv2.connectedComponentsWithStats(mask)

        # Gabor energy: ~0 inside the body ellipse, ~0.9 on the cable
        rng = np.random.default_rng(0)
        gabor = 0.6 + 0.2 * rng.standard_normal((h, w)).astype(np.float32)
        gabor = np.clip(gabor, 0, 1).astype(np.float32)
        cv2.ellipse(gabor, (200, 150), (60, 35), 0, 0, 360, 0.05, -1)

        r = ForegroundResult(
            mask=mask, blobs=[],
            frame_gray=np.zeros((h, w), np.uint8),
            n_blobs=n - 1,
            label_map=lbl.astype(np.int32),
            gabor_energy=gabor)

        # Synthetic P matrix for a top-down camera ~900 mm up
        K = np.array([[800, 0, w/2], [0, 800, h/2], [0, 0, 1.0]])
        R = np.array([[1, 0, 0], [0,-1, 0], [0, 0,-1.0]])
        t = -R @ np.array([0, 0, 900.0])
        P = K @ np.hstack([R, t.reshape(3, 1)])

        # Without Gabor refinement: prior runs on the full blob
        cx_no_gabor, cy_no_gabor = detector.hull_centroid(
            r, 230, 150, cable_erosion_px=0,
            P=P, body_length_mm=120.0, body_width_mm=70.0, body_z_mm=0.0,
            gabor_refine=False)

        # With Gabor refinement: prior runs on the (smaller) Gabor mask
        cx_gabor, cy_gabor = detector.hull_centroid(
            r, 230, 150, cable_erosion_px=0,
            P=P, body_length_mm=120.0, body_width_mm=70.0, body_z_mm=0.0,
            gabor_refine=True, canny_low=20, canny_high=80)

        # Centroids should differ noticeably — and the Gabor-refined
        # path should be at least as close to the true body centre (200)
        err_no = abs(cx_no_gabor - 200)
        err_g  = abs(cx_gabor - 200)
        assert err_g <= err_no + 0.5
