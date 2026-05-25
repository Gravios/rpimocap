"""
tests/test_gabor_mask_override.py
====================================
Tests for the blob_mask override on ForegroundDetector.gabor_body_contour.

Why this exists
---------------
Step 3 of hull_centroid erodes the cable away from the blob, producing
an `eroded_mask` of just the body. Step 3b (Gabor refinement) used to
reconstruct its working mask from `result.label_map` — i.e. the
ORIGINAL blob including the cable. Method B (percentile threshold)
could re-include cable pixels that erosion explicitly removed.

The fix gives gabor_body_contour an optional `blob_mask` parameter
that the caller can use to confine the Gabor work to a pre-cleaned
region.

These tests verify:
  1. Default behaviour unchanged (back-compat — no blob_mask).
  2. Supplied blob_mask is honoured — output ⊆ supplied mask.
  3. Shape-mismatched blob_mask is silently ignored.
  4. Various input dtypes (uint8 0/1, uint8 0/255, bool) all work.
"""
from __future__ import annotations

import numpy as np
import pytest


def _make_synthetic_result(h=200, w=200):
    """Build a ForegroundResult with a body + cable, a labelmap, and
    a Gabor-style energy map (high outside body, low on body)."""
    import cv2
    from rpimocap.detection.segment import ForegroundResult

    # Body: filled disc; cable: thin rect attached to the body.
    body_mask  = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(body_mask, (100, 100), 25, 1, -1)
    cable_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(cable_mask, (123, 95), (180, 105), 1, -1)

    full_mask = (body_mask | cable_mask)
    full_mask_255 = full_mask * 255

    # Labelmap: 0 = bg, 1 = body+cable as one blob
    label_map = full_mask.astype(np.int32)

    # Gabor energy: high (0.8) on bedding, low (0.1) on smooth body,
    # MEDIUM (0.3) on cable (so Method B's percentile threshold could
    # plausibly catch it if blob_mask includes cable region).
    energy = np.full((h, w), 0.8, dtype=np.float32)
    energy[body_mask  > 0] = 0.1
    energy[cable_mask > 0] = 0.3
    # Add small noise so the percentile threshold isn't degenerate
    rng = np.random.default_rng(0)
    energy += rng.normal(0, 0.01, energy.shape).astype(np.float32)
    energy = np.clip(energy, 0, 1)

    return ForegroundResult(
        mask=full_mask_255,
        blobs=[],
        frame_gray=np.zeros((h, w), dtype=np.uint8),
        n_blobs=1,
        label_map=label_map,
        gabor_energy=energy), body_mask, cable_mask


class TestGaborBlobMaskOverride:

    def _detector(self):
        from rpimocap.detection.segment import (
            BackgroundModel, ForegroundDetector)
        bg = BackgroundModel(np.zeros((200, 200), np.float32),
                             np.zeros((200, 200), np.float32))
        return ForegroundDetector(bg, threshold=10, min_area_px=10)

    def test_signature_back_compat(self):
        """No blob_mask = same behaviour as before the fix."""
        import inspect
        from rpimocap.detection.segment import ForegroundDetector
        sig = inspect.signature(ForegroundDetector.gabor_body_contour)
        assert "blob_mask" in sig.parameters
        # Must default to None so existing callers still work
        assert sig.parameters["blob_mask"].default is None

    def test_no_blob_mask_uses_labelmap_path(self):
        """Without an override, the function operates on the full blob
        (body + cable)."""
        det = self._detector()
        result, body_mask, cable_mask = _make_synthetic_result()
        # Centroid of the body
        out = det.gabor_body_contour(result, cx=100, cy=100,
                                     energy_pct=50)
        assert out is not None
        # Should cover the body roughly; may or may not include cable
        # (depends on energy threshold + connectivity). We just verify
        # we get a non-empty result.
        assert out.sum() > 0

    def test_blob_mask_restricts_output_to_supplied_region(self):
        """With an eroded-mask override, the output must NOT extend
        outside the supplied mask — even if the original blob did."""
        det = self._detector()
        result, body_mask, cable_mask = _make_synthetic_result()
        # Pass body-only mask (cable removed)
        body_only = (body_mask * 255).astype(np.uint8)
        out = det.gabor_body_contour(result, cx=100, cy=100,
                                     energy_pct=50,
                                     blob_mask=body_only)
        assert out is not None
        # Strict guarantee: no output pixel may fall outside body_only
        assert np.all(out[body_only == 0] == 0), (
            "gabor_body_contour produced pixels outside the supplied "
            "blob_mask — cable could leak back in")

    def test_blob_mask_accepts_bool(self):
        """bool dtype should work (callers might pass `mask > 0`)."""
        det = self._detector()
        result, body_mask, _ = _make_synthetic_result()
        out = det.gabor_body_contour(result, cx=100, cy=100,
                                     blob_mask=body_mask.astype(bool))
        assert out is not None
        assert np.all(out[body_mask == 0] == 0)

    def test_blob_mask_accepts_0_255(self):
        """uint8 0/255 (the shape eroded_mask comes in as) should work."""
        det = self._detector()
        result, body_mask, _ = _make_synthetic_result()
        out = det.gabor_body_contour(
            result, cx=100, cy=100,
            blob_mask=(body_mask * 255).astype(np.uint8))
        assert out is not None
        assert np.all(out[body_mask == 0] == 0)

    def test_shape_mismatch_silently_falls_back(self):
        """A blob_mask of wrong shape must not crash — it falls back
        to the labelmap path (back-compat-equivalent behaviour)."""
        det = self._detector()
        result, body_mask, _ = _make_synthetic_result()
        bad_mask = np.ones((50, 50), dtype=np.uint8)  # wrong shape
        out = det.gabor_body_contour(result, cx=100, cy=100,
                                     blob_mask=bad_mask)
        # Doesn't crash; same shape as labelmap
        assert out is not None
        assert out.shape == result.label_map.shape


class TestHullCentroidPassesErodedMask:
    """End-to-end: hull_centroid step 3 + 3b chain must keep the cable
    excluded even when --gabor-refine is on."""

    def test_eroded_mask_flows_through_to_gabor(self):
        """If step 3 succeeds (cable erosion produced an eroded_mask),
        step 3b's call into gabor_body_contour must receive blob_mask=
        eroded_mask, not None."""
        import inspect
        from rpimocap.detection import segment as seg_mod
        src = inspect.getsource(seg_mod.ForegroundDetector.hull_centroid)
        # The step-3b call should now pass blob_mask=eroded_mask
        assert "blob_mask=eroded_mask" in src, (
            "hull_centroid step 3b no longer forwards eroded_mask to "
            "gabor_body_contour — cable pixels may leak back into "
            "Gabor refinement")
