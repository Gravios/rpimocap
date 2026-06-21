"""Tests for the texture/edge sampling utility's core (non-GUI)
functions."""
import csv
import importlib.util
import os

import cv2
import numpy as np
import pytest

from rpimocap.detection.rat_texture import build_gabor_kernels


# Load the tool module by path (it lives in tools/, not the package)
_HERE = os.path.dirname(__file__)
_TOOL = os.path.join(_HERE, "..", "tools", "texture_edge_sampler.py")
_spec = importlib.util.spec_from_file_location("texture_edge_sampler",
                                               _TOOL)
tes = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tes)


N_ORIENT = 8
SCALES = [5, 9, 13]
KERNELS = build_gabor_kernels(
    [i * np.pi / N_ORIENT for i in range(N_ORIENT)], SCALES)


class TestIntensityStats:

    def test_flat_region_zero_std(self):
        s = tes.intensity_stats(np.full((40, 40), 100, np.uint8))
        assert s["int_std"] == 0.0
        assert s["int_mean"] == 100.0
        assert s["int_p50"] == 100.0

    def test_textured_region_nonzero_std(self):
        rng = np.random.RandomState(0)
        w = rng.randint(40, 200, (40, 40)).astype(np.uint8)
        s = tes.intensity_stats(w)
        assert s["int_std"] > 5.0
        assert s["int_min"] < s["int_p50"] < s["int_max"]


class TestStructureTensorStats:

    def test_flat_low_coherence_low_gradient(self):
        s = tes.structure_tensor_stats(np.full((48, 48), 100, np.uint8))
        assert s["grad_mag_mean"] < 1.0
        assert s["coherence"] < 0.1

    def test_edge_high_coherence(self):
        edge = np.full((48, 48), 60, np.uint8)
        edge[:, 24:] = 200
        s = tes.structure_tensor_stats(edge)
        assert s["coherence"] > 0.8       # a clean edge is coherent
        assert s["grad_mag_mean"] > 1.0


class TestCrossEdgeProfile:

    def test_sharp_edge_small_width(self):
        sharp = np.full((100, 100), 60, np.uint8)
        sharp[:, 50:] = 220
        st = tes.structure_tensor_stats(sharp[26:74, 26:74])
        prof = tes.cross_edge_profile(sharp, 50, 50, st["orient_deg"])
        assert prof["edge_contrast"] > 100      # high-contrast step
        assert prof["edge_width_px"] <= 3       # sharp

    def test_soft_edge_larger_width(self):
        soft = np.full((100, 100), 60, np.float32)
        for c in range(100):
            soft[:, c] = 60 + 160 / (1 + np.exp(-(c - 50) / 6.0))
        soft = soft.astype(np.uint8)
        st = tes.structure_tensor_stats(soft[26:74, 26:74])
        prof = tes.cross_edge_profile(soft, 50, 50, st["orient_deg"])
        assert prof["edge_contrast"] > 100
        assert prof["edge_width_px"] > 5        # soft / wide

    def test_sharp_strictly_sharper_than_soft(self):
        sharp = np.full((100, 100), 60, np.uint8)
        sharp[:, 50:] = 220
        soft = np.full((100, 100), 60, np.float32)
        for c in range(100):
            soft[:, c] = 60 + 160 / (1 + np.exp(-(c - 50) / 6.0))
        soft = soft.astype(np.uint8)
        ss = tes.structure_tensor_stats(sharp[26:74, 26:74])
        st = tes.structure_tensor_stats(soft[26:74, 26:74])
        ps = tes.cross_edge_profile(sharp, 50, 50, ss["orient_deg"])
        pt = tes.cross_edge_profile(soft, 50, 50, st["orient_deg"])
        assert ps["edge_width_px"] < pt["edge_width_px"]


class TestGaborDescriptorAt:

    def test_returns_fixed_dim(self):
        rng = np.random.RandomState(0)
        g = rng.randint(40, 200, (80, 80)).astype(np.uint8)
        d = tes.gabor_descriptor_at(g, 40, 40, KERNELS, N_ORIENT, 3)
        assert d.shape == (9,)            # rotation-invariant pooling
        assert np.all(np.isfinite(d))

    def test_clips_out_of_bounds_index(self):
        g = np.full((50, 50), 100, np.uint8)
        # click outside the array shouldn't crash
        d = tes.gabor_descriptor_at(g, 999, -5, KERNELS, N_ORIENT, 3)
        assert d.shape == (9,)


class TestExtractRecord:

    def test_record_has_all_fields(self):
        rng = np.random.RandomState(1)
        g = rng.randint(40, 200, (100, 100)).astype(np.uint8)
        rec = tes.extract_record(
            g, 50, 50, "bedding", "texture", 48,
            KERNELS, N_ORIENT, 3, cam=0, frame_idx=7)
        for f in ["cam", "frame", "x", "y", "klass", "kind", "win",
                  "int_mean", "grad_mag_mean", "coherence",
                  "edge_contrast", "edge_width_px", "desc0", "_n_desc"]:
            assert f in rec, f"missing {f}"
        assert rec["klass"] == "bedding"
        assert rec["kind"] == "texture"
        assert rec["_n_desc"] == 9

    def test_window_clipped_at_border(self):
        """A click near the border still produces a valid record."""
        g = np.full((60, 60), 100, np.uint8)
        rec = tes.extract_record(
            g, 2, 2, "wall", "texture", 48,
            KERNELS, N_ORIENT, 3, cam=1, frame_idx=0)
        assert np.isfinite(rec["int_mean"])


class TestSamplePatch:

    def test_crop_within_bounds(self):
        rng = np.random.RandomState(2)
        frame = np.zeros((1080, 2028), np.uint8)
        for _ in range(10):
            p, x0, y0 = tes.sample_patch(frame, 500, rng)
            assert p.shape == (500, 500)
            assert 0 <= x0 <= 2028 - 500
            assert 0 <= y0 <= 1080 - 500

    def test_small_frame_returns_whole(self):
        rng = np.random.RandomState(0)
        frame = np.zeros((300, 400), np.uint8)
        p, x0, y0 = tes.sample_patch(frame, 500, rng)
        assert p.shape == (300, 400)
        assert x0 == 0 and y0 == 0


class TestWriteCsv:

    def test_append_round_trip(self, tmp_path):
        rng = np.random.RandomState(0)
        recs = []
        for i in range(3):
            g = rng.randint(40, 200, (80, 80)).astype(np.uint8)
            recs.append(tes.extract_record(
                g, 40, 40, "fur", "texture", 48,
                KERNELS, N_ORIENT, 3, cam=0, frame_idx=i))
        path = str(tmp_path / "samples.csv")
        tes.write_csv(path, recs, 9)
        tes.write_csv(path, recs, 9)       # append
        with open(path) as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == 6              # header written once
        assert sum(c.startswith("desc") for c in rows[0]) == 9
        assert rows[0]["klass"] == "fur"
