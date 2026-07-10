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


class TestProvenance:

    def _full_record(self, cam=0, frame=7, px0=100, py0=50, x=40, y=30):
        rng = np.random.RandomState(3)
        g = rng.randint(40, 200, (80, 80)).astype(np.uint8)
        rec = tes.extract_record(g, x, y, "fur", "texture", 48,
                                 KERNELS, N_ORIENT, 3, cam=cam, frame_idx=frame)
        # mirror the provenance fields _save_click attaches
        rec["session"] = "20260214_021722"
        rec["src_file"] = "cam0.tif"
        rec["patch_x0"], rec["patch_y0"] = px0, py0
        rec["patch_w"], rec["patch_h"] = 80, 80
        rec["x_in_patch"], rec["y_in_patch"] = x, y
        rec["x"], rec["y"] = px0 + x, py0 + y     # absolute frame coords
        return rec

    def test_provenance_columns_round_trip(self, tmp_path):
        rec = self._full_record()
        path = str(tmp_path / "samples.csv")
        tes.write_csv(path, [rec], 9)
        row = list(csv.DictReader(open(path)))[0]
        for f in tes._BASE_FIELDS:                 # every column present
            assert f in row, f"missing column {f}"
        assert row["session"] == "20260214_021722"
        assert row["src_file"] == "cam0.tif"
        assert row["cam"] == "0" and row["frame"] == "7"
        assert row["patch_x0"] == "100" and row["patch_y0"] == "50"
        assert row["patch_w"] == "80" and row["patch_h"] == "80"
        assert row["x_in_patch"] == "40" and row["y_in_patch"] == "30"
        assert row["x"] == "140" and row["y"] == "80"   # origin + in-patch

    def test_schema_guard_rejects_old_header(self, tmp_path):
        path = str(tmp_path / "old.csv")
        with open(path, "w") as fh:                # a pre-provenance header
            fh.write("cam,frame,x,y\n0,1,2,3\n")
        with pytest.raises(SystemExit):
            tes.write_csv(path, [self._full_record()], 9)


class TestColor:

    def _rec(self, color, cx=40, cy=40, local=3):
        g = np.full((80, 80), 100, np.uint8)
        return tes.extract_record(g, cx, cy, "fur", "texture", 48,
                                  KERNELS, N_ORIENT, 3, cam=0, frame_idx=0,
                                  color_patch=color, color_local=local)

    def test_color_columns_correct(self):
        color = np.zeros((80, 80, 3), np.uint8)
        color[..., 0], color[..., 1], color[..., 2] = 100, 150, 200  # B,G,R
        rec = self._rec(color)
        assert abs(rec["col_r_win"] - 200) < 1e-3
        assert abs(rec["col_g_win"] - 150) < 1e-3
        assert abs(rec["col_b_win"] - 100) < 1e-3
        assert abs(rec["col_rb_win"] - 2.0) < 1e-3            # R/B = 200/100
        assert abs(rec["col_rmb_win"] - (100 / 300)) < 1e-3  # (R-B)/(R+B)
        assert abs(rec["col_r_loc"] - 200) < 1e-3
        assert abs(rec["col_rb_loc"] - 2.0) < 1e-3

    def test_local_preserves_thin_structure_window_dilutes(self):
        color = np.zeros((80, 80, 3), np.uint8)
        color[..., 0], color[..., 2] = 80, 80        # background R=B=80 (R/B=1)
        color[:, 39:42, 0], color[:, 39:42, 2] = 40, 200   # 3px line R/B=5
        rec = self._rec(color)
        assert rec["col_rb_loc"] > 3.0               # local sees the line
        assert rec["col_rb_win"] < 1.5               # window diluted to bg

    def test_no_color_patch_yields_no_color_keys(self):
        g = np.full((80, 80), 100, np.uint8)
        rec = tes.extract_record(g, 40, 40, "fur", "texture", 48,
                                 KERNELS, N_ORIENT, 3, cam=0, frame_idx=0)
        assert "col_rb_win" not in rec               # color_stats -> {} w/o patch
        # but the CSV still carries the columns (empty) so the schema is stable
        assert "col_rb_win" in tes._BASE_FIELDS


class TestBlobGeometry:

    def _rat_cable_frame(self):
        import cv2
        H, W = 400, 500
        g = np.full((H, W), 60, np.uint8)
        cv2.ellipse(g, (140, 200), (40, 30), 15, 0, 360, 220, -1)  # rat
        cv2.circle(g, (300, 110), 16, 220, -1)                     # headstage
        cv2.polylines(g, [np.array([[300, 110], [360, 180],
                                    [340, 260], [420, 320]], np.int32)],
                      False, 220, 6)                               # curved cable
        return g

    def test_geometry_at_click_separates_rat_cable(self):
        g = self._rat_cable_frame()
        rat = tes.blob_geometry_at(g, 140, 200)
        cab = tes.blob_geometry_at(g, 350, 200)
        assert rat["geom_elongation"] < cab["geom_elongation"]
        assert rat["geom_solidity"] > cab["geom_solidity"]
        assert rat["geom_fill"] > cab["geom_fill"]

    def test_record_has_geometry_fields(self):
        g = self._rat_cable_frame()
        rec = tes.extract_record(g, 140, 200, "fur", "texture", 48,
                                 KERNELS, N_ORIENT, 3, cam=0, frame_idx=0)
        for f in ("geom_area", "geom_fill", "geom_solidity",
                  "geom_elongation"):
            assert f in rec

    def test_seed_thresholds_from_labels(self, tmp_path):
        import cv2
        rng = np.random.RandomState(1)
        recs = []
        for _ in range(10):
            f = np.full((400, 500), 60, np.uint8)
            cx, cy = 140 + rng.randint(-6, 6), 200 + rng.randint(-6, 6)
            cv2.ellipse(f, (cx, cy), (40, 30), rng.randint(0, 180),
                        0, 360, 220, -1)
            recs.append(tes.extract_record(
                f, cx, cy, "fur", "texture", 48, KERNELS, N_ORIENT, 3,
                cam=0, frame_idx=0))
        for _ in range(10):
            f = np.full((400, 500), 60, np.uint8)
            ox, oy = rng.randint(60, 140), rng.randint(60, 140)
            cv2.circle(f, (ox, oy), 16, 220, -1)
            cv2.polylines(f, [np.array([[ox, oy], [ox + 60, oy + 70],
                                        [ox + 40, oy + 150],
                                        [ox + 120, oy + 210]], np.int32)],
                          False, 220, 6)
            recs.append(tes.extract_record(
                f, ox + 40, oy + 90, "cable", "texture", 48,
                KERNELS, N_ORIENT, 3, cam=0, frame_idx=0))
        path = str(tmp_path / "lab.csv")
        tes.write_csv(path, recs, recs[0]["_n_desc"])
        res = tes.seed_thresholds_from_csv(path)
        assert res["n_rat"] == 10 and res["n_cable"] == 10
        # elongation & solidity should separate; a threshold is proposed
        assert res["features"]["elongation"]["dprime"] > 1.0
        assert "max_elongation" in res and "min_solidity" in res
        # proposed max_elongation sits below the cable's mean
        assert (res["max_elongation"]
                < res["features"]["elongation"]["cable_mean"])


class TestClassSwitching:
    """next_class_of_kind (Tab) cycles within a kind, skipping the other."""

    class _Stub:
        def __init__(self, classes, kinds):
            self.classes, self.kinds, self.ci = classes, kinds, 0
        _render = lambda self: None
        next_class_of_kind = tes._Session.next_class_of_kind
        next_class = tes._Session.next_class

    def test_tab_cycles_texture_skipping_edges(self):
        s = self._Stub(["bedding", "fur", "wall", "maze_edge", "rat_edge"],
                       ["texture", "texture", "texture", "edge", "edge"])
        seen = []
        for _ in range(5):
            seen.append(s.classes[s.ci]); s.next_class_of_kind(+1)
        assert seen == ["bedding", "fur", "wall", "bedding", "fur"]

    def test_tab_on_edge_cycles_edges(self):
        s = self._Stub(["bedding", "fur", "maze_edge", "rat_edge"],
                       ["texture", "texture", "edge", "edge"])
        s.ci = 2
        seen = []
        for _ in range(3):
            seen.append(s.classes[s.ci]); s.next_class_of_kind(+1)
        assert seen == ["maze_edge", "rat_edge", "maze_edge"]

    def test_tab_singleton_kind_stays(self):
        s = self._Stub(["fur", "rat_edge"], ["texture", "edge"])
        s.ci = 0
        s.next_class_of_kind(+1)
        assert s.ci == 0                    # only one texture class
