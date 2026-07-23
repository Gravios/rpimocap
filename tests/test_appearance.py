"""Tests for the textured appearance model (rpimocap.model.appearance)."""
import inspect

import cv2
import numpy as np
import pytest

from rpimocap.model.appearance import (AppearanceModel, appearance_energy,
                                       bootstrap_masks, estimate_whitening,
                                       exposure_duty, gradient_covariance,
                                       image_features, interp_pose,
                                       render_coverage, roi_from_mask,
                                       whitening_report)
from rpimocap.model.rat_skeleton import RatPose


def _anisotropic(shape=(200, 200), seed=0):
    """Noise with a strong vertical-structure bias (like the sensor artefact):
    horizontal gradients much stronger than vertical."""
    rng = np.random.default_rng(seed)
    img = rng.normal(128, 12, shape)
    x = np.arange(shape[1])
    img += 30 * np.sin(x / 2.0)[None, :]        # vertical stripes
    return np.clip(img, 0, 255).astype(np.uint8)


class TestWhitening:

    def test_whitening_isotropises_background(self):
        g = _anisotropic()
        mask = np.ones(g.shape, bool)
        rep = whitening_report(g, mask)
        assert rep["eig_ratio"] > 2.0             # strongly anisotropic input
        W = estimate_whitening(g, mask)
        # after whitening the covariance of the same gradients is ~identity
        C = gradient_covariance(g, mask)
        Cw = W @ C @ W.T
        ev = np.linalg.eigvalsh(Cw)
        assert abs(ev[1] / ev[0] - 1.0) < 1e-3    # isotropic

    def test_whitening_is_symmetric_psd(self):
        g = _anisotropic()
        W = estimate_whitening(g, np.ones(g.shape, bool))
        assert np.allclose(W, W.T, atol=1e-8)
        assert (np.linalg.eigvalsh(W) > 0).all()

    def test_dominant_axis_recovered(self):
        # vertical stripes -> gradients are horizontal -> dominant grad ~0 deg
        rep = whitening_report(_anisotropic(), np.ones((200, 200), bool))
        a = rep["dominant_grad_deg"]
        assert min(abs(a - 0), abs(a - 180)) < 15


class TestFeatures:

    def test_shapes_and_finiteness(self):
        rng = np.random.default_rng(1)
        bgr = rng.integers(30, 200, (120, 160, 3), dtype=np.uint8)
        f = image_features(bgr)
        assert f.chroma.shape == (120, 160) and f.coh.shape == (120, 160)
        assert f.grain.shape == (120, 160)
        for m in (f.chroma, f.coh, f.grain, f.theta, f.rb):
            assert np.isfinite(m).all()

    def test_coherence_in_unit_range(self):
        rng = np.random.default_rng(2)
        bgr = rng.integers(30, 200, (120, 160, 3), dtype=np.uint8)
        f = image_features(bgr)
        assert f.coh.min() >= -1e-6 and f.coh.max() <= 1.0 + 1e-6

    def test_grain_is_intensity_normalised(self):
        """The regression this feature was rewritten for: two regions with the
        SAME texture contrast but different brightness must give the same grain.
        Raw bandpass energy fails this (dark => low energy => reads 'smooth')."""
        rng = np.random.default_rng(3)
        base = rng.normal(0, 1, (80, 200))
        img = np.zeros((80, 200, 3), np.float32)
        left = 60.0 + 6.0 * base[:, :100]        # dark, 10% contrast
        right = 180.0 + 18.0 * base[:, 100:]     # bright, 10% contrast
        for c in range(3):
            img[:, :100, c] = left
            img[:, 100:, c] = right
        f = image_features(np.clip(img, 0, 255).astype(np.uint8), grain_k=16)
        gl = np.median(f.grain[20:60, 20:80])
        gr = np.median(f.grain[20:60, 120:180])
        assert gl == pytest.approx(gr, rel=0.35)   # same contrast -> same grain

    def test_whitening_changes_coherence(self):
        g = _anisotropic()
        bgr = np.dstack([g, g, g])
        W = estimate_whitening(g, np.ones(g.shape, bool))
        raw = image_features(bgr, None).coh
        wht = image_features(bgr, W).coh
        assert np.median(wht) < np.median(raw)     # bias removed -> less coherent


class TestAppearanceModel:

    def _two_class(self):
        rng = np.random.default_rng(4)
        bgr = np.zeros((160, 160, 3), np.uint8)
        # background: reddish (R>B); foreground blob: bluish (R<B) — the real cue
        bgr[..., 0] = rng.normal(80, 5, (160, 160))     # B
        bgr[..., 1] = rng.normal(100, 5, (160, 160))    # G
        bgr[..., 2] = rng.normal(110, 5, (160, 160))    # R  -> R/B ~ 1.4
        fg = np.zeros((160, 160), bool)
        fg[60:100, 60:100] = True
        bgr[..., 2][fg] = rng.normal(60, 5, fg.sum())   # R/B ~ 0.75 on the blob
        bg = ~fg
        return bgr, fg, bg

    def test_posterior_separates_classes(self):
        bgr, fg, bg = self._two_class()
        f = image_features(bgr, coh_k=16, grain_k=16)
        m = AppearanceModel.from_masks(f, fg, bg, features=("chroma",))
        post = m.posterior_fg(f)
        assert np.median(post[fg]) > 0.8
        assert np.median(post[bg]) < 0.2
        assert abs(m.dprime["chroma"]) > 2.0

    def test_priors_and_hists_normalised(self):
        bgr, fg, bg = self._two_class()
        f = image_features(bgr, coh_k=16, grain_k=16)
        m = AppearanceModel.from_masks(f, fg, bg)
        assert 0.0 < m.prior_fg < 1.0
        for name in m.features:
            assert m.hist_fg[name].sum() == pytest.approx(1.0)
            assert m.hist_bg[name].sum() == pytest.approx(1.0)

    def test_rejects_tiny_samples(self):
        bgr, fg, bg = self._two_class()
        f = image_features(bgr, coh_k=16, grain_k=16)
        with pytest.raises(ValueError):
            AppearanceModel.from_masks(f, np.zeros_like(fg), bg)


class TestBootstrapMasks:

    def test_fg_bg_disjoint_and_inside_floor(self):
        floor = np.zeros((200, 200), bool); floor[20:180, 20:180] = True
        seed = np.zeros((200, 200), bool); seed[90:120, 90:120] = True
        fg, bg = bootstrap_masks(floor, seed, coh_k=8, fg_erode=3, bg_gap=11)
        assert not (fg & bg).any()             # disjoint
        assert (fg & ~floor).sum() == 0        # fg inside the floor
        assert (bg & ~floor).sum() == 0        # bg inside the floor
        assert not (bg & seed).any()           # animal excluded from background

    def test_roi_restricts_background(self):
        floor = np.zeros((200, 200), bool); floor[20:180, 20:180] = True
        seed = np.zeros((200, 200), bool); seed[90:120, 90:120] = True
        roi = np.zeros((200, 200), bool); roi[60:150, 60:150] = True
        _, bg_all = bootstrap_masks(floor, seed, coh_k=8, bg_gap=11)
        _, bg_roi = bootstrap_masks(floor, seed, coh_k=8, bg_gap=11, roi=roi)
        assert bg_roi.sum() < bg_all.sum()


class TestExposureAndInterp:

    def test_duty_from_fps(self):
        assert exposure_duty(30.0, None) == 1.0
        assert exposure_duty(30.0, 1 / 1000.0) == pytest.approx(0.03)
        assert exposure_duty(50.0, 1 / 1000.0) == pytest.approx(0.05)
        assert exposure_duty(50.0, 1 / 250.0) == pytest.approx(0.20)
        assert exposure_duty(30.0, 1.0) == 1.0          # clamped

    def test_interp_endpoints_and_midpoint(self):
        p0 = RatPose(root_pos=np.array([0.0, 0.0, 50.0]), scale=1.0,
                     joint_angles={"SpineF": (0.0, 0.0, 0.0)})
        p1 = RatPose(root_pos=np.array([10.0, 0.0, 50.0]), scale=1.2,
                     joint_angles={"SpineF": (0.0, 0.4, 0.0)})
        assert np.allclose(interp_pose(p0, p1, 0.0).root_pos, p0.root_pos)
        assert np.allclose(interp_pose(p0, p1, 1.0).root_pos, p1.root_pos)
        mid = interp_pose(p0, p1, 0.5)
        assert mid.root_pos[0] == pytest.approx(5.0)
        assert mid.scale == pytest.approx(1.1)
        assert mid.joint_angles["SpineF"][1] == pytest.approx(0.2)


class _FlatMesh:
    """Stand-in mesh whose silhouette is a square, so coverage is analytic."""


def _square_render(size=8):
    def render(mesh, pose, P, shape):
        m = np.zeros(shape, np.uint8)
        cx = int(round(pose.root_pos[0])); cy = int(round(pose.root_pos[1]))
        y0, y1 = max(0, cy - size), min(shape[0], cy + size)
        x0, x1 = max(0, cx - size), min(shape[1], cx + size)
        m[y0:y1, x0:x1] = 1
        return m
    return render


class TestCoverage:

    def test_zero_motion_reduces_to_sharp_binary(self, monkeypatch):
        import rpimocap.model.appearance as A
        monkeypatch.setattr(A, "render_mesh_pose_silhouette", _square_render())
        p = RatPose(root_pos=np.array([50.0, 50.0, 0.0]))
        sharp = A.render_coverage(None, p, np.eye(3, 4), (100, 100))
        blur = A.render_coverage(None, p, np.eye(3, 4), (100, 100),
                                 pose_next=p, fps=30.0, n_sub=7)
        assert np.array_equal(sharp, blur)
        assert set(np.unique(blur)) <= {0.0, 1.0}

    def test_motion_creates_partial_coverage(self, monkeypatch):
        import rpimocap.model.appearance as A
        monkeypatch.setattr(A, "render_mesh_pose_silhouette", _square_render())
        p0 = RatPose(root_pos=np.array([40.0, 50.0, 0.0]))
        p1 = RatPose(root_pos=np.array([60.0, 50.0, 0.0]))
        cov = A.render_coverage(None, p0, np.eye(3, 4), (100, 100),
                                pose_next=p1, fps=30.0, n_sub=9)
        partial = ((cov > 0) & (cov < 1)).sum()
        assert partial > 0                       # soft edges appeared
        assert cov.max() <= 1.0 and cov.min() >= 0.0
        assert (cov > 0).sum() > (16 * 16)       # footprint wider than sharp

    def test_shorter_exposure_means_less_blur(self, monkeypatch):
        import rpimocap.model.appearance as A
        monkeypatch.setattr(A, "render_mesh_pose_silhouette", _square_render())
        p0 = RatPose(root_pos=np.array([40.0, 50.0, 0.0]))
        p1 = RatPose(root_pos=np.array([70.0, 50.0, 0.0]))
        full = A.render_coverage(None, p0, np.eye(3, 4), (100, 100),
                                 pose_next=p1, fps=30.0, exposure_s=None, n_sub=9)
        short = A.render_coverage(None, p0, np.eye(3, 4), (100, 100),
                                  pose_next=p1, fps=30.0, exposure_s=1 / 1000.,
                                  n_sub=9)
        assert (short > 0).sum() < (full > 0).sum()


class TestEnergy:

    def test_correct_coverage_scores_better(self):
        post = np.zeros((60, 60), np.float32) + 0.02
        post[20:40, 20:40] = 0.98                 # the animal, per the features
        good = np.zeros((60, 60), np.float32); good[20:40, 20:40] = 1.0
        bad = np.zeros((60, 60), np.float32); bad[5:25, 5:25] = 1.0
        assert appearance_energy(good, post) < appearance_energy(bad, post)

    def test_soft_coverage_beats_hard_on_a_blurred_edge(self):
        """A half-covered edge pixel is genuinely a mixture; the soft coverage
        must score it better than committing to either class."""
        post = np.full((1, 1), 0.5, np.float32)
        soft = np.full((1, 1), 0.5, np.float32)
        hard = np.full((1, 1), 1.0, np.float32)
        assert appearance_energy(soft, post) == pytest.approx(
            appearance_energy(hard, post), abs=1e-6)
        post2 = np.full((1, 1), 0.75, np.float32)
        # with the pixel 75% likely fg, 75% coverage is the best explanation
        es = [appearance_energy(np.full((1, 1), m, np.float32), post2)
              for m in (0.0, 0.5, 1.0)]
        assert es[2] < es[1] < es[0]

    def test_roi_restricts_evaluation(self):
        """The ROI must exclude far-field pixels. Put a mismatch OUTSIDE it: the
        ROI energy should ignore that, the whole-image energy should not."""
        post = np.full((40, 40), 0.02, np.float32)
        post[10:20, 10:20] = 0.98
        cov = np.zeros((40, 40), np.float32)
        cov[10:20, 10:20] = 1.0                  # correct, inside the ROI
        cov[30:38, 30:38] = 1.0                  # WRONG, outside the ROI
        roi = np.zeros((40, 40), bool); roi[5:25, 5:25] = True
        e_roi = appearance_energy(cov, post, roi)
        e_all = appearance_energy(cov, post)
        assert e_roi < e_all                     # ROI blind to the far mismatch
        # and inside the ROI everything matches, so the energy is near-minimal
        assert e_roi == pytest.approx(-np.log(0.98), abs=1e-3)


class TestRoi:

    def test_roi_covers_mask_with_margin(self):
        m = np.zeros((100, 100), bool); m[40:60, 40:60] = True
        roi = roi_from_mask(m, margin=10)
        assert roi[40:60, 40:60].all()
        assert roi[30, 30] and not roi[10, 10]

    def test_empty_mask_gives_full_roi(self):
        roi = roi_from_mask(np.zeros((20, 20), bool))
        assert roi.all()


# --- tools/appearance_report.py helpers (pure functions) ---
class TestReportHelpers:

    def test_auc_perfect_and_chance(self):
        from tools.appearance_report import _auc
        assert _auc(np.array([3., 4., 5.]), np.array([0., 1., 2.])) == 1.0
        # identical distributions -> ~0.5
        rng = np.random.default_rng(0)
        a = rng.normal(0, 1, 500); b = rng.normal(0, 1, 500)
        assert abs(_auc(a, b) - 0.5) < 0.1

    def test_bright_reference_picks_the_blob(self):
        from tools.appearance_report import _bright_reference
        g = np.full((80, 80), 50, np.uint8)
        g[30:50, 30:50] = 200                       # one bright square
        floor = np.ones((80, 80), bool)
        ref = _bright_reference(g, floor, pct=90.0)
        assert ref[35:45, 35:45].all()              # blob found
        assert not ref[:10, :10].any()              # dark corner excluded


class TestChromaRobustness:
    """chroma must carry the colour cue without the raw ratio's failure mode."""

    def _frame(self, seed=7):
        rng = np.random.default_rng(seed)
        bgr = np.zeros((120, 120, 3), np.uint8)
        bgr[..., 0] = rng.normal(90, 6, (120, 120))     # B
        bgr[..., 1] = rng.normal(100, 6, (120, 120))    # G
        bgr[..., 2] = rng.normal(130, 6, (120, 120))    # R
        fg = np.zeros((120, 120), bool); fg[40:80, 40:80] = True
        bgr[..., 2][fg] = rng.normal(70, 6, fg.sum())
        return bgr, fg

    def test_chroma_is_bounded(self):
        bgr, _ = self._frame()
        f = image_features(bgr, coh_k=16, grain_k=16)
        assert f.chroma.min() >= -1.0 - 1e-6
        assert f.chroma.max() <= 1.0 + 1e-6

    def test_chroma_antisymmetric_under_channel_swap(self):
        """Swapping R and B (i.e. the wrong Bayer convention) must flip the
        SIGN of chroma and leave |d'| intact — the property that makes the
        model immune to the demosaic convention."""
        bgr, fg = self._frame()
        swapped = bgr[..., ::-1].copy()          # B <-> R
        f0 = image_features(bgr, coh_k=16, grain_k=16)
        f1 = image_features(swapped, coh_k=16, grain_k=16)
        assert np.allclose(f0.chroma, -f1.chroma, atol=1e-5)
        bg = ~fg
        d0 = AppearanceModel.from_masks(f0, fg, bg,
                                        features=("chroma",)).dprime["chroma"]
        d1 = AppearanceModel.from_masks(f1, fg, bg,
                                        features=("chroma",)).dprime["chroma"]
        assert d0 == pytest.approx(-d1, rel=1e-3)

    def test_chroma_survives_a_dim_denominator(self):
        """With B near zero the raw ratio R/B explodes; chroma stays finite."""
        rng = np.random.default_rng(3)
        bgr = np.zeros((80, 80, 3), np.uint8)
        bgr[..., 0] = rng.integers(0, 3, (80, 80))       # B ~ 0  (dim channel)
        bgr[..., 1] = 100
        bgr[..., 2] = rng.integers(90, 140, (80, 80))    # R
        f = image_features(bgr, coh_k=16, grain_k=16)
        assert np.isfinite(f.chroma).all()
        assert f.chroma.std() < 0.5              # well-behaved
        assert f.rb.std() > f.chroma.std() * 10  # the ratio is not


class TestBayerConvention:
    """The demosaic map must be shared, and must follow OpenCV's naming."""

    def test_single_source_of_truth(self):
        from rpimocap.io.export import BAYER_CODES, TiffCapture
        import tools.slice_raw_frames as srf
        assert TiffCapture._BAYER_CODES is BAYER_CODES
        # slice_raw_frames must delegate, not keep its own inverse map
        assert "COLOR_BayerRG2BGR" not in inspect.getsource(
            srf._bayer_to_preview_bgr).replace("COLOR_BayerRG2BGR``", "")

    def test_opencv_naming_is_opposite_corner(self):
        """RGGB sensors need COLOR_BayerBG2BGR (OpenCV names from the second
        row, second/third columns). Regression guard: the inverse mapping
        silently swaps R and B."""
        from rpimocap.io.export import BAYER_CODES
        assert BAYER_CODES["RGGB"] == "COLOR_BayerBG2BGR"
        assert BAYER_CODES["BGGR"] == "COLOR_BayerRG2BGR"

    def test_preview_demosaic_does_not_wrap_on_wide_data(self):
        """`raw >> 2` wrapped 100% of pixels on 12-bit-in-uint16 data."""
        import tools.slice_raw_frames as srf
        rng = np.random.default_rng(0)
        raw = rng.integers(4032, 65472, (64, 64)).astype(np.uint16)
        out = srf._bayer_to_preview_bgr(raw, "RGGB")
        assert out.dtype == np.uint8 and out.shape == (64, 64, 3)
        # a correct normalisation preserves rank order of brightness
        lin = srf._bayer_to_preview_bgr(
            np.linspace(4032, 65472, 64 * 64).reshape(64, 64).astype(np.uint16),
            "RGGB")
        g = lin[:, :, 1].astype(float).ravel()
        assert g[-1] > g[0]                      # monotone, i.e. no wraparound


class TestRenderPinholes:
    """The renderer must not leave sub-pixel rasterisation speckle.

    18% of this mesh's projected triangles round to zero area and fill nothing,
    punching ~1200 pinholes over ~8% of the body. Eroding such a mask is
    catastrophic (a 9x9 erode left 1.6% of the body), which starved the
    appearance model's foreground sample.
    """

    def _render(self, close):
        from rpimocap.model.mesh_model import (build_rat_mesh,
                                               render_mesh_silhouette,
                                               skin_mesh)
        mesh = build_rat_mesh()
        pose = RatPose(root_pos=np.array([0.0, 150.0, 50.0]),
                       root_rot=np.array([0.0, 0.0, 0.0]), scale=1.0)
        P = np.array([[1400.0, 0.0, 1000.0, 0.0],
                      [0.0, 1400.0, 540.0, 0.0],
                      [0.0, 0.0, 0.0, 700.0]])
        return render_mesh_silhouette(skin_mesh(mesh, pose), mesh.faces, P,
                                      (1080, 2028), close_pinholes=close) > 0

    @staticmethod
    def _interior_holes(m):
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        filled = np.zeros(m.shape, np.uint8)
        cv2.drawContours(filled, cnts, -1, 1, -1)
        return (filled.astype(bool) & ~m).sum()

    def test_default_render_has_no_interior_pinholes(self):
        assert self._interior_holes(self._render(True)) == 0

    def test_pinholes_exist_without_the_repair(self):
        """Regression guard: if the rasteriser is ever changed so this passes
        trivially, the close can be revisited."""
        assert self._interior_holes(self._render(False)) > 0

    def test_erosion_survives_the_repair(self):
        """The property that actually matters for bootstrap_masks: a 9x9 erode
        must retain most of the body, not be shredded by speckle."""
        m = self._render(True)
        eroded = cv2.erode(m.astype(np.uint8), np.ones((9, 9), np.uint8)).sum()
        assert eroded > 0.7 * m.sum()

    def test_repair_does_not_bridge_real_gaps(self):
        """A 3x3 close must not fill genuine background between body parts."""
        from rpimocap.model.mesh_model import render_mesh_silhouette
        m = np.zeros((100, 200), np.uint8)
        m[30:70, 20:60] = 255          # two blobs separated by a 40px gap
        m[30:70, 100:140] = 255
        closed = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        assert closed[30:70, 60:100].sum() == 0        # gap preserved
