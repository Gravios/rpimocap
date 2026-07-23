"""
rpimocap.model.appearance
=========================
**Textured** appearance model: match the rendered body against the *raw image
statistics* instead of against a thresholded silhouette.

Why
---
Silhouette fitting requires a binary detector mask as the target, and the two
cameras disagree about that mask (cam0 over-segments ~73k px, cam1
under-segments ~34k px on frame 2716) — that inconsistency is the hard ceiling
on multi-view IoU. Here the detector is only an *initialiser*: it seeds the pose
and bootstraps the appearance statistics, then never enters the objective. The
model instead predicts, per pixel, which surface is seen (rat vs floor), and the
energy is the region log-likelihood of the *observed* image features under those
two classes. No thresholding, so no mask inconsistency.

Features (the measured cues)
----------------------------
* ``chroma`` — bounded chromaticity ``(R-B)/(R+B)``. Fur and bedding separate
  at |d'| ~ 4, camera-consistent (both cameras agree). The SIGN depends on the
  Bayer convention the frames were demosaiced with, which has been inconsistent
  across this project's own tools; the magnitude does not. Use this rather than
  the raw ratio ``R/B``, which is unbounded and degenerates when the denominator
  channel is dim (measured: bedding std 3282, d' -> 0.00).
* ``coh``   — structure-tensor coherence of the median-bandpassed image, on
  **whitened** gradients (below): rat ~0.44 vs bedding ~0.11, d' ~ 3.5.
* ``grain`` — median-bandpass energy (grain density), the original cue: d' ~ 1.2.

These are combined naive-Bayes. That independence assumption is not free —
it was measured: colour and grain correlate only r ~ +0.08..+0.11 over the floor.

Whitening = cross-camera photometric calibration
------------------------------------------------
Median-bandpassed frames carry a strong near-vertical (~90 deg) gradient
anisotropy that is *camera-specific* (cam0 alignment R=0.63 vs cam1 R=0.25) and
sits on rat and bedding alike — a sensor/optical artefact, not scene structure.
So the same fur patch yields *different* raw statistics in the two views, and a
single shared surface texture cannot match both raw images.

Estimating the background gradient covariance ``Cbg`` from **floor bedding only**
and applying ``Cbg^{-1/2}`` per camera divides that transfer out: bedding
collapses toward isotropic in both views (R: 0.80->0.12 cam0, 0.73->0.24 cam1)
while the fur residual survives in both. Matching in whitened space is what makes
one surface appearance consistent across cameras.

Motion blur
-----------
At 30/50 Hz a moving rat smears within the exposure. Rendering sub-poses across
the shutter window and averaging gives a **soft coverage** map in [0, 1], which
is exactly the mixture weight the region likelihood wants: a blurred edge pixel
genuinely *is* a mixture of rat and floor, so

    E = -mean log( m * P_fg + (1 - m) * P_bg )

is the correct generative form rather than an approximation. Frame rate is an
explicit input (``fps``); exposure defaults to the full inter-frame interval.

Note the blur smear is ``speed * exposure``, so it is *independent of fps* at a
fixed shutter (measured: 1/250 s gives an identical 6351 partial px at both 30
and 50 Hz). Only the full-interval default tracks the frame rate.

Validated (frame 2716, real frames, both cameras)
-------------------------------------------------
* whitening reproduces the sensor axis: eig ratio 1.90 (cam0) / 1.55 (cam1),
  structure ~90 deg vertical, and isotropises the background to identity;
* feature d' vs an independent bright-rat reference — ``rb`` -4.4/-3.8,
  ``grain`` -4.5/-3.9, ``coh`` +1.9/+1.4;
* zero-motion coverage reduces exactly to the sharp binary silhouette.

Status: what is established, and what is not
--------------------------------------------
ESTABLISHED (frame 2716 and a 10-frame dwell sequence, both cameras):

* the per-pixel classification works — posterior AUC 0.98-0.99, median
  posterior 0.99+ on the animal and ~0.01 on bedding, symmetric across cameras;
* the bootstrap is no longer starved (fg ~18k px per camera) once the renderer's
  pinholes are repaired (see mesh_model.render_mesh_silhouette);
* the yaw optimum is temporally stable — over 10 consecutive dwell frames the
  appearance minimum moved by <=10 deg per camera, where the silhouette optimum
  jumped over a ~300 deg spread.

NOT ESTABLISHED — and an earlier claim here was wrong:

* the appearance energy does NOT resolve head-vs-tail. Rotating the body 180 deg
  costs only **1.3% of the energy range** (median over 10 frames, both cameras),
  i.e. the objective is essentially bimodal and the flip is free. Apparent
  cross-camera "agreement" on full yaw (~20 deg vs the silhouette's ~170 deg) is
  mostly the two cameras landing on the same side of that degeneracy, not
  evidence about orientation;
* on the body AXIS alone (mod 180, which is the part that is not degenerate),
  the two objectives are comparable: median cross-camera disagreement 10.0 deg
  for BOTH. The appearance objective is more consistent — no outliers, mean
  10.0 deg vs 27.5 deg, since the silhouette occasionally misses by 60-85 deg —
  but it is not sharper on the axis.

Why, and what it implies
------------------------
This model is *spatially homogeneous*: one foreground histogram for the entire
body. Nothing in it varies over the body surface, so a head-first and a
tail-first pose predict the same statistics and the flip cannot be penalised.
It is a region model that happens to use texture features, not a texture MAP.

Breaking the degeneracy needs per-region (or per-vertex) appearance — head,
trunk and rump carrying their own statistics, so the render predicts *where* on
the image each looks the way it does. That is the natural next step, and it is
also what makes the anisotropic fur-orientation cue usable, since surface
tangent is only meaningful once appearance is attached to surface location.

Still open: there is no reliable silhouette ground truth on this data. The
detector's mask is not a segmentation (see :func:`bootstrap_masks`), the
brightness threshold under-segments (5.7k px where the nominal render is ~21k),
and the seed render remains over-inclusive. The bootstrap works well enough to
calibrate a posterior, but its foreground is not verified to be the animal.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from ..detection.topo_detect import median_bandpass
from .fit import _scale_P
from .mesh_model import render_mesh_pose_silhouette
from .rat_skeleton import RatPose

FEATURES = ("chroma", "coh", "grain")
_EPS = 1e-6


# --------------------------------------------------------------------------
# whitening (per camera, estimated from floor bedding)
# --------------------------------------------------------------------------
def gradient_covariance(gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """2x2 gradient covariance of the median-bandpassed image over ``mask``."""
    mbp = median_bandpass(gray)
    gx = cv2.Sobel(mbp, cv2.CV_32F, 1, 0, ksize=3)[mask]
    gy = cv2.Sobel(mbp, cv2.CV_32F, 0, 1, ksize=3)[mask]
    return np.array([[float(np.mean(gx * gx)), float(np.mean(gx * gy))],
                     [float(np.mean(gx * gy)), float(np.mean(gy * gy))]])


def estimate_whitening(gray: np.ndarray, bedding_mask: np.ndarray) -> np.ndarray:
    """``Cbg^{-1/2}`` from floor-bedding gradients — the per-camera whitening.

    ``bedding_mask`` must be floor bedding only (exclude the rat and the walls);
    it defines what "background texture" means for this camera.
    """
    C = gradient_covariance(gray, bedding_mask.astype(bool))
    ev, evec = np.linalg.eigh(C)
    ev = np.maximum(ev, _EPS)
    return evec @ np.diag(ev ** -0.5) @ evec.T


def whitening_report(gray: np.ndarray, bedding_mask: np.ndarray) -> dict:
    """Diagnostics for the estimated whitening (anisotropy + dominant axis)."""
    C = gradient_covariance(gray, bedding_mask.astype(bool))
    ev, evec = np.linalg.eigh(C)
    return {"eig_ratio": float(ev[1] / max(ev[0], _EPS)),
            "dominant_grad_deg": float(np.degrees(
                np.arctan2(evec[1, 1], evec[0, 1])) % 180.0)}


# --------------------------------------------------------------------------
# per-pixel image features
# --------------------------------------------------------------------------
@dataclass
class FeatureMaps:
    """Per-pixel feature maps for one frame (computed ONCE per frame)."""
    chroma: np.ndarray                # (R-B)/(R+B), bounded in [-1, 1]
    coh: np.ndarray
    grain: np.ndarray
    theta: np.ndarray = None          # whitened structure orientation (deg)
    rb: np.ndarray = None             # R/B — diagnostic only, see image_features

    def get(self, name: str) -> np.ndarray:
        return getattr(self, name)

    @property
    def shape(self):
        return self.chroma.shape


def image_features(bgr: np.ndarray, W: np.ndarray = None,
                   coh_k: int = 32, grain_k: int = 32) -> FeatureMaps:
    """Feature maps from a BGR frame.

    ``W`` is the per-camera whitening from :func:`estimate_whitening`; if None
    the gradients are left raw (then ``coh`` is sensor-dominated and separates
    poorly — whitening is what makes it discriminative).

    ``coh_k=32`` is not arbitrary: the coherence window must fit *inside* the
    animal or it straddles the boundary and mixes fur with bedding. Measured on
    frame 2716 (rat bbox 143x74 px), d' peaks at 32 and falls off either side —
    16: +1.50, 24: +1.89, **32: +2.07**, 48: +1.75, 64: +1.46. Scale this with
    the animal's *narrow* dimension in pixels, not its area.
    """
    img = bgr.astype(np.float32)
    if img.ndim == 2:                       # single channel: no colour
        B = G = R = img
    else:
        B, G, R = img[..., 0], img[..., 1], img[..., 2]
    # Colour cue: bounded chromaticity, NOT the raw ratio R/B.
    #
    # R/B is unbounded and its denominator is whichever channel happens to be
    # dim, so pixels with B ~ 0 blow the ratio up: measured on frame 2716 cam0
    # the bedding std reached 3282 and d' collapsed to +0.00, i.e. the feature
    # became meaningless. (R-B)/(R+B) carries the same information confined to
    # [-1, 1] and is antisymmetric under an R/B channel swap, so it gives a
    # STABLE |d'| = 4.16 (cam0) / 3.56 (cam1) whichever Bayer convention the
    # frames were demosaiced with — which matters, because that convention has
    # been inconsistent across the project's own tools (see io.export.
    # BAYER_CODES). The sign tracks the convention; the separation does not.
    chroma = (R - B) / (R + B + _EPS)
    rb = R / (B + _EPS)               # kept for diagnostics/back-compat only

    # Grain is RMS bandpass energy normalised by local mean intensity, i.e. a
    # local CONTRAST. The raw energy cv2.boxFilter(mbp**2) is intensity-
    # confounded -- a dark region has near-zero bandpass energy simply because
    # it is dark, not because it is smooth -- which inverts the cue in shadow.
    # Measured on frame 2716 (rat vs rest-of-detection): raw energy d' = -1.15,
    # normalised contrast d' = -2.61. This is the same trap the topo detector
    # sidesteps by counting grain *peaks* rather than summing energy.
    mbp = median_bandpass(G)
    rms = np.sqrt(cv2.boxFilter(mbp * mbp, -1, (grain_k, grain_k)))
    grain = rms / (cv2.boxFilter(G, -1, (grain_k, grain_k)) + _EPS)

    gx = cv2.Sobel(mbp, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(mbp, cv2.CV_32F, 0, 1, ksize=3)
    if W is not None:
        g = np.asarray(W, float) @ np.stack([gx.ravel(), gy.ravel()])
        gx = g[0].reshape(gx.shape).astype(np.float32)
        gy = g[1].reshape(gy.shape).astype(np.float32)
    jxx = cv2.boxFilter(gx * gx, -1, (coh_k, coh_k))
    jyy = cv2.boxFilter(gy * gy, -1, (coh_k, coh_k))
    jxy = cv2.boxFilter(gx * gy, -1, (coh_k, coh_k))
    coh = np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2) / (jxx + jyy + _EPS)
    theta = (np.degrees(0.5 * np.arctan2(2 * jxy, jxx - jyy)) + 90.0) % 180.0
    return FeatureMaps(chroma=chroma, coh=coh, grain=grain, theta=theta, rb=rb)


# --------------------------------------------------------------------------
# two-class appearance model (naive Bayes over the features)
# --------------------------------------------------------------------------
def bootstrap_masks(floor_mask, fg_seed, coh_k: int = 32,
                    fg_erode: int = 9, bg_gap: int = 41, roi=None):
    """Foreground / background samples for bootstrapping the appearance model.

    ``fg_seed`` must be a *localisation* of the animal — in practice the model
    **rendered at the detector's triangulated 3D point**, not the detector's
    mask. That distinction is the whole design:

    * the detector's 3D point is reliable (frame 2716: 2.6 px reprojection
      error, landing within 12-22 px of the true rat centroid in both views);
    * the detector's *mask* is not a segmentation. On frame 2716 cam0's mask is
      73k px for a ~5.7k px animal (13x oversized; its mean intensity 85 is
      indistinguishable from the floor's 90), and cam1's is displaced by 91 px
      and overlaps only **7%** of the animal, with a floor-like interior (91 vs
      the rat's 148). Bootstrapping from the mask collapses the model
      (rb d' -4.7 -> -1.4).

    So the detector locates; it does not segment. Everything downstream keys off
    the point.

    ``bg`` is the floor eroded by ``coh_k`` (so every coherence window lies
    wholly on the floor rather than straddling the wall) minus the seed dilated
    by ``bg_gap``. Skipping the floor erosion inflates background coherence
    spread and measurably degrades separation (d' 3.1 -> 2.5).

    ``roi``, if given, restricts the background so ``prior_fg`` reflects the
    window the energy is actually evaluated over.
    """
    floor = np.asarray(floor_mask, bool)
    seed = np.asarray(fg_seed, bool).astype(np.uint8)
    fg = cv2.erode(seed, np.ones((fg_erode, fg_erode), np.uint8)).astype(bool)
    fg &= floor
    floor_in = cv2.erode(floor.astype(np.uint8),
                         np.ones((coh_k, coh_k), np.uint8)).astype(bool)
    halo = cv2.dilate(seed, np.ones((bg_gap, bg_gap), np.uint8)).astype(bool)
    bg = floor_in & (~halo)
    if roi is not None:
        bg = bg & np.asarray(roi, bool)
    return fg, bg


@dataclass
class AppearanceModel:
    """Histogram appearance model for rat (fg) vs floor (bg).

    Naive Bayes across ``features``; justified empirically by the measured
    near-orthogonality of colour and grain (r ~ +0.1), not assumed for
    convenience. Histograms (rather than Gaussians) because the feature
    distributions are skewed and heavy-tailed.
    """
    edges: dict
    hist_fg: dict
    hist_bg: dict
    prior_fg: float = 0.5
    features: tuple = FEATURES
    dprime: dict = field(default_factory=dict)

    @classmethod
    def from_masks(cls, feats: FeatureMaps, fg_mask, bg_mask,
                   features=FEATURES, bins: int = 48, smooth: float = 1.0,
                   clip_pct=(0.5, 99.5)):
        """Bootstrap the model from a foreground/background segmentation.

        In use, ``fg_mask``/``bg_mask`` come from the *detector* (the
        initialiser) — the model self-calibrates per session rather than
        hard-coding session-specific constants.
        """
        fg_mask = np.asarray(fg_mask, bool)
        bg_mask = np.asarray(bg_mask, bool)
        if fg_mask.sum() < 16 or bg_mask.sum() < 16:
            raise ValueError("need >=16 px in each of fg_mask and bg_mask")
        edges, hf, hb, dp = {}, {}, {}, {}
        for name in features:
            m = feats.get(name)
            a, b = m[fg_mask], m[bg_mask]
            lo = np.percentile(np.concatenate([a, b]), clip_pct[0])
            hi = np.percentile(np.concatenate([a, b]), clip_pct[1])
            if hi <= lo:
                hi = lo + 1.0
            e = np.linspace(lo, hi, bins + 1)
            ha, _ = np.histogram(np.clip(a, lo, hi), bins=e)
            hbb, _ = np.histogram(np.clip(b, lo, hi), bins=e)
            ha = ha.astype(np.float64) + smooth
            hbb = hbb.astype(np.float64) + smooth
            edges[name] = e
            hf[name] = ha / ha.sum()
            hb[name] = hbb / hbb.sum()
            dp[name] = float((np.median(a) - np.median(b)) /
                             (0.5 * (a.std() + b.std()) + _EPS))
        prior = float(fg_mask.sum()) / float(fg_mask.sum() + bg_mask.sum())
        return cls(edges=edges, hist_fg=hf, hist_bg=hb, prior_fg=prior,
                   features=tuple(features), dprime=dp)

    def _bin(self, name, m):
        e = self.edges[name]
        idx = np.digitize(m, e) - 1
        return np.clip(idx, 0, len(e) - 2)

    def log_odds(self, feats: FeatureMaps) -> np.ndarray:
        """Per-pixel ``log p(f|fg) - log p(f|bg)`` (naive Bayes sum)."""
        out = np.zeros(feats.shape, np.float32)
        for name in self.features:
            i = self._bin(name, feats.get(name))
            out += (np.log(self.hist_fg[name][i]) -
                    np.log(self.hist_bg[name][i])).astype(np.float32)
        return out

    def posterior_fg(self, feats: FeatureMaps, use_prior: bool = True
                     ) -> np.ndarray:
        """Per-pixel P(fg | features) in [0, 1] — computed once per frame."""
        z = self.log_odds(feats)
        if use_prior:
            p = float(np.clip(self.prior_fg, 1e-4, 1 - 1e-4))
            z = z + np.log(p / (1.0 - p))
        return (1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))).astype(np.float32)


# --------------------------------------------------------------------------
# motion-blurred coverage rendering
# --------------------------------------------------------------------------
def interp_pose(p0: RatPose, p1: RatPose, t: float) -> RatPose:
    """Linear pose interpolation. Euler angles are interpolated componentwise —
    an approximation, but exposure-scale deltas are small (a few degrees)."""
    keys = set(p0.joint_angles) | set(p1.joint_angles)
    ja = {}
    for k in keys:
        a0 = np.asarray(p0.joint_angles.get(k, (0.0, 0.0, 0.0)), float)
        a1 = np.asarray(p1.joint_angles.get(k, (0.0, 0.0, 0.0)), float)
        ja[k] = tuple(a0 + t * (a1 - a0))
    return RatPose(root_pos=p0.root_pos + t * (p1.root_pos - p0.root_pos),
                   root_rot=p0.root_rot + t * (p1.root_rot - p0.root_rot),
                   scale=float(p0.scale + t * (p1.scale - p0.scale)),
                   joint_angles=ja)


def exposure_duty(fps: float, exposure_s: float = None) -> float:
    """Fraction of the inter-frame interval the shutter is open.

    ``exposure_s=None`` means the full interval (duty 1.0) — the worst-case
    blur. At 30 Hz a 1/1000 s shutter gives duty 0.03 (effectively no blur);
    at 50 Hz the same shutter gives 0.05.
    """
    if exposure_s is None:
        return 1.0
    return float(np.clip(exposure_s * float(fps), 1e-3, 1.0))


def render_coverage(mesh, pose: RatPose, P: np.ndarray, image_shape,
                    pose_next: RatPose = None, fps: float = None,
                    exposure_s: float = None, n_sub: int = 5,
                    downscale: int = 1) -> np.ndarray:
    """Soft coverage map in [0, 1] — the fraction of the exposure each pixel
    was covered by the body.

    With ``pose_next`` (the pose one frame later) and ``fps``, the shutter spans
    ``duty = exposure_s * fps`` of the interval starting at this frame's
    timestamp, and ``n_sub`` sub-poses are rendered and averaged. Without
    motion this reduces exactly to the sharp binary silhouette.
    """
    shp = image_shape
    Pm = P
    if downscale and downscale > 1:
        Pm = _scale_P(P, 1.0 / downscale)
        shp = (image_shape[0] // downscale, image_shape[1] // downscale)

    if pose_next is None or n_sub <= 1:
        return (render_mesh_pose_silhouette(mesh, pose, Pm, shp) > 0
                ).astype(np.float32)

    duty = exposure_duty(fps if fps is not None else 1.0, exposure_s)
    end = interp_pose(pose, pose_next, duty)
    acc = np.zeros(shp, np.float32)
    for i in range(n_sub):
        t = (i + 0.5) / n_sub
        acc += (render_mesh_pose_silhouette(
            mesh, interp_pose(pose, end, t), Pm, shp) > 0).astype(np.float32)
    return acc / float(n_sub)


# --------------------------------------------------------------------------
# region energy
# --------------------------------------------------------------------------
def appearance_energy(coverage: np.ndarray, post_fg: np.ndarray,
                      roi: np.ndarray = None) -> float:
    """Region log-likelihood energy (lower is better).

    ``E = -mean log( m*P_fg + (1-m)*P_bg )`` over ``roi``. ``coverage`` may be
    soft (motion blur), in which case each pixel is correctly treated as a
    rat/floor mixture. ``post_fg`` is precomputed once per frame, so the only
    per-iteration cost is the render plus this reduction.
    """
    m = coverage.astype(np.float32)
    p = m * post_fg + (1.0 - m) * (1.0 - post_fg)
    e = -np.log(np.clip(p, 1e-6, 1.0))
    return float(e.mean() if roi is None else e[roi].mean())


def roi_from_mask(mask, margin: int = 48) -> np.ndarray:
    """Fixed evaluation window: bbox of ``mask`` dilated by ``margin``.

    The ROI is fixed for a whole fit (from the detector's initial mask) so the
    energy stays comparable across poses; without it the ~2.2 Mpx background
    swamps the ~5 kpx animal and the objective goes flat.
    """
    mask = np.asarray(mask, bool)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return np.ones(mask.shape, bool)
    out = np.zeros(mask.shape, bool)
    y0 = max(0, ys.min() - margin); y1 = min(mask.shape[0], ys.max() + margin)
    x0 = max(0, xs.min() - margin); x1 = min(mask.shape[1], xs.max() + margin)
    out[y0:y1, x0:x1] = True
    return out
