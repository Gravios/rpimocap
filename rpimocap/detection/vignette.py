"""
vignette.py — flat-field (NIR vignette) correction
====================================================
NIR-illuminated sensors and cheap M12 lenses both attenuate signal in
the corners of the image — pixels at the periphery see roughly 30–60 %
of the centre's flux for the same scene radiance. After background
subtraction this introduces a systematic bias: corner pixels can light
up as foreground at much smaller true intensity differences than
centre pixels.

This module provides two flat-field correction utilities:

  load_flat_field(path)         load a flat-field PNG / NPZ
  apply_flat_field(frame, ff)   divide-and-rescale correction

and a synthesizer for setups that didn't capture a true flat-field at
rig assembly:

  synthesize_flat_field(background_image)

which fits a smooth radial polynomial to the background image (which
is a low-pass-of-the-mean-scene approximation to the true illumination
profile) and uses that as the flat-field. This is less accurate than
a real flat-field capture, but recovers most of the vignette bias.

Correction model
----------------
``corrected = frame * ff_mean / flat_field``

where ``flat_field`` is normalised so its mean equals 1; the rescale
keeps the global brightness invariant so downstream thresholds tuned
on uncorrected frames still work.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def _to_float32(img: np.ndarray) -> np.ndarray:
    return np.asarray(img, dtype=np.float32)


def load_flat_field(path: "str | Path") -> np.ndarray:
    """Load a flat-field image from PNG, TIFF, or NPZ.

    NPZ files must contain a single ``flat`` array. The returned array
    is float32 with mean 1 (i.e. already normalised for use with
    ``apply_flat_field``).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix == ".npz":
        d = np.load(path)
        flat = d["flat"]
    else:
        import cv2
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise IOError(f"Could not read flat-field image: {path}")
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        flat = img
    flat = _to_float32(flat)
    m = float(flat.mean())
    if m <= 0.0:
        raise ValueError(
            f"Flat-field has non-positive mean ({m}); check the input file.")
    return flat / m


def apply_flat_field(
    frame:     np.ndarray,
    flat:      np.ndarray,
    clip:      bool = True,
) -> np.ndarray:
    """Divide-and-rescale flat-field correction.

    Parameters
    ----------
    frame : (H, W) or (H, W, C) raw frame (uint8 or float).
    flat  : (H, W) flat-field, mean 1, same shape as ``frame``.
    clip  : if True (default) the result is clipped to [0, 255] and
            cast back to uint8 (matching the input dtype). Set False
            to keep the float result, useful when the correction will
            be chained with further float-domain work.
    """
    frame_f = _to_float32(frame)
    flat = _to_float32(flat)
    if flat.shape[:2] != frame_f.shape[:2]:
        raise ValueError(
            f"flat shape {flat.shape} does not match frame "
            f"shape {frame_f.shape}")
    # Avoid division by zero at masked / clipped pixels of the flat-field
    safe = np.maximum(flat, 1e-3)
    if frame_f.ndim == 3:
        safe = safe[:, :, None]
    out = frame_f / safe
    # Re-anchor mean so global brightness is preserved
    out *= float(frame_f.mean()) / float(out.mean() + 1e-9)
    if clip:
        return np.clip(out, 0.0, 255.0).astype(np.uint8)
    return out


# --------------------------------------------------------------------------- #
#  Flat-field synthesis from a background image                                #
# --------------------------------------------------------------------------- #

def synthesize_flat_field(
    background: np.ndarray,
    *,
    poly_order:    int = 4,
    downsample:    int = 8,
) -> np.ndarray:
    """Fit a smooth radial polynomial flat-field to a background image.

    This is a fallback for setups that did not capture a true flat-field
    target at assembly time. The background image (built from the
    median of N animal-free frames by ``BackgroundModel.from_captures``)
    is treated as an approximation of the illumination profile times a
    constant scene albedo. A radial polynomial of order ``poly_order``
    in r² (normalised to image diagonal) is fit by least squares,
    capturing the main vignette pattern while ignoring scene structure.

    Parameters
    ----------
    background : (H, W) float or uint8 image — typically the bg image
                 from ``BackgroundModel.bg0`` or ``bg1``.
    poly_order : order of the polynomial in normalised r². 4 is a good
                 default; higher orders start to fit non-vignette
                 structure in the background.
    downsample : factor to downsample the fit (the polynomial is fit on
                 a coarse grid then evaluated on the full grid). Speeds
                 the fit up by ~64× at default; quality is unaffected
                 because the underlying signal is very smooth.

    Returns
    -------
    flat : (H, W) float32 array, mean 1.
    """
    bg = _to_float32(background)
    if bg.ndim != 2:
        raise ValueError(f"background must be 2D; got shape {bg.shape}")
    h, w = bg.shape
    ds = max(1, int(downsample))
    bgd = bg[::ds, ::ds]
    hh, ww = bgd.shape
    # Normalised radius² (0 at centre, ~0.5 at corner)
    yy, xx = np.mgrid[0:hh, 0:ww].astype(np.float32)
    cy, cx = (hh - 1) / 2.0, (ww - 1) / 2.0
    diag2  = float(cx ** 2 + cy ** 2)
    r2 = ((xx - cx) ** 2 + (yy - cy) ** 2) / diag2

    # Least-squares fit: bg ≈ Σ c_k * r²^k
    A = np.stack([r2 ** k for k in range(poly_order + 1)], axis=-1)
    A_flat = A.reshape(-1, poly_order + 1)
    b_flat = bgd.reshape(-1)
    coef, *_ = np.linalg.lstsq(A_flat, b_flat, rcond=None)

    # Evaluate on the full grid
    yy2, xx2 = np.mgrid[0:h, 0:w].astype(np.float32)
    fy, fx = (h - 1) / 2.0, (w - 1) / 2.0
    fdiag2 = float(fx ** 2 + fy ** 2)
    r2_full = ((xx2 - fx) ** 2 + (yy2 - fy) ** 2) / fdiag2
    flat = np.zeros((h, w), dtype=np.float32)
    for k, c in enumerate(coef):
        flat += float(c) * (r2_full ** k)
    flat = np.maximum(flat, 1e-3)
    flat /= float(flat.mean())
    return flat.astype(np.float32)


__all__ = [
    "load_flat_field",
    "apply_flat_field",
    "synthesize_flat_field",
]
