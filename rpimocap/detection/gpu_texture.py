"""
rpimocap.detection.gpu_texture
==============================
GPU-capable port of the dominant per-frame cost — the dense Gabor
descriptor and the texture-distance map (PROCESSING_ARCHITECTURE.md §5,
"GPU offload plan", item 1: the data-parallel bottleneck).

Design
------
The descriptor is 24 separable convolutions + pooling per frame, and the
distance is an elementwise z-score reduced across descriptor channels —
both ideal GPU workloads. This module expresses that math in an
ARRAY-MODULE-AGNOSTIC way: the same code runs on NumPy (+ SciPy ndimage)
for the CPU path or CuPy (+ cupyx.scipy.ndimage) for the GPU path, chosen
by a `device` argument.

Why a separate module (not a rewrite of texture_distance.py): the
canonical CPU implementation there uses OpenCV (cv2.filter2D / boxFilter)
and is validated by the existing suite. We don't touch it. Instead this
module mirrors the math with ndimage equivalents:
  * cv2.filter2D (correlation, BORDER_REFLECT_101)  ≡  ndimage.correlate(mode='mirror')
  * cv2.boxFilter                                   ≡  ndimage.uniform_filter
and an equivalence test asserts the ndimage/CuPy path matches the cv2
path within floating-point tolerance (and that the resulting detection
masks agree). So correctness is proven on CPU here; the GPU path is the
identical code with xp=cupy and is checked by a CuPy-guarded test on the
actual hardware.

Typical use (offline batch on the workstation GPU):
    from rpimocap.detection.gpu_texture import texture_distance_device
    dist = texture_distance_device(gray, model_mean, model_std, kernels,
                                   n_orient, n_scales, device="gpu")
The model mean/std are uploaded once and kept resident (see
upload_model); per-frame the gray frame streams up and the distance map
(or just its cropped candidate region) streams down.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


# ────────────────────────────────────────────────────────────────────
#  Device / array-module selection
# ────────────────────────────────────────────────────────────────────


def _gpu_available() -> bool:
    try:
        import cupy as cp           # noqa: F401
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def array_module(device: str = "auto"):
    """Return (xp, ndi, on_gpu) for the requested device.

    device : 'cpu' → (numpy, scipy.ndimage, False)
             'gpu' → (cupy, cupyx.scipy.ndimage, True); raises if no GPU
             'auto' → gpu if available else cpu
    """
    want_gpu = (device == "gpu"
                or (device == "auto" and _gpu_available()))
    if want_gpu:
        import cupy as cp
        import cupyx.scipy.ndimage as cndi
        if cp.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError("device='gpu' but no CUDA device found")
        return cp, cndi, True
    import scipy.ndimage as ndi
    return np, ndi, False


def to_host(a):
    """Bring an array back to host (numpy) regardless of device."""
    try:
        import cupy as cp
        if isinstance(a, cp.ndarray):
            return cp.asnumpy(a)
    except Exception:
        pass
    return np.asarray(a)


# ────────────────────────────────────────────────────────────────────
#  Descriptor + distance (device-agnostic)
# ────────────────────────────────────────────────────────────────────


def gabor_descriptor_device(
        gray, kernels: Sequence[np.ndarray],
        n_orient: int, n_scales: int,
        smooth_k: int = 7, rotation_invariant: bool = True,
        log_transform: bool = False,
        device: str = "auto", xp=None, ndi=None):
    """Dense Gabor descriptor, computed on `device`. Mirrors
    texture_distance.dense_gabor_descriptor exactly (same pooling, same
    border handling) but via ndimage so it runs on CuPy.

    Returns a (D, H, W) array on the SAME device as chosen (use to_host
    to bring it back). If xp/ndi are passed they're reused (so a caller
    can keep the model resident and avoid re-importing)."""
    if xp is None or ndi is None:
        xp, ndi, _ = array_module(device)

    gray_f = xp.asarray(gray, dtype=xp.float32)
    H, W = gray_f.shape

    def _filt(img, kern):
        # cv2.filter2D = correlation with BORDER_REFLECT_101 ≡ ndimage
        # correlate(mode='mirror'). abs() to match the |response|.
        k = xp.asarray(kern, dtype=xp.float32)
        return xp.abs(ndi.correlate(img, k, mode="mirror"))

    def _box(img):
        if smooth_k > 1:
            # cv2.boxFilter (normalized) ≡ uniform_filter (mean).
            return ndi.uniform_filter(img, size=smooth_k, mode="mirror")
        return img

    if rotation_invariant:
        D = 3 * n_scales
        out = xp.empty((D, H, W), dtype=xp.float32)
        for s in range(n_scales):
            resp = xp.empty((n_orient, H, W), dtype=xp.float32)
            for o in range(n_orient):
                resp[o] = _box(_filt(gray_f, kernels[s * n_orient + o]))
            out[s * 3 + 0] = resp.max(axis=0)
            out[s * 3 + 1] = resp.mean(axis=0)
            out[s * 3 + 2] = resp.std(axis=0)
    else:
        D = n_orient * n_scales
        out = xp.empty((D, H, W), dtype=xp.float32)
        for i, kern in enumerate(kernels):
            out[i] = _box(_filt(gray_f, kern))
    if log_transform:
        out = xp.log1p(out)        # variance-stabilize (see CPU twin)
    return out


def texture_distance_device(
        gray, model_mean, model_std,
        kernels: Sequence[np.ndarray], n_orient: int, n_scales: int,
        smooth_k: int = 7, rotation_invariant: bool = True,
        log_transform: bool = False,
        persistence_map=None, persistence_power: float = 1.0,
        anisotropy_weight=None, roi_mask=None, post_smooth_k: int = 15,
        device: str = "auto", xp=None, ndi=None, return_host: bool = True):
    """Texture-distance map on `device`. Mirrors
    texture_distance.texture_distance_map (z-score → RMS across channels
    → persistence/anisotropy/ROI gating → post-smooth).

    model_mean/model_std : the (D,H,W) background model. Pass arrays
        already on the device (see upload_model) to keep them resident;
        host arrays are uploaded each call otherwise.
    return_host : if True, return a numpy array; else leave on device.
    """
    if xp is None or ndi is None:
        xp, ndi, _ = array_module(device)

    desc = gabor_descriptor_device(
        gray, kernels, n_orient, n_scales, smooth_k=smooth_k,
        rotation_invariant=rotation_invariant,
        log_transform=log_transform, xp=xp, ndi=ndi)

    mean = xp.asarray(model_mean, dtype=xp.float32)
    std = xp.asarray(model_std, dtype=xp.float32)
    z = (desc - mean) / std
    dist = xp.sqrt(xp.mean(z * z, axis=0)).astype(xp.float32)

    if persistence_map is not None:
        damp = xp.clip(1.0 - xp.asarray(persistence_map), 0.0, 1.0)
        if persistence_power != 1.0:
            damp = damp ** float(persistence_power)
        dist = dist * damp.astype(xp.float32)
    if anisotropy_weight is not None:
        w = xp.clip(xp.asarray(anisotropy_weight), 0.0, 1.0)
        dist = dist * w.astype(xp.float32)
    if roi_mask is not None:
        dist = dist * (xp.asarray(roi_mask) > 0).astype(xp.float32)
    if post_smooth_k and post_smooth_k > 1:
        dist = ndi.uniform_filter(dist, size=post_smooth_k,
                                  mode="mirror").astype(xp.float32)

    return to_host(dist) if return_host else dist


def upload_model(model_mean, model_std, device: str = "gpu"):
    """Upload the (D,H,W) background model to the device once and return
    (mean_dev, std_dev) to pass into texture_distance_device as
    model_mean/model_std — keeping it resident across frames (the
    resident-vs-streaming split from the architecture doc)."""
    xp, _, _ = array_module(device)
    return (xp.asarray(model_mean, dtype=xp.float32),
            xp.asarray(model_std, dtype=xp.float32))
