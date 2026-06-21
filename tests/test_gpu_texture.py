"""Tests for the GPU-capable texture descriptor / distance port.

The 'cpu' device path (NumPy + scipy.ndimage) is validated here against
the canonical OpenCV implementation in texture_distance. The 'gpu' path
is the identical code with CuPy and is checked by the cupy-guarded test,
which skips cleanly when no GPU/CuPy is present (it runs on the
workstation).
"""
import numpy as np
import pytest

from rpimocap.detection import texture_distance as td
from rpimocap.detection import gpu_texture as gt
from rpimocap.detection.rat_texture import build_gabor_kernels


N_ORIENT = 8
SCALES = [5, 9, 13]
KERNELS = build_gabor_kernels(
    [i * np.pi / N_ORIENT for i in range(N_ORIENT)], SCALES)


try:
    import cupy as _cp  # noqa: F401
    _HAVE_GPU = gt._gpu_available()
except Exception:
    _HAVE_GPU = False


def _scene(seed=0, H=200, W=260):
    import cv2
    rng = np.random.RandomState(seed)
    frames = [rng.randint(60, 160, (H, W)).astype(np.uint8)
              for _ in range(6)]
    model = td.BackgroundTextureModel()
    for f in frames:
        model.accumulate(td.dense_gabor_descriptor(
            f, KERNELS, N_ORIENT, len(SCALES), smooth_k=7))
    model.finalize()
    cur = rng.randint(60, 160, (H, W)).astype(np.uint8)
    cur[80:130, 90:160] = cv2.GaussianBlur(
        rng.randint(150, 230, (50, 70)).astype(np.uint8), (5, 5), 0)
    return model, cur


# ────────────────────────────────────────────────────────────────────
#  CPU device path == canonical cv2 path
# ────────────────────────────────────────────────────────────────────


class TestCpuMatchesCanonical:

    def test_descriptor_matches(self):
        _, cur = _scene(1)
        d_cv = td.dense_gabor_descriptor(
            cur, KERNELS, N_ORIENT, len(SCALES), smooth_k=7)
        d_dev = gt.to_host(gt.gabor_descriptor_device(
            cur, KERNELS, N_ORIENT, len(SCALES), smooth_k=7,
            device="cpu"))
        rng = d_cv.max() - d_cv.min()
        assert np.abs(d_cv - d_dev).max() / (rng + 1e-9) < 1e-4

    def test_descriptor_non_rotation_invariant(self):
        _, cur = _scene(2)
        d_cv = td.dense_gabor_descriptor(
            cur, KERNELS, N_ORIENT, len(SCALES), smooth_k=7,
            rotation_invariant=False)
        d_dev = gt.to_host(gt.gabor_descriptor_device(
            cur, KERNELS, N_ORIENT, len(SCALES), smooth_k=7,
            rotation_invariant=False, device="cpu"))
        assert d_cv.shape == d_dev.shape == (N_ORIENT * len(SCALES),
                                             *cur.shape)
        rng = d_cv.max() - d_cv.min()
        assert np.abs(d_cv - d_dev).max() / (rng + 1e-9) < 1e-4

    def test_distance_matches(self):
        model, cur = _scene(3)
        dist_cv = td.texture_distance_map(
            cur, model, KERNELS, N_ORIENT, len(SCALES), smooth_k=7,
            post_smooth_k=15)
        dist_dev = gt.texture_distance_device(
            cur, model.mean, model.std, KERNELS, N_ORIENT, len(SCALES),
            smooth_k=7, post_smooth_k=15, device="cpu")
        rel = np.abs(dist_cv - dist_dev).max() / (dist_cv.max() + 1e-9)
        assert rel < 0.02

    def test_distance_masks_agree(self):
        model, cur = _scene(4)
        dist_cv = td.texture_distance_map(
            cur, model, KERNELS, N_ORIENT, len(SCALES), smooth_k=7,
            post_smooth_k=15)
        dist_dev = gt.texture_distance_device(
            cur, model.mean, model.std, KERNELS, N_ORIENT, len(SCALES),
            smooth_k=7, post_smooth_k=15, device="cpu")
        thr = dist_cv.mean() + 2 * dist_cv.std()
        a, b = dist_cv > thr, dist_dev > thr
        iou = (a & b).sum() / max((a | b).sum(), 1)
        assert iou > 0.98

    def test_gating_matches(self):
        """Persistence + anisotropy + roi gating match the canonical
        path too."""
        model, cur = _scene(5)
        H, W = cur.shape
        pers = np.random.RandomState(0).rand(H, W).astype(np.float32)
        aniso = np.linspace(0.2, 1.0, W, dtype=np.float32)[None, :] \
            .repeat(H, 0)
        roi = np.ones((H, W), np.uint8)
        roi[:20] = 0
        dist_cv = td.texture_distance_map(
            cur, model, KERNELS, N_ORIENT, len(SCALES), smooth_k=7,
            persistence_map=pers, persistence_power=2.0,
            anisotropy_weight=aniso, roi_mask=roi, post_smooth_k=15)
        dist_dev = gt.texture_distance_device(
            cur, model.mean, model.std, KERNELS, N_ORIENT, len(SCALES),
            smooth_k=7, persistence_map=pers, persistence_power=2.0,
            anisotropy_weight=aniso, roi_mask=roi, post_smooth_k=15,
            device="cpu")
        rel = np.abs(dist_cv - dist_dev).max() / (dist_cv.max() + 1e-9)
        assert rel < 0.02


class TestDeviceSelection:

    def test_array_module_cpu(self):
        xp, ndi, on_gpu = gt.array_module("cpu")
        assert xp is np
        assert on_gpu is False

    def test_to_host_passthrough(self):
        a = np.arange(6).reshape(2, 3)
        assert np.array_equal(gt.to_host(a), a)

    def test_upload_model_cpu(self):
        model, _ = _scene(6)
        mean, std = gt.upload_model(model.mean, model.std, device="cpu")
        assert mean.shape == model.mean.shape


# ────────────────────────────────────────────────────────────────────
#  GPU path == CPU path (runs only with CuPy + a GPU)
# ────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(not _HAVE_GPU,
                    reason="no CuPy/GPU available")
class TestGpuMatchesCpu:

    def test_distance_gpu_matches_cpu(self):
        model, cur = _scene(7)
        dist_cpu = gt.texture_distance_device(
            cur, model.mean, model.std, KERNELS, N_ORIENT, len(SCALES),
            smooth_k=7, post_smooth_k=15, device="cpu")
        dist_gpu = gt.texture_distance_device(
            cur, model.mean, model.std, KERNELS, N_ORIENT, len(SCALES),
            smooth_k=7, post_smooth_k=15, device="gpu")
        rel = np.abs(dist_cpu - dist_gpu).max() / (dist_cpu.max() + 1e-9)
        assert rel < 0.02

    def test_resident_model_roundtrip(self):
        model, cur = _scene(8)
        mean_d, std_d = gt.upload_model(model.mean, model.std,
                                        device="gpu")
        dist = gt.texture_distance_device(
            cur, mean_d, std_d, KERNELS, N_ORIENT, len(SCALES),
            smooth_k=7, post_smooth_k=15, device="gpu")
        assert dist.shape == cur.shape
        assert np.isfinite(dist).all()
