"""Tests for motion-aware background-frame selection."""
import cv2
import numpy as np

from rpimocap.detection import bg_select as bs


class _FakeCap:
    """Minimal cv2.VideoCapture-like wrapper over an array of frames."""
    def __init__(self, frames):
        self.frames = frames
        self.pos = 0

    def set(self, prop, v):
        self.pos = int(v)

    def read(self):
        if 0 <= self.pos < len(self.frames):
            return True, self.frames[self.pos]
        return False, None

    def get(self, prop):
        return float(len(self.frames))


def _dwell_then_move(N=200, H=120, W=160, seed=0):
    """Rat dwells at (40,60) for t<N/2, then moves rightward."""
    rng = np.random.RandomState(seed)
    bg = rng.randint(60, 110, (H, W)).astype(np.uint8)
    frames = []
    for t in range(N):
        f = bg.copy()
        if t < N // 2:
            cx, cy = 40, 60
        else:
            cx, cy = 40 + (t - N // 2), 60
        cv2.circle(f, (cx, cy), 12, 200, -1)
        frames.append(f)
    return np.array(frames), bg


class TestMotionSeries:

    def test_motion_zero_for_static(self):
        H, W = 60, 80
        static = np.full((H, W), 100, np.uint8)
        cap = _FakeCap(np.array([static] * 10))
        m = bs.frame_motion_series(cap, list(range(10)),
                                   green_channel=False, motion_downsample=2)
        assert m[0] == 0.0
        assert np.allclose(m, 0.0)

    def test_motion_high_when_moving(self):
        frames, _ = _dwell_then_move()
        cap = _FakeCap(frames)
        # sample across the whole clip
        idx = list(range(0, 200, 5))
        m = bs.frame_motion_series(cap, idx, green_channel=False,
                                   motion_downsample=2)
        # motion in the moving half >> dwell half
        idx = np.array(idx)
        assert m[idx >= 100].mean() > m[idx < 100].mean() + 0.1


class TestActiveSelection:

    def test_prefers_moving_frames(self):
        frames, _ = _dwell_then_move()
        cap = _FakeCap(frames)
        sel, cand, motion = bs.select_active_frames(
            cap, 12, 0, 200, green_channel=False, oversample=4,
            motion_downsample=2)
        # the rat dwells in the first half; selection should avoid it
        assert all(s >= 100 for s in sel)

    def test_cleaner_background_at_dwell_spot(self):
        frames, bg = _dwell_then_move()
        cap = _FakeCap(frames)
        sel, _, _ = bs.select_active_frames(
            cap, 12, 0, 200, green_channel=False, oversample=4,
            motion_downsample=2)

        def med(ix):
            return np.median(frames[np.array(ix)].astype(np.float32),
                             axis=0)

        uniform = np.linspace(0, 199, 12).astype(int)
        true_bg = float(bg[60, 40])
        err_uniform = abs(med(uniform)[60, 40] - true_bg)
        err_motion = abs(med(sel)[60, 40] - true_bg)
        # motion selection avoids the dwell → clean median there
        assert err_motion < err_uniform

    def test_returns_requested_count(self):
        frames, _ = _dwell_then_move(N=300)
        cap = _FakeCap(frames)
        sel, _, _ = bs.select_active_frames(
            cap, 20, 0, 300, green_channel=False, oversample=3,
            motion_downsample=2)
        # close to requested (dedup/empty-bin may trim slightly)
        assert 15 <= len(sel) <= 20

    def test_all_still_falls_back(self):
        """If nothing moves, don't crash — fall back to spread frames."""
        H, W = 60, 80
        static = np.full((H, W), 100, np.uint8)
        cap = _FakeCap(np.array([static] * 60))
        sel, _, _ = bs.select_active_frames(
            cap, 8, 0, 60, green_channel=False, oversample=3,
            motion_downsample=2)
        assert len(sel) >= 3

    def test_roi_restricts_motion(self):
        """Motion outside the ROI is ignored."""
        H, W = 120, 160
        rng = np.random.RandomState(1)
        bg = rng.randint(60, 110, (H, W)).astype(np.uint8)
        frames = []
        for t in range(60):
            f = bg.copy()
            # moving blob only in the RIGHT half
            cv2.circle(f, (120, 60 + (t % 20)), 10, 220, -1)
            frames.append(f)
        frames = np.array(frames)
        cap = _FakeCap(frames)
        # ROI covering only the LEFT half (no motion there)
        roi = np.zeros((H, W), np.uint8)
        roi[:, :80] = 255
        m = bs.frame_motion_series(
            cap, list(range(0, 60, 3)), green_channel=False,
            roi_mask=roi, motion_downsample=2)
        # blob is outside ROI → near-zero motion seen
        assert m.max() < 5.0
