"""
tests/test_sam2_video.py
=========================
Smoke tests for the SAM2VideoTracker scaffold. We can't exercise the
real SAM2 video predictor in CI (no GPU, no weights, sam2 package may
not be installed), so these tests only verify the graceful-fallback
behaviour and the API surface — that the class can be constructed
without error and reports availability honestly.
"""
from __future__ import annotations

import numpy as np
import pytest


def test_unavailable_when_checkpoint_missing(tmp_path):
    """Pointing at a non-existent checkpoint must result in
    ``available=False`` and must not raise."""
    from rpimocap.detection.tracker import SAM2VideoTracker
    bogus = tmp_path / "does_not_exist.pt"
    t = SAM2VideoTracker(checkpoint=str(bogus), device="cpu")
    assert t.available is False


def test_init_state_raises_when_unavailable(tmp_path):
    """Calling init_state on an unavailable tracker should fail loudly."""
    from rpimocap.detection.tracker import SAM2VideoTracker
    t = SAM2VideoTracker(checkpoint=str(tmp_path / "missing.pt"), device="cpu")
    with pytest.raises(RuntimeError):
        t.init_state([np.zeros((10, 10, 3), np.uint8)], prompts={"animal": [(5, 5)]})


def test_propagate_raises_when_uninitialised(tmp_path):
    """Even if SAM2 were available, calling propagate() before
    init_state() must raise."""
    from rpimocap.detection.tracker import SAM2VideoTracker
    t = SAM2VideoTracker(checkpoint=str(tmp_path / "missing.pt"), device="cpu")
    # If SAM2 isn't installed _available is False so we just confirm
    # the precondition guard fires.
    with pytest.raises(RuntimeError):
        next(t.propagate())


def test_attributes_present_on_class():
    """API stability: contract surface must exist."""
    from rpimocap.detection.tracker import SAM2VideoTracker
    assert hasattr(SAM2VideoTracker, "init_state")
    assert hasattr(SAM2VideoTracker, "propagate")
    assert hasattr(SAM2VideoTracker, "available")
