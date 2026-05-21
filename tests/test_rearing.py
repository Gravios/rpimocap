"""
tests/test_rearing.py
======================
Unit tests for rpimocap.reconstruction.rearing.

Verifies hysteresis, vz-based early detection, and that the anatomical
prior collapses to vertical dimensions when reared.
"""
from __future__ import annotations

import numpy as np
import pytest

from rpimocap.reconstruction.rearing import (
    PostureState,
    RearingClassifier,
    trace_postures,
)


def _state(x=0, y=0, z=0, vx=0, vy=0, vz=0):
    return np.array([x, y, z, vx, vy, vz], dtype=np.float64)


class TestRearingClassifier:

    def test_below_enter_threshold_is_horizontal(self):
        cls = RearingClassifier()
        ps = cls.classify(_state(z=50))
        assert ps.reared is False
        assert ps.body_axis_horizontal is True
        assert ps.body_length_mm == cls.horizontal_body_length_mm

    def test_above_enter_threshold_is_reared(self):
        cls = RearingClassifier()
        ps = cls.classify(_state(z=150))
        assert ps.reared is True
        assert ps.body_axis_horizontal is False
        assert ps.body_length_mm == cls.vertical_body_length_mm

    def test_vz_based_early_detection(self):
        """If z is below enter but the rat is climbing fast, switch
        into reared state immediately rather than waiting for z to
        rise above z_enter."""
        cls = RearingClassifier(z_enter=100.0, vz_enter=200.0)
        ps = cls.classify(_state(z=80, vz=300))
        assert ps.reared is True

    def test_hysteresis_keeps_reared_state(self):
        """Once reared, the state persists until z falls below z_exit,
        not just below z_enter — prevents flicker around the boundary."""
        cls = RearingClassifier(z_enter=100.0, z_exit=70.0)
        cls.classify(_state(z=120))           # enters reared
        ps = cls.classify(_state(z=80))       # below enter but above exit
        assert ps.reared is True
        ps = cls.classify(_state(z=60))       # finally below exit
        assert ps.reared is False

    def test_for_labeller_payload(self):
        cls = RearingClassifier()
        d = cls.classify(_state(z=200)).for_labeller()
        assert set(d.keys()) == {
            "reared", "body_length_mm", "body_width_mm", "horizontal"}
        assert d["reared"] is True
        assert d["horizontal"] is False

    def test_reset_clears_state(self):
        cls = RearingClassifier()
        cls.classify(_state(z=200))
        cls.reset()
        ps = cls.classify(_state(z=80))   # above exit but below enter
        assert ps.reared is False, "reset should drop hysteresis memory"


class TestTracePostures:

    def test_constant_low_z_no_rearing(self):
        states = np.zeros((20, 6))
        states[:, 2] = 30.0
        trace = trace_postures(states)
        assert len(trace) == 20
        assert all(not p.reared for p in trace)

    def test_rear_then_settle(self):
        states = np.zeros((20, 6))
        states[:5, 2]   = 30.0
        states[5:15, 2] = 200.0
        states[15:, 2]  = 30.0
        trace = trace_postures(states)
        # First 5 frames: floor
        assert all(not p.reared for p in trace[:5])
        # During rear
        assert all(p.reared for p in trace[5:15])
        # After rear
        assert all(not p.reared for p in trace[15:])

    def test_nan_frames_inherit_previous_state(self):
        states = np.zeros((10, 6))
        states[:5, 2]  = 200.0      # reared
        states[5:8, :] = np.nan     # missing
        states[8:, 2]  = 200.0      # still reared
        trace = trace_postures(states)
        assert all(p.reared for p in trace[:5])
        # NaN frames keep the previous classification
        assert all(p.reared for p in trace[5:8])
        assert all(p.reared for p in trace[8:])
