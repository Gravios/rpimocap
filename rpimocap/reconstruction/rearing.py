"""
rearing.py — body posture detection from 3D state
==================================================
Detects whether the rat is on the floor (horizontal posture) or reared
up (vertical posture) from its 3D position, and exposes anatomical-
prior adjustments that downstream labellers can use.

A reared rat has a very different floor-plane projection shape than a
walking one: the body becomes vertical and narrow in the (x, y) plane,
the nose-to-tail axis points roughly along z, and a fixed anatomical
prior tuned for "180 mm horizontal ellipse" pulls the nose centroid
toward the body's midpoint rather than the actual nose. This module
provides a single boolean+prior bundle that GeometricLabeller can
consume to switch to vertical priors when rearing is detected.

Heuristics
----------
We classify rearing from the Kalman state alone. The two signals are:

  z_threshold      : body Z above this many mm of the arena floor → rear
  vz_threshold     : recent vertical velocity above this many mm/s →
                     transitioning into a rear (used for early detection
                     before z has actually risen).

For a standard ±215 × ±140 mm × 0–388 mm arena, z > 100 mm reliably
indicates the body is at least partially up; rats rear to ~250 mm.
Hysteresis (different thresholds for entering and leaving the reared
state) prevents oscillation when z hovers near the boundary.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class PostureState:
    """Result of a posture classification."""
    reared: bool                       # True if the rat is rearing
    z:      float                      # body Z (mm) used for the call
    vz:     float                      # body Z velocity (mm/s)
    body_axis_horizontal: bool         # True when the long axis is in (x,y)
    # Anatomical prior adjustments (mm). The labeller can scale its
    # nose-vs-tail axis according to these.
    body_length_mm: float              # along the body's long axis
    body_width_mm:  float              # perpendicular

    def for_labeller(self) -> dict:
        """Return a dict consumable by GeometricLabeller's prior."""
        return {
            "reared": self.reared,
            "body_length_mm": self.body_length_mm,
            "body_width_mm":  self.body_width_mm,
            "horizontal":     self.body_axis_horizontal,
        }


@dataclass
class RearingClassifier:
    """Stateful posture classifier with hysteresis.

    Parameters
    ----------
    z_enter   : enter the reared state when z > z_enter (mm).
    z_exit    : leave the reared state when z < z_exit (mm). Should be
                less than z_enter to provide hysteresis.
    vz_enter  : alternatively enter reared state when vz > vz_enter
                (mm/s), to catch the transition early.
    horizontal_body_length_mm / horizontal_body_width_mm :
                anatomical prior when not rearing.
    vertical_body_length_mm / vertical_body_width_mm :
                anatomical prior when rearing — the floor-plane
                projection is much narrower since the body is upright.
    """
    z_enter:  float = 100.0
    z_exit:   float =  70.0
    vz_enter: float = 200.0

    horizontal_body_length_mm: float = 180.0
    horizontal_body_width_mm:  float =  60.0
    vertical_body_length_mm:   float =  90.0
    vertical_body_width_mm:    float =  45.0

    _reared: bool = False

    # ------------------------------------------------------------------ #

    def reset(self) -> None:
        """Reset the hysteresis state to 'on the floor'."""
        self._reared = False

    def classify(self, kalman_state: np.ndarray) -> PostureState:
        """Classify from a 6-state Kalman vector [x, y, z, vx, vy, vz].

        Hysteresis is applied: once entered, the reared state persists
        until z falls below ``z_exit``.
        """
        s = np.asarray(kalman_state, dtype=np.float64).reshape(6)
        z, vz = float(s[2]), float(s[5])
        if self._reared:
            if z < self.z_exit:
                self._reared = False
        else:
            if z > self.z_enter or vz > self.vz_enter:
                self._reared = True

        if self._reared:
            return PostureState(
                reared=True, z=z, vz=vz,
                body_axis_horizontal=False,
                body_length_mm=self.vertical_body_length_mm,
                body_width_mm=self.vertical_body_width_mm)
        return PostureState(
            reared=False, z=z, vz=vz,
            body_axis_horizontal=True,
            body_length_mm=self.horizontal_body_length_mm,
            body_width_mm=self.horizontal_body_width_mm)


# --------------------------------------------------------------------------- #
#  Convenience: per-frame trace                                                #
# --------------------------------------------------------------------------- #

def trace_postures(
    kalman_states: np.ndarray,
    classifier:    "Optional[RearingClassifier]" = None,
) -> list[PostureState]:
    """Classify a sequence of (n_frames, 6) Kalman states.

    Parameters
    ----------
    kalman_states : (n_frames, 6) array of state vectors. Rows
                    containing NaN are treated as missing (the classifier
                    keeps its previous state).
    classifier    : RearingClassifier instance. A fresh one is used by
                    default.

    Returns
    -------
    list of PostureState, one per frame.
    """
    if classifier is None:
        classifier = RearingClassifier()
    out: list[PostureState] = []
    last: "Optional[PostureState]" = None
    for s in kalman_states:
        if np.any(np.isnan(s)):
            if last is None:
                out.append(PostureState(
                    reared=False, z=float("nan"), vz=float("nan"),
                    body_axis_horizontal=True,
                    body_length_mm=classifier.horizontal_body_length_mm,
                    body_width_mm=classifier.horizontal_body_width_mm))
            else:
                out.append(last)
            continue
        ps = classifier.classify(s)
        out.append(ps)
        last = ps
    return out


__all__ = [
    "PostureState",
    "RearingClassifier",
    "trace_postures",
]
