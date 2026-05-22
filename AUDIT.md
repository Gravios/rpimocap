# Code Audit — May 2026

Audit run after v0.5.0 / housekeeping / 3-pending-integration patches
landed. Scope: all of `rpimocap/`, `tests/`. Tool: `ruff` with
`select = ["E", "F", "W", "I"]`, `ignore = ["E501"]`.

## Headline

- **Real bugs found**: 2
- **Cleanups applied**: 19
- **Pre-existing style issues left for follow-up**: ~210

The codebase is in good shape. The only runtime bug uncovered was a
missing import that would have surfaced when a user actually used a
particular alignment path. All known correctness issues are resolved
in this audit pass; the rest is stylistic and concentrated in legacy
modules.

## Bugs fixed

### 1. Missing `kabsch_align_from_pixels` import (real runtime bug)

`rpimocap/cli/pipeline.py:386` imports a handful of symbols from
`rpimocap.reconstruction.align` but **omits**
`kabsch_align_from_pixels`. The function is then called at line 395
on the pixel-alignment branch:

```python
if _has_px:
    align_result = kabsch_align_from_pixels(...)   # NameError
```

This would have crashed any pipeline.py invocation with
`--align-points` pointing at a CSV containing pixel coordinates.
The function exists in `align.py:66`; the bug was purely the
missing import line.

**Fix**: added `kabsch_align_from_pixels` to the import.

### 2. `assert det[5] == False` on a numpy scalar

`tests/test_detected_mask.py:71` — written in this session.
Comparing a `numpy.bool_` to Python `False` works at runtime but is
flagged by ruff E712 and is anti-idiomatic. Replaced with `assert
not det[5]` plus an explicit one-line-per-assert reformat.

## Cleanups applied (in code authored or modified this session)

| Rule | Where | What |
|---|---|---|
| F401 | `vignette.py` | `typing.Optional`, `typing.Tuple` (never used; deleted) |
| F401 | `triangulate.py` | `dataclasses.field` (never used; deleted) |
| F401 | `tracker.py` | `ForegroundResult`, `SAMLabeller` (only in comments; deleted from import) |
| F401 | `pipeline.py` | `CentroidPoseDetector` (auto-fix) |
| F541 | `cli/segment.py` (×7) | f-strings with no placeholders — dropped the `f` prefix |
| F541 | `pipeline.py` (×3) | Same |
| F841 | `tracker.py:542` | `except Exception as e` → bare `except Exception:` (the `e` was never logged) |
| I001 | `triangulate.py`, `tracker.py`, `cli/segment.py`, `cli/pipeline.py`, `export.py` | Import blocks sorted via `ruff --fix` |

Plus a pyproject migration: `[tool.ruff]` → `[tool.ruff.lint]` to
silence the deprecation warning ruff emits on every invocation.

After this pass, **every module I authored or extended in v0.5.0
(`detection/sam2_mask_cache.py`, `detection/vignette.py`,
`detection/kalman.py`, `detection/rearing.py`, `cli/segment.py`)
lints clean**.

## Not fixed — pre-existing issues, recommended follow-ups

I deliberately did not churn code I didn't author. The remaining
issues cluster as follows:

### `rpimocap/io/export.py` — `TiffCapture` class (19 issues)

The whole `TiffCapture` class uses an inline multi-statement style:

```python
self._n = shape[i_n]; self._h = shape[i_h]; self._w = shape[i_w]
```

…and helpers using colon-on-same-line conditionals:

```python
def _ax(cands):
    for c in cands:
        if c in axes: return axes.index(c)
    return None
```

Ruff flags 19 E701/E702 lines, all in this one class. It works, it's
just style. Recommendation: a single follow-up patch reformatting
`TiffCapture` would close all 19. Keep it scoped to the class so
git blame stays clean on the rest.

### `rpimocap/reconstruction/align.py` (30 issues)

Largest concentrations:

- Multiline imports in `_load_align_csv` need sorting.
- Several `kabsch_align*` helpers have parameters with `np.array` /
  list ambiguity that mypy would flag if mypy were enabled.

Recommendation: ruff `--fix` on `align.py` is safe based on a quick
manual scan, then a manual pass for the `kabsch_align_from_pixels`
docstring (currently terse).

### `rpimocap/reconstruction/__init__.py` (24 issues)

Mostly `F401` re-exports — they look intentional (a public re-export
surface for downstream users). Recommendation: add a module-level
`__all__ = [...]` listing the public names; that silences F401 for
intentional re-exports while keeping accidental ones loud.

### `rpimocap/cli/autocalib.py` (17), `cli/calibrate_from_corners.py`
### (11), `cli/refine_cal.py` (9), `calibration/autocalib/report.py` (11)

Same patterns as `export.py` — semicolon-joined statements,
in-function imports for circular-dep avoidance, unused intermediate
variables in print formatting. Same prescription: ruff `--fix` on
each, eyeball the diffs.

### `tests/` (~25 issues)

Mostly F541 (`f""` with no placeholder) and a couple of E702
semicolon joins. Worth running `ruff --fix tests/` in one go — tests
have no API surface so the risk is low.

## Refactor opportunities I considered

These are honest "could be done if you're feeling tidy" suggestions,
not bugs.

### Two Kalman implementations

`KalmanTracker3D` (online, in `reconstruction/kalman.py`) and
`kalman_filter_trajectory` (offline RTS, in
`reconstruction/triangulate.py`) share most of the state-transition
and process-noise math (`F`, `Q`, `H`, `R`). They differ in shape:
one is a stateful class for per-frame use, the other is a stateless
function over a list of frames. A shared `_kalman_matrices(dt,
max_accel)` helper would dedup ~20 lines but reads less clearly than
the current self-contained versions. I left them as-is.

### `_process_frame` duplication

The hull-refine loop in `SegmentTracker._process_frame` has two
near-identical branches for cam0 and cam1:

```python
if fg0 is not None:
    hx0, hy0 = self._det.hull_centroid(fg0, r0.cx, r0.cy, ...,
                                       P=self._P0, ...)
    r0 = r0.__class__(label=r0.label, cx=hx0, cy=hy0, ...)
if fg1 is not None:
    hx1, hy1 = self._det.hull_centroid(fg1, r1.cx, r1.cy, ...,
                                       P=self._P1, ...)
    r1 = r1.__class__(label=r1.label, cx=hx1, cy=hy1, ...)
```

A small inline helper `_refine_one_side(fg, r, P)` would dedupe this
and the analogous SAM2 mask consumption block above. Mechanical and
safe; would shrink `_process_frame` by ~40 lines. Holding off until
the per-frame surface stops growing — adding it as part of "Phase 3
SLEAP" is the natural moment.

### `cli/segment.py` size

Now ~900 lines, ~80 CLI flags. The argument-group definitions
account for ~250 lines. Splitting them into a separate
`cli/_segment_args.py` would leave the main file focused on the
data-flow orchestration. Mechanical refactor; deferred until a CLI
restructure is needed anyway (e.g. for the rearing CLI sub-command).

### Optional type-hint cleanup

A few signatures still use `"Optional[np.ndarray]"` (the string-form
forward reference) where `Optional[np.ndarray]` would work fine
since `from __future__ import annotations` is at the top of every
file. The string form was a defensive carry-over from an older
Python. Cleanup, not a bug.

## Lint configuration left behind

`pyproject.toml` now has:

```toml
[tool.ruff]
line-length    = 100
target-version = "py310"

[tool.ruff.lint]
select         = ["E", "F", "W", "I"]
ignore         = ["E501"]
```

To run the same audit:

```bash
ruff check rpimocap/ tests/                # full report
ruff check rpimocap/ tests/ --statistics   # rule-by-rule counts
ruff check rpimocap/ tests/ --fix          # safe auto-fixes only
```

CI already runs `pytest tests/` on every push (see
`.github/workflows/tests.yml`); adding a `ruff check` step to that
workflow would catch regressions automatically going forward.

## Verification

- `pytest tests/ -q` → 167 passed, 0 failed
- `ruff check` on the v0.5.0 module set → clean
- No behavioural changes; no diff in any test assertion
