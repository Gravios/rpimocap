"""
rpimocap.calibration.autocalib
===============================
Self-calibration from subject motion in a constrained arena.

Modules
-------
features    SIFT cross-view matching → FundamentalEstimate
kruppa      Essential matrix constraint → focal length (Kruppa equations)
report      Standalone HTML validation report generator
"""

from rpimocap.calibration.autocalib.features import (
    FundamentalEstimate,
    CrossViewMatcher,
    sample_frame_pairs,
    filter_estimates,
    sampson_distance,
)
from rpimocap.calibration.autocalib.kruppa import (
    KruppaResult,
    EssentialDecomposition,
    make_K,
    cost_for_f,
    essential_constraint_residual,
    estimate_focal_kruppa,
    decompose_essential,
    metric_scale_refinement,
    bundle_refine_focal,
    run_focal_estimation,
)
from rpimocap.calibration.autocalib.report import generate_report

__all__ = [
    # features
    "FundamentalEstimate",
    "CrossViewMatcher",
    "sample_frame_pairs",
    "filter_estimates",
    "sampson_distance",
    # kruppa
    "KruppaResult",
    "EssentialDecomposition",
    "make_K",
    "cost_for_f",
    "essential_constraint_residual",
    "estimate_focal_kruppa",
    "decompose_essential",
    "metric_scale_refinement",
    "bundle_refine_focal",
    "run_focal_estimation",
    # report
    "generate_report",
]
