"""
rpimocap.cli
============
Command-line entry points for the rpimocap pipeline.

Entry points (registered in pyproject.toml)
-------------------------------------------
rpimocap-calibrate      rpimocap.cli.calibrate:main
rpimocap-autocalib      rpimocap.cli.autocalib:main
rpimocap-run            rpimocap.cli.pipeline:main
"""


def parse_bounds(s: str):
    """
    Parse an arena-bounds string 'xmin,xmax,ymin,ymax,zmin,zmax' (mm).
    Shared by autocalib and pipeline CLI modules.
    """
    vals = [float(x) for x in s.split(",")]
    if len(vals) != 6:
        raise ValueError(
            "--bounds expects 6 comma-separated values: "
            "xmin,xmax,ymin,ymax,zmin,zmax"
        )
    return ((vals[0], vals[1]), (vals[2], vals[3]), (vals[4], vals[5]))
