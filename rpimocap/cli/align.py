"""rpimocap-align — Interactive arena alignment + distortion annotator (Qt6)."""
from __future__ import annotations


def main() -> None:
    """Entry point for the ``rpimocap-align`` command-line tool."""
    import sys
    from pathlib import Path
    tools_dir = Path(__file__).resolve().parent.parent.parent / "tools"
    if tools_dir.exists():
        sys.path.insert(0, str(tools_dir))
    from arena_aligner import main as _main
    _main()
