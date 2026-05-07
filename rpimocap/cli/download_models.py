"""Entry point shim for rpimocap-download-models."""
from __future__ import annotations


def main() -> None:
    """Entry point for the rpimocap-download-models command."""
    import sys
    from pathlib import Path
    tools_dir = Path(__file__).resolve().parent.parent / "tools"
    sys.path.insert(0, str(tools_dir))
    from download_models import main as _main
    _main()
