"""
rpimocap.viewer
===============
Bundled Three.js viewer for interactive 3D reconstruction playback.

The viewer is a single-file HTML application (assets/index.html) that
can be opened directly in any modern browser.  It loads viewer_data.json
from a sibling data/ directory, or via drag-and-drop.

Usage
-----
    from rpimocap.viewer import viewer_html_path
    import shutil, pathlib

    out = pathlib.Path("output/viewer")
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(viewer_html_path(), out / "index.html")
    (out / "data").mkdir(exist_ok=True)
    # then copy output/viewer_data.json → output/viewer/data/viewer_data.json
"""

from __future__ import annotations

from pathlib import Path


def viewer_html_path() -> Path:
    """Return the absolute path to the bundled index.html viewer asset."""
    return Path(__file__).parent / "assets" / "index.html"


def deploy_viewer(dest_dir: str | Path, viewer_json: str | Path | None = None):
    """
    Copy the viewer HTML into dest_dir and optionally symlink/copy the JSON.

    Parameters
    ----------
    dest_dir    : directory to copy viewer into
    viewer_json : path to viewer_data.json to copy into dest_dir/data/
    """
    import shutil

    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy(viewer_html_path(), dest / "index.html")

    if viewer_json is not None:
        data_dir = dest / "data"
        data_dir.mkdir(exist_ok=True)
        shutil.copy(viewer_json, data_dir / "viewer_data.json")

    return dest / "index.html"


__all__ = ["viewer_html_path", "deploy_viewer"]
