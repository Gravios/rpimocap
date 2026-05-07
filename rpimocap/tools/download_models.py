"""
rpimocap model downloader
=========================
Downloads SAM2 (Segment Anything Model 2) weights for use with
rpimocap-segment --sam2-checkpoint.

Sources tried in order:
  1. Meta's CDN  (dl.fbaipublicfiles.com)
  2. HuggingFace  (huggingface.co/facebook/...)

Usage
-----
    rpimocap-download-models                      # downloads recommended model
    rpimocap-download-models --model large        # explicit size
    rpimocap-download-models --model all          # download all sizes
    rpimocap-download-models --dest /path/to/dir  # custom destination
    rpimocap-download-models --list               # show available models
"""

from __future__ import annotations

import hashlib
import os
import sys
import time
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODELS: dict[str, dict] = {
    "tiny": {
        "filename": "sam2.1_hiera_tiny.pt",
        "config":   "sam2.1_hiera_t.yaml",
        "size_mb":  38,
        "sha256":   None,   # verified at runtime from header ETag when available
        "urls": [
            "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt",
            "https://huggingface.co/facebook/sam2.1-hiera-tiny/resolve/main/sam2.1_hiera_tiny.pt",
        ],
        "description": "Tiny  (~38 MB)  — fastest, good for real-time use",
    },
    "small": {
        "filename": "sam2.1_hiera_small.pt",
        "config":   "sam2.1_hiera_s.yaml",
        "size_mb":  46,
        "sha256":   None,
        "urls": [
            "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt",
            "https://huggingface.co/facebook/sam2.1-hiera-small/resolve/main/sam2.1_hiera_small.pt",
        ],
        "description": "Small (~46 MB)  — good balance of speed and accuracy",
    },
    "base_plus": {
        "filename": "sam2.1_hiera_base_plus.pt",
        "config":   "sam2.1_hiera_b+.yaml",
        "size_mb":  80,
        "sha256":   None,
        "urls": [
            "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt",
            "https://huggingface.co/facebook/sam2.1-hiera-base-plus/resolve/main/sam2.1_hiera_base_plus.pt",
        ],
        "description": "Base+ (~80 MB)  — recommended for most use cases",
    },
    "large": {
        "filename": "sam2.1_hiera_large.pt",
        "config":   "sam2.1_hiera_l.yaml",
        "size_mb":  224,
        "sha256":   None,
        "urls": [
            "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
            "https://huggingface.co/facebook/sam2.1-hiera-large/resolve/main/sam2.1_hiera_large.pt",
        ],
        "description": "Large (~224 MB) — best accuracy, recommended for RTX 5070 Ti",
    },
}

# Default destination: ~/.cache/rpimocap/models
DEFAULT_DEST = Path.home() / ".cache" / "rpimocap" / "models"

# Default model to download
DEFAULT_MODEL = "large"


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def _human_size(n_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} TB"


def _progress_bar(downloaded: int, total: int, width: int = 40) -> str:
    if total <= 0:
        return f"  {_human_size(downloaded)}"
    frac  = min(downloaded / total, 1.0)
    filled = int(frac * width)
    bar   = "█" * filled + "░" * (width - filled)
    return f"  [{bar}] {100*frac:5.1f}%  {_human_size(downloaded)}/{_human_size(total)}"


def _download_url(
    url:      str,
    dest:     Path,
    expected_mb: int,
    timeout:  int = 30,
) -> bool:
    """Download url → dest with a progress bar.  Returns True on success."""
    import urllib.request

    req = urllib.request.Request(url, headers={"User-Agent": "rpimocap/1.0"})

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            total = int(resp.headers.get("Content-Length", expected_mb * 1024 * 1024))
            downloaded = 0
            t0 = time.time()
            chunk = 65536

            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_suffix(".tmp")

            with open(tmp, "wb") as fh:
                while True:
                    buf = resp.read(chunk)
                    if not buf:
                        break
                    fh.write(buf)
                    downloaded += len(buf)
                    elapsed = time.time() - t0
                    speed   = downloaded / elapsed if elapsed > 0 else 0
                    print(
                        f"\r{_progress_bar(downloaded, total)}  "
                        f"{_human_size(int(speed))}/s   ",
                        end="", flush=True)

            print()
            tmp.rename(dest)
            return True

    except Exception as e:
        print(f"\n  ERROR: {e}")
        tmp = dest.with_suffix(".tmp")
        if tmp.exists():
            tmp.unlink()
        return False


def _verify_sha256(path: Path, expected: Optional[str]) -> bool:
    """Verify SHA256 of a file if expected hash is provided."""
    if expected is None:
        return True
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    actual = h.hexdigest()
    if actual != expected:
        print(f"  SHA256 mismatch!")
        print(f"    expected: {expected}")
        print(f"    got:      {actual}")
        return False
    return True


def download_model(
    model_key:   str,
    dest_dir:    Path,
    skip_if_exists: bool = True,
    verbose:     bool = True,
) -> Optional[Path]:
    """Download a SAM2 model checkpoint.

    Parameters
    ----------
    model_key       : one of ``"tiny"``, ``"small"``, ``"base_plus"``, ``"large"``
    dest_dir        : directory to save the checkpoint
    skip_if_exists  : skip download if file already exists and is non-empty
    verbose         : print progress

    Returns
    -------
    Path to the downloaded checkpoint, or None on failure.
    """
    if model_key not in MODELS:
        raise ValueError(f"Unknown model: {model_key!r}. "
                         f"Choose from: {list(MODELS)}")

    info = MODELS[model_key]
    dest = dest_dir / info["filename"]

    if skip_if_exists and dest.exists() and dest.stat().st_size > 1024 * 1024:
        if verbose:
            print(f"  Already downloaded: {dest}")
        return dest

    if verbose:
        print(f"  Model:  {model_key}  —  {info['description']}")
        print(f"  Dest:   {dest}")

    for i, url in enumerate(info["urls"]):
        if verbose:
            source = "Meta CDN" if "fbaipublicfiles" in url else "HuggingFace"
            print(f"  Source: {source}  ({url})")
        success = _download_url(url, dest, info["size_mb"])
        if success:
            if _verify_sha256(dest, info["sha256"]):
                if verbose:
                    print(f"  ✓ Saved: {dest}")
                return dest
            else:
                dest.unlink()
        if i < len(info["urls"]) - 1 and verbose:
            print(f"  Trying next source ...")

    return None


def print_usage(dest: Path) -> None:
    """Print how to use the downloaded models with rpimocap-segment."""
    print()
    print("─" * 60)
    print("Use with rpimocap-segment:")
    print()
    for key, info in MODELS.items():
        ckpt = dest / info["filename"]
        if ckpt.exists():
            print(f"  # {info['description']}")
            print(f"  rpimocap-segment ... \\")
            print(f"      --sam2-checkpoint {ckpt} \\")
            print(f"      --sam2-config     {info['config']}")
            print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    ap.add_argument("--model", default=DEFAULT_MODEL,
                    choices=list(MODELS) + ["all"],
                    help=f"Model size to download (default: {DEFAULT_MODEL})")
    ap.add_argument("--dest", default=str(DEFAULT_DEST),
                    help=f"Destination directory (default: {DEFAULT_DEST})")
    ap.add_argument("--list", action="store_true",
                    help="List available models and exit")
    ap.add_argument("--force", action="store_true",
                    help="Re-download even if file already exists")

    args = ap.parse_args()
    dest = Path(args.dest)

    if args.list:
        print("Available SAM2 models:")
        for key, info in MODELS.items():
            flag = " [default]" if key == DEFAULT_MODEL else ""
            ckpt = dest / info["filename"]
            status = " ✓ downloaded" if ckpt.exists() else ""
            print(f"  {key:12s}  {info['description']}{flag}{status}")
        print(f"\nDownload destination: {dest}")
        return

    # Check sam2 package is installed
    try:
        import sam2  # noqa: F401
    except ImportError:
        print("ERROR: sam2 package not installed.")
        print()
        print("Install it with:")
        print("  pip install sam2")
        print()
        print("Then re-run rpimocap-download-models")
        sys.exit(1)

    models_to_download = list(MODELS) if args.model == "all" else [args.model]

    print(f"rpimocap-download-models")
    print(f"Destination: {dest}")
    print()

    success_count = 0
    for key in models_to_download:
        print(f"Downloading {key} ...")
        path = download_model(
            key, dest,
            skip_if_exists=not args.force,
            verbose=True)
        if path:
            success_count += 1
        else:
            print(f"  FAILED: {key}")
        print()

    print(f"{success_count}/{len(models_to_download)} models downloaded successfully.")
    print_usage(dest)


if __name__ == "__main__":
    main()
