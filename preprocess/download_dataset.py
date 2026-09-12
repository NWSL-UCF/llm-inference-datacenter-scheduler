#!/usr/bin/env python3
"""Download the Azure LLM inference traces into dataset/raw/.

Files:
  AzureLLMInferenceTrace_code_1week.csv
  AzureLLMInferenceTrace_conv_1week.csv

Example (from repo root):
  python3 preprocess/download_dataset.py
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = REPO_ROOT / "dataset" / "raw"

BASE_URL = "https://github.com/Azure/AzurePublicDataset/releases/download/dataset-llm-2024"
FILES = (
    "AzureLLMInferenceTrace_code_1week.csv",
    "AzureLLMInferenceTrace_conv_1week.csv",
)


def _progress(prefix: str):
    last_pct = [-1]

    def hook(block_num: int, block_size: int, total_size: int) -> None:
        if total_size <= 0:
            return
        downloaded = block_num * block_size
        pct = min(100, downloaded * 100 // total_size)
        if pct == last_pct[0]:
            return
        last_pct[0] = pct
        mb = min(downloaded, total_size) / (1024 * 1024)
        total_mb = total_size / (1024 * 1024)
        print(f"\r{prefix} {mb:.1f}/{total_mb:.1f} MiB ({pct}%)", end="", file=sys.stderr)
        if pct >= 100:
            print(file=sys.stderr)

    return hook


def download_file(url: str, dest: Path, force: bool) -> None:
    if dest.is_file() and not force:
        size_mb = dest.stat().st_size / (1024 * 1024)
        print(f"skip {dest.name} ({size_mb:.1f} MiB already present)", file=sys.stderr)
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"download {dest.name}", file=sys.stderr)
    try:
        urllib.request.urlretrieve(url, tmp, reporthook=_progress(dest.name))
        tmp.replace(dest)
    except BaseException:
        if tmp.exists():
            tmp.unlink()
        raise


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help=f"destination directory (default: {DEFAULT_OUT_DIR})")
    p.add_argument("--force", action="store_true", help="re-download even if the file already exists")
    return p.parse_args(argv)


def main(argv: Optional[list] = None) -> None:
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        download_file(f"{BASE_URL}/{name}", args.out_dir / name, args.force)
    print(f"done -> {args.out_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
