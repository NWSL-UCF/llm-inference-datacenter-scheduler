#!/usr/bin/env python3
"""Add prefill, KV-cache, and transfer columns to the Azure traces.

Reads dataset/raw/*.csv and writes dataset/input/*.csv.

Prefill (ms):
  2 * N_params * ContextTokens / (peak_FLOPS * MFU) * 1000

KV cache (MiB):
  ContextTokens * KV_bytes_per_token / 1024^2

Transfer (ms), 80% of a 100/200/400 Gbps link:
  KV_bits / (0.8 * BW_Gbps * 1e9) * 1000

Example (from repo root):
  python3 preprocess/enrich_dataset.py
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_IN_DIR = REPO_ROOT / "dataset" / "raw"
DEFAULT_OUT_DIR = REPO_ROOT / "dataset" / "input"

FILES = (
    "AzureLLMInferenceTrace_code_1week.csv",
    "AzureLLMInferenceTrace_conv_1week.csv",
)

REQUIRED = ("TIMESTAMP", "ContextTokens", "GeneratedTokens")

MIB = 1024 * 1024
MFU = 0.6
LINK_EFFICIENCY = 0.8

MODEL_PARAMS = {
    "Llama3.1_8B": 8e9,
    "Llama3.1_70B": 70e9,
}
GPU_PEAK_FLOPS = {
    "H100PCIe": 756e12,
    "H100SXM": 989e12,
}
# Llama 3.1 GQA, BF16/FP16 (b=2 B). KV_tok = 2 * L * H_kv * d_h * b
KV_BYTES_PER_TOKEN = {
    "Llama3.1_8B": 2 * 32 * 8 * 128 * 2,
    "Llama3.1_70B": 2 * 80 * 8 * 128 * 2,
}
LINK_GBPS = (100, 200, 400)
MODELS = ("Llama3.1_8B", "Llama3.1_70B")
GPUS = ("H100PCIe", "H100SXM")

PREFILL_COLUMNS = tuple(f"PrefillMs__{gpu}__{model}" for gpu in GPUS for model in MODELS)
KV_COLUMNS = tuple(f"KVCacheMiB__{model}" for model in MODELS)
TRANSFER_COLUMNS = tuple(f"TransferMs__{bw}Gbps__{model}" for bw in LINK_GBPS for model in MODELS)
EXTRA_COLUMNS = PREFILL_COLUMNS + KV_COLUMNS + TRANSFER_COLUMNS

PREFILL_MS_PER_TOKEN = {
    (gpu, model): 2.0 * MODEL_PARAMS[model] / (GPU_PEAK_FLOPS[gpu] * MFU) * 1000.0
    for gpu in GPUS
    for model in MODELS
}
KV_MIB_PER_TOKEN = {model: KV_BYTES_PER_TOKEN[model] / MIB for model in MODELS}
TRANSFER_MS_PER_TOKEN = {
    (bw, model): KV_BYTES_PER_TOKEN[model] * 8.0 / (LINK_EFFICIENCY * bw * 1e9) * 1000.0
    for bw in LINK_GBPS
    for model in MODELS
}


def extra_values(context_tokens: int) -> list[str]:
    n = float(context_tokens)
    out = [f"{n * PREFILL_MS_PER_TOKEN[(gpu, model)]:.6f}" for gpu in GPUS for model in MODELS]
    out.extend(f"{n * KV_MIB_PER_TOKEN[model]:.6f}" for model in MODELS)
    out.extend(f"{n * TRANSFER_MS_PER_TOKEN[(bw, model)]:.6f}" for bw in LINK_GBPS for model in MODELS)
    return out


def enrich_file(src: Path, dest: Path, force: bool) -> int:
    if dest.is_file() and not force:
        size_mb = dest.stat().st_size / (1024 * 1024)
        print(f"skip {dest.name} ({size_mb:.1f} MiB already present)", file=sys.stderr)
        return 0

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"enrich {src.name} -> {dest}", file=sys.stderr)
    n = 0
    try:
        with src.open(newline="") as fin, tmp.open("w", newline="") as fout:
            reader = csv.DictReader(fin)
            names = reader.fieldnames or []
            missing = [c for c in REQUIRED if c not in names]
            if missing:
                raise SystemExit(f"{src}: missing {missing}; have: {names}")
            writer = csv.writer(fout)
            writer.writerow([*REQUIRED, *EXTRA_COLUMNS])
            for row in reader:
                writer.writerow(
                    [
                        row["TIMESTAMP"],
                        row["ContextTokens"],
                        row["GeneratedTokens"],
                        *extra_values(int(row["ContextTokens"])),
                    ]
                )
                n += 1
                if n % 1_000_000 == 0:
                    print(f"  {src.name}: {n:,} rows", file=sys.stderr)
        tmp.replace(dest)
    except BaseException:
        if tmp.exists():
            tmp.unlink()
        raise
    print(f"  {src.name}: {n:,} rows written", file=sys.stderr)
    return n


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in-dir", type=Path, default=DEFAULT_IN_DIR)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--force", action="store_true", help="overwrite existing output files")
    return p.parse_args(argv)


def main(argv: Optional[list] = None) -> None:
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        src = args.in_dir / name
        if not src.is_file():
            raise SystemExit(f"missing raw trace: {src}")
        enrich_file(src, args.out_dir / name, args.force)
    print(f"done -> {args.out_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
