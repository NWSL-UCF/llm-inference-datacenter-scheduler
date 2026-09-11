#!/usr/bin/env python3
"""Run JSQ simulation for every GPU × model pair (90 servers by default).

Writes one assignment CSV per (trace, gpu, model) under --out-dir.

Example (from repo root):
  python3 discrete_event_simulator/runner.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from simulator import (
    GPU_CHOICES,
    MODEL_CHOICES,
    iter_arrivals,
    latency_column,
    simulate,
)

REPO_ROOT = _HERE.parent
DEFAULT_TRACES = {
    "code": REPO_ROOT / "dataset" / "AzureLLMInferenceTrace_code_1week_prefill_ms.csv",
    "conv": REPO_ROOT / "dataset" / "AzureLLMInferenceTrace_conv_1week_prefill_ms.csv",
}
DEFAULT_OUT_DIR = _HERE / "results"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--servers", type=int, default=90)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument(
        "--traces",
        nargs="+",
        default=list(DEFAULT_TRACES.keys()),
        choices=list(DEFAULT_TRACES.keys()),
        help="which traces to run (default: code conv)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.servers < 1:
        raise SystemExit("--servers must be >= 1")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    jobs = [
        (trace_name, gpu, model)
        for trace_name in args.traces
        for gpu in GPU_CHOICES
        for model in MODEL_CHOICES
    ]
    total = len(jobs)
    print(f"{total} runs, {args.servers} servers, out={args.out_dir}", file=sys.stderr)

    for i, (trace_name, gpu, model) in enumerate(jobs, start=1):
        trace_path = DEFAULT_TRACES[trace_name]
        if not trace_path.is_file():
            raise SystemExit(f"missing trace: {trace_path}")
        out_path = args.out_dir / f"{trace_name}_{gpu}_{model}.csv"
        column = latency_column(gpu, model)
        print(f"[{i}/{total}] {trace_name} {gpu} {model} -> {out_path.name}", file=sys.stderr)
        n = simulate(iter_arrivals(str(trace_path), column), args.servers, str(out_path), model)
        print(f"         {n} requests", file=sys.stderr)


if __name__ == "__main__":
    main()
