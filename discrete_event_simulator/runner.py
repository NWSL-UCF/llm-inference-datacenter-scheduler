#!/usr/bin/env python3
"""Run JSQ for every trace × GPU × model × link × latency-mode.

Default sweep is 2 × 2 × 2 × 3 × 3 = 72 runs (90 servers).
Writes one assignment CSV per combination under --out-dir.

Example (from repo root):
  python3 discrete_event_simulator/runner.py
  python3 discrete_event_simulator/runner.py --jobs 8
  python3 discrete_event_simulator/runner.py --sample --traces code --links 200

Outputs go to dataset/outputs/.
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Tuple

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from simulator import (
    GPU_CHOICES,
    LATENCY_MODE_CHOICES,
    LINK_CHOICES,
    MODEL_CHOICES,
    iter_arrivals,
    simulate,
)

REPO_ROOT = _HERE.parent
INPUT_DIR = REPO_ROOT / "dataset" / "input"
DEFAULT_OUT_DIR = REPO_ROOT / "dataset" / "outputs"

TRACE_FILES = {
    "code": "AzureLLMInferenceTrace_code_1week.csv",
    "conv": "AzureLLMInferenceTrace_conv_1week.csv",
}
SAMPLE_FILES = {
    "code": "AzureLLMInferenceTrace_code_1week_sample1000.csv",
    "conv": "AzureLLMInferenceTrace_conv_1week_sample1000.csv",
}

Job = Tuple[int, int, str, str, str, str, int, str, int, str]


def job_name(trace_name: str, gpu: str, model: str, link: int, mode: str) -> str:
    mode_tag = mode.replace("+", "plus")
    return f"{trace_name}_{gpu}_{model}_{link}Gbps_{mode_tag}.csv"


def run_job(job: Job) -> Tuple[int, int, str]:
    index, total, trace_path, out_path, gpu, model, link, mode, n_servers, label = job
    print(f"[{index}/{total}] start {label}", file=sys.stderr, flush=True)
    n = simulate(
        iter_arrivals(trace_path, gpu=gpu, model=model, link=link, latency_mode=mode),
        n_servers,
        out_path,
    )
    print(f"[{index}/{total}] done  {label}  {n} requests", file=sys.stderr, flush=True)
    return index, n, label


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--servers", type=int, default=90)
    p.add_argument("--jobs", type=int, default=8, help="parallel worker processes (default 8)")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--input-dir", type=Path, default=INPUT_DIR)
    p.add_argument("--sample", action="store_true", help="use the 1000-row sample CSVs instead of the full week")
    p.add_argument("--traces", nargs="+", default=list(TRACE_FILES), choices=list(TRACE_FILES))
    p.add_argument("--gpus", nargs="+", default=list(GPU_CHOICES), choices=list(GPU_CHOICES))
    p.add_argument("--models", nargs="+", default=list(MODEL_CHOICES), choices=list(MODEL_CHOICES))
    p.add_argument("--links", nargs="+", type=int, default=list(LINK_CHOICES), choices=list(LINK_CHOICES))
    p.add_argument(
        "--latency-modes",
        nargs="+",
        default=list(LATENCY_MODE_CHOICES),
        choices=list(LATENCY_MODE_CHOICES),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.servers < 1:
        raise SystemExit("--servers must be >= 1")
    if args.jobs < 1:
        raise SystemExit("--jobs must be >= 1")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    files = SAMPLE_FILES if args.sample else TRACE_FILES

    combos = [
        (trace_name, gpu, model, link, mode)
        for trace_name in args.traces
        for gpu in args.gpus
        for model in args.models
        for link in args.links
        for mode in args.latency_modes
    ]
    total = len(combos)
    print(
        f"{total} runs, {args.jobs} processes, {args.servers} servers, "
        f"{'sample' if args.sample else 'full-week'}, out={args.out_dir}",
        file=sys.stderr,
        flush=True,
    )

    jobs: list[Job] = []
    for i, (trace_name, gpu, model, link, mode) in enumerate(combos, start=1):
        trace_path = args.input_dir / files[trace_name]
        if not trace_path.is_file():
            raise SystemExit(f"missing trace: {trace_path}")
        out_path = args.out_dir / job_name(trace_name, gpu, model, link, mode)
        label = f"{trace_name} {gpu} {model} {link}Gbps {mode} -> {out_path.name}"
        jobs.append(
            (
                i,
                total,
                str(trace_path),
                str(out_path),
                gpu,
                model,
                link,
                mode,
                args.servers,
                label,
            )
        )

    finished = 0
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(run_job, job) for job in jobs]
        for fut in as_completed(futures):
            index, n, label = fut.result()
            finished += 1
            print(f"[{finished}/{total}] finished #{index}  {n} requests  {label}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
