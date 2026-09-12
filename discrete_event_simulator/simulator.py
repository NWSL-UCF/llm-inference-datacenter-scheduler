#!/usr/bin/env python3
"""Join-shortest-queue discrete-event simulator for LLM prefill serving.

One global event timeline (arrivals from the sorted trace + worker
completions). Each worker has a simple FIFO of waiting requests. On arrival
the scheduler assigns the request to the server with the fewest unfinished
jobs (waiting + in-flight). Ties go to the lowest index. serverId is 1-based.

Service time is one of:
  computeonly        PrefillMs__{gpu}__{model}
  transferonly       TransferMs__{link}Gbps__{model}
  compute+transfer   prefill + transfer

Example:
  python discrete_event_simulator/simulator.py \\
    --trace dataset/input/AzureLLMInferenceTrace_code_1week_sample1000.csv \\
    --gpu H100SXM --model Llama3.1_8B --link 200 \\
    --latency-mode compute+transfer --servers 90 \\
    --out assignments.csv
"""

from __future__ import annotations

import argparse
import csv
import heapq
import sys
from collections import deque
from datetime import datetime
from typing import Deque, Iterator, List, Optional, Tuple

GPU_CHOICES = ("H100PCIe", "H100SXM")
MODEL_CHOICES = ("Llama3.1_8B", "Llama3.1_70B")
LINK_CHOICES = (100, 200, 400)
LATENCY_MODE_CHOICES = ("computeonly", "transferonly", "compute+transfer")

COMPLETION = 0

Arrival = Tuple[str, float, float, int, int, float, float, float]


def prefill_column(gpu: str, model: str) -> str:
    return f"PrefillMs__{gpu}__{model}"


def transfer_column(link_gbps: int, model: str) -> str:
    return f"TransferMs__{link_gbps}Gbps__{model}"


def kv_column(model: str) -> str:
    return f"KVCacheMiB__{model}"


def latency_column(gpu: str, model: str) -> str:
    """Prefill compute column (new traces)."""
    return prefill_column(gpu, model)


def split_latencies(compute: float, transfer: float, mode: str) -> Tuple[float, float, float]:
    """Return (compute_ms, transfer_ms, service_ms) for the selected mode.

    Unused components are zeroed so output matches what the worker actually ran.
    """
    if mode == "computeonly":
        return compute, 0.0, compute
    if mode == "transferonly":
        return 0.0, transfer, transfer
    if mode == "compute+transfer":
        return compute, transfer, compute + transfer
    raise SystemExit(f"unknown --latency-mode {mode!r}")


def parse_timestamp_ms(ts_str: str) -> float:
    """Numeric ms, or ISO-8601 from the Azure traces."""
    try:
        return float(ts_str)
    except ValueError:
        return datetime.fromisoformat(ts_str).timestamp() * 1000.0


def iter_arrivals(
    path: str,
    column: Optional[str] = None,
    gpu: Optional[str] = None,
    model: Optional[str] = None,
    link: int = 200,
    latency_mode: str = "computeonly",
) -> Iterator[Arrival]:
    """Yield (timestamp_str, timestamp_ms, service_ms, context_tokens, generated_tokens, kv_mib, compute_ms, transfer_ms).

    timestamp_ms is relative to the first row (0 at the first arrival).
    Latencies and KV size are read from the enriched trace; they are not recomputed.
    If `column` is set, that column is the service time (legacy compute-only path).
    Otherwise service comes from gpu/model/link/latency_mode.
    """
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        names = reader.fieldnames or []
        if column is not None:
            needed = [column, "ContextTokens", "GeneratedTokens"]
        else:
            if gpu is None or model is None:
                raise SystemExit("iter_arrivals needs --gpu/--model (or a legacy column)")
            needed = ["ContextTokens", "GeneratedTokens", kv_column(model)]
            if latency_mode in ("computeonly", "compute+transfer"):
                needed.append(prefill_column(gpu, model))
            if latency_mode in ("transferonly", "compute+transfer"):
                needed.append(transfer_column(link, model))
        missing = [c for c in needed if c not in names]
        if missing:
            available = ", ".join(names)
            raise SystemExit(f"missing {missing} in trace; have: {available}")

        t0: Optional[float] = None
        for row in reader:
            ts_str = row["TIMESTAMP"]
            t_abs = parse_timestamp_ms(ts_str)
            if t0 is None:
                t0 = t_abs
            context_tokens = int(row["ContextTokens"])
            generated_tokens = int(row["GeneratedTokens"]) if "GeneratedTokens" in names else 0
            kv_name = kv_column(model) if model is not None else ""
            kv_mib = float(row[kv_name]) if kv_name and kv_name in names else 0.0
            if column is not None:
                compute, transfer, service = split_latencies(float(row[column]), 0.0, "computeonly")
            else:
                assert gpu is not None and model is not None
                pref = prefill_column(gpu, model)
                xfer = transfer_column(link, model)
                raw_compute = float(row[pref]) if pref in names else 0.0
                raw_transfer = float(row[xfer]) if xfer in names else 0.0
                compute, transfer, service = split_latencies(raw_compute, raw_transfer, latency_mode)
            yield ts_str, t_abs - t0, service, context_tokens, generated_tokens, kv_mib, compute, transfer


def pick_server(n_servers: int, waiting: List[Deque], busy: List[bool]) -> int:
    best_i = 0
    best = len(waiting[0]) + (1 if busy[0] else 0)
    for i in range(1, n_servers):
        load = len(waiting[i]) + (1 if busy[i] else 0)
        if load < best:
            best = load
            best_i = i
    return best_i


def simulate(
    arrivals: Iterator[Arrival],
    n_servers: int,
    out_path: str,
) -> int:
    waiting: List[Deque[float]] = [deque() for _ in range(n_servers)]
    busy = [False] * n_servers
    free_at = [0.0] * n_servers
    # completions: (time, seq, COMPLETION, worker)
    heap: List[Tuple[float, int, int, int]] = []
    seq = 0
    n_assigned = 0

    arrival_iter = iter(arrivals)
    nxt: Optional[Arrival] = next(arrival_iter, None)

    with open(out_path, "w", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(
            [
                "relative_timestamp",
                "arrival_to_server_timestamp",
                "ContextTokens",
                "GeneratedTokens",
                "waiting_in_queue_ms",
                "compute_ms",
                "transfer_ms",
                "completion_timestamp",
                "serverId",
            ]
        )

        while nxt is not None or heap:
            t_arr = nxt[1] if nxt is not None else float("inf")
            t_cmp = heap[0][0] if heap else float("inf")
            # At equal time, completions run first so JSQ sees the freed worker.
            handle_arrival = t_arr < t_cmp

            if handle_arrival:
                assert nxt is not None
                ts_str, now, service, context_tokens, generated_tokens, _kv_mib, compute, transfer = nxt
                worker = pick_server(n_servers, waiting, busy)
                if not busy[worker]:
                    wait = 0.0
                    done = now + service
                    busy[worker] = True
                    seq += 1
                    heapq.heappush(heap, (done, seq, COMPLETION, worker))
                else:
                    wait = free_at[worker] - now
                    done = free_at[worker] + service
                    waiting[worker].append(service)
                free_at[worker] = done
                writer.writerow(
                    [
                        f"{now:.6f}",
                        f"{now:.6f}",
                        context_tokens,
                        generated_tokens,
                        f"{wait:.6f}",
                        f"{compute:.6f}",
                        f"{transfer:.6f}",
                        f"{done:.6f}",
                        worker + 1,
                    ]
                )
                n_assigned += 1
                nxt = next(arrival_iter, None)
            else:
                now, _, _, worker = heapq.heappop(heap)
                if waiting[worker]:
                    service = waiting[worker].popleft()
                    seq += 1
                    heapq.heappush(heap, (now + service, seq, COMPLETION, worker))
                else:
                    busy[worker] = False

    return n_assigned


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trace", required=True, help="enriched Azure CSV (TIMESTAMP + PrefillMs/TransferMs columns)")
    p.add_argument("--gpu", required=True, choices=GPU_CHOICES)
    p.add_argument("--model", required=True, choices=MODEL_CHOICES)
    p.add_argument("--link", type=int, default=200, choices=LINK_CHOICES, help="prefill→decode link Gbps (default 200)")
    p.add_argument(
        "--latency-mode",
        default="computeonly",
        choices=LATENCY_MODE_CHOICES,
        help="per-request service time (default computeonly)",
    )
    p.add_argument("--servers", type=int, default=90, help="number of GPU servers (default 90)")
    p.add_argument(
        "--out",
        required=True,
        help="output CSV: relative_timestamp,arrival_to_server_timestamp,ContextTokens,GeneratedTokens,waiting_in_queue_ms,compute_ms,transfer_ms,completion_timestamp,serverId",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    if args.servers < 1:
        raise SystemExit("--servers must be >= 1")
    arrivals = iter_arrivals(
        args.trace,
        gpu=args.gpu,
        model=args.model,
        link=args.link,
        latency_mode=args.latency_mode,
    )
    n = simulate(arrivals, args.servers, args.out)
    print(
        f"assigned {n} requests -> {args.out}  "
        f"({args.gpu} / {args.model} / {args.link}Gbps / {args.latency_mode}, {args.servers} servers)",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
