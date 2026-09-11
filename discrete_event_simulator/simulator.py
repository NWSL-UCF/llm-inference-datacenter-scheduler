#!/usr/bin/env python3
"""Join-shortest-queue discrete-event simulator for LLM prefill serving.

One global event timeline (arrivals from the sorted trace + worker
completions). Each worker has a simple FIFO of waiting requests. On arrival
the scheduler assigns the request to the server with the fewest unfinished
jobs (waiting + in-flight). Ties go to the lowest index. serverId is 1-based.

Example:
  python discrete_event_simulator/simulator.py \\
    --trace dataset_insights/prefill_ms_sample_10rows.csv \\
    --gpu H100SXM --model Llama3.1_8B --servers 90 \\
    --out assignments.csv
"""

from __future__ import annotations

import argparse
import csv
import heapq
import sys
from collections import deque
from typing import Deque, Iterator, List, Optional, Tuple

GPU_CHOICES = ("H100PCIe", "H100SXM")
MODEL_CHOICES = ("Llama3.1_8B", "Llama3.1_70B")

COMPLETION = 0
MIB = 1024 * 1024

# Llama 3.1 GQA, BF16/FP16 (b=2 B). KV_tok = 2 * L * H_kv * d_h * b
# 8B:  L=32, H_kv=8, d_h=128 → 131072 B/token
# 70B: L=80, H_kv=8, d_h=128 → 327680 B/token
KV_BYTES_PER_TOKEN = {
    "Llama3.1_8B": 2 * 32 * 8 * 128 * 2,
    "Llama3.1_70B": 2 * 80 * 8 * 128 * 2,
}


def latency_column(gpu: str, model: str) -> str:
    return f"{gpu}___{model}"


def kv_cache_mb(context_tokens: int, model: str) -> float:
    """Prefill KV cache in MiB: N_p * KV_tok / 1024^2 (no block rounding)."""
    return context_tokens * KV_BYTES_PER_TOKEN[model] / MIB


def iter_arrivals(path: str, column: str) -> Iterator[Tuple[str, float, float, int]]:
    """Yield (timestamp_str, timestamp_ms, prefill_ms, context_tokens) in file order."""
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        names = reader.fieldnames or []
        missing = [c for c in (column, "ContextTokens") if c not in names]
        if missing:
            available = ", ".join(names)
            raise SystemExit(f"missing {missing} in trace; have: {available}")
        for row in reader:
            ts_str = row["TIMESTAMP"]
            yield ts_str, float(ts_str), float(row[column]), int(row["ContextTokens"])


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
    arrivals: Iterator[Tuple[str, float, float, int]],
    n_servers: int,
    out_path: str,
    model: str,
) -> int:
    waiting: List[Deque[float]] = [deque() for _ in range(n_servers)]
    busy = [False] * n_servers
    free_at = [0.0] * n_servers
    # completions: (time, seq, COMPLETION, worker)
    heap: List[Tuple[float, int, int, int]] = []
    seq = 0
    n_assigned = 0

    arrival_iter = iter(arrivals)
    nxt: Optional[Tuple[str, float, float, int]] = next(arrival_iter, None)

    with open(out_path, "w", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(
            [
                "timestamp",
                "serverArrivalTimestamp",
                "contextTokens",
                "KVCache_MB",
                "completionTimestamp",
                "waitingMs",
                "processingMs",
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
                ts_str, now, service, context_tokens = nxt
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
                        ts_str,
                        f"{now:.6f}",
                        context_tokens,
                        f"{kv_cache_mb(context_tokens, model):.6f}",
                        f"{done:.6f}",
                        f"{wait:.6f}",
                        f"{service:.6f}",
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
    p.add_argument("--trace", required=True, help="prefill_ms CSV (TIMESTAMP + latency columns)")
    p.add_argument("--gpu", required=True, choices=GPU_CHOICES)
    p.add_argument("--model", required=True, choices=MODEL_CHOICES)
    p.add_argument("--servers", type=int, default=90, help="number of GPU servers (default 90)")
    p.add_argument(
        "--out",
        required=True,
        help="output CSV: timestamp,serverArrivalTimestamp,contextTokens,KVCache_MB,completionTimestamp,waitingMs,processingMs,serverId",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    if args.servers < 1:
        raise SystemExit("--servers must be >= 1")
    column = latency_column(args.gpu, args.model)
    n = simulate(iter_arrivals(args.trace, column), args.servers, args.out, args.model)
    print(f"assigned {n} requests -> {args.out}  ({args.gpu} / {args.model}, {args.servers} servers)", file=sys.stderr)


if __name__ == "__main__":
    main()
