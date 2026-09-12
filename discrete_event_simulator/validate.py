#!/usr/bin/env python3
"""Replay assignments and check they match join-shortest-queue.

Checks:
  1. Same row count as the trace; timestamps match in order.
  2. serverId is in 1..N.
  3. At every arrival, the assigned server had the minimum unfinished
     count (waiting + in-flight). Ties must be the lowest index.
  4. Each server runs at most one prefill at a time (FIFO).
  5. completion_timestamp is start + service, after any jobs already on that server.
  6. waiting_in_queue_ms is queue delay; compute_ms + transfer_ms is service time.
  7. arrival_to_server_timestamp is when the request is enqueued on the assigned worker.

Example:
  python3 discrete_event_simulator/validate.py \\
    --trace dataset_insights/code_prefill_ms_10min.csv \\
    --assignments discrete_event_simulator/results/code_H100PCIe_Llama3.1_70B_10min.csv \\
    --gpu H100PCIe --model Llama3.1_70B --servers 90
"""

from __future__ import annotations

import argparse
import csv
import heapq
import sys
from collections import deque
from pathlib import Path
from typing import List, Optional, Tuple

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from simulator import (
    GPU_CHOICES,
    MODEL_CHOICES,
    iter_arrivals,
    latency_column,
    pick_server,
)


def load_assignments(
    path: str,
) -> List[
    Tuple[
        str,
        int,
        Optional[float],
        Optional[float],
        Optional[float],
        Optional[float],
        Optional[int],
        Optional[float],
    ]
]:
    rows: List[
        Tuple[
            str,
            int,
            Optional[float],
            Optional[float],
            Optional[float],
            Optional[float],
            Optional[int],
            Optional[float],
        ]
    ] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        ts_name = "relative_timestamp" if "relative_timestamp" in (reader.fieldnames or []) else "timestamp"
        if reader.fieldnames is None or ts_name not in reader.fieldnames or "serverId" not in reader.fieldnames:
            raise SystemExit(f"assignments need relative_timestamp,serverId; have: {reader.fieldnames}")
        names = set(reader.fieldnames)
        for row in reader:
            done_name = "completion_timestamp" if "completion_timestamp" in names else "completionTimestamp"
            wait_name = "waiting_in_queue_ms" if "waiting_in_queue_ms" in names else "waitingMs"
            arr_name = "arrival_to_server_timestamp" if "arrival_to_server_timestamp" in names else "serverArrivalTimestamp"
            done = float(row[done_name]) if done_name in names else None
            wait = float(row[wait_name]) if wait_name in names else None
            if "compute_ms" in names or "transfer_ms" in names:
                proc = float(row.get("compute_ms") or 0) + float(row.get("transfer_ms") or 0)
            elif "processingMs" in names:
                proc = float(row["processingMs"])
            else:
                proc = None
            srv_arr = float(row[arr_name]) if arr_name in names else None
            ctx = int(row["contextTokens"]) if "contextTokens" in names else None
            kv = float(row["KVCache_MB"]) if "KVCache_MB" in names else None
            rows.append((row[ts_name], int(row["serverId"]), done, wait, proc, srv_arr, ctx, kv))
    return rows


def validate(trace: str, assignments_path: str, column: str, n_servers: int, model: str) -> int:
    assigned = load_assignments(assignments_path)
    waiting = [deque() for _ in range(n_servers)]
    busy = [False] * n_servers
    free_at = [0.0] * n_servers
    heap: List[Tuple[float, int, int]] = []  # (time, seq, worker)
    seq = 0
    errors = 0
    n = 0

    def drain(until: float) -> None:
        nonlocal seq
        while heap and heap[0][0] <= until:
            now, _, worker = heapq.heappop(heap)
            if waiting[worker]:
                service = waiting[worker].popleft()
                seq += 1
                heapq.heappush(heap, (now + service, seq, worker))
            else:
                busy[worker] = False

    for i, (
        (ts_str, t, service, context_tokens, _generated_tokens, kv_mib, *_),
        (out_ts, server_id, out_done, out_wait, out_proc, out_srv_arr, out_ctx, out_kv),
    ) in enumerate(zip(iter_arrivals(trace, column), assigned)):
        n += 1
        try:
            out_t = float(out_ts)
        except ValueError:
            out_t = None
        if out_t is None:
            if ts_str != out_ts:
                print(f"row {n}: timestamp mismatch trace={ts_str!r} out={out_ts!r}", file=sys.stderr)
                errors += 1
        elif abs(out_t - t) > 1e-6:
            print(f"row {n}: relative_timestamp {out_ts} != expected {t}", file=sys.stderr)
            errors += 1
        if not (1 <= server_id <= n_servers):
            print(f"row {n}: serverId {server_id} not in 1..{n_servers}", file=sys.stderr)
            errors += 1
            continue

        # Completions at t happen before this arrival (same rule as the simulator).
        drain(t)

        expected = pick_server(n_servers, waiting, busy) + 1
        if server_id != expected:
            loads = [len(waiting[j]) + (1 if busy[j] else 0) for j in range(n_servers)]
            print(
                f"row {n} t={t}: assigned {server_id}, JSQ expected {expected}; "
                f"loads={loads[: max(expected, server_id) + 2]}",
                file=sys.stderr,
            )
            errors += 1

        worker = server_id - 1
        if not busy[worker]:
            wait = 0.0
            done = t + service
            busy[worker] = True
            seq += 1
            heapq.heappush(heap, (done, seq, worker))
        else:
            wait = free_at[worker] - t
            done = free_at[worker] + service
            waiting[worker].append(service)
        free_at[worker] = done
        if out_done is not None and abs(out_done - done) > 1e-6:
            print(
                f"row {n} t={t}: completion_timestamp {out_done} != expected {done}",
                file=sys.stderr,
            )
            errors += 1
        if out_wait is not None and abs(out_wait - wait) > 1e-6:
            print(f"row {n} t={t}: waiting_in_queue_ms {out_wait} != expected {wait}", file=sys.stderr)
            errors += 1
        if out_proc is not None and abs(out_proc - service) > 1e-6:
            print(f"row {n} t={t}: compute_ms+transfer_ms {out_proc} != expected {service}", file=sys.stderr)
            errors += 1
        if out_srv_arr is not None and abs(out_srv_arr - t) > 1e-6:
            print(
                f"row {n} t={t}: arrival_to_server_timestamp {out_srv_arr} != expected {t}",
                file=sys.stderr,
            )
            errors += 1
        if out_ctx is not None and out_ctx != context_tokens:
            print(
                f"row {n} t={t}: contextTokens {out_ctx} != expected {context_tokens}",
                file=sys.stderr,
            )
            errors += 1
        if out_kv is not None:
            expected_kv = kv_mib
            if abs(out_kv - expected_kv) > 1e-6:
                print(
                    f"row {n} t={t}: KVCache_MB {out_kv} != expected {expected_kv}",
                    file=sys.stderr,
                )
                errors += 1

    extra = sum(1 for _ in assigned[n:])
    # Count remaining trace rows if assignments ran out.
    leftover_trace = 0
    if n == len(assigned):
        leftover_trace = sum(1 for _ in iter_arrivals(trace, column)) - n

    if extra:
        print(f"assignments have {extra} extra rows", file=sys.stderr)
        errors += 1
    if leftover_trace > 0:
        print(f"trace has {leftover_trace} extra rows", file=sys.stderr)
        errors += 1

    print(
        f"checked {n} rows against {assignments_path}: "
        f"{'OK' if errors == 0 else f'{errors} error(s)'}",
        file=sys.stderr,
    )
    return errors


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trace", required=True)
    p.add_argument("--assignments", required=True)
    p.add_argument("--gpu", required=True, choices=GPU_CHOICES)
    p.add_argument("--model", required=True, choices=MODEL_CHOICES)
    p.add_argument("--servers", type=int, default=90)
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    if args.servers < 1:
        raise SystemExit("--servers must be >= 1")
    errors = validate(
        args.trace,
        args.assignments,
        latency_column(args.gpu, args.model),
        args.servers,
        args.model,
    )
    raise SystemExit(1 if errors else 0)


if __name__ == "__main__":
    main()
