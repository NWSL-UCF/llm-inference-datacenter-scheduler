# Simulation algorithm

The discrete-event simulator replays an enriched Azure prefill trace onto `N` GPU workers. Service times are **read from the CSV**; the engine never recomputes FLOPs or KV size. Assignment is join-shortest-queue — see [SCHEDULING.md](SCHEDULING.md). Latency columns: [preprocess/README.md](../preprocess/README.md). How to run: [README.md](README.md).

Implemented in `simulator.py` (`simulate`, `iter_arrivals`).

## What is simulated

One request = one prefill job. Each worker is a single GPU that runs **at most one job at a time**. Waiting jobs sit in a per-worker FIFO. There is no decode phase, no batching, no preemption, and no extra network hop onto the assigned server.

Time is milliseconds from the first trace row (`t = 0`).

## Service time

`--latency-mode` selects which precomputed columns become the job’s service time `S`. Unused components are written as `0` in the output so the CSV matches what the worker actually ran.

| Mode | `S` | `compute_ms` | `transfer_ms` |
|---|---|---|---|
| `computeonly` | prefill | prefill | `0` |
| `transferonly` | transfer | `0` | transfer |
| `compute+transfer` | prefill + transfer (sequential) | prefill | transfer |

`compute+transfer` is one contiguous busy interval, not two events. The worker stays busy for `S` and then becomes free (or starts the next queued job).

`GeneratedTokens` is copied to the output and is not used for `S`.

## State

For each worker `i ∈ {0 … N−1}`:

| Field | Meaning |
|---|---|
| `busy[i]` | True while a job is in service |
| `waiting[i]` | FIFO of **service times** of jobs not yet started |
| `free_at[i]` | Time the worker will finish every job already assigned to it (in-flight + queued) |

Global:

| Field | Meaning |
|---|---|
| Event heap | Completions only: `(done_time, seq, COMPLETION, worker)` |
| `seq` | Monotonic push counter so equal-time heap entries stay ordered |
| Arrival cursor | Next unread trace row |

Arrivals are **not** on the heap. The engine always knows the next arrival time from the sorted trace and the next completion time from `heap[0]`.

## Event rule

At every step, compare the next arrival time `t_arr` to the next completion time `t_cmp`:

```
handle arrival  if  t_arr < t_cmp
handle completion otherwise
```

**Equal times run as completions first.** A worker that finishes at `t` is idle (or has already started its next queued job) before JSQ sees an arrival at the same `t`. `validate.py` uses the same rule (`drain(until=t)` before `pick_server`).

## Arrival (`t_arr < t_cmp`)

1. Read `(now, S, compute, transfer, …)` from the next trace row. `now` is relative arrival time.
2. Choose a worker with JSQ (`pick_server`). Output `serverId` is `worker + 1`.
3. If the worker is idle:
   - Wait `W = 0`
   - Start immediately: completion `C = now + S`
   - Mark busy, push a completion event at `C`
4. If the worker is busy:
   - Wait `W = free_at[worker] − now`
   - Completion `C = free_at[worker] + S`
   - Append `S` to that worker’s FIFO (do **not** push a completion yet)
5. Set `free_at[worker] = C`
6. Write one output row (below)
7. Advance the arrival cursor

`arrival_to_server_timestamp` is the same as `relative_timestamp`. There is no dispatch or fabric delay.

## Completion (heap head)

Pop `(now, worker)`.

- If that worker’s FIFO is non-empty: pop the next `S`, start it at `now`, push a new completion at `now + S`. The worker stays busy.
- Else: mark the worker idle.

`free_at` is already correct from assignment time; completions only start the next queued job or free the GPU.

## Output row

All times are milliseconds from the first request.

| Column | Value |
|---|---|
| `relative_timestamp` | Arrival `now` |
| `arrival_to_server_timestamp` | Same as `now` |
| `ContextTokens` / `GeneratedTokens` | From the trace |
| `waiting_in_queue_ms` | `W` |
| `compute_ms` / `transfer_ms` | Mode-filtered components of `S` |
| `completion_timestamp` | `C = now + W + S` |
| `serverId` | Assigned worker, `1 … N` |

Identity used everywhere:

```
completion = arrival + wait + service
service    = compute_ms + transfer_ms
```

## Trace ingest

`iter_arrivals` reads the **enriched** CSV (`dataset/input/…`), not the raw Azure files.

- `TIMESTAMP` is a float (ms) or ISO-8601; values are shifted so the first row is `0`.
- Prefill column: `PrefillMs__{gpu}__{model}`
- Transfer column: `TransferMs__{link}Gbps__{model}`
- KV column `KVCacheMiB__{model}` is read but not written (kept for validation / later use)

Rows are processed in file order. The Azure traces are already time-sorted.

## What is not modeled

- Decode tokens or decode-server queues
- Continuous batching / iteration-level scheduling
- Prefill/decode colocation or KV-cache memory capacity
- Preemption, migration, or jockeying after assignment
- Heterogeneous workers (every server has the same GPU and model)
- Network contention among transfers (transfer time is a constant per request)
- Dispatch delay between the scheduler and the GPU

Those omissions are intentional: the engine is a single-timeline JSQ replay of precomputed service times.

## Checking a run

`validate.py` replays an assignment CSV against the same JSQ + FIFO rules and checks timestamps, `serverId`, wait, service, and completion. Use the same `--gpu`, `--model`, and `--servers` as the original run. See [README.md](README.md).
