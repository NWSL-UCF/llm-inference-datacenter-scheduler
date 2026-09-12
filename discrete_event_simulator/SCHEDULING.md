# Scheduling policy

The simulator is built around **join-shortest-queue (JSQ)** assignment of prefill requests onto homogeneous GPU workers. The event engine in [SIMULATOR.md](SIMULATOR.md) exists to make that policy well-defined in continuous time: each arrival sees a consistent unfinished-job count, and each worker runs a FIFO of the jobs JSQ sent it.

Implemented in `simulator.py` (`pick_server`) and re-checked by `validate.py`.

## Placement

On every arrival the scheduler looks at all `N` workers and sends the request to the one with the **fewest unfinished jobs**:

```
load(i) = |waiting[i]| + (1 if busy[i] else 0)
```

- `busy[i]`: a job is in service on worker `i`
- `waiting[i]`: jobs assigned to `i` that have not started

That is a **job-count** load, not remaining work. Two queues of length 3 are equal even if one holds long prefills and the other holds short ones.

Ties go to the **lowest worker index** (`0 … N−1` internally, `serverId = index + 1` in the CSV). The scan in `pick_server` is left-to-right and only updates on a strict `<`, so the first minimum wins.

Assignment is instantaneous and irrevocable. There is no later migration, steal, or jockeying.

## Why JSQ for this design

The system under study is prefill serving on a pool of identical GPUs. Each request’s service time is already on the trace (compute, transfer, or both). The scheduler’s only job is **which server**, not **how long** — it does not recompute FLOPs, KV size, or link rate.

JSQ is the policy the rest of the simulator is shaped around:

| Design choice | Why it matches JSQ |
|---|---|
| Per-worker FIFO | Jobs stay on the server JSQ chose; order on that server is arrival order |
| One job in service | Load is an integer job count, not a share of a batched GPU |
| Completions before same-time arrivals | An arrival at time `t` sees workers that finished at `t` as already freed |
| `free_at` tracked at assign time | Wait and completion are determined the moment JSQ places the job |
| Homogeneous `--servers N` | JSQ assumes interchangeable workers; GPU/model/link are sweep parameters, not per-server types |

Without those rules, “shortest queue” would depend on event order or leftover work and would not be reproducible.

## What the scheduler does not use

JSQ ignores every per-request size field except insofar as those fields already determined `S` in the CSV:

- `ContextTokens`, `GeneratedTokens`
- `PrefillMs__…`, `TransferMs__…`, `KVCacheMiB__…`
- Remaining service on the busy job
- Historical arrival rate

So a huge 70B prefill and a tiny 8B prefill look the same at decision time if they would add one job to the same queues. Size only shows up later, as wait for whoever lands behind them.

## Decision at an arrival

Let `now` be the arrival time and `i*` the JSQ worker.

**Idle (`busy[i*] = false`).** The job starts at `now`. Wait is `0`. Completion is `now + S`. That worker’s load becomes 1.

**Busy.** The job is appended to `waiting[i*]`. It starts when every job already on `i*` has finished (`free_at[i*]`). Wait is `free_at[i*] − now`. Completion is `free_at[i*] + S`. Load increases by 1.

After the decision, `free_at[i*]` is set to that completion so the next arrival that picks the same worker sees the new backlog.

## Interaction with the event timeline

JSQ is evaluated only on arrivals. Completions never reassign work; they only start the next FIFO job or mark the GPU idle.

When an arrival and a completion share a timestamp, **the completion is applied first**. Example: worker 3 finishes its last job at `t = 100` and a new request also arrives at `t = 100`. After the completion, worker 3 has load 0. If it is (tied for) the shortest queue, JSQ can send the new request there with wait 0. If arrivals were processed first, worker 3 would still look busy and the request would go elsewhere.

`validate.py` drains all completions with `time <= t` before calling `pick_server`, which is the same order.

## FIFO on each worker

Once JSQ has placed a job, that worker is an ordinary single-server queue:

- At most one job in service
- Waiting jobs start in the order they were assigned
- No preemption
- Service time does not change after assignment

Global JSQ plus local FIFO is the whole scheduling stack. There is no second-level policy (priority, size-based, or decode-aware).

## Load vs work

JSQ here is **not** join-the-shortest-work / least-work-left. Least-work-left would compare `free_at[i] − now` (remaining busy time) instead of job counts. That would need the scheduler to use `S` at decision time. This codebase does not: `pick_server` only sees queue lengths and busy bits.

Consequences:

- A worker with one long job looks lighter than a worker with two short jobs, even if the long job finishes later.
- Tie-breaking is by server index, not by leftover time.

That is the intended baseline: cheap, deterministic, and independent of the latency-mode columns.

## Homogeneous pool

Every worker has the same GPU, model, and link. `--gpu`, `--model`, `--link`, and `--latency-mode` change **every** request’s `S` the same way; they do not create a mixed cluster. JSQ therefore never compares “faster server” vs “slower server.”

Default `N = 90` (`--servers`). `serverId` in the output is `1 … N`.

## Checks

`validate.py` replays the assignment file and requires, at every arrival:

1. `serverId` is the JSQ choice (min load, lowest index on ties)
2. That server never runs two prefills at once
3. `waiting_in_queue_ms` and `completion_timestamp` match FIFO on the assigned server
4. `compute_ms + transfer_ms` equals the service time used for that row

If those hold, the CSV is a correct JSQ schedule for the trace and `N`.
