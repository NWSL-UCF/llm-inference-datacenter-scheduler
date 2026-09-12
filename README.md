# LLM inference datacenter scheduler

This repository simulates **how a single central scheduler assigns incoming LLM inference requests to prefill GPU servers** in a large datacenter. Requests come from public Azure traces of two workloads: **code** and **conversation**. The scheduler sees every arrival first, then places each request on one prefill worker with **join-shortest-queue (JSQ)**.

The assignment policy is documented in [discrete_event_simulator/SCHEDULING.md](discrete_event_simulator/SCHEDULING.md). The event engine that makes that policy well-defined in time is in [discrete_event_simulator/SIMULATOR.md](discrete_event_simulator/SIMULATOR.md).

## Scope of the simulation

**In scope.** One homogeneous pool of prefill GPUs (same GPU type, same model, same prefill→decode link in a given run). A central scheduler receives the full arrival stream and, for each request, chooses **which prefill server** should run the prefill. Service time is read from precomputed columns on an enriched Azure trace (prefill compute, KV transfer, or both). Each worker runs at most one prefill at a time, with a FIFO of waiting jobs.

**Out of scope.** Decode-token generation and decode-server queues. Continuous batching. Mixing models or NIC speeds inside one run. Prefill/decode colocation, KV-cache capacity limits, preemption, and job migration after assignment.

The Azure files only give `TIMESTAMP`, `ContextTokens`, and `GeneratedTokens`. This simulator does not generate text. It only answers: **which prefill GPU should take this request, and when does that prefill finish?**

## Prefill and decode

An LLM inference request has two phases.

**Prefill** reads the whole prompt (`ContextTokens`) in one forward pass and builds the key–value (KV) cache for those tokens. Cost grows with prompt length. In a disaggregated datacenter, this work often runs on a dedicated prefill GPU.

**Decode** then emits output tokens one by one (`GeneratedTokens`). Each step attends using the KV cache. After prefill, that cache may be shipped over the network to a decode server.

This project models **prefill placement only**. Decode length is kept on the trace and in the output CSV, but it is not used as service time. Optional transfer time is the cost of sending the KV cache off the prefill server, not a full decode schedule.

## Data

Raw traces: [Azure LLM Inference Trace 2024](https://github.com/Azure/AzurePublicDataset/releases/tag/dataset-llm-2024).

| Trace | Workload | Dates |
|---|---|---|
| `AzureLLMInferenceTrace_code_1week.csv` | Code | 10–16 May 2024 |
| `AzureLLMInferenceTrace_conv_1week.csv` | Conversation | 12–18 May 2024 |

Each row is one request: arrival time, prompt length, generated length. Prefill milliseconds, KV size, and transfer milliseconds are added in preprocess (from `ContextTokens` only).

## How to navigate this repo

Work top to bottom: understand the policy, build the input traces, then emit a scheduling trace.

### 1. Scheduling policy

[discrete_event_simulator/SCHEDULING.md](discrete_event_simulator/SCHEDULING.md) — JSQ by unfinished job count, lowest-index ties, per-worker FIFO, and what the scheduler does not look at (token length, leftover work). One run is one GPU, one model, and one link on every worker.

### 2. Preprocess (raw Azure → enriched input)

[preprocess/README.md](preprocess/README.md) — download the Azure CSVs and add the 12 latency / KV / transfer columns the simulator reads. Equations and constants live there.

```bash
python3 preprocess/download_dataset.py
python3 preprocess/enrich_dataset.py
```

| | Path |
|---|---|
| Raw download | `dataset/raw/` |
| Enriched input | `dataset/input/` |

1000-row samples (`*_sample1000.csv`) sit next to the full-week files for smoke tests.

Hourly plots of the raw arrivals (not required to run the simulator): [dataset_insights/README.md](dataset_insights/README.md).

### 3. Discrete-event simulation (enriched input → scheduling trace)

[discrete_event_simulator/README.md](discrete_event_simulator/README.md) — CLI, latency modes, output columns, and the 72-run sweep.

[discrete_event_simulator/SIMULATOR.md](discrete_event_simulator/SIMULATOR.md) — arrivals vs completions, worker state, and how wait / completion are written.

One run (sample trace):

```bash
python3 discrete_event_simulator/simulator.py \
  --trace dataset/input/AzureLLMInferenceTrace_code_1week_sample1000.csv \
  --gpu H100SXM --model Llama3.1_8B --link 200 \
  --latency-mode compute+transfer --servers 90 \
  --out assignments.csv
```

The output CSV is the **scheduling trace**: each request’s arrival, assigned `serverId`, queue wait, compute/transfer used, and completion time.

Full grid (code and conv × GPUs × models × links × latency modes):

```bash
mkdir -p dataset/outputs
python3 discrete_event_simulator/runner.py --jobs 8
```

Outputs: `dataset/outputs/{trace}_{gpu}_{model}_{link}Gbps_{mode}.csv`.

Replay-check an assignment file with `discrete_event_simulator/validate.py` (same `--gpu` / `--model` / `--servers` as the run). Details are in the [simulator README](discrete_event_simulator/README.md).

## Layout

```
preprocess/                      download + enrich Azure traces
discrete_event_simulator/        JSQ engine, runner, validator, policy docs
dataset/raw/                     downloaded Azure CSVs
dataset/input/                   enriched traces (simulator input)
dataset/outputs/                 scheduling traces (simulator output)
dataset_insights/                plots of raw request rates
```
