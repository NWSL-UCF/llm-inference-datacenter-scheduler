# Discrete-event simulator

Join-shortest-queue (JSQ) assignment of LLM prefill requests onto GPU servers. Each request’s service time is read from an enriched Azure trace (compute, transfer, or both). The scheduler does not recompute FLOPs or KV size.

Simulation algorithm: [SIMULATOR.md](SIMULATOR.md). Assignment policy: [SCHEDULING.md](SCHEDULING.md). How those latency columns are built: [preprocess/README.md](../preprocess/README.md).

## Input CSV

`simulator.py` reads the **enriched** traces under `dataset/input/`, not the raw Azure files.

```
dataset/input/AzureLLMInferenceTrace_code_1week.csv
dataset/input/AzureLLMInferenceTrace_conv_1week.csv
```

1000-row samples: `*_sample1000.csv` in the same folder.

Required columns:

- `TIMESTAMP`, `ContextTokens`, `GeneratedTokens`
- `PrefillMs__{gpu}__{model}` for compute modes
- `TransferMs__{link}Gbps__{model}` for transfer modes
- `KVCacheMiB__{model}`

Generate them from the repo root:

```bash
python3 preprocess/download_dataset.py
python3 preprocess/enrich_dataset.py
```

See [preprocess/README.md](../preprocess/README.md) for the 12 derived columns and equations.

## Run one simulation

```bash
python3 discrete_event_simulator/simulator.py \
  --trace dataset/input/AzureLLMInferenceTrace_code_1week_sample1000.csv \
  --gpu H100SXM --model Llama3.1_8B --link 200 \
  --latency-mode compute+transfer --servers 90 \
  --out assignments.csv
```

| Arg | Meaning |
|---|---|
| `--trace` | Enriched CSV path (code or conv, full week or sample) |
| `--gpu` | `H100PCIe` or `H100SXM`. Selects the `PrefillMs__…` column |
| `--model` | `Llama3.1_8B` or `Llama3.1_70B`. Selects prefill, KV, and transfer columns |
| `--link` | `100`, `200`, or `400` (Gbps). Selects `TransferMs__{link}Gbps__…`. Default `200` |
| `--latency-mode` | Per-request service time. Default `computeonly` |
| `--servers` | Number of GPU workers. Default `90`. `serverId` is 1-based |
| `--out` | Assignment CSV path |

`--latency-mode`:

| Value | Service time | `compute_ms` | `transfer_ms` |
|---|---|---|---|
| `computeonly` | prefill | prefill | `0` |
| `transferonly` | transfer | `0` | transfer |
| `compute+transfer` | prefill + transfer (sequential) | prefill | transfer |

## Output columns

Times are milliseconds from the first request (`t = 0`).

| Column | Meaning |
|---|---|
| `relative_timestamp` | Request arrival (ms from first row) |
| `arrival_to_server_timestamp` | Enqueue time on the assigned worker (same as arrival; no extra hop delay) |
| `ContextTokens` | Prompt length from the trace |
| `GeneratedTokens` | Decode length from the trace (not used for service time) |
| `waiting_in_queue_ms` | Queue delay on that worker |
| `compute_ms` | Prefill time used in this mode (`0` if `transferonly`) |
| `transfer_ms` | KV-ship time used in this mode (`0` if `computeonly`) |
| `completion_timestamp` | `arrival + wait + service` |
| `serverId` | Assigned worker, `1 … N` |

## Sweep all combinations (`runner.py`)

Default grid is **72** runs: 2 traces × 2 GPUs × 2 models × 3 links × 3 latency modes. Eight processes, 90 servers.

```bash
mkdir -p dataset/outputs
python3 discrete_event_simulator/runner.py --jobs 8
```

Background:

```bash
nohup python3 discrete_event_simulator/runner.py --jobs 8 \
  > dataset/outputs/runner.log 2>&1 &
```

Outputs: `dataset/outputs/{trace}_{gpu}_{model}_{link}Gbps_{mode}.csv`  
(`compute+transfer` is spelled `computeplustransfer` in the filename.)

| Runner flag | Default |
|---|---|
| `--jobs` | `8` |
| `--servers` | `90` |
| `--input-dir` | `dataset/input` |
| `--out-dir` | `dataset/outputs` |
| `--traces` | `code conv` |
| `--gpus` | both |
| `--models` | both |
| `--links` | `100 200 400` |
| `--latency-modes` | all three |
| `--sample` | off (use `*_sample1000.csv` when set) |

```bash
python3 discrete_event_simulator/runner.py --sample --traces code --links 200
```

Replay-check one assignment file with `validate.py` (same `--gpu` / `--model` / `--servers` as the run).
