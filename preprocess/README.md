# Preprocess

This folder turns the public Azure LLM inference traces into per-request **prefill compute time**, **KV-cache size**, and **prefill→decode transfer time**. Those columns are the inputs for the discrete-event scheduler.

The raw Azure CSVs only have arrival time, prompt length, and generated length. They do not include GPU latency or network transfer time, so we compute those from `ContextTokens` only (`GeneratedTokens` is kept but not used).

## Pipeline

From the repo root:

```bash
python3 preprocess/download_dataset.py
python3 preprocess/enrich_dataset.py
```

| Step | Script | Input | Output |
|---|---|---|---|
| Download | `download_dataset.py` | [Azure LLM Inference Trace 2024](https://github.com/Azure/AzurePublicDataset/releases/tag/dataset-llm-2024) | `dataset/raw/` |
| Enrich | `enrich_dataset.py` | `dataset/raw/*.csv` | `dataset/input/*.csv` (original 3 columns + 12 computed) |

Traces:

- `AzureLLMInferenceTrace_code_1week.csv` — code, 10–16 May 2024
- `AzureLLMInferenceTrace_conv_1week.csv` — conversation, 12–18 May 2024

1000-row samples (header + first 1000 requests) live next to the full files:

- `dataset/input/AzureLLMInferenceTrace_code_1week_sample1000.csv`
- `dataset/input/AzureLLMInferenceTrace_conv_1week_sample1000.csv`

Re-download or rebuild with `--force`.

## The 12 added columns

Kept from Azure: `TIMESTAMP`, `ContextTokens`, `GeneratedTokens`.

**Prefill compute (ms)** — 4 columns, GPU × model:

| Column | GPU | Model |
|---|---|---|
| `PrefillMs__H100PCIe__Llama3.1_8B` | H100 PCIe | Llama 3.1 8B |
| `PrefillMs__H100PCIe__Llama3.1_70B` | H100 PCIe | Llama 3.1 70B |
| `PrefillMs__H100SXM__Llama3.1_8B` | H100 SXM | Llama 3.1 8B |
| `PrefillMs__H100SXM__Llama3.1_70B` | H100 SXM | Llama 3.1 70B |

**KV cache after prefill (MiB)** — 2 columns:

| Column | Model |
|---|---|
| `KVCacheMiB__Llama3.1_8B` | Llama 3.1 8B |
| `KVCacheMiB__Llama3.1_70B` | Llama 3.1 70B |

**KV transfer from prefill server to decode server (ms)** — 6 columns, link × model:

| Column | Link | Model |
|---|---|---|
| `TransferMs__100Gbps__Llama3.1_8B` | 100 Gbps | Llama 3.1 8B |
| `TransferMs__100Gbps__Llama3.1_70B` | 100 Gbps | Llama 3.1 70B |
| `TransferMs__200Gbps__Llama3.1_8B` | 200 Gbps | Llama 3.1 8B |
| `TransferMs__200Gbps__Llama3.1_70B` | 200 Gbps | Llama 3.1 70B |
| `TransferMs__400Gbps__Llama3.1_8B` | 400 Gbps | Llama 3.1 8B |
| `TransferMs__400Gbps__Llama3.1_70B` | 400 Gbps | Llama 3.1 70B |

## Equations

`N_p` is `ContextTokens`.

### 1. Prefill compute time

A dense transformer **forward** pass costs about **2 FLOPs per parameter per token** (one multiply-add). Training is often written as `6 * N * D` (forward + backward); inference keeps only the forward `2 * N * D`. That count is from Kaplan et al., *Scaling Laws for Neural Language Models* (2020), and is the usual prefill-cost model in serving papers (Splitwise, DistServe, Sarathi).

```
FLOPs = 2 * N_params * N_p
```

Wall-clock time uses **model FLOPs utilization (MFU)**: achieved throughput divided by GPU peak, as in Chowdhery et al., *PaLM* (2022) / Megatron-LM.

```
t_prefill_s = (2 * N_params * N_p) / (F_peak * MFU)
```

We store milliseconds:

```
PrefillMs = (2 * N_params * N_p) / (F_peak * 0.6) * 1000
```

This is compute-only. It does not model memory-bandwidth limits, kernel launch, or batching.

### 2. KV cache size

After prefill, the KV cache holds **K and V** for every layer and every prompt token. Llama 3.1 uses **GQA**, so the KV head count is smaller than the query head count. Meta’s Llama 3.1 model card gives the architecture; the per-token byte count is the standard GQA formula (same as vLLM / Hugging Face):

```
KV_tok = 2 * L * H_kv * d_h * b
```

- `2`: key and value
- `L`: layers
- `H_kv`: KV heads
- `d_h`: head dimension
- `b = 2`: BF16 / FP16 bytes

```
KVCacheMiB = (N_p * KV_tok) / (1024 * 1024)
```

No paged-attention block rounding.

### 3. Prefill → decode transfer time

The prefill server ships the KV cache to the decode server. Size in bits over an Ethernet link, with **80% of line rate** usable (headers, credit, less-than-peak goodput):

```
KV_bits    = KVCacheMiB * 1024 * 1024 * 8
TransferMs = KV_bits / (0.8 * BW_Gbps * 1e9) * 1000
```

`BW_Gbps` is **decimal Gbps** (`1e9` bit/s), matching 100 / 200 / 400 GbE, not gibibits.

## Constants

| Symbol | Value | Source |
|---|---|---|
| `N_params` 8B | `8e9` | Meta Llama 3.1 8B (nominal size) |
| `N_params` 70B | `70e9` | Meta Llama 3.1 70B (nominal size) |
| `F_peak` H100 PCIe | `756e12` FLOP/s | NVIDIA H100 datasheet, dense BF16 (no 2:4 sparsity) |
| `F_peak` H100 SXM | `989e12` FLOP/s | NVIDIA H100 datasheet, dense BF16 (no 2:4 sparsity) |
| `MFU` | `0.6` | Chosen operating point for this project |
| Llama 3.1 8B `L`, `H_kv`, `d_h` | 32, 8, 128 | Meta Llama 3.1 model card (GQA) |
| Llama 3.1 70B `L`, `H_kv`, `d_h` | 80, 8, 128 | Meta Llama 3.1 model card (GQA) |
| `b` | 2 bytes | BF16 / FP16 |
| `KV_tok` 8B | `2 * 32 * 8 * 128 * 2` = 131072 B/token | GQA formula above |
| `KV_tok` 70B | `2 * 80 * 8 * 128 * 2` = 327680 B/token | GQA formula above |
| Link rates | 100, 200, 400 Gbps | IEEE 100 / 200 / 400 GbE |
| Link efficiency | `0.8` | Assumed achievable fraction of line rate |
| Tokens used | `ContextTokens` only | Prefill, not decode |

## Example (first code-trace request)

`ContextTokens` = 2162

| Column | Value |
|---|---|
| `PrefillMs__H100PCIe__Llama3.1_8B` | 76.261023 ms |
| `PrefillMs__H100PCIe__Llama3.1_70B` | 667.283951 ms |
| `PrefillMs__H100SXM__Llama3.1_8B` | 58.294574 ms |
| `PrefillMs__H100SXM__Llama3.1_70B` | 510.077519 ms |
| `KVCacheMiB__Llama3.1_8B` | 270.25 MiB |
| `KVCacheMiB__Llama3.1_70B` | 675.625 MiB |
| `TransferMs__100Gbps__Llama3.1_8B` | 28.337766 ms |
| `TransferMs__100Gbps__Llama3.1_70B` | 70.844416 ms |
| `TransferMs__200Gbps__Llama3.1_8B` | 14.168883 ms |
| `TransferMs__200Gbps__Llama3.1_70B` | 35.422208 ms |
| `TransferMs__400Gbps__Llama3.1_8B` | 7.084442 ms |
| `TransferMs__400Gbps__Llama3.1_70B` | 17.711104 ms |

## References

- Azure LLM inference traces: [AzurePublicDataset `dataset-llm-2024`](https://github.com/Azure/AzurePublicDataset/releases/tag/dataset-llm-2024)
- Kaplan et al., *Scaling Laws for Neural Language Models*, 2020 — `2 * N` FLOPs per token for a forward pass
- Chowdhery et al., *PaLM: Scaling Language Modeling with Pathways*, 2022 — MFU
- [Meta Llama 3.1 model card](https://github.com/meta-llama/llama-models) — layers, GQA head counts, head dim
- [NVIDIA H100 Tensor Core GPU datasheet](https://www.nvidia.com/en-us/data-center/h100/) — 756 / 989 TFLOPS dense BF16
