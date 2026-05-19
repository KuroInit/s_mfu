# s_mfu

Sweep harness for [MoE-CAP](https://github.com/Auto-CAP/MoE-CAP) benchmarks across multiple models, batch sizes, and datasets.

MoE-CAP evaluates sparse Mixture-of-Experts inference along **Cost**, **Accuracy**, and **Performance**. This repo provides the orchestration layer: it manages the SGLang server lifecycle, invokes MoE-CAP's runner for each `(model × batch_size × dataset)` triple, checkpoints progress, and plots MoE-CAP-derived S-MFU / S-MBU metrics.

## Prerequisites

- NVIDIA GPU(s) — the active sweep targets H100 NVL-class hosts. The sweep uses a conservative `gpu_memory_gb: 94` preflight guardrail for the 96 GB SKU, with TP=1 for the active models. See [`BENCHMARK_GUIDE.md`](BENCHMARK_GUIDE.md) for memory planning.
- A HuggingFace token with access to the models in `sweep_config.yaml`
- Either Docker with the NVIDIA container toolkit, or a local Python env (see below)

## Quick Start (local, no Docker)

```bash
export HF_TOKEN=hf_...
./run_sweep.sh
```

`run_sweep.sh` validates `HF_TOKEN`, clones/installs MoE-CAP (editable), installs `sglang[all]` and `matplotlib`, then runs `orchestrator.py` followed by `analyze.py`. Re-running the script resumes from the last checkpoint.

## Quick Start (Docker)

```bash
export HF_TOKEN=hf_...
export HF_CACHE_DIR=/path/to/hf_cache
export RESULTS_DIR=/path/to/results
docker compose up
python analyze.py "$RESULTS_DIR"
```

The Docker entrypoint runs the sweep. Run `analyze.py` afterward from the repo checkout to produce CSV and plots.

## Environment variables

**Required**
- `HF_TOKEN` — HuggingFace token (gated model access).

**Paths (optional, with defaults)**
- `HF_HOME` — HF cache dir (default: `~/.cache/huggingface`)
- `RESULTS_DIR` — output root (default: `./results`)
- `SWEEP_CONFIG` — sweep YAML (default: `sweep_config.yaml`)
- `CHECKPOINT_PATH` — resume state (default: `$RESULTS_DIR/checkpoint.yaml`)
- `SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR` — expert-activation dumps (default: `$RESULTS_DIR/expert_records`)

**Runtime tuning**
- `MOE_CAP_PROFILING_ONLY=1` — skip the per-forward-pass expert distribution recorder for faster runs (no expert-activation data).
- `SKIP_INSTALL=1` — skip pip installs in `run_sweep.sh`.
- `ANALYZE_ONLY=1` — skip the sweep and run `analyze.py` only.
- `DISABLE_RADIX_CACHE=0` — leave SGLang radix/prefix caching enabled for debugging. The default is disabled for S-MFU/S-MBU measurement.
- `AUTO_SELECT_GPUS=0` — disable automatic idle-GPU selection and leave `CUDA_VISIBLE_DEVICES` unchanged.
- `GPU_FREE_MEMORY_USED_MB` — memory-used threshold for an idle GPU (default: `1024`).
- `GPU_RETRY_INTERVAL_SECONDS` — wait between idle-GPU checks (default: `15`).
- `GPU_MAX_IDLE_CHECKS` — retry budget before a cell is skipped without checkpointing (default: `3`).
- `ANALYZE_GPU_TYPE` — override GPU type for old result files whose records say `"Unknown"`; for example `NVIDIA-H100-NVL-96GB`.

## `sweep_config.yaml`

Current batched-prefill scope:

```yaml
batch_sizes: [2, 4, 8, 16, 32, 64, 128, 256]
datasets: [batched_prefill]
port: 30000
gpu_memory_gb: 94

models:
  - id: Qwen/Qwen1.5-MoE-A2.7B-Chat
    tp: 1
    slug: qwen1_5_moe
    max_context_tokens: 32768
    weight_gb_per_gpu: 28.6
    kv_bytes_per_token_per_gpu: 12288
  - id: Qwen/Qwen3-30B-A3B
    tp: 1
    slug: qwen3_30b
    max_context_tokens: 40960
    weight_gb_per_gpu: 60.0
    kv_bytes_per_token_per_gpu: 98304
  - id: deepseek-ai/DeepSeek-V2-Lite-Chat
    tp: 1
    slug: deepseek_v2_lite
    max_context_tokens: 163840
    weight_gb_per_gpu: 31.4
    kv_bytes_per_token_per_gpu: 31744
  - id: deepseek-ai/deepseek-moe-16b-chat
    tp: 1
    slug: deepseek_moe_16b_chat
    max_context_tokens: 4096
    weight_gb_per_gpu: 32.0
    kv_bytes_per_token_per_gpu: 65536
```

- `tp` is set by necessity only — TP=1 unless the model won't fit on one GPU.
- `gpu_memory_gb`, `max_context_tokens`, `weight_gb_per_gpu`, and `kv_bytes_per_token_per_gpu` are harness preflight guardrails. They make impossible batch cells fail before SGLang loads weights.
- Some models are capped below the global sweep max with `max_batch_size`
  in their active batched-prefill configs.
- Optional `chunked_prefill_size`, `max_prefill_tokens`, and `mem_fraction_static` can be set per model or dataset when an operator needs explicit SGLang scheduling overrides.
- SGLang radix/prefix caching is disabled by default with `--disable-radix-cache`; set `DISABLE_RADIX_CACHE=0` only for debugging cached-prefix behavior, not for S-MFU/S-MBU measurements.

The active sweep includes Qwen MoE, DeepSeek-V2-Lite, and the original
DeepSeekMoE 16B chat model.

## `configs/`

Each active `<dataset>_<slug>.yaml` is a MoE-CAP benchmark config passed to the runner.

Active batched-prefill configs use fixed-length prompts and `target_output_tokens: 1`, so the run stays focused on packed prefill while MoE-CAP can still record decode steps when present.

| Config | Input tokens | Chunking / caps |
| --- | ---: | --- |
| `batched_prefill_qwen1_5_moe.yaml` | 2048 | `chunked_prefill_size: 131072`, `max_prefill_tokens: 131072`, `mem_fraction_static: 0.9` |
| `batched_prefill_qwen3_30b.yaml` | 2048 | `chunked_prefill_size: 16384`, `max_batch_size: 128` |
| `batched_prefill_deepseek_v2_lite.yaml` | 1024 | `chunked_prefill_size: 32768` |
| `batched_prefill_deepseek_moe_16b_chat.yaml` | 2048 | `chunked_prefill_size: 32768`, `max_batch_size: 128` |

`batched_prefill_qwen3_next_80b.yaml` is kept for optional TP=2 experiments and is paired with the commented Qwen3-Next model entry in `sweep_config.yaml`.

## How it works

For each `(slug, batch_size, dataset)` triple not yet marked success in the checkpoint:

1. **Pre-flight** — validates all MoE-CAP configs, memory guardrails, and HF model IDs up front.
2. **Pick GPUs** — by default, uses `nvidia-smi` to bind each SGLang server to enough low-memory physical GPUs. Non-numeric `CUDA_VISIBLE_DEVICES` values are treated as unmanaged and left alone.
3. **Start SGLang** — launches `moe_cap.systems.sglang` with `--enable-metrics`, `--enable-expert-distribution-metrics`, and `--disable-radix-cache`; polls `/health` until ready (25 min cap). `--chunked-prefill-size`, `--max-prefill-tokens`, and `--mem-fraction-static` are passed only when explicitly configured.
4. **Run benchmark** — invokes `moe_cap.runner.openai_api_profile` with the configured `--server-batch-size`.
5. **Preserve server records** — copies MoE-CAP/SGLang expert-distribution records next to the result files so analysis can use MoE-CAP's continuous-batching metric function accurately.
6. **Checkpoint and failure artifacts** — successful cells are checkpointed. Runner, startup/OOM, and missing-server-record failures are written as `failure_<dataset>_<ts>.json` artifacts and marked failed so they can be retried later.
7. **Shutdown** — SIGTERM -> wait -> SIGKILL; wait for the port to clear before the next triple.

SGLang is restarted per triple so detailed_results and expert_records can't bleed across datasets (Audit Finding #21).

## Results layout

```
$RESULTS_DIR/
├── checkpoint.yaml
├── expert_records/<model_id>/expert_distribution_record.jsonl
└── <slug>/bs<N>/<dataset>/
    └── <org>/<model_name>/
        ├── metadata_<dataset>_<ts>.json
        ├── metrics_<dataset>_<ts>.json
        ├── detailed_results_<dataset>_<ts>.jsonl
        ├── server_records_<dataset>_<ts>.jsonl
        └── failure_<dataset>_<ts>.json
```

`failure_*.json` appears only for cells that fail before valid MoE-CAP metrics and server records are available.

## Analysis

```bash
python analyze.py $RESULTS_DIR
```

Loads every leaf directory, re-derives S-MFU / S-MBU / tokens-per-sec using `moe_cap.utils.continuous_batching_utils._calculate_continuous_metrics`, and reconstructs raw TFLOPS from MoE-CAP S-MFU and MoE-CAP peak FLOPS.

The vendored MoE-CAP checkout includes a local fix for packed prefill records:
when SGLang provides `per_req_info`, prefill throughput is computed from the
packed forward record's `seq_lens_sum / latency` instead of averaging
per-request token counts over the shared forward latency. Keep MoE-CAP installed
editable from this checkout (`pip install -e ./MoE-CAP`) before analyzing runs.

If old result files recorded `"Unknown"` as the record-level GPU type, `analyze.py` falls back to metadata and also accepts `ANALYZE_GPU_TYPE=NVIDIA-H100-NVL-96GB` as an explicit override.

Outputs (to `$RESULTS_DIR/`):
- `raw_values.csv` — CSV dump of every MoE-CAP-derived metric per `(dataset, slug, bs)`, including run status, selected GPUs, scheduling overrides, decoding metrics, and failure reasons when present.
- `smfu_<dataset>.png`, `smbu_<dataset>.png`, `raw_flops_<dataset>.png`, `tokens_per_sec_<dataset>.png` — combined prefill figures per dataset with one line per model.
- `decoding_smfu_<dataset>.png`, `decoding_smbu_<dataset>.png`, `decoding_raw_flops_<dataset>.png`, `decoding_tokens_per_sec_<dataset>.png` — combined decoding figures when decoding metrics are present.
- `<slug>_smfu_smbu_<dataset>.png` — per-model prefill S-MFU/S-MBU figure.
- `qwen3_next_80b_legacy_{smfu,smbu,raw_flops}_<dataset>.png` — dedicated current-path vs legacy-Qwen3-path comparison for the optional 80B model.

On the combined plots a dotted line parallel to the `qwen3_next_80b` line shows the Qwen3 (legacy) path as a sanity reference.

## Tests

```bash
pip install pytest
pytest tests/
```

Covers checkpoint persistence, SGLang lifecycle (start/health/kill), idle-GPU selection, pre-flight model/config validation, runner invocation, failure artifacts, sweep loop behavior, and analyze.py aggregation — all via mocks, no GPU required.
