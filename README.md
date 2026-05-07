# s_mfu

Sweep harness for [MoE-CAP](https://github.com/Auto-CAP/MoE-CAP) benchmarks across multiple models, batch sizes, and datasets.

MoE-CAP evaluates sparse Mixture-of-Experts inference along **Cost**, **Accuracy**, and **Performance**. This repo provides the orchestration layer: it manages the SGLang server lifecycle, invokes MoE-CAP's runner for each `(model × batch_size × dataset)` triple, checkpoints progress, and plots MoE-CAP-derived S-MFU / S-MBU metrics.

## Prerequisites

- NVIDIA GPU(s) — the current sweep targets H100 NVL 94 GB (single-GPU for 30B-class, TP=2 for 80B). See [`BENCHMARK_GUIDE.md`](BENCHMARK_GUIDE.md) for memory planning.
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
```

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

## `sweep_config.yaml`

Current Qwen1.5-MoE batched-prefill scope:

```yaml
batch_sizes: [2, 4, 8, 16, 32, 64, 128]
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
```

- `tp` is set by necessity only — TP=1 unless the model won't fit on one GPU.
- `gpu_memory_gb`, `max_context_tokens`, `weight_gb_per_gpu`, and `kv_bytes_per_token_per_gpu` are harness preflight guardrails. They make impossible batch cells fail before SGLang loads weights.
- Optional `chunked_prefill_size` and `max_prefill_tokens` can be set per model or dataset when an operator needs explicit SGLang scheduling overrides.
- SGLang radix/prefix caching is disabled by default with `--disable-radix-cache`; set `DISABLE_RADIX_CACHE=0` only for debugging cached-prefix behavior, not for S-MFU/S-MBU measurements.

**Dropped in this sweep:** DeepSeek and Qwen3 variants. The active sweep is intentionally Qwen1.5-MoE only, using LongBench V2 prompts fixed to 1K input tokens and 1 output token.

## `configs/`

Each active `<dataset>_<slug>.yaml` is a MoE-CAP benchmark config passed to the runner. The current active config is `configs/batched_prefill_qwen1_5_moe.yaml`: `target_input_tokens: 1024`, `target_output_tokens: 1`, `chunked_prefill_size: 32768`, and `max_prefill_tokens: 32768`.

## How it works

For each `(slug, batch_size, dataset)` triple not yet marked success in the checkpoint:

1. **Pre-flight** — validates all HF model IDs up front (`orchestrator.py:validate_models`).
2. **Start SGLang** — launches `moe_cap.systems.sglang` with `--enable-metrics` and `--disable-radix-cache`; polls `/health` until ready (25 min cap). `--chunked-prefill-size` and `--max-prefill-tokens` are passed only when explicitly configured.
3. **Run benchmark** — invokes `moe_cap.runner.openai_api_profile` with the configured `--server-batch-size`.
4. **Preserve server records** — copies MoE-CAP/SGLang expert-distribution records next to the result files so analysis can use MoE-CAP's continuous-batching metric function accurately.
5. **Checkpoint** — success or failure written to `$CHECKPOINT_PATH`.
6. **Shutdown** — SIGTERM → wait → SIGKILL; wait for the port to clear before the next triple.

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
        └── server_records_<dataset>_<ts>.jsonl
```

## Analysis

```bash
python analyze.py $RESULTS_DIR
```

Loads every leaf directory, re-derives S-MFU / S-MBU / tokens-per-sec using `moe_cap.utils.continuous_batching_utils._calculate_continuous_metrics`, and reconstructs raw TFLOPS from MoE-CAP S-MFU and MoE-CAP peak FLOPS.

If old result files recorded `"Unknown"` as the record-level GPU type, `analyze.py` falls back to metadata and also accepts `ANALYZE_GPU_TYPE=NVIDIA-H100-NVL-94GB` as an explicit override.

Outputs (to `$RESULTS_DIR/`):
- `raw_values.csv` — CSV dump of every MoE-CAP-derived metric per `(dataset, slug, bs)`.
- `smfu_<dataset>.png`, `smbu_<dataset>.png`, `raw_flops_<dataset>.png`, `tokens_per_sec_<dataset>.png` — one combined figure per dataset with one line per model.
- `qwen3_next_80b_legacy_{smfu,smbu,raw_flops}_<dataset>.png` — dedicated current-path vs legacy-Qwen3-path comparison for the 80B model.

On the combined plots a dotted line parallel to the `qwen3_next_80b` line shows the Qwen3 (legacy) path as a sanity reference.

## Tests

```bash
pip install pytest
pytest tests/
```

Covers checkpoint persistence, SGLang lifecycle (start/health/kill), pre-flight model validation, runner invocation, sweep loop behavior, and analyze.py aggregation — all via mocks, no GPU required.
