# s_mfu

Sweep harness for [MoE-CAP](https://github.com/Auto-CAP/MoE-CAP) benchmarks across multiple models, batch sizes, and datasets.

MoE-CAP evaluates sparse Mixture-of-Experts inference along **Cost**, **Accuracy**, and **Performance**. This repo provides the orchestration layer: it manages the SGLang server lifecycle, drives a benchmark runner for each `(model × batch_size × dataset)` triple, scrapes server-side metrics as ground truth, checkpoints progress, and produces S-MFU / S-MBU plots.

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
- `BATCH_RUNNER=strict` — use the strict-serial runner in `batch_runner.py` instead of MoE-CAP's default asyncio runner (see **Runners** below).
- `BATCH_RUNNER_REQUEST_TIMEOUT` — per-request timeout in seconds (default: 3600).
- `METRICS_POLL_INTERVAL` — Tier-5 `/metrics` poll period in seconds (default: 1.0; `0` disables).
- `MOE_CAP_PROFILING_ONLY=1` — skip the per-forward-pass expert distribution recorder for faster runs (no expert-activation data).
- `SKIP_INSTALL=1` — skip pip installs in `run_sweep.sh`.
- `ANALYZE_ONLY=1` — skip the sweep and run `analyze.py` only.

## `sweep_config.yaml`

Current Run 3 scope:

```yaml
batch_sizes: [1, 2, 4, 8, 16, 32]
datasets: [longbench_v2, longbench_v2_maxctx]
port: 30000

models:
  - id: Qwen/Qwen1.5-MoE-A2.7B-Chat
    tp: 1
    slug: qwen1_5_moe
    chunk_size_cap: 32768
  - id: Qwen/Qwen3-30B-A3B
    tp: 1
    slug: qwen3_30b
    chunk_size_cap: 32768
  - id: Qwen/Qwen3-Next-80B-A3B-Instruct
    tp: 2
    slug: qwen3_next_80b
    chunk_size_cap: 16384
```

- `tp` is set by necessity only — TP=1 unless the model won't fit on one GPU.
- `chunk_size_cap` caps `--chunked-prefill-size` per model; tight caps (e.g. 16 K on the 80B) claw back KV room.

**Dropped in Run 3:** DeepSeek-V2-Lite (MLA FLOPS upstream bug), Mixtral-8x22B-AWQ (no `ExpertLocationMetadata`). Datasets `prefill` and `batched_prefill` also removed — the former duplicates `maxctx`, the latter suffers carryover + scheduler under-fill contamination (Audit Findings #21/#22).

## `configs/`

Each `<dataset>_<slug>.yaml` is a MoE-CAP benchmark config passed to the runner. They set `fixed_length_mode: true`, `target_output_tokens: 1`, and `target_input_tokens` per model to hold context length roughly constant across batch sizes. Some configs also set `max_batch_size` to cap the sweep for that model × dataset.

## How it works

For each `(slug, batch_size, dataset)` triple not yet marked success in the checkpoint:

1. **Pre-flight** — validates all HF model IDs up front (`orchestrator.py:validate_models`).
2. **Start SGLang** — launches `moe_cap.systems.sglang` with `--enable-metrics` and per-model `--chunked-prefill-size`; polls `/health` until ready (25 min cap).
3. **Start Tier-5 poller** — `sglang_metrics.py` scrapes `/metrics` every `METRICS_POLL_INTERVAL` seconds and writes `sglang_metrics_bs<N>.jsonl` next to the results.
4. **Run benchmark** — either `moe_cap.runner.openai_api_profile` (default) or `batch_runner.py` (strict).
5. **Checkpoint** — success or failure written to `$CHECKPOINT_PATH`.
6. **Shutdown** — SIGTERM → wait → SIGKILL; wait for the port to clear before the next triple.

SGLang is restarted per triple so detailed_results and expert_records can't bleed across datasets (Audit Finding #21).

## Runners

**Default** (`moe_cap.runner.openai_api_profile`): MoE-CAP's upstream asyncio runner. Two known client-side pathologies:
- at `bs=1`, all N prompts are fired concurrently (threshold `bs // 2 == 0`);
- at `bs ≥ 2`, the next wave launches when half the current wave completes, so peak concurrency is ~1.5 × bs.

**Strict** (`batch_runner.py`, enabled with `BATCH_RUNNER=strict`): sends N, awaits all N, then sends the next N. No overlap. Reuses MoE-CAP's loaders so prompts are identical, pulls per-forward-pass records via `/dump_expert_distribution_record`, and writes the same `detailed_results_*.jsonl` + `metadata_*.json` schema.

Use strict when you want batch size to equal server concurrency (e.g. per-batch S-MFU / S-MBU). Use default for continuity with prior runs.

## Results layout

```
$RESULTS_DIR/
├── checkpoint.yaml
├── expert_records/<model_id>/expert_distribution_record.jsonl
└── <slug>/bs<N>/<dataset>/
    ├── sglang_metrics_bs<N>.jsonl          # Tier-5 /metrics snapshots
    └── <org>/<model_name>/
        ├── metadata_<dataset>_<ts>.json
        └── detailed_results_<dataset>_<ts>.jsonl
```

## Analysis

```bash
python analyze.py $RESULTS_DIR
```

Loads every leaf directory, re-derives per-prefill S-MFU / S-MBU / raw TFLOPS / tokens-per-sec using `moe_cap.utils.continuous_batching_utils._calculate_continuous_metrics`, and cross-checks against Tier-5 server counters (`prompt_tokens_total`, `num_running_reqs`, `cache_hit_rate`) — warning if client/server throughputs diverge > 5 %, if `peak_running_reqs > batch_size + 1` (serial-wave contract broken), or if the prefix cache shows contamination.

Outputs (to `$RESULTS_DIR/`):
- `raw_values.txt` — plaintext dump of every computed metric per `(slug, bs, dataset)`.
- `smfu_<dataset>.png`, `smbu_<dataset>.png`, `raw_flops_<dataset>.png`, `tokens_per_sec_<dataset>.png` — one combined figure per dataset with one line per model.
- `qwen3_next_80b_legacy_{smfu,smbu,raw_flops}_<dataset>.png` — dedicated current-path vs legacy-Qwen3-path comparison for the 80B model.

On the combined plots a dotted line parallel to the `qwen3_next_80b` line shows the Qwen3 (legacy) path as a sanity reference.

## Tests

```bash
pip install pytest
pytest tests/
```

Covers checkpoint persistence, SGLang lifecycle (start/health/kill), pre-flight model validation, runner invocation, sweep loop behavior, and analyze.py aggregation — all via mocks, no GPU required.
