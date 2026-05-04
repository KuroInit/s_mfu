# Test Flow

This document describes the full benchmark test from setup through analysis. The goal is to measure prefill S-MFU, S-MBU, achieved prefill TFLOPS, and prefill tokens/sec across a sweep of batch sizes, using fixed-length prompts and exactly one generated token.

## Measurement Contract

Every valid run must satisfy these conditions:

1. The model is supported by MoE-CAP's testing framework.
2. The server runs on H100 NVL hardware, with H100 NVL present in MoE-CAP's hardware lookup tables.
3. The prompt length is fixed near 32K tokens.
4. `target_output_tokens` is exactly `1`.
5. SGLang prefix/radix cache is disabled.
6. The benchmark runs each configured `(model, batch_size, dataset)` triple independently.
7. Any value MoE-CAP can provide is treated as the ground truth and is taken from MoE-CAP.
8. Values MoE-CAP does not provide directly are computed in `analyze.py` from MoE-CAP records, MoE-CAP hardware lookup values, or SGLang `/metrics`.
9. This harness only orchestrates repeated measurement, preflight validation, server lifecycle, checkpointing, Tier-5 sanity checks, missing-value aggregation, and plotting.

For Qwen1.5-MoE, the active input length is `32,767`, not `32,768`, because the model context window is exactly `32,768` total tokens and one token is reserved for generation.

## Step 1: Environment Setup

The user starts the benchmark with:

```bash
export HF_TOKEN=hf_...
./run_sweep.sh
```

`run_sweep.sh` performs these setup actions:

1. Verifies `HF_TOKEN` is set.
2. Sets default paths:
   - `HF_HOME`, defaulting to `~/.cache/huggingface`
   - `RESULTS_DIR`, defaulting to `./results`
   - `SWEEP_CONFIG`, defaulting to `sweep_config.yaml`
   - `CHECKPOINT_PATH`, defaulting to `$RESULTS_DIR/checkpoint.yaml`
   - `SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR`, defaulting to `$RESULTS_DIR/expert_records`
3. Creates the result and expert-record directories.
4. Unless `SKIP_INSTALL=1`, installs:
   - MoE-CAP in editable mode
   - `sglang[all]`
   - `matplotlib`
5. Runs `python orchestrator.py`.
6. Runs `python analyze.py "$RESULTS_DIR"`.

## Step 2: Sweep Configuration Is Loaded

`orchestrator.py` reads `sweep_config.yaml`.

The active sweep currently contains:

```yaml
batch_sizes: [1, 2, 4, 8, 16, 32]
datasets: [longbench_v2]
gpu_memory_gb: 94
```

Each model entry supplies:

- `id`: HuggingFace model ID.
- `slug`: result directory name.
- `tp`: tensor-parallel size.
- `max_context_tokens`: context-window guardrail.
- `weight_gb_per_gpu`: estimated per-GPU weight footprint.
- `kv_bytes_per_token_per_gpu`: estimated per-token KV footprint per GPU.

These resource values are preflight guardrails only. They do not replace MoE-CAP metric calculations.

## Ground Truth Policy

The project policy is:

```text
Use MoE-CAP values wherever MoE-CAP provides them.
Compute only the missing experiment-level values in s_mfu/analyze.py.
```

MoE-CAP is treated as the ground truth because this repository is an experimental harness around MoE-CAP, not a replacement for MoE-CAP's metric implementation.

Values taken from MoE-CAP or MoE-CAP outputs:

1. `prefill_smfu`.
2. `prefill_smbu`.
3. `prefill_tokens_per_sec`, taken from MoE-CAP's `prefill_tp`.
4. Model architecture metadata.
5. MoE metadata.
6. Attention metadata.
7. Precision metadata.
8. GPU peak FLOPS and bandwidth lookup.
9. Per-request and per-forward records such as `seq_lens_sum`, `ttft`, `tpot`, `forward_mode`, `gpu_raw_type`, and hardware metadata.

Values computed by `s_mfu/analyze.py` because MoE-CAP does not provide them in the needed experiment-level form:

1. `prefill_tokens_per_sec_aggregate = sum(seq_lens_sum) / sum(prefill_latency)`, used as a harness cross-check.
2. `prefill_total_tokens = sum(seq_lens_sum)`.
3. `prefill_total_latency = sum(prefill_latency)`.
4. `prefill_raw_tflops`, reconstructed from MoE-CAP S-MFU and MoE-CAP peak FLOPS.
5. Averages across result leaves.
6. Plot-ready series grouped by model, batch size, and dataset.

Values taken from SGLang `/metrics`, not MoE-CAP:

1. `server_tokens_per_sec`.
2. `peak_running_reqs`.
3. `peak_cache_hit_rate`.
4. `client_vs_server_delta_pct`.

These SGLang values are sanity checks. They do not replace MoE-CAP's S-MFU or S-MBU values.

## Step 3: Dataset Configs Are Validated

For each active model and dataset, the harness expects a config file named:

```text
configs/<dataset>_<slug>.yaml
```

For example:

```text
configs/longbench_v2_qwen3_30b.yaml
```

Each active config must satisfy:

1. `model_id` matches the model ID in `sweep_config.yaml`.
2. `fixed_length_mode: true`.
3. `target_input_tokens` is set.
4. `target_output_tokens: 1`.
5. `target_input_tokens + target_output_tokens <= max_context_tokens`.
6. The estimated batch memory fits within `gpu_memory_gb`.

If a configured batch size exceeds the dataset config's `max_batch_size`, that batch is skipped before SGLang starts.

## Step 4: Model IDs Are Validated

Before the sweep begins, `orchestrator.py` checks each active HuggingFace model ID with `huggingface_hub.model_info`.

If any model is missing or inaccessible, the sweep exits before loading SGLang.

## Step 5: Checkpoint State Is Read

The harness reads:

```text
$CHECKPOINT_PATH
```

Each completed triple is keyed by:

```text
(slug, batch_size, dataset)
```

If a triple is already marked with `status: success`, it is skipped. Failed triples are eligible to run again.

## Step 6: The Harness Iterates Through Triples

For every active combination of:

```text
model slug x batch size x dataset
```

the harness performs one independent run. A run corresponds to exactly one result leaf such as:

```text
$RESULTS_DIR/qwen3_30b/bs4/longbench_v2/
```

SGLang is restarted for every triple so results and expert records do not bleed across batch sizes or models.

## Step 7: SGLang Is Launched

For each triple, `orchestrator.py` starts:

```bash
python -m moe_cap.systems.sglang \
  --model-path <model_id> \
  --port <port> \
  --expert-distribution-recorder-mode stat \
  --tp-size <tp> \
  --max-running-requests <batch_size> \
  --enable-metrics \
  --disable-radix-cache
```

Important details:

1. `--max-running-requests` is set to the current batch size.
2. `--enable-metrics` exposes SGLang `/metrics` for Tier-5 cross-checks.
3. `--disable-radix-cache` prevents prefix-cache contamination.
4. `--chunked-prefill-size` is passed only if explicitly configured.

`DISABLE_RADIX_CACHE=0` removes `--disable-radix-cache`, but that is only for debugging cached-prefix behavior. It should not be used for valid S-MFU or S-MBU measurements.

## Step 8: The Server Health Check Runs

After launching SGLang, the harness polls:

```text
http://localhost:<port>/health
```

It waits until the server returns HTTP 200 or the startup timeout expires. The default startup timeout is 25 minutes to allow large models to load and warm up.

If health never succeeds, the triple is marked failed and the harness moves on after shutdown cleanup.

## Step 9: Tier-5 Metrics Polling Starts

Once SGLang is healthy, `sglang_metrics.py` starts polling:

```text
http://localhost:<port>/metrics
```

The poll interval is controlled by:

```text
METRICS_POLL_INTERVAL
```

The default is `1.0` second. Set `METRICS_POLL_INTERVAL=0` to disable Tier-5 polling.

The poller writes snapshots into the triple's output directory:

```text
sglang_metrics_bs<N>.jsonl
```

These snapshots are used later to cross-check:

- server-side prompt tokens/sec
- peak running requests
- prefix-cache hit rate

## Step 10: MoE-CAP Runner Sends Requests

By default, the harness invokes MoE-CAP's upstream runner:

```bash
python -m moe_cap.runner.openai_api_profile \
  --config-file configs/<dataset>_<slug>.yaml \
  --api-url http://localhost:<port>/v1/completions \
  --backend sglang \
  --server-batch-size <batch_size> \
  --output_dir <triple_output_dir>
```

The runner uses the dataset config to produce fixed-length prompts:

- Qwen3 models: `32,768` input tokens and `1` output token.
- Qwen1.5-MoE: `32,767` input tokens and `1` output token.

MoE-CAP records per-request and per-forward-pass data in its normal output schema.

## Step 11: Optional Strict Runner Mode

If:

```bash
BATCH_RUNNER=strict
```

the harness uses `batch_runner.py` instead of the upstream MoE-CAP runner.

Strict mode sends exactly `N` requests, waits for those `N` requests to finish, then sends the next `N`. It is useful for debugging exact client request waves. The default measurement path remains the MoE-CAP runner.

## Step 12: Result Files Are Written

Each triple produces files under:

```text
$RESULTS_DIR/<slug>/bs<N>/<dataset>/
```

The expected layout is:

```text
$RESULTS_DIR/
├── checkpoint.yaml
├── expert_records/<model_id>/expert_distribution_record.jsonl
└── <slug>/bs<N>/<dataset>/
    ├── sglang_metrics_bs<N>.jsonl
    └── <org>/<model_name>/
        ├── metadata_<dataset>_<timestamp>.json
        └── detailed_results_<dataset>_<timestamp>.jsonl
```

The metadata and detailed-results files are produced by MoE-CAP. The `sglang_metrics_bs<N>.jsonl` file is produced by this harness.

## Step 13: The Triple Is Checkpointed

After the runner exits:

1. The metrics poller stops.
2. The triple is marked `success` if the runner succeeded.
3. The triple is marked `failed` if the runner failed or timed out.
4. The checkpoint file is written immediately.

This allows interrupted sweeps to resume without re-running successful triples.

## Step 14: SGLang Is Shut Down

The harness shuts down SGLang after every triple:

1. Sends SIGTERM.
2. Waits for the configured grace period.
3. Sends SIGKILL if the process does not exit.
4. Waits until the TCP port is free before the next triple starts.

This per-triple restart is required so detailed results and expert records remain isolated.

## Step 15: Analysis Loads Result Leaves

After the sweep, `run_sweep.sh` calls:

```bash
python analyze.py "$RESULTS_DIR"
```

`analyze.py` walks every result leaf, finds the newest:

- `metadata_*.json`
- `detailed_results_*.jsonl`

It normalizes MoE-CAP records so prefill records use `ttft` as latency and decoding records use `tpot` as latency.

## Step 16: GPU Type Is Resolved

For each result leaf, `analyze.py` resolves the GPU type used by MoE-CAP's hardware lookup.

Resolution order:

1. `ANALYZE_GPU_TYPE`, if set.
2. Record-level `gpu_raw_type`, if it is not `"Unknown"`.
3. Metadata-level `hardware.gpu_type`, if it is not `"Unknown"`.
4. The first available unknown value, which will usually be skipped by MoE-CAP.

For older result folders with bad GPU metadata, run:

```bash
ANALYZE_GPU_TYPE=NVIDIA-H100-NVL-94GB python analyze.py "$RESULTS_DIR"
```

This only fixes hardware lookup. It does not make cache-contaminated runs valid.

## Step 17: Missing Experiment-Level Values Are Aggregated

For each result leaf, `analyze.py` computes values that MoE-CAP does not provide directly in the required experiment-level form.

The harness aggregate throughput cross-check is:

```text
prefill_tokens_per_sec_aggregate = sum(seq_lens_sum) / sum(prefill_latency)
```

This is a raw aggregate across MoE-CAP prefill records. The input values come from MoE-CAP output files; only the aggregation is done by `analyze.py`. The primary reported `prefill_tokens_per_sec` value is still taken from MoE-CAP's `prefill_tp`.

## Step 18: MoE-CAP Computes S-MFU, S-MBU, and FLOPS

`analyze.py` calls MoE-CAP's:

```python
moe_cap.utils.continuous_batching_utils._calculate_continuous_metrics
```

MoE-CAP receives:

- model architecture data
- MoE metadata
- attention metadata
- GPU type
- precision
- tensor-parallel GPU count
- normalized output records

MoE-CAP returns the S-MFU, S-MBU, and `prefill_tp` values. These are the authoritative S-MFU, S-MBU, and prefill tokens/sec values for this project.

`prefill_raw_tflops` is reconstructed from:

```text
prefill_smfu * num_gpus * peak_dense_tflops_per_gpu
```

This reconstruction is done in `analyze.py`, but both `prefill_smfu` and the GPU peak FLOPS lookup come from MoE-CAP.

## Step 19: Tier-5 Sanity Checks Run

If `sglang_metrics_bs<N>.jsonl` exists, `analyze.py` cross-checks the client-side result against server-side SGLang counters.

It warns when:

1. Client tokens/sec and server prompt tokens/sec differ by more than 5%.
2. `peak_running_reqs > batch_size + 1`.
3. `peak_cache_hit_rate > 0.05`.

A valid run should have:

```text
peak_cache_hit_rate ~= 0
client/server tokens/sec divergence <= 5%
prefill S-MFU <= 100%
```

If S-MFU exceeds 100%, or cache hit rate is high, discard the run and rerun with radix cache disabled and a recognized GPU type.

## Step 20: Outputs Are Written

`analyze.py` writes:

```text
$RESULTS_DIR/raw_values.txt
$RESULTS_DIR/smfu_<dataset>.png
$RESULTS_DIR/smbu_<dataset>.png
$RESULTS_DIR/raw_flops_<dataset>.png
$RESULTS_DIR/tokens_per_sec_<dataset>.png
```

For Qwen3-Next, it also writes legacy-comparison plots:

```text
$RESULTS_DIR/qwen3_next_80b_legacy_smfu_<dataset>.png
$RESULTS_DIR/qwen3_next_80b_legacy_smbu_<dataset>.png
$RESULTS_DIR/qwen3_next_80b_legacy_raw_flops_<dataset>.png
```

## Step 21: Unit Tests Validate Harness Behavior

The local unit test suite runs with:

```bash
.venv/bin/python -m pytest -q
```

The tests validate:

1. Checkpoint persistence and resume behavior.
2. SGLang launch command construction.
3. Health-check and shutdown logic.
4. Config and resource preflight checks.
5. Runner invocation.
6. Result walking and analysis aggregation.
7. H100 NVL hardware glue.
8. GPU fallback and override behavior in `analyze.py`.

The unit tests do not require GPUs and do not measure real S-MFU. Real S-MFU/S-MBU validity comes from running the full sweep on the target H100 NVL server and passing the Tier-5 sanity checks.
