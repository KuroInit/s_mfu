# Benchmark Guide

Operational guide for running the `s_mfu` sweep: GPU planning, adding models/datasets, interpreting outputs, and known gotchas.

## GPU memory planning

The current sweep targets **H100 NVL 94 GB** (same GH100 die as the 80 GB HBM3 part, but with 6 HBM3 stacks → 3.9 TB/s BW; FLOPS unchanged). Other Hopper/Ampere parts work but may need different chunk / batch caps.

Rough per-GPU budget for prefill:

```
free_kv  =  gpu_mem  −  weights/tp  −  activation_workspace(chunked_prefill_size)
```

If `free_kv ≤ 0`, SGLang auto-sets `mem_fraction_static < weight_footprint` and KV pool allocation fails at startup. That's what forced the 80B and 30B `maxctx` context shrinks on 2026-04-16:

| Model | TP | Weights/GPU | Fits at (input, bs) | Chunk cap |
|---|---|---|---|---|
| Qwen1.5-MoE-A2.7B-Chat | 1 | 28.6 GB | 24 K × bs=128 | 32 K |
| Qwen3-30B-A3B | 1 | ~60 GB | 4 K × bs=128 / 16 K × bs=16 | 32 K |
| Qwen3-Next-80B-A3B | 2 | 74.3 GB | 8 K × bs=32 | 16 K |

If a new model OOMs at startup, **halve the chunk cap first** (`chunk_size_cap` in `sweep_config.yaml`), then reduce `target_input_tokens`, then set `max_batch_size` in the dataset config.

## Adding a model to the sweep

1. Append an entry to `sweep_config.yaml`:
   ```yaml
   - id: org/new-moe-model
     tp: 1
     slug: new_moe
     chunk_size_cap: 32768
   ```
2. Create one dataset config per dataset you want to sweep for it (e.g. `configs/longbench_v2_new_moe.yaml` and `configs/longbench_v2_maxctx_new_moe.yaml`). Copy an existing file and adjust `model_id`, `target_input_tokens`, and `max_batch_size`.
3. Run a dry smoke at the smallest batch size: `batch_sizes: [1]` temporarily — watch SGLang startup log for `mem_fraction_static` and KV pool size.
4. Restore the full batch list once it fits.

**Known incompatibilities**
- AWQ-quantised MoEs (Mixtral-8x22B-Instruct-AWQ): SGLang's MoE architecture detector returns `None` → `AssertionError: ExpertLocationMetadata is required`. No workaround; drop from the sweep.
- DeepSeek-V2-Lite (MLA): upstream FLOPS calculation in MoE-CAP is wrong for MLA attention, so S-MFU is not comparable (Audit Finding #15). Dropped from Run 3.

## Adding a dataset

1. Add a loader in `MoE-CAP/moe_cap/data_loader/` that subclasses `DataLoader` and implements `get_input()`.
2. Register it in `loader_registry.py`: `_REGISTRY["my_dataset"] = (MyLoader, default_max_new_tokens)`.
3. Create `configs/my_dataset_<slug>.yaml` for each model.
4. Add `my_dataset` to `datasets:` in `sweep_config.yaml`.

## Running a single triple (debug loop)

The fastest way to iterate without running the full sweep is to invoke the runner against an already-running SGLang:

```bash
# terminal 1 — start SGLang manually
python -m moe_cap.systems.sglang \
    --model-path Qwen/Qwen3-30B-A3B \
    --port 30000 \
    --expert-distribution-recorder-mode stat \
    --tp-size 1 \
    --max-running-requests 1 \
    --enable-metrics \
    --chunked-prefill-size 32768

# terminal 2 — drive one config
python batch_runner.py \
    --config-file configs/longbench_v2_qwen3_30b.yaml \
    --api-url http://localhost:30000/v1/completions \
    --backend sglang \
    --server-batch-size 1 \
    --output_dir ./results/qwen3_30b/bs1/longbench_v2
```

Drop the `--enable-metrics` flag if you don't need the Tier-5 cross-check.

## Choosing the runner

| Situation | Runner |
|---|---|
| Per-batch S-MFU / S-MBU (Run 3 goal) | `BATCH_RUNNER=strict` |
| Peak throughput / real serving workload | default (asyncio overlap) |
| Continuity with prior Run 3 partial data | default |

The default runner has two client-side pathologies that break the contract "batch size = server concurrency":
- At `bs=1` it flood-fires all N prompts at once (`threshold = bs//2 = 0`).
- At `bs ≥ 2` the next wave is launched when 50 % of the current wave completes, so peak concurrency is ~1.5 × bs.

`batch_runner.py` sends N, awaits all N, then sends the next N — no overlap.

## Profiling-only mode

Set `MOE_CAP_PROFILING_ONLY=1` to skip the per-forward-pass `ExpertDistributionRecorder`. Runs faster and produces the same S-MFU / S-MBU numbers; you lose per-layer expert activations (which analyze.py currently doesn't use anyway).

## Interpreting the outputs

`analyze.py` writes one set of files per dataset:

- `smfu_<dataset>.png` — **S-MFU** is *achieved dense FLOPS / (peak dense FLOPS × num_gpus)*. MoE S-MFU differs from dense MFU because the denominator is the dense-equivalent compute, not the sparse-activated compute.
- `smbu_<dataset>.png` — **S-MBU** is achieved HBM bandwidth / peak HBM bandwidth. Sensitive to KV size and activation footprint.
- `raw_flops_<dataset>.png` — achieved prefill TFLOPS (the numerator of S-MFU, before normalisation).
- `tokens_per_sec_<dataset>.png` — raw throughput: `Σ seq_lens_sum / Σ latency` across prefill records.
- `raw_values.txt` — text dump of everything, easier to diff across runs.
- `qwen3_next_80b_legacy_{metric}_<dataset>.png` — current (Qwen3-Next path) vs legacy (Qwen3 path) for the 80B model only, since the two architectures share enough that the legacy math is a useful sanity line.

**Reading a smfu / smbu curve**
- Low S-MFU at high bs + low S-MBU → compute-bound but stuck on something else (launch overhead, kernel dispatch, expert routing imbalance).
- Low S-MFU at high bs + high S-MBU → memory-bound (KV eviction, HBM traffic dominates).
- Flat at low values across bs → single-forward overhead dominates; step up `target_input_tokens` or check that batch size is actually reaching the GPU.

## Tier-5 sanity warnings

`analyze.py` emits warnings when:
- `|client_tps − server_tps| / server_tps > 5 %` — client-side aggregation disagrees with SGLang's own counters. Usually means dataset records are being dropped or duplicated.
- `peak_running_reqs > batch_size + 1` — serial-wave contract broken. Either you're using the default runner or the server scheduled more requests than it should.
- `peak_cache_hit_rate > 0.05` — prefix cache is contaminating runs. Between-triple restart should prevent this; if it triggers, check the SGLang restart loop.

## Known findings

(See project notes / QMD knowledge base for the full audit.)

- **#15** MLA FLOPS calculation upstream is wrong → DeepSeek-V2-Lite excluded from Run 3.
- **#21** Continuous SGLang server across datasets leaks detailed_results / expert_records. Mitigation: per-triple restart (current behaviour).
- **#22** `batched_prefill` dataset scheduler underfills chunks → dropped from Run 3.
- **#24** `longbench_v2_maxctx` collapses to the same length as `longbench_v2` for `qwen1_5_moe` (24 K both) and `qwen3_next_80b` (8 K both); only `qwen3_30b` has a true 4 K vs 16 K comparison.

## Common failure modes

| Symptom | Cause | Fix |
|---|---|---|
| `AssertionError: ExpertLocationMetadata is required` | AWQ / unsupported MoE arch for expert recorder | Remove model from sweep, or set `MOE_CAP_PROFILING_ONLY=1` |
| `KeyError: '<GPU-name>'` in `hardware_utils.py` | New GPU not in dicts | Add entry to `MEM_BW_DICT` and `PEAK_FLOPS_DICT` |
| `GET /metrics → 404` loop | SGLang launched without `--enable-metrics` | Orchestrator adds it automatically; if debugging manually, pass the flag yourself |
| `AttributeError: ... has no attribute 'load'` in batch_runner | Loader API is `get_input()`, not `load()` | Already fixed; ensure you're running the latest `batch_runner.py` |
| OOM at SGLang startup, `mem_fraction_static` < weight fraction | Chunk workspace + weights exceed GPU | Lower `chunk_size_cap` or `target_input_tokens` |
| `analyze.py` reports "No valid results found" | `walk_results` didn't descend into `<org>/<model>` leaves | Already fixed; re-run after pulling latest |
