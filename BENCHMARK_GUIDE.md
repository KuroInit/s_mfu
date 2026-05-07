# Benchmark Guide

This harness is intentionally thin: it starts SGLang through MoE-CAP, invokes
MoE-CAP's `openai_api_profile` runner for each configured sweep cell, preserves
the server records MoE-CAP needs for continuous-batching metrics, and plots
MoE-CAP-derived values.

## Runner Contract

- `orchestrator.py` always uses `python -m moe_cap.runner.openai_api_profile`.
- The sweep `batch_size` is passed to MoE-CAP as `--server-batch-size`.
- The deleted harness-owned strict runner is no longer supported.
- SGLang is restarted for every `(model, batch_size, dataset)` cell so expert
  records do not bleed between runs.

## Analysis Contract

`analyze.py` computes S-MFU, S-MBU, throughput, TTFT, TPOT, and raw TFLOPS via
MoE-CAP's `moe_cap.utils.continuous_batching_utils._calculate_continuous_metrics`.
The harness only adds CSV/plot formatting and a small amount of run metadata.

Outputs:

- `raw_values.csv`
- `smfu_<dataset>.png`
- `smbu_<dataset>.png`
- `raw_flops_<dataset>.png`
- `tokens_per_sec_<dataset>.png`
- matching decoding plots when decoding metrics are present

## Memory Guardrails

The optional `gpu_memory_gb`, `max_context_tokens`, `weight_gb_per_gpu`, and
`kv_bytes_per_token_per_gpu` sweep fields are preflight guardrails only. They do
not participate in MoE-CAP metric math.

If a run OOMs, lower `max_batch_size` or `target_input_tokens`. Use
`chunked_prefill_size`, `max_prefill_tokens`, and `mem_fraction_static` only as
explicit SGLang scheduling overrides.
