# S-MFU / S-MBU Calculation Chain and Local MoE-CAP Changes

This note documents how this harness computes S-MFU and S-MBU from runner
records, and what local changes exist in the editable `MoE-CAP` checkout used
by `.venv`.

## Environment

The repo-local virtual environment uses the editable MoE-CAP checkout:

```text
Python: /home/kur0/Documents/s_mfu/.venv/bin/python
MoE-CAP source: /home/kur0/Documents/s_mfu/MoE-CAP
continuous_batching_utils:
  /home/kur0/Documents/s_mfu/MoE-CAP/moe_cap/utils/continuous_batching_utils.py
```

The default shell Python did not have `moe_cap`, but `.venv` does.

## Harness Flow

The harness runs MoE-CAP's OpenAI API runner:

```text
python -m moe_cap.runner.openai_api_profile
  --config-file <config>
  --api-url http://localhost:<port>/v1/completions
  --backend sglang
  --server-batch-size <batch_size>
  --output_dir <output_dir>
```

SGLang/MoE-CAP writes full server records to:

```text
expert_records/<model_id>/expert_distribution_record.jsonl
```

The harness preserves those records in each result leaf as:

```text
server_records_<dataset>_<timestamp>.jsonl
```

Then `analyze.py` loads the records and computes:

```text
records
  -> normalize_records(records)
  -> compute_smfu_smbu(records, metadata)
  -> _run_metrics(...)
  -> moe_cap.utils.continuous_batching_utils._calculate_continuous_metrics(...)
```

`analyze.py` normalizes latency like this:

```text
if forward_mode == "prefill":
    latency = ttft if ttft exists else latency or tpot
else:
    latency = tpot if tpot exists else latency or ttft
```

MoE-CAP returns S-MFU and S-MBU as fractions. The harness multiplies them by
`100` for CSV and plots:

```text
prefill_smfu_percent = result["prefill_smfu"] * 100
prefill_smbu_percent = result["prefill_smbu"] * 100
decoding_smfu_percent = result["decoding_smfu"] * 100
decoding_smbu_percent = result["decoding_smbu"] * 100
```

## Runner Variables Used

The experiment records provide these runtime variables:

```text
forward_mode          -> "prefill" or decode mode
seq_lens_sum          -> total tokens in the packed forward step
ttft                  -> prefill latency, normalized to latency
tpot                  -> decode latency, normalized to latency
latency               -> MoE-CAP expected timing field
batch_size            -> batch size for the server step
per_req_info          -> per-request packed prefill details
expert_activation     -> activated expert count / routing activity
gpu_raw_type          -> GPU name for peak FLOPS/BW lookup
```

Metadata and HuggingFace config provide model and hardware constants:

```text
model_name
precision
num_gpus
num_hidden_layers
hidden_size
num_attention_heads
head_dim
num_key_value_heads
ffn_dim / moe_intermediate_size
shared_expert_intermediate_size
DeepSeek-specific KV/LoRA/attention fields when present
```

## Current Prefill Chain

For each valid prefill record:

```text
ttft = out["latency"]
prefill_activation = out["expert_activation"]
prefill_tp = out["seq_lens_sum"] / ttft
```

Then continuous output processing does:

```text
ctx_len = out["seq_lens_sum"]
processed_tokens =
  sum(req.extend_len for req in per_req_info)
  if per_req_info exists
  else ctx_len

kv_size = processed_tokens * per_token_kv_size / 1e12
true_kv_size =
  (processed_tokens * per_token_kv_size + per_token_kv_size) / 1e9 * 1e3
```

Attention score:

```text
total_attention_score = _calculate_attention_score(ctx_len, output_len=1)

if forward_mode == "prefill":
    attention_score = total_attention_score / ctx_len
else:
    attention_score = total_attention_score / batch_size
```

This means prefill S-MFU receives per-token attention work and packed forward
tokens/sec.

## Generic Prefill Formulas

For generic models, MoE-CAP calculates:

```text
prefill_smbu_i =
  ((n_layers * (prefill_activation * expert_size + attention_size_per_token)
    + kv_size)
   * precision_bytes / ttft)
  / (num_gpus * peak_bandwidth_tb)
```

The final prefill S-MBU is latency weighted:

```text
prefill_smbu =
  sum(prefill_smbu_i * ttft_i) / sum(ttft_i)
```

This is equivalent to:

```text
prefill_smbu =
  sum(bytes_i) / sum(ttft_i) / (num_gpus * peak_bandwidth_tb)
```

```text
prefill_smfu =
  ((n_layers * (attention_size_per_token + expert_size)
    + attention_score)
   * 2 * prefill_tp)
  / (num_gpus * peak_flops_tf / 2)
```

Dependency chain:

```text
prefill_smfu
  = f(sparse_work_per_token, prefill_tp, num_gpus, peak_flops_tf)

prefill_tp
  = seq_lens_sum / ttft

seq_lens_sum
  = obtained from runner/server record

ttft
  = runner ttft normalized to latency

sparse_work_per_token
  = n_layers * (attention_size_per_token + expert_size) + attention_score

attention_score
  = total_attention_score(seq_lens_sum, model attention config) / seq_lens_sum

expert_size
  = f(ffn_dim or moe_intermediate_size, hidden_size)
```

## Model-Specific Variants

### Qwen

Qwen adds shared experts:

```text
prefill_smbu =
  (n_layers * (prefill_activation * expert_size
               + shared_experts_size_total
               + attention_size_per_token)
   + kv_size)
  * precision_bytes / ttft
  / (num_gpus * peak_bandwidth_tb)
```

```text
prefill_smfu =
  (n_layers * (attention_size_per_token
               + expert_size
               + shared_experts_size_total)
   + attention_score)
  * 2 * prefill_tp
  / (num_gpus * peak_flops_tf / 2)
```

### Qwen3

Qwen3 separates MoE layers from dense FFN layers:

```text
num_moe_layers =
  count(layer_idx not in mlp_only_layers
        and (layer_idx + 1) % decoder_sparse_step == 0)

num_dense_layers = num_hidden_layers - num_moe_layers
```

```text
moe_bandwidth = num_moe_layers * prefill_activation * expert_size
dense_bandwidth = num_dense_layers * dense_ffn_size
attention_bandwidth = n_layers * attention_size_per_token

prefill_smbu =
  (moe_bandwidth + dense_bandwidth + attention_bandwidth + kv_size)
  * precision_bytes / ttft
  / (num_gpus * peak_bandwidth_tb)
```

```text
moe_flops = num_moe_layers * expert_size
dense_flops = num_dense_layers * dense_ffn_size

prefill_smfu =
  (moe_flops + dense_flops + attention_bandwidth + attention_score)
  * 2 * prefill_tp
  / (num_gpus * peak_flops_tf / 2)
```

### DeepSeek

DeepSeek separates sparse MoE layers from dense replacement layers:

```text
deepseek_num_dense_layer = first_k_dense_replace
deepseek_sparse_layer_num = n_layers - deepseek_num_dense_layer
```

```text
prefill_smbu =
  (n_layers * attention_size_per_token
   + deepseek_sparse_layer_num
     * (prefill_activation * expert_size + shared_experts_size_total)
   + deepseek_num_dense_layer * deepseek_dense_ffn_size
   + kv_size)
  * precision_bytes / ttft
  / (num_gpus * peak_bandwidth_tb)
```

```text
prefill_smfu =
  (n_layers * attention_size_per_token
   + deepseek_sparse_layer_num
     * (expert_size + shared_experts_size_total)
   + deepseek_num_dense_layer * deepseek_dense_ffn_size
   + attention_score)
  * 2 * prefill_tp
  / (num_gpus * peak_flops_tf / 2)
```

## Local MoE-CAP Changes

`git status --short` inside `MoE-CAP` shows:

```text
 M moe_cap/utils/basic_utils.py
 M moe_cap/utils/continuous_batching_utils.py
 M moe_cap/utils/hardware_utils.py
```

Diff summary:

```text
moe_cap/utils/basic_utils.py               | updates attention and processed-token KV sizing
moe_cap/utils/continuous_batching_utils.py | packed prefill path and S-MBU time weighting
moe_cap/utils/hardware_utils.py            |  8 +++
```

### Change 1: Packed Prefill Throughput for S-MFU/S-MBU

Original `per_req_info` prefill path:

```text
for each packed prefill record:
  for each request in per_req_info:
    latency_sum += out["latency"]
    seq_lens_sum += req_info["extend_len"]
    total_tokens = req_info["total_len"]

after request completes:
  ttft = latency_sum
  prefill_tp = total_tokens / ttft
  aggregated_out["seq_lens_sum"] = total_tokens
  smfu = f(prefill_tp, aggregated_out)
```

For a packed batch:

```text
batch_size = B
tokens per request = L
record latency = t

original prefill_tp per request = L / t
average prefill_tp = L / t
```

This undercounts packed prefill throughput because the physical forward pass
processed all `B * L` tokens during the same latency `t`.

Local version:

```text
ttft = out["latency"]
prefill_tp = out["seq_lens_sum"] / ttft
metrics_data = _process_outputs_continuous(out, ...)
```

So now:

```text
seq_lens_sum = B * L
ttft = t
prefill_tp = (B * L) / t
```

Then S-MFU receives the packed forward-pass throughput:

```text
prefill_smfu =
  sparse_work_per_token * 2 * prefill_tp
  / (num_gpus * peak_flops_tf / 2)
```

Effect:

```text
old S-MFU was roughly low by factor B for packed prefill throughput
new S-MFU uses the actual physical forward-pass token mass
```

### Change 2: Attention Score Normalized to Per-Token Work

Original `_process_outputs_continuous`:

```text
ctx_len = out["seq_lens_sum"]
attention_score = _calculate_attention_score(ctx_len, output_len=1)
```

That returns total attention work for the whole context. But S-MFU later
multiplies by `prefill_tp`, so the value must be per-token work.

Local version:

```text
ctx_len = out["seq_lens_sum"]
attention_score = _calculate_attention_score(ctx_len, output_len=1)

if forward_mode == "prefill":
    attention_score = attention_score / max(ctx_len, 1)
else:
    attention_score = attention_score / max(batch_size, 1)
```

New chain:

```text
seq_lens_sum
  -> total attention score
  -> attention_score_per_token = total_attention_score / seq_lens_sum
  -> S-MFU = (... + attention_score_per_token) * 2 * prefill_tp / peak_flops
```

Effect:

```text
old S-MFU mixed total attention work with tokens/sec
new S-MFU uses work per token * tokens/sec
```

### Change 3: H100 NVL Hardware Specs

`hardware_utils.py` adds:

```text
NVIDIA-H100-NVL-96GB
```

Bandwidth:

```text
peak_bandwidth = 3900e9 bytes/sec
```

Peak FLOPS:

```text
float32  = 835e12
float16  = 1671e12
bfloat16 = 1671e12
int8     = 3341e12
fp8      = 3341e12
fp4      = 3341e12
int4     = 3341e12
```

For S-MFU, this changes the denominator chain:

```text
gpu_raw_type = "NVIDIA-H100-NVL-96GB"
  -> peak_flops_tf = 1671 for bfloat16/float16
  -> prefill_smfu denominator = num_gpus * 1671 / 2
```

Without this local change, H100 NVL runs can hit an unknown GPU lookup and get
skipped during analysis.

### Change 4: S-MBU Uses Processed Tokens and Time-Weighted Aggregation

Previous S-MBU aggregation:

```text
prefill_smbu = mean(prefill_smbu_i)
```

Current S-MBU aggregation:

```text
prefill_smbu =
  sum(prefill_smbu_i * ttft_i) / sum(ttft_i)
```

Because:

```text
prefill_smbu_i =
  bytes_i / ttft_i / peak_bandwidth
```

the time-weighted form is equivalent to:

```text
prefill_smbu =
  sum(bytes_i) / sum(ttft_i) / peak_bandwidth
```

For packed or chunked prefill, KV bytes now use the actual newly processed
tokens when available:

```text
processed_tokens_i =
  sum(req.extend_len for req in per_req_info_i)
  if per_req_info_i exists
  else seq_lens_sum_i

kv_size_i = processed_tokens_i * per_token_kv_size / 1e12
```

This keeps `seq_lens_sum` available for attention/S-MFU context work while
making S-MBU KV traffic reflect the new KV written by that forward pass.

## Latest Patch Summary

The latest implementation update changed S-MBU without changing the S-MFU
packed-prefill throughput fix.

Files changed:

```text
MoE-CAP/moe_cap/utils/basic_utils.py
MoE-CAP/moe_cap/utils/continuous_batching_utils.py
tests/test_moe_cap_glue.py
```

### `basic_utils.py`

`_process_outputs_continuous()` now separates the token count used for
attention/S-MFU context work from the token count used for S-MBU KV traffic.

Current behavior:

```text
ctx_len = out["seq_lens_sum"]

attention_score =
  _calculate_attention_score(ctx_len, output_len=1)
```

For prefill S-MFU:

```text
attention_score = attention_score / ctx_len
```

For S-MBU KV traffic:

```text
kv_tokens = out.get("processed_tokens", ctx_len)
kv_size = kv_tokens * per_token_kv_size / 1e12
true_kv_size = (kv_tokens * per_token_kv_size + per_token_kv_size) / 1e9 * 1e3
```

Reason:

```text
seq_lens_sum may describe the full context mass
processed_tokens describes the newly written KV tokens for this forward pass
```

### `continuous_batching_utils.py`

For packed or chunked prefill records with `per_req_info`, the code now derives
`processed_tokens` from request-level extend lengths:

```text
processed_tokens =
  sum(req_info["extend_len"] for req_info in out["per_req_info"])
```

It passes this value into `_process_outputs_continuous()` as:

```text
metrics_out = dict(out)
metrics_out["processed_tokens"] = processed_tokens
```

S-MFU is unchanged:

```text
prefill_tp = out["seq_lens_sum"] / out["latency"]
```

S-MBU aggregation changed from:

```text
prefill_smbu = mean(prefill_smbu_i)
decoding_smbu = mean(decoding_smbu_i)
```

to latency-weighted aggregation:

```text
prefill_smbu =
  sum(prefill_smbu_i * ttft_i) / sum(ttft_i)

decoding_smbu =
  sum(decoding_smbu_i * tpot_i) / sum(tpot_i)
```

This makes the final S-MBU equivalent to:

```text
sum(estimated_bytes_i) / sum(latency_i) / peak_bandwidth
```

instead of:

```text
mean(estimated_bytes_i / latency_i / peak_bandwidth)
```

### Tests Added

`tests/test_moe_cap_glue.py` now includes:

```text
test_continuous_prefill_kv_size_uses_processed_tokens_when_present
test_prefill_smbu_is_latency_weighted
```

Verification:

```text
.venv/bin/python -m pytest tests/test_moe_cap_glue.py tests/test_analyze.py

55 passed, 4 warnings
```

## Final S-MFU Chain After Local Changes

For packed prefill:

```text
runner record:
  seq_lens_sum = physical packed tokens in forward pass
  ttft/latency = forward pass latency
  expert_activation = activated expert count
  gpu_raw_type = GPU name

prefill_tp = seq_lens_sum / latency

attention_score =
  _calculate_attention_score(seq_lens_sum, output_len=1)
  / seq_lens_sum

model_work_per_token =
  model-specific sparse work
  + attention_score

prefill_smfu_fraction =
  model_work_per_token * 2 * prefill_tp
  / (num_gpus * peak_flops_tf / 2)

prefill_smfu_percent =
  prefill_smfu_fraction * 100
```
