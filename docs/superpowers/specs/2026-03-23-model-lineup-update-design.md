# Design: Model Lineup Update & README Overhaul

**Date:** 2026-03-23
**Scope:** H100 NVL Docker environment only. NSCC deferred.

---

## Context

The project profiles MoE LLM models by sweeping batch sizes and measuring S-MFU and S-MBU using MoE-CAP. The original model lineup (Qwen1.5-MoE, DeepSeek-V2-Lite, Qwen3-30B, qwen3-235b) is being replaced/expanded with 7 models. Several bugs were found in the existing code that are **not yet fixed**:

1. `sweep.sh` and `submit_sweep.sh` referenced `qwen3_next_80b` but the configs pointed to `qwen3-235b` — scripts and configs were never in sync.
2. `run_single.sh` injects `LABEL` (a short display name like `qwen1.5-moe`) as `model_id` into the config. MoE-CAP dispatches model-family FLOP logic by substring-matching `model_id` against canonical HF prefixes (`"Qwen/Qwen1.5"`, `"Qwen3"`, `"DeepSeek"`, etc.). A short label never matches, causing silent fallthrough to a wrong/generic FLOP formula. Injecting `MODEL` (a local path) also fails since the HF org prefix is absent. **Fix:** add a canonical HF model ID as a 4th field in the MODELS array and pass it to `run_single.sh` as a dedicated argument.
3. `sweep.sh` has a hardcoded `HF_CACHE` default pointing to an NSCC NTU user path — must be replaced.
4. `configs/qwen3_235b_*.yaml` and `configs/llama3_8b_*.yaml` are orphaned — must be removed.
5. `scripts/nscc/submit_sweep.sh` has inconsistent model references — deferred to NSCC work.

**Note on model availability:** `Qwen/Qwen3.5-9B` and `Qwen/Qwen3-Next-80B-A3B-Instruct` are confirmed by the project owner. Verify availability on HuggingFace before the first sweep run.

---

## Model Roster (7 models)

| Label | HuggingFace ID | Params (total/active) | TP (H100 NVL) | MoE-CAP Status |
|-------|---------------|----------------------|---------------|----------------|
| `olmoe-1b` | `allenai/OLMoE-1B-7B-0924` | 6.9B / 1B | 1 | Fully supported |
| `qwen1.5-moe` | `Qwen/Qwen1.5-MoE-A2.7B-Chat` | 14.3B / 2.7B | 1 | Fully supported |
| `dsv2-lite` | `deepseek-ai/DeepSeek-V2-Lite-Chat` | 16B / 2.4B | 1 | Fully supported |
| `qwen3.5-9b` | `Qwen/Qwen3.5-9B` | 9B / 9B (dense) | 1 | Not applicable — exploratory only |
| `qwen3-30b` | `Qwen/Qwen3-30B-A3B` | 30B / 3B | 1 | Fully supported |
| `phi-3.5-moe` | `microsoft/Phi-3.5-MoE-instruct` | 41.9B / 6.6B | 1 | Supported (verify MoE-CAP dispatch branch at runtime) |
| `qwen3-next-80b` | `Qwen/Qwen3-Next-80B-A3B-Instruct` | 80B / 3B | 2 | Partial — see caveats |

### Caveats

**`qwen3.5-9b`:** Dense model (no MoE) with Gated DeltaNet + Gated Attention instead of standard transformer attention. Two compounding problems: (a) no MoE — S-MFU/S-MBU are undefined; (b) DeltaNet is linear attention with a fundamentally different FLOP profile — even standard MFU estimates are inaccurate. Included to observe MoE-CAP behaviour on an unsupported architecture. Results must be clearly labelled exploratory in the README and plots.

**`qwen3-next-80b`:** Has MoE (512 experts, top-k=10, 1 shared expert) but 3 out of 4 attention layers are Gated DeltaNet, not standard attention. MoE-CAP's `F_attn` term assumes standard attention and is inaccurate for those layers. MoE expert utilisation metrics are directionally useful; absolute FLOP numbers are not validated.

**`phi-3.5-moe`:** Standard transformer attention and MoE — should be fully supported. CLAUDE.md documents MoE-CAP dispatch for Qwen/Qwen1.5, Qwen3, and DeepSeek families explicitly; Phi-3.5 dispatch path is not documented. Verify at runtime that MoE-CAP produces plausible metrics for this model.

**VRAM note for `qwen3-next-80b`:** 80B params × 2 bytes (BF16) ≈ 160GB for weights alone. 2× H100 NVL at 95GB = 190GB total, leaving ~30GB for KV cache and activations. This does not account for batch size or sequence length. May OOM at batch sizes ≥ 64. Treat large-batch results as best-effort.

---

## Hardware

- **Environment:** Ubuntu server, Docker
- **GPUs:** 2× NVIDIA H100 NVL, 95GB VRAM each (190GB total)
- **Max TP:** 2
- **All TP values are for this environment only.** NSCC A100 deployment is deferred.

---

## Implementation Tasks

### Task 1 — Fix model_id injection in run_single.sh (NOT YET IMPLEMENTED)

**Current broken code (run_single.sh line 50):**
```bash
sed "s|^model_id:.*|model_id: ${LABEL}|" "$TEMPLATE" > "$TEMP_CONFIG"
```

**Target state:** Add `HF_MODEL_ID` as a 6th positional argument. Use it for `model_id` injection. `LABEL` remains for logging and output file naming only.

```bash
# New signature
MODEL=${1:?...}
BATCH_SIZE=${2:?...}
DATASET=${3:?...}
TP=${4:-1}
LABEL=${5:-$(basename "$MODEL")}
HF_MODEL_ID=${6:-$MODEL}   # canonical HF ID for MoE-CAP dispatch

# sed fix
sed "s|^model_id:.*|model_id: ${HF_MODEL_ID}|" "$TEMPLATE" > "$TEMP_CONFIG"
```

Also update the inline comment on the old line 9 (`[label] - display name written as model_id in the result JSON`) to say that LABEL is for logging/file naming only, and HF_MODEL_ID (arg 6) is written as `model_id`.

---

### Task 2 — Update sweep.sh MODELS array (NOT YET IMPLEMENTED)

Add a 4th pipe-delimited field (canonical HF ID) to every MODELS entry. Update the `IFS` read line to handle 4 fields. Update the `run_single.sh` call to pass the HF ID as arg 6. Replace the hardcoded `HF_CACHE`.

**Current broken loop:**
```bash
HF_CACHE="/home/users/ntu/ashwin01/scratch/fyp/sglang_pcie/models/hf_cache"  # wrong
MODELS=(
    "${qwen1_5_moe:-$HF_CACHE/Qwen1.5-MoE-A2.7B-Chat}|qwen1.5-moe|1"  # 3 fields only
    ...
)
for entry in "${MODELS[@]}"; do
    IFS='|' read -r model_path model_label model_tp <<< "$entry"  # 3 vars — 4th field bleeds into model_tp
    ...
    bash scripts/run_single.sh "$model_path" "$batch_size" "$dataset" "$model_tp" "$model_label"  # no HF ID arg
```

**Target state:**
```bash
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"

# Format: "local_path_or_hf_id|label|tp|canonical_hf_id"
MODELS=(
    "${olmoe_1b:-allenai/OLMoE-1B-7B-0924}|olmoe-1b|1|allenai/OLMoE-1B-7B-0924"
    "${qwen1_5_moe:-Qwen/Qwen1.5-MoE-A2.7B-Chat}|qwen1.5-moe|1|Qwen/Qwen1.5-MoE-A2.7B-Chat"
    "${dsv2_lite:-deepseek-ai/DeepSeek-V2-Lite-Chat}|dsv2-lite|1|deepseek-ai/DeepSeek-V2-Lite-Chat"
    "${qwen3_5_9b:-Qwen/Qwen3.5-9B}|qwen3.5-9b|1|Qwen/Qwen3.5-9B"
    "${qwen3_30b:-Qwen/Qwen3-30B-A3B}|qwen3-30b|1|Qwen/Qwen3-30B-A3B"
    "${phi_3_5_moe:-microsoft/Phi-3.5-MoE-instruct}|phi-3.5-moe|1|microsoft/Phi-3.5-MoE-instruct"
    "${qwen3_next_80b:-Qwen/Qwen3-Next-80B-A3B-Instruct}|qwen3-next-80b|2|Qwen/Qwen3-Next-80B-A3B-Instruct"
)

for entry in "${MODELS[@]}"; do
    IFS='|' read -r model_path model_label model_tp model_hf_id <<< "$entry"  # 4 vars
    ...
    bash scripts/run_single.sh "$model_path" "$batch_size" "$dataset" "$model_tp" "$model_label" "$model_hf_id"
```

---

### Task 3 — Remove orphaned configs (NOT YET IMPLEMENTED)

Delete:
- `configs/qwen3_235b_4k1k.yaml`
- `configs/qwen3_235b_13k1k.yaml`
- `configs/llama3_8b_4k1k.yaml`
- `configs/llama3_8b_13k1k.yaml`

---

### Task 4 — Verify generic templates (VERIFY BEFORE PROCEEDING)

Confirm these files match exactly before marking the config work done:

**`configs/4k1k.yaml` (expected):**
```yaml
dataset_names: ["gsm8k"]
metrics: []
model_id: PLACEHOLDER
fixed_length_mode: true
target_input_tokens: 4000
target_output_tokens: 1000
num_samples: 256
```

**`configs/13k1k.yaml` (expected):**
```yaml
dataset_names: ["longbench_v2"]
metrics: []
model_id: PLACEHOLDER
fixed_length_mode: true
target_input_tokens: 13000
target_output_tokens: 1000
num_samples: 256
```

The `sed` substitution targets `^model_id:` — if this key is absent or differently named, substitution fails silently.

---

### Task 5 — Update README (NOT YET IMPLEMENTED)

Structure:
1. **Overview** — project goal, S-MFU/S-MBU metric definitions
2. **Model table** — all 7 models, TP, MoE-CAP status; `qwen3.5-9b` and `qwen3-next-80b` marked with caveat indicators
3. **Known limitations** — dedicated section: DeltaNet incompatibility for both models; VRAM risk for `qwen3-next-80b`; Phi-3.5 dispatch unverified
4. **Quickstart** — `docker compose run --rm bench bash` → `bash scripts/sweep.sh`
5. **Running a single job** — `run_single.sh` usage with examples
6. **Generating plots** — `python plot_metrics.py`
7. **Overriding defaults** — env var table for all 7 models + `HF_CACHE`, `BATCH_SIZES`, `DATASETS`
8. **Future: NSCC deployment** — stub paragraph: NSCC scripts exist in `scripts/nscc/`; TP values differ per GPU type; `submit_sweep.sh` needs updating before use

---

## Env Var Override Names

| Env var | Default (HF ID) |
|---------|-----------------|
| `olmoe_1b` | `allenai/OLMoE-1B-7B-0924` |
| `qwen1_5_moe` | `Qwen/Qwen1.5-MoE-A2.7B-Chat` |
| `dsv2_lite` | `deepseek-ai/DeepSeek-V2-Lite-Chat` |
| `qwen3_5_9b` | `Qwen/Qwen3.5-9B` |
| `qwen3_30b` | `Qwen/Qwen3-30B-A3B` |
| `phi_3_5_moe` | `microsoft/Phi-3.5-MoE-instruct` |
| `qwen3_next_80b` | `Qwen/Qwen3-Next-80B-A3B-Instruct` |
| `HF_CACHE` | `${HF_HOME:-$HOME/.cache/huggingface}` |
| `BATCH_SIZES` | `1 4 8 16 32 64 128` |
| `DATASETS` | `4k1k 13k1k` |

---

## Out of Scope

- NSCC PBS submission, Singularity, resource scaling — deferred
- `scripts/nscc/submit_sweep.sh` known model inconsistencies — deferred
- `scripts/nscc/build_image.pbs` duplicate `module load singularity` — deferred
- `plot_metrics.py` — no changes needed
- `docker-compose.yml` — no changes needed
- Existing model-specific configs (`qwen1_5_moe_*.yaml`, `dsv2_lite_*.yaml`, `qwen3_30b_moe_*.yaml`) — kept as convenience files, not used by `sweep.sh`
