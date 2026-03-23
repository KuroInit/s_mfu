# Model Lineup Update Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Update the project to profile 7 MoE/dense models on H100 NVL via Docker, fix a silent MoE-CAP model dispatch bug, remove orphaned configs, and rewrite the README as H100/Docker-focused documentation.

**Architecture:** `sweep.sh` drives all runs by iterating a 4-field MODELS array (`path|label|tp|hf_id`) and calling `run_single.sh` with 6 arguments; `run_single.sh` injects the canonical HF model ID (not the display label) into the config template so MoE-CAP can dispatch the correct FLOP formula per model family.

**Tech Stack:** Bash, Docker, MoE-CAP, SGLang 0.5.8, Python (plot_metrics.py)

**Spec:** `docs/superpowers/specs/2026-03-23-model-lineup-update-design.md`

---

## File Map

| Action | File | What changes |
|--------|------|-------------|
| Modify | `scripts/run_single.sh` | Add `HF_MODEL_ID` as arg 6; fix `sed` to use it; update header comment |
| Modify | `scripts/sweep.sh` | 7-model MODELS array with 4 fields; fix `IFS` read; pass HF ID as arg 6; fix `HF_CACHE` default |
| Delete | `configs/qwen3_235b_4k1k.yaml` | Orphaned |
| Delete | `configs/qwen3_235b_13k1k.yaml` | Orphaned |
| Delete | `configs/llama3_8b_4k1k.yaml` | Orphaned |
| Delete | `configs/llama3_8b_13k1k.yaml` | Orphaned |
| Rewrite | `README.md` | H100/Docker-focused; 7-model table with caveats; known limitations section |

---

## Task 0: Verify generic config templates

**Files:** `configs/4k1k.yaml`, `configs/13k1k.yaml` (read-only verification)

- [ ] **Step 1: Confirm templates contain a `model_id:` key**

```bash
grep "model_id" configs/4k1k.yaml configs/13k1k.yaml
```
Expected:
```
configs/4k1k.yaml:model_id: PLACEHOLDER
configs/13k1k.yaml:model_id: PLACEHOLDER
```
If either file is missing the `model_id:` key, the `sed` substitution in `run_single.sh` will silently produce a broken config. Do not proceed until this passes.

---

## Task 1: Fix model_id injection in run_single.sh

**Files:**
- Modify: `scripts/run_single.sh` (lines 3, 9, 23–24, 50, 52)

- [ ] **Step 1: Verify the bug exists**

```bash
grep "model_id" scripts/run_single.sh
```
Expected output shows `model_id: ${LABEL}` — confirms the bug.

- [ ] **Step 2: Update the header comment block**

In `scripts/run_single.sh`, replace the Usage comment block (lines 3–15).

Before (current lines 3–15):
```bash
# Run one (model, batch_size, dataset) profiling job.
# Usage: run_single.sh <model_path> <batch_size> <dataset> [tp_size] [label]
#
#   <model_path>  - HF model ID or local filesystem path (e.g. /scratch/models/Qwen1.5-MoE)
#   <batch_size>  - integer (e.g. 1, 4, 8, 16, 32, 64, 128)
#   <dataset>     - dataset key ("4k1k", "13k1k") or path to a full .yaml config file
#   [tp_size]     - tensor parallel size (default: 1)
#   [label]       - display name written as model_id in the result JSON
#                   (default: basename of model_path)
#
# Environment:
#   OUTPUT_DIR    - where to write result JSONs (default: /workspace/results)
#   CONFIGS_DIR   - where to find dataset templates (default: /workspace/configs)
#   PORT          - SGLang server port (default: 30000)
```

After:

```bash
# Run one (model, batch_size, dataset) profiling job.
# Usage: run_single.sh <model_path> <batch_size> <dataset> [tp_size] [label] [hf_model_id]
#
#   <model_path>   - HF model ID or local filesystem path (e.g. /hf_cache/Qwen1.5-MoE-A2.7B-Chat)
#   <batch_size>   - integer (e.g. 1, 4, 8, 16, 32, 64, 128)
#   <dataset>      - dataset key ("4k1k", "13k1k") or path to a full .yaml config file
#   [tp_size]      - tensor parallel size (default: 1)
#   [label]        - display name for logging and output file naming (default: basename of model_path)
#   [hf_model_id]  - canonical HuggingFace model ID written as model_id in the MoE-CAP config
#                    (default: model_path). Must match HF ID so MoE-CAP dispatches the correct
#                    FLOP formula (e.g. "Qwen/Qwen1.5-MoE-A2.7B-Chat", not "qwen1.5-moe").
#
# Environment:
#   OUTPUT_DIR    - where to write result JSONs (default: /workspace/results)
#   CONFIGS_DIR   - where to find dataset templates (default: /workspace/configs)
#   PORT          - SGLang server port (default: 30000)
```

- [ ] **Step 3: Add HF_MODEL_ID argument and fix sed**

Replace lines 22–23 (the argument parsing block):
```bash
TP=${4:-1}
LABEL=${5:-$(basename "$MODEL")}
```
with:
```bash
TP=${4:-1}
LABEL=${5:-$(basename "$MODEL")}
HF_MODEL_ID=${6:-$MODEL}
```

Replace line 50 (the sed substitution):
```bash
    sed "s|^model_id:.*|model_id: ${LABEL}|" "$TEMPLATE" > "$TEMP_CONFIG"
```
with:
```bash
    sed "s|^model_id:.*|model_id: ${HF_MODEL_ID}|" "$TEMPLATE" > "$TEMP_CONFIG"
```

Replace line 52 (the log line):
```bash
    log "Generated config: dataset=$DATASET  model_id=$LABEL"
```
with:
```bash
    log "Generated config: dataset=$DATASET  model_id=$HF_MODEL_ID  label=$LABEL"
```

- [ ] **Step 4: Syntax-check the file**

```bash
bash -n scripts/run_single.sh
```
Expected: no output (no syntax errors).

- [ ] **Step 5: Verify the fix with a dry-run grep**

```bash
grep -n "model_id\|HF_MODEL_ID\|LABEL" scripts/run_single.sh
```
Confirm `HF_MODEL_ID` is used in the `sed` line, not `LABEL`.

- [ ] **Step 6: Commit**

```bash
git add scripts/run_single.sh
git commit -m "fix: inject canonical HF model ID into MoE-CAP config, not display label

LABEL (e.g. 'qwen1.5-moe') does not match MoE-CAP's model-family dispatch
substrings ('Qwen/Qwen1.5', 'DeepSeek', etc.), causing silent fallthrough to
wrong FLOP formula. New arg 6 (HF_MODEL_ID) carries the canonical HF ID;
LABEL remains for logging and output file naming only."
```

---

## Task 2: Update sweep.sh — 7 models, 4-field MODELS array, fix HF_CACHE

**Files:**
- Modify: `scripts/sweep.sh`

- [ ] **Step 1: Replace the entire MODELS array and HF_CACHE default**

Replace lines 14–24 (the `HF_CACHE` line through the closing `)` of the MODELS array) with:

```bash
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"

# Format: "local_path_or_hf_id|display_label|tp|canonical_hf_id"
# local_path_or_hf_id: override via env var (e.g. olmoe_1b=/data/OLMoE); defaults to HF ID
#   (SGLang will download from HF if not already cached under HF_HOME)
# canonical_hf_id: used as model_id in MoE-CAP config for correct FLOP dispatch
MODELS=(
    "${olmoe_1b:-allenai/OLMoE-1B-7B-0924}|olmoe-1b|1|allenai/OLMoE-1B-7B-0924"
    "${qwen1_5_moe:-Qwen/Qwen1.5-MoE-A2.7B-Chat}|qwen1.5-moe|1|Qwen/Qwen1.5-MoE-A2.7B-Chat"
    "${dsv2_lite:-deepseek-ai/DeepSeek-V2-Lite-Chat}|dsv2-lite|1|deepseek-ai/DeepSeek-V2-Lite-Chat"
    "${qwen3_5_9b:-Qwen/Qwen3.5-9B}|qwen3.5-9b|1|Qwen/Qwen3.5-9B"
    "${qwen3_30b:-Qwen/Qwen3-30B-A3B}|qwen3-30b|1|Qwen/Qwen3-30B-A3B"
    "${phi_3_5_moe:-microsoft/Phi-3.5-MoE-instruct}|phi-3.5-moe|1|microsoft/Phi-3.5-MoE-instruct"
    "${qwen3_next_80b:-Qwen/Qwen3-Next-80B-A3B-Instruct}|qwen3-next-80b|2|Qwen/Qwen3-Next-80B-A3B-Instruct"
)
```

- [ ] **Step 2: Fix the IFS read line to handle 4 fields**

Replace:
```bash
    IFS='|' read -r model_path model_label model_tp <<< "$entry"
```
with:
```bash
    IFS='|' read -r model_path model_label model_tp model_hf_id <<< "$entry"
```

- [ ] **Step 3: Pass model_hf_id as arg 6 to run_single.sh**

Replace:
```bash
            if OUTPUT_DIR="$OUTPUT_DIR" bash scripts/run_single.sh \
                    "$model_path" "$batch_size" "$dataset" "$model_tp" "$model_label" \
                    >> "$LOG" 2>&1; then
```
with:
```bash
            if OUTPUT_DIR="$OUTPUT_DIR" bash scripts/run_single.sh \
                    "$model_path" "$batch_size" "$dataset" "$model_tp" "$model_label" "$model_hf_id" \
                    >> "$LOG" 2>&1; then
```

- [ ] **Step 4: Update the header comment to reflect 7 models and new env vars**

Replace the existing header comment block (lines 1–8) with:
```bash
#!/bin/bash
# Sweep 7 models across batch sizes and datasets on H100 NVL (Docker).
# Run inside the Docker container: bash scripts/sweep.sh
#
# Override any model path (defaults to HF model ID — downloaded automatically):
#   olmoe_1b=/local/path         qwen1_5_moe=/local/path   dsv2_lite=/local/path
#   qwen3_5_9b=/local/path       qwen3_30b=/local/path      phi_3_5_moe=/local/path
#   qwen3_next_80b=/local/path
# Override batch sizes:  BATCH_SIZES="1 8 32" bash scripts/sweep.sh
# Override datasets:     DATASETS="4k1k" bash scripts/sweep.sh
```

- [ ] **Step 5: Syntax-check the file**

```bash
bash -n scripts/sweep.sh
```
Expected: no output.

- [ ] **Step 6: Verify the array and loop**

```bash
grep -n "MODELS\|IFS\|model_hf_id\|HF_CACHE\|HF_HOME" scripts/sweep.sh
```
Confirm: `HF_CACHE` uses `HF_HOME`, MODELS has 7 entries with 4 fields, `IFS` read has 4 variables, arg 6 is passed.

- [ ] **Step 7: Commit**

```bash
git add scripts/sweep.sh
git commit -m "feat: update sweep to 7 models with canonical HF IDs for MoE-CAP dispatch

Models: OLMoE-1B, Qwen1.5-MoE, DSv2-Lite, Qwen3.5-9B, Qwen3-30B,
Phi-3.5-MoE, Qwen3-Next-80B (TP=2). 4-field MODELS array passes
canonical HF ID to run_single.sh. HF_CACHE now defaults to HF_HOME
instead of a hardcoded NTU NSCC path."
```

---

## Task 3: Remove orphaned config files

**Files:**
- Delete: `configs/qwen3_235b_4k1k.yaml`
- Delete: `configs/qwen3_235b_13k1k.yaml`
- Delete: `configs/llama3_8b_4k1k.yaml`
- Delete: `configs/llama3_8b_13k1k.yaml`

- [ ] **Step 1: Confirm these files exist and are not referenced anywhere**

```bash
ls configs/qwen3_235b_*.yaml configs/llama3_8b_*.yaml
grep -r "qwen3_235b\|llama3_8b" scripts/ --include="*.sh"
```
Expected: files exist; grep returns no matches in scripts (they are unused).

- [ ] **Step 2: Delete them**

```bash
rm configs/qwen3_235b_4k1k.yaml configs/qwen3_235b_13k1k.yaml \
   configs/llama3_8b_4k1k.yaml configs/llama3_8b_13k1k.yaml
```

- [ ] **Step 3: Verify configs directory**

```bash
ls configs/
```
Expected:
```
13k1k.yaml
4k1k.yaml
dsv2_lite_13k1k.yaml
dsv2_lite_4k1k.yaml
qwen1_5_moe_13k1k.yaml
qwen1_5_moe_4k1k.yaml
qwen3_30b_moe_13k1k.yaml
qwen3_30b_moe_4k1k.yaml
```

- [ ] **Step 4: Commit**

```bash
git add -u configs/
git commit -m "chore: remove orphaned qwen3-235b and llama3-8b config files

These models are no longer in the sweep. Qwen3-235B was replaced by
Qwen3-Next-80B; Llama3-8B was dropped. Generic templates (4k1k.yaml,
13k1k.yaml) cover all current models."
```

---

## Task 4: Rewrite README.md

**Files:**
- Rewrite: `README.md`

- [ ] **Step 1: Write the new README**

Replace the entire contents of `README.md` with:

````markdown
# MoE S-MFU / S-MBU Profiling Sweep

Profiles 7 LLM models (MoE and dense) by sweeping batch sizes and measuring
**S-MFU** (Sparse Model FLOP Utilization) and **S-MBU** (Sparse Memory Bandwidth Utilization)
using the [MoE-CAP framework](https://github.com/Auto-CAP/MoE-CAP).

**Hardware:** 2× NVIDIA H100 NVL (95 GB VRAM each) · Docker on Ubuntu

---

## Models

| Label | Model | Params (total / active) | TP | MoE-CAP |
|-------|-------|------------------------|----|---------|
| `olmoe-1b` | `allenai/OLMoE-1B-7B-0924` | 6.9B / 1B | 1 | ✅ Full |
| `qwen1.5-moe` | `Qwen/Qwen1.5-MoE-A2.7B-Chat` | 14.3B / 2.7B | 1 | ✅ Full |
| `dsv2-lite` | `deepseek-ai/DeepSeek-V2-Lite-Chat` | 16B / 2.4B | 1 | ✅ Full |
| `qwen3-30b` | `Qwen/Qwen3-30B-A3B` | 30B / 3B | 1 | ✅ Full |
| `phi-3.5-moe` | `microsoft/Phi-3.5-MoE-instruct` | 41.9B / 6.6B | 1 | ⚠️ Unverified |
| `qwen3-next-80b` | `Qwen/Qwen3-Next-80B-A3B-Instruct` | 80B / 3B | 2 | ⚠️ Partial |
| `qwen3.5-9b` | `Qwen/Qwen3.5-9B` | 9B / 9B (dense) | 1 | ❌ Exploratory |

TP = tensor parallel size. Models default to HF IDs and are downloaded automatically on first run.

---

## Known Limitations

### `qwen3.5-9b` — Dense model, metrics are not meaningful
This is a **dense** model (no Mixture-of-Experts) using Gated DeltaNet linear attention layers
instead of standard transformer attention. Two compounding problems:
- No MoE component: S-MFU and S-MBU are undefined for dense models.
- DeltaNet has a fundamentally different FLOP profile from standard attention:
  even standard MFU estimates will be inaccurate.

Included to observe MoE-CAP behaviour on an unsupported architecture. **Do not compare
its S-MFU/S-MBU values against the MoE models.**

### `qwen3-next-80b` — Hybrid attention, partially inaccurate FLOP accounting
This model has a valid MoE component (512 experts, top-k=10, 1 shared expert) but 3 out of
every 4 attention layers use Gated DeltaNet (linear attention), not standard attention.
MoE-CAP's `F_attn` term in S-MFU assumes standard attention and is inaccurate for DeltaNet layers.
MoE expert utilisation trends are directionally useful; absolute FLOP numbers are not validated.

**VRAM note:** 80B params × 2 bytes (BF16) ≈ 160 GB for weights, leaving ~30 GB headroom
across 2× H100 NVL. May OOM at batch sizes ≥ 64. Large-batch results are best-effort.

### `phi-3.5-moe` — MoE-CAP dispatch unverified
Standard transformer attention with MoE — should be fully supported. MoE-CAP's documented
dispatch paths cover Qwen/Qwen1.5, Qwen3, and DeepSeek families explicitly; Phi-3.5 dispatch
has not been verified in practice. Check results for plausibility on first run.

---

## Quickstart

```bash
# Build image and open a shell
docker compose run --rm bench bash

# Inside container: run the full sweep (all 7 models × 7 batch sizes × 2 datasets)
bash scripts/sweep.sh

# Generate plots from results
python plot_metrics.py
```

Models are downloaded from HuggingFace automatically on first run (stored under `HF_HOME`).
To use a local cache, mount it via `HF_CACHE_DIR`:

```bash
HF_CACHE_DIR=~/.cache/huggingface docker compose run --rm bench bash
```

---

## Running a Single Job

```bash
# Inside the container
bash scripts/run_single.sh <model_path_or_hf_id> <batch_size> <dataset> [tp] [label] [hf_model_id]
```

Examples:

```bash
# OLMoE-1B, batch size 8, 4k input / 1k output
bash scripts/run_single.sh allenai/OLMoE-1B-7B-0924 8 4k1k 1 olmoe-1b allenai/OLMoE-1B-7B-0924

# Qwen3-Next-80B from a local path, batch size 16, TP=2
bash scripts/run_single.sh /hf_cache/Qwen3-Next-80B-A3B-Instruct 16 4k1k 2 qwen3-next-80b Qwen/Qwen3-Next-80B-A3B-Instruct
```

**Argument 6 (`hf_model_id`) is important when using a local path for arg 1.**
MoE-CAP uses the model ID to select the correct FLOP formula for each model family.
If omitted, it defaults to arg 1 (the path), which may not be recognised.

---

## Generating Plots

```bash
# From the repo root (inside or outside the container)
python plot_metrics.py
# Writes: plots/smfu_prefill.png  plots/smfu_decoding.png
#         plots/smbu_prefill.png  plots/smbu_decoding.png
```

Override directories:

```bash
RESULTS_DIR=/other/results PLOTS_DIR=/other/plots python plot_metrics.py
```

---

## Overriding Defaults

| Variable | Default | Description |
|----------|---------|-------------|
| `olmoe_1b` | `allenai/OLMoE-1B-7B-0924` | Path or HF ID for OLMoE-1B |
| `qwen1_5_moe` | `Qwen/Qwen1.5-MoE-A2.7B-Chat` | Path or HF ID for Qwen1.5-MoE |
| `dsv2_lite` | `deepseek-ai/DeepSeek-V2-Lite-Chat` | Path or HF ID for DeepSeek-V2-Lite |
| `qwen3_5_9b` | `Qwen/Qwen3.5-9B` | Path or HF ID for Qwen3.5-9B |
| `qwen3_30b` | `Qwen/Qwen3-30B-A3B` | Path or HF ID for Qwen3-30B |
| `phi_3_5_moe` | `microsoft/Phi-3.5-MoE-instruct` | Path or HF ID for Phi-3.5-MoE |
| `qwen3_next_80b` | `Qwen/Qwen3-Next-80B-A3B-Instruct` | Path or HF ID for Qwen3-Next-80B |
| `HF_CACHE` | `$HF_HOME` or `~/.cache/huggingface` | Root of downloaded model weights |
| `BATCH_SIZES` | `1 4 8 16 32 64 128` | Space-separated batch sizes |
| `DATASETS` | `4k1k 13k1k` | Space-separated dataset keys |
| `OUTPUT_DIR` | `/workspace/results` | Result JSON output directory |

Examples:

```bash
# One dataset, two batch sizes
DATASETS="4k1k" BATCH_SIZES="1 8" bash scripts/sweep.sh

# Use a local model path for Qwen3-Next-80B
qwen3_next_80b=/data/models/Qwen3-Next-80B-A3B-Instruct bash scripts/sweep.sh
```

---

## Repository Layout

```
sglang_s_mfu/
├── Dockerfile                   # nvcr.io/nvidia/pytorch:24.01-py3 + sglang==0.5.8 + MoE-CAP
├── docker-compose.yml           # H100 NVL Docker dev with HF cache mount
├── configs/
│   ├── 4k1k.yaml                # Generic template: gsm8k, 4000→1000 tok (model_id injected at runtime)
│   ├── 13k1k.yaml               # Generic template: longbench_v2, 13000→1000 tok
│   ├── qwen1_5_moe_4k1k.yaml    # Model-specific convenience configs (not used by sweep.sh)
│   └── ...
├── scripts/
│   ├── run_single.sh            # One (model, bs, dataset) run: start server → profile → kill
│   ├── sweep.sh                 # Full sweep: all 7 models × batch sizes × datasets
│   └── export_image.sh          # docker save → .tar.gz for transfer
├── scripts/nscc/                # NSCC PBS scripts (deferred — see below)
├── results/                     # JSON outputs (gitignored)
├── plots/                       # Generated figures (gitignored)
└── plot_metrics.py              # Reads results/ → batch_size vs S-MFU/S-MBU plots
```

---

## Metrics

**S-MFU (Sparse Model FLOP Utilization)**
```
S-MFU = T_token × (F_attn + 2·N_router + 2·k_expert·N_expert) / F_peak
```
- `T_token = batch_size / forward_pass_latency` (per forward pass)
- Does not require expert activation distribution

**S-MBU (Sparse Memory Bandwidth Utilization)**
```
S-MBU = (S_activated + S_KV) / (TPOT × B_peak)
```
- `S_activated`: memory of activated experts only (`k/E` fraction of total weights)
- Requires expert activation distribution from SGLang hooks

Both metrics correct the overestimation of standard MFU/MBU by counting only *k* activated experts.
Output JSON values are fractions (0–1); `plot_metrics.py` multiplies by 100 for percentages.

---

## Future: NSCC Deployment

PBS job scripts for the NSCC A100 SXM4 cluster are in `scripts/nscc/`. They are not
currently maintained — TP values differ from H100 (larger models need more GPUs on 40 GB cards)
and the model list in `submit_sweep.sh` needs updating before use. NSCC deployment is deferred.
````

- [ ] **Step 2: Verify the README renders correctly**

```bash
# Check for broken markdown (mismatched backticks, etc.)
python3 -c "
import re, sys
text = open('README.md').read()
# Count backtick fences — must be even
fences = re.findall(r'^```', text, re.MULTILINE)
print(f'Code fences: {len(fences)} (should be even)')
sys.exit(0 if len(fences) % 2 == 0 else 1)
"
```
Expected: `Code fences: N (should be even)` with no exit error.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: rewrite README for H100 NVL Docker environment with 7 models

- 7-model table with MoE-CAP support status (full/partial/exploratory)
- Dedicated Known Limitations section for qwen3.5-9b (dense+DeltaNet),
  qwen3-next-80b (DeltaNet FLOP inaccuracy + VRAM risk), phi-3.5-moe (unverified)
- Updated env var override table for all 7 models
- NSCC deployment deferred to future work stub
- Metrics definitions section added"
```

---

## Final Check

- [ ] **Verify all four tasks are complete**

```bash
# Scripts pass syntax check
bash -n scripts/run_single.sh && echo "run_single.sh OK"
bash -n scripts/sweep.sh && echo "sweep.sh OK"

# Orphaned configs are gone
ls configs/ | grep -E "qwen3_235b|llama3_8b" && echo "FAIL: orphaned configs remain" || echo "Orphaned configs removed OK"

# run_single.sh uses HF_MODEL_ID in sed
grep "HF_MODEL_ID" scripts/run_single.sh && echo "run_single.sh fix OK"

# sweep.sh has 7 model entries (count lines that match the 4-field MODELS entry pattern)
grep -c '^\s*"${' scripts/sweep.sh
```

- [ ] **Git log sanity check**

```bash
git log --oneline -5
```
Expected: 4 commits from this implementation visible at the top.
