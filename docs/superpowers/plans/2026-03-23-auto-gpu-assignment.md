# Auto GPU Assignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add ~12 lines of bash to `run_single.sh` so it automatically selects the GPU with the most free VRAM, avoiding OOM when GPU 0 is occupied on the local 2×H100 machine.

**Architecture:** Query `nvidia-smi` for per-GPU free VRAM at script start, pick the highest-VRAM GPU index, and export `CUDA_VISIBLE_DEVICES` before starting the SGLang server. For TP>1 models, unset `CUDA_VISIBLE_DEVICES` so SGLang can span both GPUs. The `|| true` guard prevents `set -euo pipefail` from aborting if `nvidia-smi` is unavailable.

**Tech Stack:** Bash, `nvidia-smi`, `awk`, `sort`

---

## File Map

| File | Action | What changes |
|------|--------|-------------|
| `scripts/run_single.sh` | Modify (line 38) | Insert GPU selection block after `log()` definition |

No new files. No other files touched.

---

### Task 1: Verify the `nvidia-smi` pipeline produces the correct GPU index

Before touching the script, confirm the pipeline works correctly in the local environment.

**Files:**
- No file changes — shell verification only

- [ ] **Step 1: Run the pipeline manually and check output**

```bash
nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits \
    | awk '{print NR-1, $1}' | sort -k2 -rn | head -1 | awk '{print $1}'
```

Expected: a single integer (`0` or `1`) — the index of the GPU with the most free VRAM.

If you have 2 GPUs and GPU 0 is busy, you should see `1`. If both are idle with equal VRAM, you'll see `0` (deterministic tie-break).

- [ ] **Step 2: Verify the fallback when `nvidia-smi` is unavailable**

```bash
BEST_GPU=$(false | awk '{print NR-1, $1}' | sort -k2 -rn | head -1 | awk '{print $1}') || true
echo "BEST_GPU='$BEST_GPU'"
```

Expected output: `BEST_GPU=''` — empty string, no error, script continues.

---

### Task 2: Insert the GPU selection block into `run_single.sh`

**Files:**
- Modify: `scripts/run_single.sh` — insert after line 37 (`log() { ... }`)

- [ ] **Step 1: Open `scripts/run_single.sh` and locate the insertion point**

Line 37 reads:
```bash
log() { echo "[$(date '+%H:%M:%S')] $*"; }
```

The new block goes immediately after this line, before the `# ── Resolve config file` comment on line 39.

- [ ] **Step 2: Insert the GPU selection block**

Add the following between line 37 and line 39:

```bash

# ── GPU selection ─────────────────────────────────────────────────────────────
if [ "$TP" -gt 1 ]; then
    # Multi-GPU model: unset so SGLang can see all GPUs and span them via TP.
    # Note: this overrides any caller-supplied CUDA_VISIBLE_DEVICES; acceptable
    # for the 2-GPU H100 local environment this feature targets.
    unset CUDA_VISIBLE_DEVICES
else
    # Single-GPU model: pick GPU with most free VRAM.
    # || true prevents set -euo pipefail from aborting if nvidia-smi is missing.
    BEST_GPU=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits \
        | awk '{print NR-1, $1}' | sort -k2 -rn | head -1 | awk '{print $1}') || true
    if [ -n "$BEST_GPU" ]; then
        export CUDA_VISIBLE_DEVICES="$BEST_GPU"
        log "Selected GPU $BEST_GPU (most free VRAM)"
    else
        log "WARNING: nvidia-smi unavailable; not setting CUDA_VISIBLE_DEVICES"
    fi
fi
```

- [ ] **Step 3: Verify the script is syntactically valid**

```bash
bash -n scripts/run_single.sh
```

Expected: no output, exit code 0.

- [ ] **Step 4: Do a dry-run smoke test (TP=1 path)**

```bash
bash -c '
TP=1
log() { echo "[$(date +%H:%M:%S)] $*"; }
BEST_GPU=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits \
    | awk "{print NR-1, \$1}" | sort -k2 -rn | head -1 | awk "{print \$1}") || true
if [ -n "$BEST_GPU" ]; then
    export CUDA_VISIBLE_DEVICES="$BEST_GPU"
    log "Selected GPU $BEST_GPU (most free VRAM)"
else
    log "WARNING: nvidia-smi unavailable; not setting CUDA_VISIBLE_DEVICES"
fi
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
'
```

Expected: prints `Selected GPU <N> (most free VRAM)` and `CUDA_VISIBLE_DEVICES=<N>`.

- [ ] **Step 5: Do a dry-run smoke test (TP=2 path)**

```bash
bash -c '
export CUDA_VISIBLE_DEVICES=0
TP=2
if [ "$TP" -gt 1 ]; then
    unset CUDA_VISIBLE_DEVICES
fi
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
'
```

Expected: `CUDA_VISIBLE_DEVICES=<unset>` — confirms the `unset` cleared the caller-supplied value.

- [ ] **Step 6: Commit**

```bash
git add scripts/run_single.sh
git commit -m "feat: auto-select GPU with most free VRAM in run_single.sh"
```
