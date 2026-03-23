# Auto GPU Assignment Design

**Date:** 2026-03-23
**Status:** Approved

## Problem

On a local machine with 2 H100 GPUs, `run_single.sh` launches SGLang without setting
`CUDA_VISIBLE_DEVICES`. PyTorch defaults to GPU 0, causing OOM failures when GPU 0 is
already occupied. The fix must work for sequential local runs only — NSCC PBS jobs are
unaffected (each job gets its own isolated GPU allocation).

## Goal

Automatically select the GPU with the most free VRAM before starting SGLang, so that
sequential sweep runs land on whichever GPU is currently least loaded.

## Scope

- **In scope:** `scripts/run_single.sh` — single change, inline bash
- **Out of scope:** `sweep.sh`, `docker-compose.yml`, NSCC scripts (`submit_sweep.sh`,
  `job.pbs`) — no changes needed

## Design

Add the following block to `run_single.sh` after the `log` function definition (after
line 37, before the `cleanup` function), so `CUDA_VISIBLE_DEVICES` is set for the entire
process including server start and profiler:

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

### How it works

- `nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits` outputs one line per
  GPU with free VRAM in MiB (e.g. `38000` for GPU 0, `79000` for GPU 1)
- `awk '{print NR-1, $1}'` prepends the 0-based GPU index
- `sort -k2 -rn | head -1` picks the GPU with the highest free VRAM; if two GPUs report
  equal VRAM, `sort` preserves input order so GPU 0 is chosen — deterministic and acceptable
- `export CUDA_VISIBLE_DEVICES="$BEST_GPU"` constrains both the SGLang server process and
  the profiler subprocess to that single GPU for the full duration of the run

### Multi-GPU case (TP > 1)

Currently only `qwen3_next_80b` uses `tp=2`. For these, `CUDA_VISIBLE_DEVICES` is unset
so SGLang can discover and use both GPUs. This unconditionally overrides any caller-set
`CUDA_VISIBLE_DEVICES` — this is a known limitation, acceptable for the 2×H100 local
environment where no fine-grained GPU pinning is expected.

## Error handling

The `nvidia-smi` pipeline is guarded with `|| true` to prevent `set -euo pipefail` from
aborting the script on failure. If `nvidia-smi` is unavailable or returns no output,
`BEST_GPU` will be empty; the `-n` guard skips the `export` and logs a warning instead.
`CUDA_VISIBLE_DEVICES` is left unset in that case, so PyTorch falls back to its default
GPU selection. Setting it to an empty string (which CUDA interprets as "no visible devices")
is explicitly avoided.

## Files changed

| File | Change |
|------|--------|
| `scripts/run_single.sh` | Add ~12-line GPU selection block after `log` function definition (line 37) |
