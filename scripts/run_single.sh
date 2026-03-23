#!/bin/bash
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

set -euo pipefail

MODEL=${1:?Usage: run_single.sh <model_path> <batch_size> <dataset> [tp_size] [label]}
BATCH_SIZE=${2:?missing batch_size}
DATASET=${3:?missing dataset}
TP=${4:-1}
LABEL=${5:-$(basename "$MODEL")}
HF_MODEL_ID=${6:-$MODEL}

PORT=${PORT:-30000}
OUTPUT_DIR=${OUTPUT_DIR:-/workspace/results}
CONFIGS_DIR=${CONFIGS_DIR:-/workspace/configs}
EXPERT_RECORDS_DIR=${SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR:-/workspace/expert_records}

export SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR="$EXPERT_RECORDS_DIR"

mkdir -p "$OUTPUT_DIR" "$EXPERT_RECORDS_DIR"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ── GPU selection (local only — PBS manages GPU assignment on NSCC) ───────────
if [ -n "${PBS_JOBID:-}" ]; then
    log "PBS job detected: skipping GPU selection (PBS manages GPU assignment)"
elif [ "$TP" -gt 1 ]; then
    # Multi-GPU model: unset so SGLang can see all GPUs and span them via TP.
    # Note: this overrides any caller-supplied CUDA_VISIBLE_DEVICES; acceptable
    # for the 2-GPU H100 local environment this feature targets.
    unset CUDA_VISIBLE_DEVICES
    log "TP=$TP: unsetting CUDA_VISIBLE_DEVICES so SGLang can span all GPUs"
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

# ── Resolve config file ───────────────────────────────────────────────────────
# If DATASET ends in .yaml or is an existing file, use it directly.
# Otherwise treat it as a dataset key and generate a temp config with model_id injected.
TEMP_CONFIG=""
if [[ "$DATASET" == *.yaml ]] || [[ -f "$DATASET" ]]; then
    # Direct yaml path used as-is; HF_MODEL_ID (arg 6) is not substituted in this case.
    CONFIG_FILE="$DATASET"
else
    TEMPLATE="${CONFIGS_DIR}/${DATASET}.yaml"
    if [ ! -f "$TEMPLATE" ]; then
        log "ERROR: Dataset template not found: $TEMPLATE"
        log "Available templates: $(ls "$CONFIGS_DIR"/*.yaml 2>/dev/null | xargs -n1 basename || echo none)"
        exit 1
    fi
    TEMP_CONFIG=$(mktemp /tmp/cap_config_XXXXXX.yaml)
    sed "s|^model_id:.*|model_id: ${HF_MODEL_ID}|" "$TEMPLATE" > "$TEMP_CONFIG"
    CONFIG_FILE="$TEMP_CONFIG"
    log "Generated config: dataset=$DATASET  model_id=$HF_MODEL_ID  label=$LABEL"
fi

# ── Kill any existing server on our port ─────────────────────────────────────
cleanup() {
    if [ -n "$TEMP_CONFIG" ] && [ -f "$TEMP_CONFIG" ]; then
        rm -f "$TEMP_CONFIG"
    fi
    log "Cleaning up server on port $PORT..."
    fuser -k "${PORT}/tcp" 2>/dev/null || true
    pkill -f "moe_cap.systems.sglang" 2>/dev/null || true
    sleep 5
}

wait_for_server() {
    local max=120  # 10 min
    local i=0
    log "Waiting for SGLang server on port $PORT..."
    while [ $i -lt $max ]; do
        if curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1 || \
           curl -sf "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then
            log "Server ready."
            sleep 3
            return 0
        fi
        i=$((i+1))
        sleep 5
    done
    log "ERROR: Server did not start within timeout."
    return 1
}

# ── Start server ──────────────────────────────────────────────────────────────
cleanup

log "Starting SGLang: model=$MODEL  bs=$BATCH_SIZE  tp=$TP"
python -m moe_cap.systems.sglang \
    --model-path "$MODEL" \
    --port "$PORT" \
    --expert-distribution-recorder-mode stat \
    --tp-size "$TP" \
    --max-running-requests "$BATCH_SIZE" \
    > "${OUTPUT_DIR}/server_$(basename "$MODEL")_bs${BATCH_SIZE}.log" 2>&1 &

SERVER_PID=$!
trap 'cleanup' EXIT

wait_for_server

# ── Run profiler ──────────────────────────────────────────────────────────────
log "Running profiler: dataset=$DATASET  batch_size=$BATCH_SIZE"
python -m moe_cap.runner.openai_api_profile \
    --config-file "$CONFIG_FILE" \
    --api-url "http://localhost:${PORT}/v1/completions" \
    --backend sglang \
    --ignore-eos \
    --server-batch-size "$BATCH_SIZE" \
    --output_dir "$OUTPUT_DIR"

log "Done: label=$LABEL  bs=$BATCH_SIZE  dataset=$DATASET"
