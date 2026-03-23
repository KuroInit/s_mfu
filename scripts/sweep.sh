#!/bin/bash
# Sweep 7 models across batch sizes and datasets on H100 NVL (Docker).
# Run inside the Docker container: bash scripts/sweep.sh
#
# Override any model path (defaults to HF model ID — downloaded automatically):
#   olmoe_1b=/local/path         qwen1_5_moe=/local/path   dsv2_lite=/local/path
#   qwen3_5_9b=/local/path       qwen3_30b=/local/path     phi_3_5_moe=/local/path
#   qwen3_235b=/local/path
# Override batch sizes:  BATCH_SIZES="1 8 32" bash scripts/sweep.sh
# Override datasets:     DATASETS="4k1k" bash scripts/sweep.sh

set -euo pipefail

BATCH_SIZES=${BATCH_SIZES:-"1 4 8 16 32 64 128"}
DATASETS=${DATASETS:-"4k1k 13k1k"}
OUTPUT_DIR=${OUTPUT_DIR:-/workspace/results}
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"

mkdir -p "$OUTPUT_DIR"

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
    "${qwen3_235b:-Qwen/Qwen3-235B-A22B}|qwen3-235b|8|Qwen/Qwen3-235B-A22B"
)

LOG="$OUTPUT_DIR/sweep_$(date +%Y%m%d_%H%M%S).log"
log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

n_bs=$(echo "$BATCH_SIZES" | wc -w)
n_ds=$(echo "$DATASETS" | wc -w)
total=$(( ${#MODELS[@]} * n_bs * n_ds ))
run=0; failed=0

log "===== Sweep start: ${#MODELS[@]} models, $total total runs ====="
log "Batch sizes: $BATCH_SIZES  |  Datasets: $DATASETS"

for entry in "${MODELS[@]}"; do
    IFS='|' read -r model_path model_label model_tp model_hf_id <<< "$entry"
    for batch_size in $BATCH_SIZES; do
        for dataset in $DATASETS; do
            run=$((run+1))
            log "[$run/$total] $model_label  bs=$batch_size  dataset=$dataset  tp=$model_tp"
            if OUTPUT_DIR="$OUTPUT_DIR" bash scripts/run_single.sh \
                    "$model_path" "$batch_size" "$dataset" "$model_tp" "$model_label" "$model_hf_id" \
                    >> "$LOG" 2>&1; then
                log "[$run/$total] OK"
            else
                log "[$run/$total] FAILED (exit $?)"
                failed=$((failed+1))
            fi
            sleep 3
        done
    done
done

log "===== Done: $((total-failed))/$total succeeded ====="
