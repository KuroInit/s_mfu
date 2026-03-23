#!/bin/bash
# Submit one PBS job per (model, batch_size, dataset) on NSCC.
#
# Prerequisites (on NSCC login node):
#   1. cd /scratch/<project>/sglang_s_mfu
#   2. singularity build moe-cap-sweep.sif docker-archive://moe-cap-sweep.tar.gz
#   3. bash scripts/nscc/submit_sweep.sh
#
# Override any model path:   qwen3_30b=/other/path bash scripts/nscc/submit_sweep.sh
# Override batch sizes:      BATCH_SIZES="1 8 32" bash scripts/nscc/submit_sweep.sh
# Override datasets:         DATASETS="4k1k" bash scripts/nscc/submit_sweep.sh

set -euo pipefail

SCRATCH_DIR="${SCRATCH_DIR:-$(pwd)}"
IMAGE_PATH="${IMAGE_PATH:-$SCRATCH_DIR/moe-cap-sweep.sif}"
BATCH_SIZES="${BATCH_SIZES:-1 4 8 16 32 64 128}"
DATASETS="${DATASETS:-4k1k 13k1k}"
HF_CACHE="${HF_CACHE:-/home/users/ntu/ashwin01/scratch/fyp/sglang_pcie/models/hf_cache}"

[ -f "$IMAGE_PATH" ] || { echo "ERROR: Singularity image not found: $IMAGE_PATH"; exit 1; }

# "path|label|tp" — override path via env var; tp scales PBS resource request
MODELS=(
    "${qwen1_5_moe:-$HF_CACHE/Qwen1.5-MoE-A2.7B-Chat}|qwen1.5-moe|1"
    "${dsv2_lite:-$HF_CACHE/DeepSeek-V2-Lite-Chat}|dsv2-lite|1"
    "${qwen3_30b:-$HF_CACHE/Qwen3-30B-A3B}|qwen3-30b|1"
    "${qwen3_235b:-$HF_CACHE/Qwen3-235B-A22B}|qwen3-235b|8"
)

submitted=0

for entry in "${MODELS[@]}"; do
    IFS='|' read -r model_path model_label tp <<< "$entry"
    job_label=$(echo "$model_label" | tr '.-' '__')
    ncpus=$((tp * 16))
    mem=$((tp * 64))

    for batch_size in $BATCH_SIZES; do
        for dataset in $DATASETS; do
            job_name="cap_${job_label}_bs${batch_size}_${dataset}"
            echo "Submitting: $job_name  (tp=$tp, ${ncpus}cpu, ${mem}gb, ${tp}gpu)"
            qsub \
                -N "$job_name" \
                -l "select=1:ncpus=${ncpus}:mem=${mem}gb:ngpus=${tp}:gpu_model=A100_SXM4_40GB" \
                -v "MODEL=$model_path,LABEL=$model_label,BATCH_SIZE=$batch_size,DATASET=$dataset,TP=$tp,IMAGE_PATH=$IMAGE_PATH,SCRATCH_DIR=$SCRATCH_DIR" \
                "$SCRATCH_DIR/scripts/nscc/job.pbs"
            submitted=$((submitted+1))
        done
    done
done

echo "Submitted $submitted jobs."
echo "Monitor with: qstat -u \$USER"
