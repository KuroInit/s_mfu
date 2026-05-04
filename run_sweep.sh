#!/usr/bin/env bash
# run_sweep.sh — Run the MoE-CAP benchmark sweep locally (no Docker).
#
# Usage:
#   HF_TOKEN=hf_... ./run_sweep.sh
#
# Optional overrides (environment variables):
#   HF_HOME           — HuggingFace cache dir   (default: ~/.cache/huggingface)
#   RESULTS_DIR       — where results are saved  (default: ./results)
#   SWEEP_CONFIG      — sweep config file        (default: sweep_config.yaml)
#   BATCH_RUNNER      — upstream or strict       (default: upstream)
#   SKIP_INSTALL      — set to 1 to skip pip install steps
#   ANALYZE_ONLY      — set to 1 to skip the sweep and run analyze.py only

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MOE_CAP_DIR="${SCRIPT_DIR}/MoE-CAP"

# ─── Validate HF_TOKEN ─────────────────────────────────────────────────────
if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "ERROR: HF_TOKEN is not set. Export it before running:"
    echo "  export HF_TOKEN=hf_..."
    exit 1
fi
export HF_TOKEN

# ─── Configurable paths ────────────────────────────────────────────────────
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export RESULTS_DIR="${RESULTS_DIR:-${SCRIPT_DIR}/results}"
export SWEEP_CONFIG="${SWEEP_CONFIG:-sweep_config.yaml}"
export CHECKPOINT_PATH="${CHECKPOINT_PATH:-${RESULTS_DIR}/checkpoint.yaml}"
export SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR="${RESULTS_DIR}/expert_records"

echo "=== Environment ==="
echo "  HF_HOME      = ${HF_HOME}"
echo "  RESULTS_DIR   = ${RESULTS_DIR}"
echo "  SWEEP_CONFIG  = ${SWEEP_CONFIG}"
echo "  CHECKPOINT    = ${CHECKPOINT_PATH}"
echo "  EXPERT_RECORDS= ${SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR}"
echo "==================="

# ─── Create output directories ─────────────────────────────────────────────
mkdir -p "${RESULTS_DIR}"
mkdir -p "${SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR}"

# ─── Install dependencies ──────────────────────────────────────────────────
if [[ "${SKIP_INSTALL:-0}" != "1" ]]; then
    echo ""
    echo "[setup] Checking MoE-CAP clone..."
    if [[ ! -d "${MOE_CAP_DIR}" ]]; then
        echo "[setup] Cloning MoE-CAP..."
        git clone --depth 1 https://github.com/Auto-CAP/MoE-CAP "${MOE_CAP_DIR}"
    fi

    echo "[setup] Installing MoE-CAP (editable)..."
    pip install -e "${MOE_CAP_DIR}"

    echo "[setup] Installing sglang..."
    pip install "sglang[all]"

    echo "[setup] Installing matplotlib (for analyze.py)..."
    pip install matplotlib

    echo "[setup] Dependencies ready."
fi

# ─── Run from MoE-CAP workdir ──────────────────────────────────────────────
# orchestrator.py expects to be run from a directory containing configs/ and
# sweep_config.yaml. The Dockerfile copies them into MoE-CAP's workdir, but
# locally they live in SCRIPT_DIR. We cd there instead.
cd "${SCRIPT_DIR}"

if [[ "${ANALYZE_ONLY:-0}" != "1" ]]; then
    echo ""
    echo "=== Starting sweep ==="
    python orchestrator.py
    echo ""
    echo "=== Sweep finished ==="
fi

# ─── Post-processing ───────────────────────────────────────────────────────
echo ""
echo "=== Running analysis ==="
python analyze.py "${RESULTS_DIR}"
echo ""
echo "=== Done ==="
