FROM nvcr.io/nvidia/pytorch:24.01-py3

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl psmisc && \
    rm -rf /var/lib/apt/lists/*

# SGLang 0.5.8 with flashinfer (pre-built for CUDA 12.3 / PyTorch 2.2)
# If this fails due to a flashinfer ABI mismatch, pin manually:
#   pip install flashinfer-python --find-links https://flashinfer.ai/whl/cu123/torch2.2/
RUN pip install --no-cache-dir "sglang[all]==0.5.8"

# MoE-CAP (includes all metric calculation and profiler runner)
RUN git clone https://github.com/Auto-CAP/MoE-CAP /opt/MoE-CAP && \
    cd /opt/MoE-CAP && \
    pip install --no-cache-dir -e .

# Plotting
RUN pip install --no-cache-dir matplotlib pandas

WORKDIR /workspace

COPY configs/ ./configs/
COPY scripts/ ./scripts/
COPY plot_metrics.py ./

RUN chmod +x ./scripts/*.sh && \
    find ./scripts/nscc -name "*.sh" -exec chmod +x {} \; 2>/dev/null || true

# MoE-CAP writes expert distribution records here (profiler reads from same path)
ENV SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR=/workspace/expert_records

RUN mkdir -p /workspace/results /workspace/plots /workspace/expert_records
