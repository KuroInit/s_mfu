# Pin this tag to a specific digest before running in production.
# Find the latest digest: docker pull lmsys/sglang:latest && docker inspect lmsys/sglang:latest
FROM lmsys/sglang:latest

# Clone MoE-CAP and install it as an editable package.
# This makes `python -m moe_cap.systems.sglang` and
# `python -m moe_cap.runner.openai_api_profile` available.
# Pin to a specific commit SHA for reproducibility before production use.
# Example: RUN git clone --depth 1 https://github.com/Auto-CAP/MoE-CAP /workspace/MoE-CAP && \
#              cd /workspace/MoE-CAP && git checkout <commit-sha>
RUN git clone --depth 1 https://github.com/Auto-CAP/MoE-CAP /workspace/MoE-CAP

WORKDIR /workspace/MoE-CAP

RUN pip install --no-cache-dir -e .

# Copy orchestrator files into the image.
# The MoE-CAP working directory is the CWD, so relative paths in
# orchestrator.py (e.g. "configs/gsm8k_qwen3_30b.yaml") resolve correctly.
COPY orchestrator.py .
COPY sweep_config.yaml .
COPY configs/ configs/

# Expert distribution records and HF model cache are mounted at runtime.
ENV SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR=/results/expert_records
ENV HF_HOME=/hf_cache

# Default entry point. Override with `docker run ... python orchestrator.py --help`
# if you add argument parsing in future.
ENTRYPOINT ["python", "orchestrator.py"]
