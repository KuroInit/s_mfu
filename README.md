# s_mfu

Sweep harness for running [MoE-CAP](https://github.com/Auto-CAP/MoE-CAP) benchmarks across multiple models, batch sizes, and datasets.

MoE-CAP evaluates sparse Mixture-of-Experts inference systems along three dimensions — **Cost**, **Accuracy**, and **Performance** — producing a unified CAP score. This repo provides the orchestration layer: it manages the SGLang server lifecycle, runs the benchmark runner for each (model × batch_size × dataset) combination, and checkpoints progress so a crashed run can resume from where it left off.

## Prerequisites

- Docker with NVIDIA container toolkit (`--gpus all` support)
- A HuggingFace token with access to the models in `sweep_config.yaml`
- Sufficient GPU memory for the models you intend to run (see [`BENCHMARK_GUIDE.md`](MoE-CAP/BENCHMARK_GUIDE.md))

## Quick Start

```bash
export HF_TOKEN=hf_...
export HF_CACHE_DIR=/path/to/hf_cache   # mounted read-only into the container
export RESULTS_DIR=/path/to/results      # where results and checkpoints are written

docker compose up
```

The sweep runs all combinations defined in `sweep_config.yaml`. If interrupted, re-running the same command resumes from the last checkpoint.

## Configuration

### `sweep_config.yaml`

Controls which models, batch sizes, and datasets are swept:

```yaml
batch_sizes: [1, 32, 64, 128]
datasets: [gsm8k, numinamath]
port: 30000

models:
  - id: Qwen/Qwen1.5-MoE-A2.7B-Chat
    tp: 1          # tensor parallelism — number of GPUs
    slug: qwen1_5_moe
```

- `id`: HuggingFace model ID (validated against HF before any run starts)
- `tp`: tensor parallel degree; must match the number of GPUs available
- `slug`: short name used to look up per-model benchmark configs in `configs/`

### `configs/`

Each `<dataset>_<slug>.yaml` file is a MoE-CAP benchmark config passed directly to the runner. See the [MoE-CAP README](MoE-CAP/README.md) for the full config schema and fixed-length benchmarking options.

## How It Works

For each `(model, batch_size, dataset)` triple:

1. **Pre-flight** — validates all model IDs on HuggingFace before starting anything
2. **Start SGLang** — launches `moe_cap.systems.sglang` as a subprocess and polls `/health` until ready (up to 10 minutes)
3. **Run benchmarks** — invokes `moe_cap.runner.openai_api_profile` for each dataset
4. **Checkpoint** — records success or failure to `/results/checkpoint.yaml`
5. **Shutdown** — gracefully terminates SGLang (SIGTERM → wait → SIGKILL) and waits for the port to clear before starting the next model

Skips any triple already marked `success` in the checkpoint, so restarts are safe.

## Running Without Docker

```bash
git clone --depth 1 https://github.com/Auto-CAP/MoE-CAP
cd MoE-CAP && pip install -e . && cd ..

export HF_HOME=/path/to/hf_cache
export SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR=/path/to/results/expert_records
export CHECKPOINT_PATH=/path/to/results/checkpoint.yaml

python orchestrator.py
```

## Tests

```bash
pip install pytest
pytest tests/
```

The test suite covers checkpoint persistence, SGLang lifecycle (start/health/kill), pre-flight model validation, runner invocation, and sweep loop behavior — all via mocks, no GPU required.
