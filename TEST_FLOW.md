# Test Flow

The harness flow is:

1. Load `sweep_config.yaml`.
2. Validate model IDs and benchmark configs.
3. For each `(model, batch_size, dataset)` cell, start
   `moe_cap.systems.sglang`.
4. Wait for `/health`.
5. Run `moe_cap.runner.openai_api_profile` with `--server-batch-size`.
6. Preserve MoE-CAP/SGLang server records beside MoE-CAP's output files.
7. Mark the checkpoint and restart SGLang for the next cell.
8. Run `analyze.py` to compute and plot MoE-CAP-derived metrics.

The harness does not implement its own benchmark runner or alternate metric
definitions. Its job is orchestration, checkpointing, result preservation, and
plot/export formatting.
