# Test Flow

The harness flow is:

1. Load `sweep_config.yaml`.
2. Validate model IDs and benchmark configs.
3. For each `(model, batch_size, dataset)` cell, wait for enough idle GPUs when
   automatic GPU selection is enabled.
4. Start `moe_cap.systems.sglang`.
5. Wait for `/health`.
6. Run `moe_cap.runner.openai_api_profile` with `--server-batch-size`.
7. Preserve MoE-CAP/SGLang server records beside MoE-CAP's output files.
8. Write an analyzable `failure_*.json` artifact for startup/OOM, runner, or
   missing-server-record failures.
9. Mark the checkpoint and restart SGLang for the next cell.
10. Run `analyze.py` to compute CSV rows and plots from MoE-CAP-derived metrics.

The harness does not implement its own benchmark runner or alternate metric
definitions. Its job is orchestration, checkpointing, result preservation, and
plot/export formatting.
