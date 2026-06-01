# S-MFU Harness TODO

## Completed

- Standardized the sweep entry point around `benchmark_types`.
  - `prefill: [batched_prefill]`
  - `chat: [sharegpt, azure_chat]`
  - `reasoning: [mmlu_pro]`
  - `agentic: []`
- Removed the separate top-level `datasets` key from `sweep_config.yaml`.
- Updated the harness so `run_sweep()` derives active datasets from `benchmark_types`.
- Added validation for supported benchmark lanes: `prefill`, `chat`, `reasoning`, and `agentic`.
- Added validation that each dataset slug appears in only one benchmark lane.
- Added validation that each active config's `benchmark_type` matches its lane.
- Standardized YAML config keys across user-built configs.
- Shrunk active configs to one shared `configs/<dataset>.yaml` per dataset.
- Added `model_overrides` for the few fields that differ by model.
- Added `benchmark_type` to benchmark configs.
- Added `configs/mmlu_pro.yaml` for the active reasoning lane.
- Added chat configs for ShareGPT and Azure-style chat traces.
- Added a packaged harness-side MoE-CAP runner wrapper for chat datasets.
- Added harness-side `sharegpt` and `azure_chat` loaders without modifying MoE-CAP.
- Kept reasoning configs non-fixed-length so the MoE-CAP dataset loader controls prompts and generation caps.
- Kept chat configs non-fixed-length so real trace prompts drive request shape.
- Kept batched-prefill configs fixed-length for S-MFU/S-MBU packed-prefill measurement.
- Removed DeepSeek MoE 16B chat from the active sweep.
- Deleted DeepSeek MoE 16B chat configs.
- Updated README and tests for the `benchmark_types` workflow.
- Verified the current harness state with unit and acceptance tests.

## Remaining

- Harden chat benchmark support.
  - Run ShareGPT with the explicit default HuggingFace JSON source.
  - Run Azure chat with a real local trace via `S_MFU_AZURE_CHAT_PATH` or an explicit `S_MFU_AZURE_CHAT_HF_DATASET`.
  - Confirm whether flattened multi-turn transcripts are sufficient for measurement or whether true multi-message preservation is required.
  - Add stricter chat-specific validation if new config fields become necessary.

- Implement agentic benchmark support.
  - Add or confirm MoE-CAP loader support for SWE-Bench.
  - Decide the canonical dataset slug, for example `swe_bench`.
  - Add SWE-Bench configs for each active model.
  - Enable SWE-Bench under `benchmark_types.agentic`.
  - Confirm output artifacts have enough task-level metadata for later analysis.

- Tighten benchmark-type behavior.
  - Add lane-specific validation rules for agentic once SWE-Bench config shape is known.
  - Add acceptance tests for agentic configs after loaders/configs exist.

- Expand analysis coverage.
  - Confirm S-MFU/S-MBU aggregation behaves correctly for non-prefill reasoning runs.
  - Decide whether chat and agentic runs should report the same S-MFU/S-MBU views or separate lane-specific summaries.
  - Add analysis tests using sample chat and SWE-Bench result artifacts.

- Run real benchmark smoke tests.
  - Run a small `batched_prefill` smoke test.
  - Run a small `mmlu_pro` smoke test.
  - Run `sharegpt` and `azure_chat` smoke tests.
  - Run SWE-Bench smoke tests after agentic support exists.

## Guardrails

- Do not modify MoE-CAP code unless the harness cannot support the required benchmark through configuration or wrapper changes.
- If MoE-CAP changes become necessary, ask for permission first.
- Keep `benchmark_types` as the single source of truth for active datasets.
- Keep one shared config present for every dataset enabled in `benchmark_types`.
