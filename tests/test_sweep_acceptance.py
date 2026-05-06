import yaml
from pathlib import Path


def _load_yaml(path: str):
    with Path(path).open() as f:
        return yaml.safe_load(f) or {}


def test_active_sweep_requests_fit_model_context_windows():
    sweep = _load_yaml("sweep_config.yaml")

    assert [model["slug"] for model in sweep["models"]] == ["qwen1_5_moe"]

    for model in sweep["models"]:
        for dataset in sweep["datasets"]:
            cfg = _load_yaml(f"configs/{dataset}_{model['slug']}.yaml")
            input_tokens = cfg["target_input_tokens"]
            output_tokens = cfg["target_output_tokens"]
            max_context = model["max_context_tokens"]

            assert input_tokens + output_tokens <= max_context, (
                f"{cfg['model_id']} requests {input_tokens}+{output_tokens} tokens, "
                f"but max_position_embeddings is {max_context}"
            )


def test_qwen1_5_uses_75_percent_context_window():
    cfg = _load_yaml("configs/longbench_v2_qwen1_5_moe.yaml")

    assert cfg["target_input_tokens"] == 24576
    assert cfg["target_output_tokens"] == 1
    assert cfg["target_input_tokens"] + cfg["target_output_tokens"] <= 32768


def test_active_batched_prefill_sweep_can_exercise_high_batch_with_chunking():
    cfg = _load_yaml("configs/batched_prefill_qwen1_5_moe.yaml")

    assert cfg["prefill_mode"] == "batched"
    assert cfg["target_input_tokens"] == 1024
    assert cfg["target_output_tokens"] == 1
    assert cfg["chunked_prefill_size"] == 150000
    assert cfg["max_prefill_tokens"] == 150000
    # Prompts tokenize to roughly 1,050 tokens, so keep slack above 128 * 1K.
    assert 128 * cfg["target_input_tokens"] <= cfg["chunked_prefill_size"]
    assert 128 * cfg["target_input_tokens"] <= cfg["max_prefill_tokens"]


def test_active_sweep_batch_cells_fit_h100_nvl_memory_estimate():
    from orchestrator import _effective_batch_sizes, _estimate_per_gpu_memory_gb

    sweep = _load_yaml("sweep_config.yaml")
    gpu_memory_gb = sweep["gpu_memory_gb"]
    for model in sweep["models"]:
        for dataset in sweep["datasets"]:
            cfg = _load_yaml(f"configs/{dataset}_{model['slug']}.yaml")
            for bs in _effective_batch_sizes(sweep["batch_sizes"], cfg):
                estimated = _estimate_per_gpu_memory_gb(model, cfg, bs)
                assert estimated is not None
                assert estimated <= gpu_memory_gb, (
                    f"{model['id']} bs={bs} needs ~{estimated:.1f}GB/GPU "
                    f"on a {gpu_memory_gb}GB H100 NVL"
                )
