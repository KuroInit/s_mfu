import yaml
from pathlib import Path


def _load_yaml(path: str):
    with Path(path).open() as f:
        return yaml.safe_load(f) or {}


def _active_datasets(sweep: dict):
    from orchestrator import _active_datasets

    return _active_datasets(sweep)


def _effective_config(dataset: str, model: dict):
    from orchestrator import _load_effective_dataset_config

    _, cfg = _load_effective_dataset_config(dataset, model)
    return cfg


def test_active_sweep_requests_fit_model_context_windows():
    sweep = _load_yaml("sweep_config.yaml")

    assert [model["slug"] for model in sweep["models"]] == [
        "qwen1_5_moe",
        "qwen3_30b",
        "qwen3_next_80b",
    ]

    for model in sweep["models"]:
        for dataset in _active_datasets(sweep):
            cfg = _effective_config(dataset, model)
            input_tokens = cfg["target_input_tokens"]
            output_tokens = cfg["target_output_tokens"]
            max_context = model["max_context_tokens"]
            if input_tokens is None and output_tokens is None:
                continue

            assert input_tokens + output_tokens <= max_context, (
                f"{cfg['model_id']} requests {input_tokens}+{output_tokens} tokens, "
                f"but max_position_embeddings is {max_context}"
            )


def test_qwen1_5_uses_75_percent_context_window():
    model = {"id": "Qwen/Qwen1.5-MoE-A2.7B-Chat", "slug": "qwen1_5_moe"}
    cfg = _effective_config("longbench_v2", model)

    assert cfg["target_input_tokens"] == 24576
    assert cfg["target_output_tokens"] == 1
    assert cfg["target_input_tokens"] + cfg["target_output_tokens"] <= 32768


def test_active_batched_prefill_sweep_can_exercise_high_batch_with_chunking():
    sweep = _load_yaml("sweep_config.yaml")
    model = next(model for model in sweep["models"] if model["slug"] == "qwen1_5_moe")
    cfg = _effective_config("batched_prefill", model)

    assert cfg["prefill_mode"] == "batched"
    assert cfg["target_input_tokens"] == 2048
    assert cfg["target_output_tokens"] == 1
    assert cfg["chunked_prefill_size"] == 131072
    assert cfg["max_prefill_tokens"] == 131072
    assert cfg["mem_fraction_static"] == 0.9
    assert 64 * cfg["target_input_tokens"] <= cfg["chunked_prefill_size"]
    assert 64 * cfg["target_input_tokens"] <= cfg["max_prefill_tokens"]


def test_active_sweep_batch_cells_fit_h100_nvl_memory_estimate():
    from orchestrator import _effective_batch_sizes, _estimate_per_gpu_memory_gb

    sweep = _load_yaml("sweep_config.yaml")
    gpu_memory_gb = sweep["gpu_memory_gb"]
    for model in sweep["models"]:
        for dataset in _active_datasets(sweep):
            cfg = _effective_config(dataset, model)
            for bs in _effective_batch_sizes(sweep["batch_sizes"], cfg):
                estimated = _estimate_per_gpu_memory_gb(model, cfg, bs)
                if cfg["target_input_tokens"] is None:
                    assert estimated is None
                    continue
                assert estimated <= gpu_memory_gb, (
                    f"{model['id']} bs={bs} needs ~{estimated:.1f}GB/GPU "
                    f"on a {gpu_memory_gb}GB H100 NVL"
                )
