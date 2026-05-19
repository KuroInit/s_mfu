def test_h100_nvl_hardware_specs_are_registered():
    from moe_cap.utils.hardware_utils import get_peak_bw, get_peak_flops

    gpu_name = "NVIDIA-H100-NVL-96GB"
    assert get_peak_bw(gpu_name) == 3900e9
    assert get_peak_flops(gpu_name, "bfloat16") == 1671e12
    assert get_peak_flops(gpu_name, "float16") == 1671e12
    assert get_peak_flops(gpu_name, "float32") == 835e12


def test_per_req_prefill_uses_packed_forward_throughput(monkeypatch):
    from moe_cap.utils import continuous_batching_utils as cb

    monkeypatch.setattr(cb, "_get_hardware_specs", lambda *_: {})
    monkeypatch.setattr(cb, "_calculate_kv_size", lambda *_, **__: 1)
    monkeypatch.setattr(cb, "_calculate_attention_size", lambda *_, **__: 1)
    monkeypatch.setattr(cb, "_calculate_expert_config", lambda *_, **__: {})
    monkeypatch.setattr(
        cb,
        "_process_outputs_continuous",
        lambda out, *_, **__: {"true_kv_size": out["seq_lens_sum"], "attention_score": 0},
    )

    seen_prefill_tps = []

    def fake_prefill_metrics(**kwargs):
        seen_prefill_tps.append(kwargs["prefill_tp"])
        return 0.0, kwargs["prefill_tp"]

    monkeypatch.setattr(cb, "_calculate_prefill_metrics", fake_prefill_metrics)

    result = cb._calculate_continuous_metrics(
        n_layers=1,
        d_model=1,
        gpu_raw_type="gpu",
        n_attn_heads=1,
        d_head=1,
        n_kv_heads=1,
        d_ff=1,
        hf_config=None,
        num_gpus=1,
        model_name="model",
        used_dtype="bfloat16",
        precision=2,
        output_data=[
            {
                "forward_mode": "prefill",
                "expert_activation": 0.5,
                "latency": 2.0,
                "batch_size": 4,
                "seq_lens_sum": 400,
                "per_req_info": [
                    {"req_id": f"r{i}", "extend_len": 100, "total_len": 100, "is_last_chunk": True}
                    for i in range(4)
                ],
            }
        ],
    )

    assert seen_prefill_tps == [200.0]
    assert result["prefill_tp"] == 200.0
    assert result["prefill_smfu"] == 200.0


def test_continuous_attention_score_is_per_token_for_prefill():
    from moe_cap.utils.basic_utils import _process_outputs_continuous

    metrics = _process_outputs_continuous(
        {
            "forward_mode": "prefill",
            "seq_lens_sum": 100,
            "batch_size": 4,
        },
        per_token_kv_size=1,
        attention_size_per_token=1,
        model_name="model",
        hf_config=None,
        n_layers=1,
        n_attn_heads=1,
        d_head=1,
    )

    # Total attention score would be 200 / 1e12 for this toy prefill.
    # The continuous-batching S-MFU formulas multiply attention_score by
    # tokens/sec, so this must be normalized to per-token work.
    assert metrics["attention_score"] == 2e-12


def test_continuous_attention_score_is_per_generated_token_for_decode():
    from moe_cap.utils.basic_utils import _process_outputs_continuous

    metrics = _process_outputs_continuous(
        {
            "forward_mode": "decode",
            "seq_lens_sum": 100,
            "batch_size": 4,
        },
        per_token_kv_size=1,
        attention_size_per_token=1,
        model_name="model",
        hf_config=None,
        n_layers=1,
        n_attn_heads=1,
        d_head=1,
    )

    assert metrics["attention_score"] == 50e-12


def test_continuous_prefill_kv_size_uses_processed_tokens_when_present():
    from moe_cap.utils.basic_utils import _process_outputs_continuous

    metrics = _process_outputs_continuous(
        {
            "forward_mode": "prefill",
            "seq_lens_sum": 100,
            "processed_tokens": 40,
            "batch_size": 4,
        },
        per_token_kv_size=2,
        attention_size_per_token=1,
        model_name="model",
        hf_config=None,
        n_layers=1,
        n_attn_heads=1,
        d_head=1,
    )

    assert metrics["kv_size"] == 80e-12
    assert metrics["true_kv_size"] == 82e-6
    # Attention still uses the full context length for prefill work.
    assert metrics["attention_score"] == 2e-12


def test_prefill_smbu_is_latency_weighted(monkeypatch):
    from moe_cap.utils import continuous_batching_utils as cb

    monkeypatch.setattr(cb, "_get_hardware_specs", lambda *_: {})
    monkeypatch.setattr(cb, "_calculate_kv_size", lambda *_, **__: 1)
    monkeypatch.setattr(cb, "_calculate_attention_size", lambda *_, **__: 1)
    monkeypatch.setattr(cb, "_calculate_expert_config", lambda *_, **__: {})
    monkeypatch.setattr(
        cb,
        "_process_outputs_continuous",
        lambda out, *_, **__: {"true_kv_size": out["seq_lens_sum"], "attention_score": 0},
    )

    def fake_prefill_metrics(**kwargs):
        return kwargs["ttft"], 0.0

    monkeypatch.setattr(cb, "_calculate_prefill_metrics", fake_prefill_metrics)

    result = cb._calculate_continuous_metrics(
        n_layers=1,
        d_model=1,
        gpu_raw_type="gpu",
        n_attn_heads=1,
        d_head=1,
        n_kv_heads=1,
        d_ff=1,
        hf_config=None,
        num_gpus=1,
        model_name="model",
        used_dtype="bfloat16",
        precision=2,
        output_data=[
            {
                "forward_mode": "prefill",
                "expert_activation": 0.5,
                "latency": 1.0,
                "batch_size": 1,
                "seq_lens_sum": 100,
            },
            {
                "forward_mode": "prefill",
                "expert_activation": 0.5,
                "latency": 3.0,
                "batch_size": 1,
                "seq_lens_sum": 300,
            },
        ],
    )

    assert result["prefill_smbu"] == 2.5
