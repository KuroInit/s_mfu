import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

# ── find_latest_file ───────────────────────────────────────────────────────────

def test_find_latest_file_returns_most_recent(tmp_path):
    from analyze import find_latest_file
    (tmp_path / "results_gsm8k_20240101_100000.jsonl").write_text("a")
    (tmp_path / "results_gsm8k_20240101_120000.jsonl").write_text("b")
    result = find_latest_file(tmp_path, "results_gsm8k_*.jsonl")
    assert result.name == "results_gsm8k_20240101_120000.jsonl"

def test_find_latest_file_returns_none_when_missing(tmp_path):
    from analyze import find_latest_file
    assert find_latest_file(tmp_path, "results_gsm8k_*.jsonl") is None

def test_find_latest_file_single_file(tmp_path):
    from analyze import find_latest_file
    f = tmp_path / "metadata_gsm8k_20240101_090000.json"
    f.write_text("{}")
    assert find_latest_file(tmp_path, "metadata_gsm8k_*.json") == f

# ── load_triple ────────────────────────────────────────────────────────────────

def _write_metadata(path: Path, batch_size=32, gpu_type="NVIDIA-H100-HBM3-80GB",
                    num_gpus=1, model_name="org/model", precision="bfloat16"):
    path.write_text(json.dumps({
        "hardware": {"gpu_type": gpu_type, "num_gpus": num_gpus},
        "model_config": {"model_name": model_name, "precision": precision},
        "system_environment": {"batch_size": batch_size, "inference_engine": "sglang"},
    }))

def _write_jsonl(path: Path, records):
    path.write_text("\n".join(json.dumps(r) for r in records))

def test_load_triple_reads_metadata_and_records(tmp_path):
    from analyze import load_triple
    _write_metadata(tmp_path / "metadata_gsm8k_20240101_100000.json")
    _write_jsonl(tmp_path / "detailed_results_gsm8k_20240101_100000.jsonl",
                 [{"forward_mode": "decoding", "expert_activation": 0.5,
                   "batch_size": 32, "seq_lens_sum": 100, "tpot": 0.02}])
    meta, records = load_triple(tmp_path, "bs32")
    assert meta["hardware"]["gpu_type"] == "NVIDIA-H100-HBM3-80GB"
    assert meta["system_environment"]["batch_size"] == 32
    assert len(records) == 1

def test_load_triple_prefers_full_server_records(tmp_path):
    from analyze import load_triple
    _write_metadata(tmp_path / "metadata_gsm8k_20240101_100000.json")
    _write_jsonl(tmp_path / "detailed_results_gsm8k_20240101_100000.jsonl",
                 [{"forward_mode": "prefill", "seq_lens_sum": 10, "ttft": 0.1}])
    _write_jsonl(tmp_path / "server_records_gsm8k_20240101_100000.jsonl",
                 [{"forward_mode": "prefill", "seq_lens_sum": 100, "latency": 0.2,
                   "per_req_info": [{"req_id": 1, "extend_len": 100, "is_last_chunk": True}]}])
    _, records = load_triple(tmp_path, "bs32")
    assert records[0]["seq_lens_sum"] == 100
    assert "per_req_info" in records[0]

def test_load_triple_batch_size_fallback_to_dir_name(tmp_path):
    from analyze import load_triple
    (tmp_path / "metadata_gsm8k_20240101_100000.json").write_text(json.dumps({
        "hardware": {"gpu_type": "NVIDIA-H100-HBM3-80GB", "num_gpus": 1},
        "model_config": {"model_name": "org/model", "precision": "bfloat16"},
        "system_environment": {},
    }))
    _write_jsonl(tmp_path / "detailed_results_gsm8k_20240101_100000.jsonl", [])
    meta, _ = load_triple(tmp_path, "bs64")
    assert meta["system_environment"]["batch_size"] == 64

def test_load_triple_returns_none_when_no_metadata(tmp_path):
    from analyze import load_triple
    _write_jsonl(tmp_path / "detailed_results_gsm8k_20240101_100000.jsonl", [])
    meta, records = load_triple(tmp_path, "bs32")
    assert meta is None and records is None

def test_load_triple_returns_none_when_no_records(tmp_path):
    from analyze import load_triple
    _write_metadata(tmp_path / "metadata_gsm8k_20240101_100000.json")
    meta, records = load_triple(tmp_path, "bs32")
    assert meta is None and records is None

# ── normalize_records ──────────────────────────────────────────────────────────

def test_normalize_prefill_record_uses_ttft():
    from analyze import normalize_records
    records = [{"forward_mode": "prefill", "ttft": 0.05, "batch_size": 1,
                "seq_lens_sum": 100, "expert_activation": 0.3}]
    result = normalize_records(records)
    assert result[0]["latency"] == 0.05

def test_normalize_decoding_record_uses_tpot():
    from analyze import normalize_records
    records = [{"forward_mode": "decoding", "tpot": 0.02, "batch_size": 32,
                "seq_lens_sum": 0, "expert_activation": 0.5}]
    result = normalize_records(records)
    assert result[0]["latency"] == 0.02

def test_normalize_zero_ttft_does_not_fall_through_to_tpot():
    from analyze import normalize_records
    records = [{"forward_mode": "prefill", "ttft": 0.0, "tpot": 0.02,
                "batch_size": 1, "seq_lens_sum": 100, "expert_activation": 0.3}]
    result = normalize_records(records)
    assert result[0]["latency"] == 0.0

def test_normalize_does_not_mutate_originals():
    from analyze import normalize_records
    original = [{"forward_mode": "decoding", "tpot": 0.01, "batch_size": 1,
                 "seq_lens_sum": 0, "expert_activation": 0.0}]
    normalize_records(original)
    assert "latency" not in original[0]

def test_normalize_prefill_preserves_native_latency():
    from analyze import normalize_records
    records = [{"forward_mode": "prefill", "latency": 0.07,
                "batch_size": 1, "seq_lens_sum": 100}]
    result = normalize_records(records)
    assert result[0]["latency"] == 0.07

# ── compute_smfu_smbu ──────────────────────────────────────────────────────────

def _make_meta(gpu_type="NVIDIA-H100-HBM3-80GB", num_gpus=1,
               model_name="org/model", precision="bfloat16", batch_size=32):
    return {
        "hardware": {"gpu_type": gpu_type, "num_gpus": num_gpus},
        "model_config": {"model_name": model_name, "precision": precision},
        "system_environment": {"batch_size": batch_size},
    }

def _make_records():
    return [{"forward_mode": "decoding", "tpot": 0.02, "latency": 0.02,
             "batch_size": 32, "seq_lens_sum": 0, "expert_activation": 0.5,
             "gpu_raw_type": "NVIDIA-H100-HBM3-80GB", "gpu_num": 1}]

def _mock_retriever(n_layers=32, d_model=2048, n_attn_heads=16, d_head=128,
                    n_kv_heads=8, d_ff=4096, precision_bytes=2.0):
    r = MagicMock()
    r.hf_config = MagicMock()
    r.get_model_precision_bytes.return_value = precision_bytes
    r.get_architecture_info.return_value = {
        "hidden_size": d_model, "num_hidden_layers": n_layers}
    r.get_moe_info.return_value = {"ffn_dim": d_ff}
    r.get_attention_info.return_value = {
        "num_attention_heads": n_attn_heads,
        "num_key_value_heads": n_kv_heads,
        "head_dim": d_head,
    }
    return r

def test_compute_smfu_smbu_returns_percentages():
    from analyze import compute_smfu_smbu
    meta = _make_meta()
    records = _make_records()
    fake_result = {
        "prefill_smfu": 0.42, "prefill_smbu": 0.55,
        "prefill_tp": 1234.0,
        "decoding_smfu": 0.30, "decoding_smbu": 0.88,
    }
    with patch("analyze.HFModelInfoRetriever", return_value=_mock_retriever()), \
         patch("analyze._calculate_continuous_metrics", return_value=fake_result):
        result = compute_smfu_smbu(records, meta)
    assert result["prefill_smfu"] == pytest.approx(42.0)
    assert result["prefill_smbu"] == pytest.approx(55.0)
    assert result["prefill_tokens_per_sec"] == pytest.approx(1234.0)
    assert result["decoding_smfu"] == pytest.approx(30.0)
    assert result["decoding_smbu"] == pytest.approx(88.0)

def test_compute_smfu_smbu_returns_none_on_key_error():
    from analyze import compute_smfu_smbu
    with patch("analyze.HFModelInfoRetriever", return_value=_mock_retriever()), \
         patch("analyze._calculate_continuous_metrics", side_effect=KeyError("NVIDIA-RTX-Unknown")):
        result = compute_smfu_smbu(_make_records(), _make_meta())
    assert result is None

def test_compute_smfu_smbu_returns_none_on_empty_records():
    from analyze import compute_smfu_smbu
    with patch("analyze.HFModelInfoRetriever", return_value=_mock_retriever()), \
         patch("analyze._calculate_continuous_metrics", return_value={}):
        result = compute_smfu_smbu([], _make_meta())
    assert result is None

def test_resolve_gpu_raw_type_falls_back_from_unknown_record():
    from analyze import resolve_gpu_raw_type
    records = [{"gpu_raw_type": "Unknown"}]
    meta = _make_meta(gpu_type="NVIDIA-H100-NVL-94GB")
    assert resolve_gpu_raw_type(records, meta) == "NVIDIA-H100-NVL-94GB"

def test_resolve_gpu_raw_type_env_override(monkeypatch):
    from analyze import resolve_gpu_raw_type
    monkeypatch.setenv("ANALYZE_GPU_TYPE", "NVIDIA-H100-NVL-94GB")
    records = [{"gpu_raw_type": "Unknown"}]
    meta = _make_meta(gpu_type="Unknown")
    assert resolve_gpu_raw_type(records, meta) == "NVIDIA-H100-NVL-94GB"

# ── aggregate_results ─────────────────────────────────────────────────────────

def test_aggregate_single_entry():
    from analyze import aggregate_results
    raw = [("model_a", 32, {"prefill_smfu": 40.0, "prefill_smbu": 50.0,
                             "prefill_raw_tflops": 400.0})]
    agg = aggregate_results(raw)
    assert agg["model_a"][32]["prefill_smfu"] == pytest.approx(40.0)

def test_aggregate_averages_across_datasets():
    from analyze import aggregate_results
    raw = [
        ("model_a", 32, {"prefill_smfu": 40.0, "prefill_smbu": 50.0,
                          "prefill_raw_tflops": 400.0}),
        ("model_a", 32, {"prefill_smfu": 60.0, "prefill_smbu": 70.0,
                          "prefill_raw_tflops": 600.0}),
    ]
    agg = aggregate_results(raw)
    assert agg["model_a"][32]["prefill_smfu"] == pytest.approx(50.0)
    assert agg["model_a"][32]["prefill_smbu"] == pytest.approx(60.0)

def test_aggregate_multiple_models_and_batch_sizes():
    from analyze import aggregate_results
    raw = [
        ("model_a", 1,  {"prefill_smfu": 10.0, "prefill_smbu": 90.0,
                          "prefill_raw_tflops": 100.0}),
        ("model_a", 32, {"prefill_smfu": 40.0, "prefill_smbu": 60.0,
                          "prefill_raw_tflops": 400.0}),
        ("model_b", 1,  {"prefill_smfu": 20.0, "prefill_smbu": 80.0,
                          "prefill_raw_tflops": 200.0}),
    ]
    agg = aggregate_results(raw)
    assert set(agg.keys()) == {"model_a", "model_b"}
    assert set(agg["model_a"].keys()) == {1, 32}

# ── plot_single_metric ────────────────────────────────────────────────────────

def test_plot_single_metric_saves_file(tmp_path):
    from analyze import plot_single_metric
    bs_data = {1: {"prefill_smfu": 10.0},
               32: {"prefill_smfu": 40.0}}
    out = tmp_path / "smfu_model_a.png"
    plot_single_metric("model_a", bs_data, "Prefill S-MFU (%)",
                       "prefill_smfu", out)
    assert out.exists()

def test_plot_single_metric_overwrites_existing(tmp_path):
    from analyze import plot_single_metric
    bs_data = {1: {"prefill_smfu": 10.0}}
    out = tmp_path / "smfu_model_a.png"
    out.write_bytes(b"old content")
    plot_single_metric("model_a", bs_data, "Prefill S-MFU (%)",
                       "prefill_smfu", out)
    assert out.stat().st_size != len(b"old content")

def test_plot_single_metric_no_data_does_nothing(tmp_path):
    from analyze import plot_single_metric
    out = tmp_path / "smfu_empty.png"
    plot_single_metric("empty", {}, "Prefill S-MFU (%)",
                       "prefill_smfu", out)
    assert not out.exists()

def test_plot_single_metric_with_legacy_keys(tmp_path):
    from analyze import plot_single_metric
    bs_data = {
        1: {"prefill_smfu": 10.0, "prefill_smfu_legacy": 12.0},
        32: {"prefill_smfu": 40.0, "prefill_smfu_legacy": 45.0},
    }
    out = tmp_path / "smfu_qwen3_next.png"
    plot_single_metric("Qwen3-Next-80B", bs_data, "Prefill S-MFU (%)",
                       "prefill_smfu", out, "prefill_smfu_legacy")
    assert out.exists()


# ── normalize_records forward_mode dispatch ──────────────────────────────────

def test_normalize_decoding_with_both_ttft_and_tpot_uses_tpot():
    """Decoding records that have both ttft and tpot must use tpot."""
    from analyze import normalize_records
    records = [{"forward_mode": "decoding", "ttft": 0.5, "tpot": 0.02,
                "batch_size": 32, "seq_lens_sum": 0, "expert_activation": 0.5}]
    result = normalize_records(records)
    assert result[0]["latency"] == 0.02

def test_normalize_prefill_with_both_ttft_and_tpot_uses_ttft():
    """Prefill records that have both ttft and tpot must use ttft."""
    from analyze import normalize_records
    records = [{"forward_mode": "prefill", "ttft": 0.05, "tpot": 0.02,
                "batch_size": 1, "seq_lens_sum": 100, "expert_activation": 0.3}]
    result = normalize_records(records)
    assert result[0]["latency"] == 0.05

def test_normalize_decoding_without_tpot_falls_back_to_ttft():
    """Decoding records missing tpot should fall back to ttft."""
    from analyze import normalize_records
    records = [{"forward_mode": "decoding", "ttft": 0.5,
                "batch_size": 32, "seq_lens_sum": 0, "expert_activation": 0.5}]
    result = normalize_records(records)
    assert result[0]["latency"] == 0.5

def test_normalize_unknown_mode_uses_tpot():
    """Records with unknown forward_mode should use tpot (decoding path)."""
    from analyze import normalize_records
    records = [{"forward_mode": "unknown", "tpot": 0.03,
                "batch_size": 1, "seq_lens_sum": 0, "expert_activation": 0.0}]
    result = normalize_records(records)
    assert result[0]["latency"] == 0.03


# ── compute_smfu_smbu raw TFLOPS ─────────────────────────────────────────────

def test_compute_smfu_smbu_includes_raw_tflops():
    from analyze import compute_smfu_smbu
    meta = _make_meta()
    records = _make_records()
    fake_result = {
        "prefill_smfu": 0.42, "prefill_smbu": 0.55,
        "decoding_smfu": 0.30, "decoding_smbu": 0.88,
    }
    with patch("analyze.HFModelInfoRetriever", return_value=_mock_retriever()), \
         patch("analyze._calculate_continuous_metrics", return_value=fake_result), \
         patch("analyze.get_peak_flops", return_value=1979e12):
        result = compute_smfu_smbu(records, meta)
    assert "prefill_raw_tflops" in result
    assert "decoding_raw_tflops" in result
    # raw_tflops = smfu_fraction * num_gpus * peak_tflops / 2
    expected_prefill = 0.42 * 1 * 1979 / 2
    expected_decode = 0.30 * 1 * 1979 / 2
    assert result["prefill_raw_tflops"] == pytest.approx(expected_prefill, rel=1e-3)
    assert result["decoding_raw_tflops"] == pytest.approx(expected_decode, rel=1e-3)


# ── compute_smfu_smbu Qwen3-Next legacy comparison ───────────────────────────

def test_compute_smfu_smbu_qwen3_next_includes_legacy():
    from analyze import compute_smfu_smbu
    meta = _make_meta(model_name="Qwen/Qwen3-Next-80B-A3B-Instruct")
    records = _make_records()
    fake_primary = {
        "prefill_smfu": 0.20, "prefill_smbu": 0.30,
        "decoding_smfu": 0.15, "decoding_smbu": 0.40,
    }
    fake_legacy = {
        "prefill_smfu": 0.50, "prefill_smbu": 0.60,
        "decoding_smfu": 0.45, "decoding_smbu": 0.70,
    }
    call_count = {"n": 0}
    original_results = [fake_primary, fake_legacy]

    def fake_run_metrics(records, model_name, *args, **kwargs):
        idx = call_count["n"]
        call_count["n"] += 1
        return original_results[idx]

    with patch("analyze._run_metrics", side_effect=fake_run_metrics), \
         patch("analyze.get_peak_flops", return_value=1979e12):
        result = compute_smfu_smbu(records, meta)

    # Primary metrics (Qwen3-Next correct path)
    assert result["prefill_smfu"] == pytest.approx(20.0)
    assert result["decoding_smfu"] == pytest.approx(15.0)
    # Legacy metrics (Qwen3 path for comparison)
    assert result["prefill_smfu_legacy"] == pytest.approx(50.0)
    assert result["prefill_smbu_legacy"] == pytest.approx(60.0)
    assert result["decoding_smfu_legacy"] == pytest.approx(45.0)


def test_compute_smfu_smbu_non_qwen3_next_has_no_legacy():
    from analyze import compute_smfu_smbu
    meta = _make_meta(model_name="Qwen/Qwen3-30B-A3B")
    records = _make_records()
    fake_result = {
        "prefill_smfu": 0.42, "prefill_smbu": 0.55,
        "decoding_smfu": 0.30, "decoding_smbu": 0.88,
    }
    with patch("analyze._run_metrics", return_value=fake_result), \
         patch("analyze.get_peak_flops", return_value=1979e12):
        result = compute_smfu_smbu(records, meta)

    assert "prefill_smfu_legacy" not in result


# ── aggregate_results with FLOPS keys ────────────────────────────────────────

def test_aggregate_includes_raw_tflops():
    from analyze import aggregate_results
    raw = [
        ("model_a", 32, {"prefill_smfu": 40.0, "prefill_smbu": 50.0,
                          "prefill_raw_tflops": 400.0}),
        ("model_a", 32, {"prefill_smfu": 60.0, "prefill_smbu": 70.0,
                          "prefill_raw_tflops": 600.0}),
    ]
    agg = aggregate_results(raw)
    assert agg["model_a"][32]["prefill_raw_tflops"] == pytest.approx(500.0)


# ── plot_single_metric for raw TFLOPS ─────────────────────────────────────────

def test_plot_single_metric_tflops_saves_file(tmp_path):
    from analyze import plot_single_metric
    bs_data = {
        1: {"prefill_raw_tflops": 10.0},
        32: {"prefill_raw_tflops": 200.0},
    }
    out = tmp_path / "raw_flops_model_a.png"
    plot_single_metric("model_a", bs_data, "Prefill TFLOPS",
                       "prefill_raw_tflops", out)
    assert out.exists()

def test_plot_single_metric_tflops_no_data(tmp_path):
    from analyze import plot_single_metric
    out = tmp_path / "raw_flops_empty.png"
    plot_single_metric("empty", {}, "Prefill TFLOPS",
                       "prefill_raw_tflops", out)
    assert not out.exists()


# ── aggregate_by_dataset ──────────────────────────────────────────────────────

def test_aggregate_by_dataset_groups_by_dataset_then_slug_then_bs():
    from analyze import aggregate_by_dataset
    raw = [
        ("model_a", 1,  "longbench_v2", {"prefill_smfu": 30.0}),
        ("model_a", 32, "longbench_v2", {"prefill_smfu": 40.0}),
        ("model_b", 1,  "longbench_v2", {"prefill_smfu": 20.0}),
        ("model_a", 1,  "longbench_v2_maxctx", {"prefill_smfu": 55.0}),
    ]
    agg = aggregate_by_dataset(raw)
    assert set(agg.keys()) == {"longbench_v2", "longbench_v2_maxctx"}
    assert agg["longbench_v2"]["model_a"][1]["prefill_smfu"] == pytest.approx(30.0)
    assert agg["longbench_v2"]["model_a"][32]["prefill_smfu"] == pytest.approx(40.0)
    assert agg["longbench_v2"]["model_b"][1]["prefill_smfu"] == pytest.approx(20.0)
    assert agg["longbench_v2_maxctx"]["model_a"][1]["prefill_smfu"] == pytest.approx(55.0)

def test_aggregate_by_dataset_averages_duplicate_cells():
    """Same (slug, bs, dataset) repeated: values must be averaged, not overwritten."""
    from analyze import aggregate_by_dataset
    raw = [
        ("model_a", 1, "longbench_v2", {"prefill_smfu": 40.0}),
        ("model_a", 1, "longbench_v2", {"prefill_smfu": 60.0}),
    ]
    agg = aggregate_by_dataset(raw)
    assert agg["longbench_v2"]["model_a"][1]["prefill_smfu"] == pytest.approx(50.0)


def test_aggregate_by_dataset_keeps_metadata_strings():
    from analyze import aggregate_by_dataset
    raw = [
        ("model_a", 1, "longbench_v2",
         {"prefill_smfu": 40.0, "runner_mode": "moe_cap.openai_api_profile"}),
        ("model_a", 1, "longbench_v2",
         {"prefill_smfu": 60.0, "runner_mode": "moe_cap.openai_api_profile"}),
    ]
    agg = aggregate_by_dataset(raw)
    cell = agg["longbench_v2"]["model_a"][1]
    assert cell["prefill_smfu"] == pytest.approx(50.0)
    assert cell["runner_mode"] == "moe_cap.openai_api_profile"


# ── plot_metric_per_dataset ───────────────────────────────────────────────────

def test_plot_metric_per_dataset_saves_file(tmp_path):
    from analyze import plot_metric_per_dataset
    per_slug = {
        "model_a": {1: {"prefill_smfu": 10.0}, 32: {"prefill_smfu": 40.0}},
        "model_b": {1: {"prefill_smfu": 20.0}, 32: {"prefill_smfu": 50.0}},
    }
    out = tmp_path / "smfu_longbench_v2.png"
    plot_metric_per_dataset("longbench_v2", per_slug, "Prefill S-MFU (%)",
                            "prefill_smfu", out)
    assert out.exists()

def test_plot_metric_per_dataset_no_models_does_nothing(tmp_path):
    from analyze import plot_metric_per_dataset
    out = tmp_path / "smfu_empty.png"
    plot_metric_per_dataset("empty", {}, "Prefill S-MFU (%)",
                            "prefill_smfu", out)
    assert not out.exists()

def test_plot_metric_per_dataset_with_legacy_line(tmp_path):
    from analyze import plot_metric_per_dataset
    per_slug = {
        "qwen3_next_80b": {
            64:  {"prefill_smfu": 14.9, "prefill_smfu_legacy": 14.9},
            128: {"prefill_smfu": 15.1, "prefill_smfu_legacy": 15.1},
        },
    }
    out = tmp_path / "smfu_longbench_v2.png"
    plot_metric_per_dataset("longbench_v2", per_slug, "Prefill S-MFU (%)",
                            "prefill_smfu", out, legacy_key="prefill_smfu_legacy")
    assert out.exists()


def test_plot_smfu_smbu_for_model_saves_file(tmp_path):
    from analyze import plot_smfu_smbu_for_model
    bs_data = {
        8: {"prefill_smfu": 29.6, "prefill_smbu": 3.0},
        32: {"prefill_smfu": 27.1, "prefill_smbu": 2.9},
        128: {"prefill_smfu": 26.0, "prefill_smbu": 2.9},
    }
    out = tmp_path / "qwen1_5_moe_smfu_smbu_longbench_v2.png"
    plot_smfu_smbu_for_model("qwen1_5_moe", "longbench_v2", bs_data, out)
    assert out.exists()


# ── write_raw_values ──────────────────────────────────────────────────────────

def test_write_raw_values_emits_every_cell(tmp_path):
    from analyze import write_raw_values
    import csv
    raw = [
        ("model_a", 1, "longbench_v2", {"prefill_smfu": 30.0,
                                        "prefill_smbu": 1.2,
                                         "prefill_raw_tflops": 300.0}),
        ("model_b", 1, "longbench_v2", {"prefill_smfu": 20.0,
                                         "prefill_smbu": 1.1,
                                         "prefill_raw_tflops": 200.0}),
        ("model_a", 1, "longbench_v2_maxctx", {"prefill_smfu": 55.0,
                                                 "prefill_smbu": 2.0,
                                                 "prefill_raw_tflops": 550.0}),
    ]
    out = tmp_path / "raw_values.csv"
    write_raw_values(raw, out)
    with out.open(newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 3
    assert {r["dataset"] for r in rows} == {"longbench_v2", "longbench_v2_maxctx"}
    assert {r["slug"] for r in rows} == {"model_a", "model_b"}
    assert {float(r["prefill_smfu"]) for r in rows} == {20.0, 30.0, 55.0}


def test_write_raw_values_includes_run_metadata_columns(tmp_path):
    from analyze import write_raw_values
    import csv
    raw = [
        ("qwen1_5_moe", 2, "batched_prefill",
         {"prefill_smfu": 12.0, "prefill_smbu": 4.0,
          "runner_mode": "moe_cap.openai_api_profile",
          "chunked_prefill_size": 32768,
          "max_prefill_tokens": 32768,
          "mem_fraction_static": 0.9}),
    ]
    out = tmp_path / "raw_values.csv"
    write_raw_values(raw, out)
    with out.open(newline="") as f:
        rows = list(csv.DictReader(f))

    assert rows[0]["runner_mode"] == "moe_cap.openai_api_profile"
    assert rows[0]["chunked_prefill_size"] == "32768"
    assert rows[0]["max_prefill_tokens"] == "32768"
    assert rows[0]["mem_fraction_static"] == "0.9"


def test_write_raw_values_includes_legacy_columns_when_present(tmp_path):
    from analyze import write_raw_values
    import csv
    raw = [
        ("qwen3_next_80b", 64, "longbench_v2",
         {"prefill_smfu": 14.9, "prefill_smbu": 6.6,
          "prefill_raw_tflops": 295.4,
          "prefill_smfu_legacy": 14.9, "prefill_smbu_legacy": 6.6,
          "prefill_raw_tflops_legacy": 295.4}),
    ]
    out = tmp_path / "raw_values.csv"
    write_raw_values(raw, out)
    with out.open(newline="") as f:
        reader = csv.DictReader(f)
        assert "prefill_smfu_legacy" in reader.fieldnames
        assert "prefill_raw_tflops_legacy" in reader.fieldnames

def test_write_raw_values_skips_absent_legacy_columns(tmp_path):
    """Non-Qwen3-Next rows have no legacy keys — the column header should not appear."""
    from analyze import write_raw_values
    import csv
    raw = [
        ("qwen1_5_moe", 1, "longbench_v2",
         {"prefill_smfu": 30.0, "prefill_smbu": 1.2,
          "prefill_raw_tflops": 300.0}),
    ]
    out = tmp_path / "raw_values.csv"
    write_raw_values(raw, out)
    with out.open(newline="") as f:
        reader = csv.DictReader(f)
        assert "prefill_smfu_legacy" not in reader.fieldnames
