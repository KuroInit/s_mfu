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
        "decoding_smfu": 0.30, "decoding_smbu": 0.88,
    }
    with patch("analyze.HFModelInfoRetriever", return_value=_mock_retriever()), \
         patch("analyze._calculate_continuous_metrics", return_value=fake_result):
        result = compute_smfu_smbu(records, meta)
    assert result["prefill_smfu"] == pytest.approx(42.0)
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

# ── aggregate_results ─────────────────────────────────────────────────────────

def test_aggregate_single_entry():
    from analyze import aggregate_results
    raw = [("model_a", 32, {"prefill_smfu": 40.0, "decoding_smfu": 30.0,
                             "prefill_smbu": 50.0, "decoding_smbu": 80.0})]
    agg = aggregate_results(raw)
    assert agg["model_a"][32]["prefill_smfu"] == pytest.approx(40.0)

def test_aggregate_averages_across_datasets():
    from analyze import aggregate_results
    raw = [
        ("model_a", 32, {"prefill_smfu": 40.0, "decoding_smfu": 30.0,
                          "prefill_smbu": 50.0, "decoding_smbu": 80.0}),
        ("model_a", 32, {"prefill_smfu": 60.0, "decoding_smfu": 50.0,
                          "prefill_smbu": 70.0, "decoding_smbu": 60.0}),
    ]
    agg = aggregate_results(raw)
    assert agg["model_a"][32]["prefill_smfu"] == pytest.approx(50.0)
    assert agg["model_a"][32]["decoding_smbu"] == pytest.approx(70.0)

def test_aggregate_multiple_models_and_batch_sizes():
    from analyze import aggregate_results
    raw = [
        ("model_a", 1,  {"prefill_smfu": 10.0, "decoding_smfu": 5.0,
                          "prefill_smbu": 90.0, "decoding_smbu": 85.0}),
        ("model_a", 32, {"prefill_smfu": 40.0, "decoding_smfu": 35.0,
                          "prefill_smbu": 60.0, "decoding_smbu": 55.0}),
        ("model_b", 1,  {"prefill_smfu": 20.0, "decoding_smfu": 15.0,
                          "prefill_smbu": 80.0, "decoding_smbu": 75.0}),
    ]
    agg = aggregate_results(raw)
    assert set(agg.keys()) == {"model_a", "model_b"}
    assert set(agg["model_a"].keys()) == {1, 32}

# ── plot_metric ───────────────────────────────────────────────────────────────

def test_plot_metric_saves_file(tmp_path):
    from analyze import plot_metric
    data = {
        "model_a": {1: {"prefill_smfu": 10.0, "decoding_smfu": 5.0},
                    32: {"prefill_smfu": 40.0, "decoding_smfu": 35.0}},
    }
    out = tmp_path / "smfu.png"
    plot_metric(data, "smfu", prefill_key="prefill_smfu",
                decoding_key="decoding_smfu", out_path=out)
    assert out.exists()

def test_plot_metric_overwrites_existing(tmp_path):
    from analyze import plot_metric
    data = {"model_a": {1: {"prefill_smfu": 10.0, "decoding_smfu": 5.0}}}
    out = tmp_path / "smfu.png"
    out.write_bytes(b"old content")
    plot_metric(data, "smfu", prefill_key="prefill_smfu",
                decoding_key="decoding_smfu", out_path=out)
    assert out.stat().st_size != len(b"old content")

def test_plot_metric_exits_on_empty_data(tmp_path):
    from analyze import plot_metric
    with pytest.raises(SystemExit):
        plot_metric({}, "smfu", prefill_key="prefill_smfu",
                    decoding_key="decoding_smfu", out_path=tmp_path / "smfu.png")
