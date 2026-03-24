import json
import pytest
from pathlib import Path

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
