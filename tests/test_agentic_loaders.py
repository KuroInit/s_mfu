import json


def test_swe_bench_loader_formats_local_jsonl(tmp_path, monkeypatch):
    from s_mfu.agentic_loaders import SWEBenchLoader

    path = tmp_path / "swe.jsonl"
    row = {
        "repo": "sqlfluff/sqlfluff",
        "instance_id": "sqlfluff__sqlfluff-1625",
        "base_commit": "14e1a23",
        "problem_statement": "Rule L031 incorrectly triggers.",
        "hints_text": "Look at join aliases.",
        "FAIL_TO_PASS": '["test_file.py::test_case"]',
        "patch": "diff --git a/file.py b/file.py",
    }
    path.write_text(json.dumps(row) + "\n")

    config = type("Config", (), {"num_samples": 1})()
    monkeypatch.setenv("S_MFU_SWEBENCH_PATH", str(path))
    loader = SWEBenchLoader(config)

    prompt = loader.get_input()[0]
    assert "Repository: sqlfluff/sqlfluff" in prompt
    assert "Instance ID: sqlfluff__sqlfluff-1625" in prompt
    assert "Rule L031 incorrectly triggers." in prompt
    assert "Return only the patch." in prompt
    assert loader.get_target() == ["diff --git a/file.py b/file.py"]
    assert loader.get_uids() == ["sqlfluff__sqlfluff-1625"]


def test_swe_bench_default_uses_lite_hf_dataset(monkeypatch):
    import s_mfu.agentic_loaders as agentic_loaders

    calls = []

    def fake_load_hf_dataset(*args, **kwargs):
        calls.append((args, kwargs))
        return [
            {
                "repo": "pvlib/pvlib-python",
                "instance_id": "pvlib__pvlib-python-1707",
                "base_commit": "40e9e97",
                "problem_statement": "Fix physical IAM.",
                "patch": "",
            }
        ]

    monkeypatch.delenv("S_MFU_SWEBENCH_PATH", raising=False)
    monkeypatch.delenv("S_MFU_SWEBENCH_HF_DATASET", raising=False)
    monkeypatch.setattr(agentic_loaders, "_load_hf_dataset", fake_load_hf_dataset)
    config = type("Config", (), {"dataset_split": "test", "num_samples": 1})()

    loader = agentic_loaders.SWEBenchLoader(config)

    assert "Fix physical IAM." in loader.get_input()[0]
    assert calls[0][0] == ("princeton-nlp/SWE-bench_Lite",)
    assert calls[0][1]["split"] == "test"
    assert calls[0][1]["streaming"] is True
