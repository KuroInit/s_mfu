"""Harness-side agentic dataset loaders for MoE-CAP."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable

from moe_cap.data_loader.base_data_loader import DataLoader

from s_mfu.chat_loaders import _load_hf_dataset


SWEBENCH_DATASET = "princeton-nlp/SWE-bench_Lite"


def _read_json_or_jsonl(path: str) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"agentic dataset file not found: {source}")
    if source.suffix.lower() == ".jsonl":
        rows = []
        with source.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    with source.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        raise ValueError(f"unsupported agentic dataset JSON shape in {source}")
    if _looks_like_swe_bench_example(payload):
        return [payload]
    for key in ("data", "rows", "examples", "instances"):
        if isinstance(payload.get(key), list):
            return payload[key]
    raise ValueError(f"unsupported agentic dataset JSON shape in {source}")


def _looks_like_swe_bench_example(row: dict[str, Any]) -> bool:
    return any(
        row.get(key)
        for key in (
            "problem_statement",
            "issue",
            "repo",
            "instance_id",
            "base_commit",
            "patch",
            "test_patch",
        )
    )


def _format_swe_bench_prompt(row: dict[str, Any]) -> str:
    repo = row.get("repo") or ""
    instance_id = row.get("instance_id") or ""
    base_commit = row.get("base_commit") or ""
    problem = row.get("problem_statement") or row.get("issue") or ""
    hints = row.get("hints_text") or ""
    fail_to_pass = row.get("FAIL_TO_PASS") or row.get("fail_to_pass") or ""

    parts = [
        "You are a software engineering agent. Produce a minimal git patch "
        "that fixes the reported issue.",
        f"Repository: {repo}",
        f"Instance ID: {instance_id}",
        f"Base commit: {base_commit}",
        "",
        "Issue:",
        str(problem).strip(),
    ]
    if hints:
        parts.extend(["", "Hints:", str(hints).strip()])
    if fail_to_pass:
        parts.extend(["", "Failing tests:", str(fail_to_pass).strip()])
    parts.extend(["", "Return only the patch."])
    return "\n".join(part for part in parts if part is not None)


class SWEBenchLoader(DataLoader):
    env_path_var = "S_MFU_SWEBENCH_PATH"
    env_hf_dataset_var = "S_MFU_SWEBENCH_HF_DATASET"
    env_hf_config_var = "S_MFU_SWEBENCH_HF_CONFIG"
    hf_dataset = SWEBENCH_DATASET

    def __init__(self, config):
        super().__init__(config)
        self.prompts: list[str] = []
        self.targets: list[str] = []
        self.uids: list[str] = []
        self._process(self._load_rows(config))

    def _load_rows(self, config) -> Iterable[dict[str, Any]]:
        path = os.environ.get(self.env_path_var)
        if path:
            return _read_json_or_jsonl(path)

        hf_dataset = os.environ.get(self.env_hf_dataset_var) or self.hf_dataset
        hf_config = os.environ.get(self.env_hf_config_var) or None
        split = getattr(config, "dataset_split", "test") or "test"
        return _load_hf_dataset(
            hf_dataset,
            split=split,
            dataset_config=hf_config,
            streaming=True,
        )

    def _process(self, rows: Iterable[dict[str, Any]]) -> None:
        limit = getattr(self.config, "num_samples", None)
        for row in rows:
            row = dict(row)
            if not row.get("problem_statement") and not row.get("issue"):
                continue
            prompt = _format_swe_bench_prompt(row)
            if not prompt.strip():
                continue
            self.prompts.append(prompt)
            self.targets.append(str(row.get("patch") or ""))
            self.uids.append(str(row.get("instance_id") or len(self.uids)))
            if limit is not None and limit > 0 and len(self.prompts) >= limit:
                break

    def get_input(self):
        return self.prompts

    def get_target(self):
        return self.targets

    def get_uids(self):
        return self.uids


def register_agentic_loaders() -> None:
    from moe_cap.data_loader import loader_registry

    loader_registry._REGISTRY["swe_bench"] = (SWEBenchLoader, 4096)
