"""Harness-side chat dataset loaders for MoE-CAP.

These loaders are registered by ``s_mfu.moe_cap_runner`` at process startup so
the harness can support chat traces without editing the MoE-CAP checkout.
"""

from __future__ import annotations

import json
import os
import csv
from pathlib import Path
from typing import Any, Iterable, Iterator

from moe_cap.data_loader.base_data_loader import DataLoader


SHAREGPT_DATASET = "anon8231489123/ShareGPT_Vicuna_unfiltered"
SHAREGPT_FILE = "ShareGPT_V3_unfiltered_cleaned_split.json"


def _read_json_or_jsonl(path: str) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"chat dataset file not found: {source}")
    if source.suffix.lower() == ".csv":
        with source.open("r", encoding="utf-8", newline="") as f:
            return [dict(row) for row in csv.DictReader(f)]
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
        raise ValueError(f"unsupported chat dataset JSON shape in {source}")
    if _looks_like_chat_example(payload):
        return [payload]
    for key in ("data", "rows", "examples"):
        if isinstance(payload.get(key), list):
            return payload[key]
    raise ValueError(f"unsupported chat dataset JSON shape in {source}")


def _load_hf_dataset(
    dataset_name: str,
    split: str,
    dataset_config: str | None = None,
    data_files: str | list[str] | dict[str, str] | None = None,
    streaming: bool = True,
):
    from datasets import load_dataset

    kwargs: dict[str, Any] = {"split": split, "streaming": streaming}
    if data_files is not None:
        kwargs["data_files"] = data_files
    if dataset_config:
        return load_dataset(dataset_name, dataset_config, **kwargs)
    return load_dataset(dataset_name, **kwargs)


def _iter_limited(
    rows: Iterable[dict[str, Any]],
    limit: int | None,
) -> Iterator[dict[str, Any]]:
    for i, row in enumerate(rows):
        if limit is not None and limit > 0 and i >= limit:
            break
        yield dict(row)


def _message_text(message: dict[str, Any]) -> str:
    content = message.get("content") or message.get("value") or message.get("text") or ""
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    return str(content)


def _message_role(message: dict[str, Any]) -> str:
    role = message.get("role") or message.get("from") or message.get("speaker") or "user"
    role = str(role).lower()
    if role in {"human", "user"}:
        return "user"
    if role in {"gpt", "assistant", "bot"}:
        return "assistant"
    if role == "system":
        return "system"
    return role


def _looks_like_chat_example(example: dict[str, Any]) -> bool:
    if any(
        key in example
        for key in (
            "messages",
            "conversation",
            "conversations",
            "turns",
            "prompt",
            "input",
            "query",
            "question",
            "text",
            "instruction",
            "input_ids",
            "prompt_token_ids",
            "new_input_ids",
            "ContextTokens",
            "GeneratedTokens",
            "context_tokens",
            "generated_tokens",
        )
    ):
        return True
    meta = example.get("meta")
    return isinstance(meta, dict) and any(
        key in meta for key in ("input_length", "output_length")
    )


def _extract_messages(example: dict[str, Any]) -> list[dict[str, str]]:
    raw_messages = (
        example.get("messages")
        or example.get("conversation")
        or example.get("conversations")
        or example.get("turns")
    )
    if isinstance(raw_messages, list):
        messages = []
        for raw in raw_messages:
            if isinstance(raw, dict):
                content = _message_text(raw)
                if content:
                    messages.append({"role": _message_role(raw), "content": content})
            elif isinstance(raw, str):
                content = raw.strip()
                if content:
                    messages.append({"role": "user", "content": content})
        if messages:
            return messages

    instruction = example.get("instruction")
    input_text = example.get("input")
    if instruction or input_text:
        prompt = "\n".join(str(x) for x in (instruction, input_text) if x)
        return [{"role": "user", "content": prompt}]

    for key in ("prompt", "input", "query", "question", "text"):
        if example.get(key):
            return [{"role": "user", "content": str(example[key])}]
    return []


def _split_system_prompt(messages: list[dict[str, str]]) -> tuple[str, str]:
    system_parts = [m["content"] for m in messages if m["role"] == "system"]
    system = "\n".join(system_parts)
    body = [m for m in messages if m["role"] != "system"]

    if not body:
        return system, ""
    if len(body) == 1 and body[0]["role"] == "user":
        return system, body[0]["content"]

    transcript = []
    for message in body:
        role = "Assistant" if message["role"] == "assistant" else "User"
        transcript.append(f"{role}: {message['content']}")
    return system, "\n".join(transcript)


def _prompt_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """Return request messages, excluding assistant answer text at the end."""
    prompt = list(messages)
    while prompt and prompt[-1]["role"] == "assistant":
        prompt.pop()
    return prompt


def _messages_to_text(messages: list[dict[str, str]]) -> str:
    _, prompt = _split_system_prompt(messages)
    return prompt


def _coerce_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _first_int(row: dict[str, Any], keys: tuple[str, ...]) -> int | None:
    lower = {str(key).lower(): value for key, value in row.items()}
    for key in keys:
        value = lower.get(key.lower())
        if value is not None:
            parsed = _coerce_int(value)
            if parsed is not None:
                return parsed
    return None


def _token_ids_from_value(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, list):
        return [int(token) for token in value]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("["):
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [int(token) for token in parsed]
        return [int(token) for token in text.replace(",", " ").split()]
    return []


def _first_token_ids(row: dict[str, Any]) -> list[int]:
    for key in ("input_ids", "prompt_token_ids", "new_input_ids"):
        if key in row:
            token_ids = _token_ids_from_value(row[key])
            if token_ids:
                return token_ids
    return []


def _repeat_to_length(token_ids: list[int], length: int) -> list[int]:
    if length <= 0:
        return []
    if not token_ids:
        return [1000] * length
    repeats = (length + len(token_ids) - 1) // len(token_ids)
    return (token_ids * repeats)[:length]


def _extract_azure_token_request(row: dict[str, Any]) -> dict[str, Any] | None:
    meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
    flat = {**meta, **row}
    prompt_len = _first_int(
        flat,
        (
            "ContextTokens",
            "context_tokens",
            "input_length",
            "prompt_length",
            "prompt_len",
            "total_input_tokens",
        ),
    )
    output_len = _first_int(
        flat,
        (
            "GeneratedTokens",
            "generated_tokens",
            "output_length",
            "output_len",
            "max_tokens",
            "max_new_tokens",
        ),
    )
    token_ids = _first_token_ids(row)
    if prompt_len is None and token_ids:
        prompt_len = len(token_ids)
    if prompt_len is None or output_len is None or prompt_len <= 0 or output_len <= 0:
        return None
    return {
        "prompt_token_ids": _repeat_to_length(token_ids, prompt_len),
        "prompt_len": prompt_len,
        "max_tokens": output_len,
        "timestamp": row.get("TIMESTAMP") or row.get("timestamp"),
    }


class _ChatTraceLoader(DataLoader):
    env_path_var = ""
    env_hf_dataset_var = ""
    env_hf_config_var = ""
    env_hf_data_files_var = ""
    hf_dataset = ""
    hf_config = None
    hf_data_files = None

    def __init__(self, config):
        super().__init__(config)
        self.system_prompts: list[str] = []
        self.chat_messages: list[list[dict[str, str]]] = []
        self.prompts: list[str] = []
        rows = self._load_rows(config)
        self._process(rows)

    def _load_rows(self, config) -> Iterable[dict[str, Any]]:
        path = os.environ.get(self.env_path_var)
        if path:
            return _read_json_or_jsonl(path)

        configured_hf_dataset = os.environ.get(self.env_hf_dataset_var)
        hf_dataset = configured_hf_dataset or self.hf_dataset
        hf_config = os.environ.get(self.env_hf_config_var) or self.hf_config
        hf_data_files = os.environ.get(self.env_hf_data_files_var) or (
            self.hf_data_files if configured_hf_dataset is None else None
        )
        if not hf_dataset:
            raise ValueError(
                f"{self.env_path_var} must point to a JSON/JSONL chat trace file, "
                f"or {self.env_hf_dataset_var} must name a HuggingFace dataset"
            )

        split = getattr(config, "dataset_split", "train") or "train"
        return _load_hf_dataset(
            hf_dataset,
            split=split,
            dataset_config=hf_config,
            data_files=hf_data_files,
            streaming=True,
        )

    def _process(self, rows: Iterable[dict[str, Any]]) -> None:
        limit = getattr(self.config, "num_samples", None)
        for row in rows:
            messages = _extract_messages(row)
            if not messages:
                continue
            prompt_messages = _prompt_messages(messages)
            prompt = _messages_to_text(prompt_messages)
            if prompt_messages and prompt:
                self.system_prompts.append("")
                self.chat_messages.append(prompt_messages)
                self.prompts.append(prompt)
            if limit is not None and limit > 0 and len(self.prompts) >= limit:
                break

    def get_input(self):
        return self.chat_messages

    def get_target(self):
        return [""] * len(self.prompts)

    def get_uids(self):
        return [str(i) for i in range(len(self.prompts))]


class ShareGPTLoader(_ChatTraceLoader):
    env_path_var = "S_MFU_SHAREGPT_PATH"
    env_hf_dataset_var = "S_MFU_SHAREGPT_HF_DATASET"
    env_hf_config_var = "S_MFU_SHAREGPT_HF_CONFIG"
    env_hf_data_files_var = "S_MFU_SHAREGPT_HF_DATA_FILES"
    hf_dataset = "json"
    hf_data_files = (
        f"https://huggingface.co/datasets/{SHAREGPT_DATASET}/resolve/main/"
        f"{SHAREGPT_FILE}"
    )


class AzureChatLoader(_ChatTraceLoader):
    env_path_var = "S_MFU_AZURE_CHAT_PATH"
    env_hf_dataset_var = "S_MFU_AZURE_CHAT_HF_DATASET"
    env_hf_config_var = "S_MFU_AZURE_CHAT_HF_CONFIG"
    env_hf_data_files_var = "S_MFU_AZURE_CHAT_HF_DATA_FILES"
    hf_dataset = ""

    def __init__(self, config):
        self._input_mode: str | None = None
        super().__init__(config)
        self.max_tokens_by_request = [
            item["max_tokens"] for item in self.chat_messages
            if isinstance(item, dict) and "max_tokens" in item
        ]

    def _process(self, rows: Iterable[dict[str, Any]]) -> None:
        limit = getattr(self.config, "num_samples", None)
        for row in rows:
            token_request = _extract_azure_token_request(row)
            if token_request is not None:
                self._set_input_mode("token")
                self.system_prompts.append("")
                self.chat_messages.append(token_request)
                self.prompts.append(str(token_request["prompt_len"]))
            else:
                messages = _extract_messages(row)
                if not messages:
                    continue
                self._set_input_mode("chat")
                prompt_messages = _prompt_messages(messages)
                prompt = _messages_to_text(prompt_messages)
                if prompt_messages and prompt:
                    self.system_prompts.append("")
                    self.chat_messages.append(prompt_messages)
                    self.prompts.append(prompt)
            if limit is not None and limit > 0 and len(self.prompts) >= limit:
                break

    def _set_input_mode(self, mode: str) -> None:
        if self._input_mode is None:
            self._input_mode = mode
            return
        if self._input_mode != mode:
            raise ValueError(
                "Azure chat source mixes token-trace rows and text chat rows; "
                "use one schema per run"
            )


def register_chat_loaders() -> None:
    from moe_cap.data_loader import loader_registry

    loader_registry._REGISTRY["sharegpt"] = (ShareGPTLoader, 1024)
    loader_registry._REGISTRY["azure_chat"] = (AzureChatLoader, 1024)
