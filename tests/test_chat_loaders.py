import json


def test_extracts_sharegpt_conversation_roles():
    from s_mfu.chat_loaders import _extract_messages

    messages = _extract_messages(
        {
            "conversations": [
                {"from": "human", "value": "hello"},
                {"from": "gpt", "value": "hi"},
            ]
        }
    )

    assert messages == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]


def test_extracts_openai_messages():
    from s_mfu.chat_loaders import _extract_messages

    messages = _extract_messages(
        {
            "messages": [
                {"role": "system", "content": "be terse"},
                {"role": "user", "content": "hello"},
            ]
        }
    )

    assert messages == [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "hello"},
    ]


def test_reads_local_jsonl_and_limits_samples(tmp_path, monkeypatch):
    from s_mfu.chat_loaders import AzureChatLoader

    path = tmp_path / "azure.jsonl"
    rows = [
        {"messages": [{"role": "user", "content": "one"}]},
        {"messages": [{"role": "user", "content": "two"}]},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows))

    config = type("Config", (), {"num_samples": 1})()
    monkeypatch.setenv("S_MFU_AZURE_CHAT_PATH", str(path))
    loader = AzureChatLoader(config)

    assert loader.get_input() == [[{"role": "user", "content": "one"}]]


def test_reads_single_object_openai_json(tmp_path, monkeypatch):
    from s_mfu.chat_loaders import AzureChatLoader

    path = tmp_path / "azure.json"
    path.write_text(json.dumps({"messages": [{"role": "user", "content": "one"}]}))

    config = type("Config", (), {"num_samples": 1})()
    monkeypatch.setenv("S_MFU_AZURE_CHAT_PATH", str(path))
    loader = AzureChatLoader(config)

    assert loader.get_input() == [[{"role": "user", "content": "one"}]]


def test_reads_single_object_sharegpt_json(tmp_path, monkeypatch):
    from s_mfu.chat_loaders import ShareGPTLoader

    path = tmp_path / "sharegpt.json"
    path.write_text(
        json.dumps(
            {
                "conversations": [
                    {"from": "human", "value": "one"},
                    {"from": "gpt", "value": "answer"},
                ]
            }
        )
    )

    config = type("Config", (), {"num_samples": 1})()
    monkeypatch.setenv("S_MFU_SHAREGPT_PATH", str(path))
    loader = ShareGPTLoader(config)

    assert loader.get_input() == [[{"role": "user", "content": "one"}]]


def test_chat_num_samples_counts_valid_prompts_after_filtering(tmp_path, monkeypatch):
    from s_mfu.chat_loaders import ShareGPTLoader

    path = tmp_path / "sharegpt.jsonl"
    rows = [
        {"conversations": [{"from": "gpt", "value": "assistant only"}]},
        {"conversations": [{"from": "human", "value": "one"}]},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows))

    config = type("Config", (), {"num_samples": 1})()
    monkeypatch.setenv("S_MFU_SHAREGPT_PATH", str(path))
    loader = ShareGPTLoader(config)

    assert loader.get_input() == [[{"role": "user", "content": "one"}]]


def test_azure_reads_official_token_count_csv(tmp_path, monkeypatch):
    from s_mfu.chat_loaders import AzureChatLoader

    path = tmp_path / "azure.csv"
    path.write_text(
        "TIMESTAMP,ContextTokens,GeneratedTokens\n"
        "2023-11-16 18:15:46.6805900,374,44\n"
        "2023-11-16 18:15:50.9951690,396,109\n"
    )

    config = type("Config", (), {"num_samples": 1})()
    monkeypatch.setenv("S_MFU_AZURE_CHAT_PATH", str(path))
    loader = AzureChatLoader(config)

    request = loader.get_input()[0]
    assert len(request["prompt_token_ids"]) == 374
    assert request["prompt_len"] == 374
    assert request["max_tokens"] == 44


def test_azure_reads_parsed_token_id_jsonl(tmp_path, monkeypatch):
    from s_mfu.chat_loaders import AzureChatLoader

    path = tmp_path / "trace_parsed.jsonl"
    row = {
        "timestamp": 0.0,
        "meta": {
            "input_length": 6,
            "output_length": 3,
            "prev_total_len": 4,
            "new_input_len": 2,
        },
        "new_input_ids": [111, 222],
        "max_tokens": 3,
    }
    path.write_text(json.dumps(row) + "\n")

    config = type("Config", (), {"num_samples": 1})()
    monkeypatch.setenv("S_MFU_AZURE_CHAT_PATH", str(path))
    loader = AzureChatLoader(config)

    request = loader.get_input()[0]
    assert request["prompt_token_ids"] == [111, 222, 111, 222, 111, 222]
    assert request["prompt_len"] == 6
    assert request["max_tokens"] == 3


def test_azure_rejects_mixed_token_and_chat_rows(tmp_path, monkeypatch):
    from s_mfu.chat_loaders import AzureChatLoader

    path = tmp_path / "azure.jsonl"
    rows = [
        {"ContextTokens": 4, "GeneratedTokens": 2},
        {"messages": [{"role": "user", "content": "one"}]},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows))

    config = type("Config", (), {"num_samples": 2})()
    monkeypatch.setenv("S_MFU_AZURE_CHAT_PATH", str(path))

    try:
        AzureChatLoader(config)
    except ValueError as exc:
        assert "mixes token-trace rows and text chat rows" in str(exc)
    else:
        raise AssertionError("AzureChatLoader should reject mixed source schemas")


def test_chat_loader_drops_trailing_assistant_answer(tmp_path, monkeypatch):
    from s_mfu.chat_loaders import ShareGPTLoader

    path = tmp_path / "sharegpt.json"
    path.write_text(
        json.dumps(
            [
                {
                    "conversations": [
                        {"from": "human", "value": "first question"},
                        {"from": "gpt", "value": "first answer"},
                        {"from": "human", "value": "second question"},
                        {"from": "gpt", "value": "second answer"},
                    ]
                }
            ]
        )
    )

    config = type("Config", (), {"num_samples": 1})()
    monkeypatch.setenv("S_MFU_SHAREGPT_PATH", str(path))
    loader = ShareGPTLoader(config)

    assert loader.get_input() == [
        [
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": "first answer"},
            {"role": "user", "content": "second question"},
        ]
    ]


def test_sharegpt_default_uses_explicit_hf_json(monkeypatch):
    import s_mfu.chat_loaders as chat_loaders

    calls = []

    def fake_load_hf_dataset(*args, **kwargs):
        calls.append((args, kwargs))
        return [{"conversations": [{"from": "human", "value": "hello"}]}]

    monkeypatch.delenv("S_MFU_SHAREGPT_PATH", raising=False)
    monkeypatch.delenv("S_MFU_SHAREGPT_HF_DATASET", raising=False)
    monkeypatch.setattr(chat_loaders, "_load_hf_dataset", fake_load_hf_dataset)
    config = type("Config", (), {"dataset_split": "train", "num_samples": 1})()

    loader = chat_loaders.ShareGPTLoader(config)

    assert loader.get_input() == [[{"role": "user", "content": "hello"}]]
    assert calls[0][0] == ("json",)
    assert "ShareGPT_V3_unfiltered_cleaned_split.json" in calls[0][1]["data_files"]
    assert calls[0][1]["streaming"] is True


def test_azure_requires_path_or_explicit_hf_dataset(monkeypatch):
    from s_mfu.chat_loaders import AzureChatLoader

    monkeypatch.delenv("S_MFU_AZURE_CHAT_PATH", raising=False)
    monkeypatch.delenv("S_MFU_AZURE_CHAT_HF_DATASET", raising=False)
    config = type("Config", (), {"dataset_split": "train", "num_samples": 1})()

    try:
        AzureChatLoader(config)
    except ValueError as exc:
        assert "S_MFU_AZURE_CHAT_PATH" in str(exc)
    else:
        raise AssertionError("AzureChatLoader should require an explicit source")
