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
