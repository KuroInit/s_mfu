import asyncio


def test_token_trace_runner_respects_batch_size_concurrency(monkeypatch):
    import moe_cap.runner.openai_api_profile as profile
    from s_mfu.moe_cap_runner import _patch_chat_message_inputs

    class DummyPbar:
        def __init__(self, *args, **kwargs):
            pass

        def update(self, _):
            pass

        def close(self):
            pass

    monkeypatch.setattr(profile, "async_tqdm", DummyPbar)
    monkeypatch.setattr(profile.torch.cuda, "synchronize", lambda: None)

    active = 0
    max_active = 0
    output_lens = []

    async def fake_request(request_input, _pbar):
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        output_lens.append(request_input.output_len)
        await asyncio.sleep(0)
        active -= 1
        return profile.RequestFuncOutput(success=True)

    _patch_chat_message_inputs()

    profiler = type(
        "Profiler",
        (),
        {
            "api_url": "http://localhost:30000/v1/completions",
            "hf_model_name": "org/model",
            "ignore_eos": True,
        },
    )()
    profiler.request_fn = fake_request

    prompts = [[1], [2], [3], [4]]
    results, _ = asyncio.run(
        profile.OpenAIAPIMoEProfiler.run_benchmark(
            profiler,
            prompts,
            [10, 20, 30, 40],
            batch_size=1,
        )
    )

    assert len(results) == 4
    assert max_active == 1
    assert output_lens == [10, 20, 30, 40]
