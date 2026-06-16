"""MoE-CAP runner entrypoint with harness-side dataset registrations."""

import asyncio
import json
import time
from typing import Optional

from s_mfu.agentic_loaders import register_agentic_loaders
from s_mfu.chat_loaders import register_chat_loaders


def _patch_chat_message_inputs() -> None:
    import torch
    import moe_cap.runner.openai_api_profile as profile
    from moe_cap.runner.openai_api_profile import OpenAIAPIMoEProfiler

    original_prepare_inputs = OpenAIAPIMoEProfiler._prepare_inputs
    original_run_benchmark = OpenAIAPIMoEProfiler.run_benchmark

    def prepare_inputs(self, all_input_raw, max_new_tokens, system_prompts=None):
        if (
            all_input_raw
            and isinstance(all_input_raw[0], dict)
            and "prompt_token_ids" in all_input_raw[0]
        ):
            prompts = [item["prompt_token_ids"] for item in all_input_raw]
            prompt_lengths = [int(item["prompt_len"]) for item in all_input_raw]
            output_lengths = [int(item["max_tokens"]) for item in all_input_raw]
            if self.use_chat_api:
                self.use_chat_api = False
                self.request_fn = profile.async_request_openai_completions
                self.api_url = self.api_url.replace(
                    "/chat/completions", "/completions"
                )
            return prompts, prompt_lengths, output_lengths

        if self.use_chat_api and all_input_raw and isinstance(all_input_raw[0], list):
            prompts = [json.dumps(messages) for messages in all_input_raw]
            prompt_lengths = [
                sum(
                    len(self.tokenizer.encode(message.get("content", "")))
                    for message in messages
                )
                for messages in all_input_raw
            ]
            return prompts, prompt_lengths, max_new_tokens
        return original_prepare_inputs(
            self, all_input_raw, max_new_tokens, system_prompts
        )

    async def run_benchmark(
        self,
        prompts,
        max_output_len,
        batch_size: Optional[int] = None,
    ):
        if not isinstance(max_output_len, list):
            return await original_run_benchmark(self, prompts, max_output_len, batch_size)

        def output_len_for(idx: int) -> int:
            return int(max_output_len[idx])

        concurrency = len(prompts) if batch_size is None else max(1, int(batch_size))
        concurrency = min(concurrency, len(prompts))
        semaphore = asyncio.Semaphore(concurrency)
        pbar = profile.async_tqdm(total=len(prompts), desc="Processing requests")

        async def run_one(idx: int):
            async with semaphore:
                request_input = profile.RequestFuncInput(
                    prompt=prompts[idx],
                    api_url=self.api_url,
                    output_len=output_len_for(idx),
                    model=self.hf_model_name,
                    extra_request_body={},
                    ignore_eos=self.ignore_eos,
                )
                return await self.request_fn(request_input, pbar)

        tasks = [asyncio.create_task(run_one(idx)) for idx in range(len(prompts))]
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        results = await asyncio.gather(*tasks)
        torch.cuda.synchronize()
        total_time = time.perf_counter() - start_time
        pbar.close()

        return results, total_time

    OpenAIAPIMoEProfiler._prepare_inputs = prepare_inputs
    OpenAIAPIMoEProfiler.run_benchmark = run_benchmark


def main() -> None:
    register_agentic_loaders()
    register_chat_loaders()
    _patch_chat_message_inputs()

    from moe_cap.runner.openai_api_profile import main as moe_cap_main

    moe_cap_main()


if __name__ == "__main__":
    main()
