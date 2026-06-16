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

        if batch_size is None or batch_size >= len(prompts):
            tasks = []
            pbar = profile.async_tqdm(total=len(prompts), desc="Processing requests")

            for idx, prompt in enumerate(prompts):
                request_input = profile.RequestFuncInput(
                    prompt=prompt,
                    api_url=self.api_url,
                    output_len=output_len_for(idx),
                    model=self.hf_model_name,
                    extra_request_body={},
                    ignore_eos=self.ignore_eos,
                )
                tasks.append(self.request_fn(request_input, pbar))

            torch.cuda.synchronize()
            start_time = time.perf_counter()
            results = await asyncio.gather(*tasks)
            torch.cuda.synchronize()
            total_time = time.perf_counter() - start_time
            pbar.close()
            return results, total_time

        all_results = [None] * len(prompts)
        pbar = profile.async_tqdm(total=len(prompts), desc="Processing requests")

        torch.cuda.synchronize()
        start_time = time.perf_counter()

        batch_start_idx = 0
        active_tasks = {}

        while batch_start_idx < len(prompts) or active_tasks:
            if batch_start_idx < len(prompts):
                batch_end_idx = min(batch_start_idx + batch_size, len(prompts))

                for idx in range(batch_start_idx, batch_end_idx):
                    request_input = profile.RequestFuncInput(
                        prompt=prompts[idx],
                        api_url=self.api_url,
                        output_len=output_len_for(idx),
                        model=self.hf_model_name,
                        extra_request_body={},
                        ignore_eos=self.ignore_eos,
                    )
                    task = asyncio.create_task(self.request_fn(request_input, pbar))
                    active_tasks[task] = idx

                current_batch_size = batch_end_idx - batch_start_idx
                threshold = current_batch_size // 2
                completed_in_batch = 0

                while completed_in_batch < threshold and active_tasks:
                    done, _ = await asyncio.wait(
                        active_tasks.keys(), return_when=asyncio.FIRST_COMPLETED
                    )
                    for task in done:
                        result = await task
                        idx = active_tasks.pop(task)
                        all_results[idx] = result
                        if batch_start_idx <= idx < batch_end_idx:
                            completed_in_batch += 1

                batch_start_idx = batch_end_idx
            else:
                done, _ = await asyncio.wait(
                    active_tasks.keys(), return_when=asyncio.FIRST_COMPLETED
                )
                for task in done:
                    result = await task
                    idx = active_tasks.pop(task)
                    all_results[idx] = result

        torch.cuda.synchronize()
        total_time = time.perf_counter() - start_time
        pbar.close()

        return all_results, total_time

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
