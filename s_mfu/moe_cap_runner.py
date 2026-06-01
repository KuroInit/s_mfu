"""MoE-CAP runner entrypoint with harness-side dataset registrations."""

import json

from s_mfu.chat_loaders import register_chat_loaders


def _patch_chat_message_inputs() -> None:
    from moe_cap.runner.openai_api_profile import OpenAIAPIMoEProfiler

    original_prepare_inputs = OpenAIAPIMoEProfiler._prepare_inputs

    def prepare_inputs(self, all_input_raw, max_new_tokens, system_prompts=None):
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

    OpenAIAPIMoEProfiler._prepare_inputs = prepare_inputs


def main() -> None:
    register_chat_loaders()
    _patch_chat_message_inputs()

    from moe_cap.runner.openai_api_profile import main as moe_cap_main

    moe_cap_main()


if __name__ == "__main__":
    main()
