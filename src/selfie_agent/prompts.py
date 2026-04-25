from __future__ import annotations

from typing import Sequence

from .compat import apply_chat_template_with_thinking


class InterpretationPrompt:
    def __init__(
        self,
        tokenizer,
        user_prompt_sequence: Sequence[object],
        system_prompt: str | None = None,
        placeholder: str = "- ",
        enable_thinking: bool = False,
    ) -> None:
        self.tokenizer = tokenizer
        self.placeholder = placeholder
        self.system_prompt = system_prompt
        self.enable_thinking = enable_thinking

        user_content = ""
        self.insert_locations = []

        for part in user_prompt_sequence:
            if isinstance(part, str):
                user_content += part
                continue

            before_messages = self._build_messages(user_content)
            before_text = apply_chat_template_with_thinking(
                tokenizer,
                before_messages,
                enable_thinking=self.enable_thinking,
                tokenize=False,
                add_generation_prompt=True,
            )
            before_ids = tokenizer.encode(before_text, add_special_tokens=False)

            user_content += placeholder

            after_messages = self._build_messages(user_content)
            after_text = apply_chat_template_with_thinking(
                tokenizer,
                after_messages,
                enable_thinking=self.enable_thinking,
                tokenize=False,
                add_generation_prompt=True,
            )
            after_ids = tokenizer.encode(after_text, add_special_tokens=False)

            self.insert_locations.extend(
                self._find_insert_locations(before_ids, after_ids)
            )

        self.messages = self._build_messages(user_content)

        encoded = apply_chat_template_with_thinking(
            tokenizer,
            self.messages,
            enable_thinking=self.enable_thinking,
            return_tensors="pt",
            add_generation_prompt=True,
        )

        if hasattr(encoded, "input_ids"):
            self.interpretation_prompt_model_inputs = {
                "input_ids": encoded.input_ids,
                "attention_mask": getattr(encoded, "attention_mask", None),
            }
        else:
            self.interpretation_prompt_model_inputs = {
                "input_ids": encoded,
                "attention_mask": None,
            }

    def _build_messages(self, user_content: str):
        if self.system_prompt is None:
            return [{"role": "user", "content": user_content}]
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]

    @staticmethod
    def _find_insert_locations(before_ids, after_ids):
        left = 0
        while (
            left < len(before_ids)
            and left < len(after_ids)
            and before_ids[left] == after_ids[left]
        ):
            left += 1

        right_before = len(before_ids)
        right_after = len(after_ids)

        while (
            right_before > left
            and right_after > left
            and before_ids[right_before - 1] == after_ids[right_after - 1]
        ):
            right_before -= 1
            right_after -= 1

        if right_after <= left:
            raise ValueError("Could not locate placeholder token span.")

        return list(range(left, right_after))