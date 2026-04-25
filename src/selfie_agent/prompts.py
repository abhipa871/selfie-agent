from __future__ import annotations

from typing import Any, Dict, Sequence

from .compat import apply_chat_template_with_thinking


class InterpretationPrompt:
    def __init__(
        self,
        tokenizer,
        user_prompt_sequence: Sequence[object],
        system_prompt: str | None = None,
        placeholder: str = "_ ",
        enable_thinking: bool = False,
        assistant_prefill: str | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.placeholder = placeholder
        self.system_prompt = system_prompt
        self.enable_thinking = enable_thinking
        # When set, ``suffix`` is the *assistant* prefill. We use ``add_generation_prompt=False`` so
        # ``apply_chat_template`` does not add a second assistant header after the prefilled turn;
        # we pass ``continue_final_message=True`` when the tokenizer supports it (HF) so generation
        # continues the partial assistant message.
        self.assistant_prefill = assistant_prefill

        user_content = ""
        self.insert_locations: list[int] = []

        for part in user_prompt_sequence:
            if isinstance(part, str):
                user_content += part
                continue

            before_ids = self._encode_for_insert_diff(user_content)

            user_content += placeholder

            after_ids = self._encode_for_insert_diff(user_content)

            insert_locations = self._find_exact_one_insert_location(
                before_ids=before_ids,
                after_ids=after_ids,
                placeholder=placeholder,
            )

            self.insert_locations.extend(insert_locations)

        self.messages = self._conversation_for_final_encode(user_content)

        # If we already have an assistant prefill, do not append a second "generation" assistant header
        # (``add_generation_prompt=True``) — that duplicates roles and can cause repeated "assistant" junk.
        # Optional ``continue_final_message`` (HF) keeps generation continuing the last assistant turn.
        encoded = apply_chat_template_with_thinking(
            tokenizer,
            self.messages,
            enable_thinking=self.enable_thinking,
            return_tensors="pt",
            add_generation_prompt=(self.assistant_prefill is None),
            **self._assistant_prefill_apply_chat_extras(),
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

    def _build_messages_with_assistant(self, user_content: str, assistant_content: str):
        if self.system_prompt is None:
            return [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content},
            ]
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]

    def _conversation_for_final_encode(self, user_content: str):
        if self.assistant_prefill is not None:
            return self._build_messages_with_assistant(user_content, self.assistant_prefill)
        return self._build_messages(user_content)

    def _assistant_prefill_apply_chat_extras(self) -> Dict[str, Any]:
        """Kwargs for ``apply_chat_template`` when using assistant prefill (see class doc)."""
        if self.assistant_prefill is not None:
            return {"continue_final_message": True}
        return {}

    def _encode_for_insert_diff(self, user_content: str) -> list[int]:
        """Same chat framing as the final prompt, so insert indices match tokenized model input."""
        convo = self._conversation_for_final_encode(user_content)
        text = apply_chat_template_with_thinking(
            self.tokenizer,
            convo,
            enable_thinking=self.enable_thinking,
            tokenize=False,
            add_generation_prompt=(self.assistant_prefill is None),
            **self._assistant_prefill_apply_chat_extras(),
        )
        return self.tokenizer.encode(text, add_special_tokens=False)

    @staticmethod
    def _find_changed_span(before_ids: list[int], after_ids: list[int]) -> list[int]:
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

        return list(range(left, right_after))

    def _find_exact_one_insert_location(
        self,
        *,
        before_ids: list[int],
        after_ids: list[int],
        placeholder: str,
    ) -> list[int]:
        changed = self._find_changed_span(before_ids, after_ids)

        if len(changed) != 1:
            changed_tokens = [
                self.tokenizer.decode([after_ids[i]], skip_special_tokens=False)
                for i in changed
                if 0 <= i < len(after_ids)
            ]

            standalone = self.tokenizer.encode(
                placeholder,
                add_special_tokens=False,
            )

            raise ValueError(
                f"Placeholder {placeholder!r} must correspond to exactly one token "
                f"inside the final chat-templated prompt, but it corresponded to "
                f"{len(changed)} changed token position(s): {changed}. "
                f"Changed token text: {changed_tokens!r}. "
                f"Standalone placeholder tokenization: {standalone}. "
                "Choose a different placeholder for this tokenizer."
            )

        return changed