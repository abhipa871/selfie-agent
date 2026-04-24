from __future__ import annotations

import re
from typing import Any, Sequence

# See HuggingFace model cards for Gemma 4: opening ``<|channel>thought`` … closing ``<channel|>``.
_RE_GEMMA4_THOUGHT_BLOCK = re.compile(
    r"<\s*\|\s*channel\s*>\s*thought"
    r"[\s\S]*?"
    r"<\s*/?channel\s*\|>\s*",
    re.IGNORECASE,
)
_RE_GEMMA4_LEADING = re.compile(
    r"^\s*<\s*\|\s*channel\s*>\s*thought"
    r"[\s\S]*?"
    r"<\s*/?channel\s*\|>\s*",
    re.IGNORECASE,
)


def strip_gemma4_thought_channel(text: str) -> str:
    """Remove Gemma 4 *thought* channel blocks (``<|channel>thought`` … ``<channel|>``) from a string."""
    if not text:
        return text
    return _RE_GEMMA4_THOUGHT_BLOCK.sub("", text).strip()


def _leading_block_end_char(assistant_text: str) -> int:
    m = _RE_GEMMA4_LEADING.match(assistant_text)
    return m.end() if m else 0


def _prefix_token_count_by_decode_length(
    tokenizer: Any, answer_ids: Sequence[int], end_char: int
) -> int:
    if end_char <= 0:
        return 0
    for k in range(1, len(answer_ids) + 1):
        part = tokenizer.decode(list(answer_ids[:k]), skip_special_tokens=False)
        if len(part) >= end_char:
            return k
    return 0


def count_leading_gemma4_thought_tokens(
    tokenizer: Any,
    answer_token_ids: Sequence[int],
) -> int:
    """Count initial assistant tokens that belong to the first Gemma 4 *thought* channel only."""
    if not answer_token_ids:
        return 0
    text = tokenizer.decode(list(answer_token_ids), skip_special_tokens=False)
    end = _leading_block_end_char(text)
    if end <= 0:
        return 0
    return _prefix_token_count_by_decode_length(tokenizer, answer_token_ids, end)
