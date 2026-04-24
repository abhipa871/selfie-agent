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
# When ``decode(..., skip_special_tokens=True)`` runs *before* stripping, channel tags vanish but
# a plain ``thought`` line (vocab token) often remains — remove those whole lines.
_RE_ORPHAN_THOUGHT_LINE = re.compile(r"(?m)^\s*thought\s*$(?:\r\n?|\n)?", re.IGNORECASE)


def _dedupe_consecutive_duplicate_lines(s: str, *, min_len: int = 16) -> str:
    """Drop back-to-back identical non-empty lines (Gemma sometimes echoes the final answer)."""
    lines = s.splitlines()
    out: list[str] = []
    for line in lines:
        if (
            out
            and out[-1] == line
            and line.strip()
            and len(line.strip()) >= min_len
        ):
            continue
        out.append(line)
    return "\n".join(out)


def _dedupe_repeated_content_lines(s: str, *, min_len: int = 12) -> str:
    """If every non-empty line is the same (e.g. answer / thought / answer / thought), keep one line."""
    compact = [x.strip() for x in s.splitlines() if x.strip()]
    if not compact or len(set(compact)) != 1:
        return s
    u = compact[0]
    if len(u) < min_len or len(compact) < 2:
        return s
    return u


def strip_gemma4_thought_channel(text: str) -> str:
    """Remove Gemma 4 *thought* channel blocks and leftover channel artifacts from a string.

    Pass text decoded with ``skip_special_tokens=False`` so ``<|channel>`` / ``<channel|>`` are
    still present; this also repairs the common case where those tags were already skipped and only
    standalone ``thought`` lines and echoed answer text remain.
    """
    if not text:
        return text
    s = _RE_GEMMA4_THOUGHT_BLOCK.sub("", text)
    # Repair path if tags were already removed by ``skip_special_tokens=True`` elsewhere.
    s = _RE_ORPHAN_THOUGHT_LINE.sub("", s)
    s = _dedupe_repeated_content_lines(s)
    s = _dedupe_consecutive_duplicate_lines(s)
    # If we decoded with ``skip_special_tokens=False``, bare turn markers can remain.
    s = re.sub(r"<\s*end_of_turn\s*>\s*", "", s, flags=re.IGNORECASE)
    return s.strip()


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
