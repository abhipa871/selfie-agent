"""
Gemma 4: when *thinking* is disabled (all IT variants **except** E2B / E4B), the model can still
emit an empty *thought* channel, then the real answer in a *final* channel — roughly::

    <|...|> thought ... (empty) ... <|...|> final ... <actual answer>

We locate the *final* channel opener in token space and return indices only for tokens **after**
that prefix so hidden-state injection aligns with the visible final answer, not the empty thought
block or channel scaffolding.

Exact strings are tokenizer-dependent; we try a small set of common Hugging Face template spellings.
If nothing matches, callers should fall back to the full assistant span.
"""
from __future__ import annotations

from typing import List, Sequence, Union

import torch

# Spelling candidates for the start of the *final* content channel (after *thought* is closed).
# Extend if your checkpoint uses a different template string in tokenizer_config / chat template.
_GEMMA4_FINAL_CHANNEL_PREFIXES: tuple[str, ...] = (
    "<|channel|>final",
    "<|channel|> final",
    "<|channel|>final\n",
    # Alternate bracketing seen in some docs / variants
    "<|channel>final",
    "<|channel|> final\n",
)


def _find_subsequence(haystack: List[int], needle: List[int]) -> int | None:
    if not needle or len(needle) > len(haystack):
        return None
    n = len(needle)
    for i in range(0, len(haystack) - n + 1):
        if haystack[i : i + n] == needle:
            return i
    return None


def first_global_index_after_gemma4_final_channel_prefix(
    tokenizer,
    input_ids_row: Union[torch.Tensor, Sequence[int], List[int]],
    prompt_len: int,
) -> int | None:
    """Return the first **global** token index *after* a recognized *final* channel header.

    Searches the generated span ``input_ids[prompt_len:]`` for encoded prefixes from
    ``_GEMMA4_FINAL_CHANNEL_PREFIXES``.  If multiple match, the **earliest** in the stream wins.

    Returns ``None`` if no prefix matches (caller should use the full generated span).
    """
    if isinstance(input_ids_row, torch.Tensor):
        row = input_ids_row.tolist()
    else:
        row = list(input_ids_row)
    gen = row[prompt_len:]
    if not gen:
        return None

    best: int | None = None
    for s in _GEMMA4_FINAL_CHANNEL_PREFIXES:
        try:
            needle = tokenizer.encode(s, add_special_tokens=False)
        except Exception:  # noqa: BLE001
            continue
        if not needle:
            continue
        pos = _find_subsequence(gen, needle)
        if pos is None:
            continue
        g = prompt_len + pos + len(needle)
        if best is None or g < best:
            best = g
    return best
