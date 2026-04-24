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


def _norm_stanza(x: str) -> str:
    return " ".join(x.split())


def _collapse_duplicate_stanzas(s: str, *, min_stanza_len: int = 12) -> str:
    """If the text is the same paragraph repeated, separated by blank line(s), keep one copy."""
    if not s or len(s) < min_stanza_len:
        return s
    parts = re.split(r"\n{2,}", s)
    stripped = [p.strip() for p in parts if p.strip()]
    if len(stripped) < 2:
        return s
    norms = [_norm_stanza(t) for t in stripped]
    if len(set(norms)) == 1 and len(norms[0]) >= min_stanza_len:
        return stripped[0]
    return s


def is_gemma4_layout_or_control_piece(piece: str) -> bool:
    """True if a single token string is only a Gemma 4 *layout* / control fragment."""
    t = piece.strip()
    if not t:
        return True
    if t.lower() == "thought" and len(t) < 24:
        return True
    if t in "\n" or (len(t) == 1 and t.isspace()):
        return True
    if t == "``" or t in ("<|channel>",) or re.match(
        r"^<\s*(?:turn\|>|\|turn\|>|\|channel>)$", t, re.IGNORECASE
    ):
        return True
    if re.match(
        r"^<\s*(?:/?turn|/?channel|/?pad|/?eos|/?bos|end_of_turn|/?s)\b[^<]{0,40}\|>?\s*$",
        t,
        re.IGNORECASE,
    ):
        return True
    if re.match(r"^<\s*turn\s*\|>\s*$", t, re.IGNORECASE):
        return True
    if re.match(r"^<\s*/?channel[^>]*\|>\s*$", t, re.IGNORECASE):
        return True
    if re.match(r"^<\s*\|[^<]{0,36}>\s*$", t, re.IGNORECASE):
        return True
    if re.search(r"<\s*pad|<\s*eos|``", t, re.IGNORECASE):
        return True
    return False


def strip_gemma4_display(text: str) -> str:
    """Remove empty thought block, then turn/channel/pad/specials that leak into decoded strings.

    Use with ``skip_special_tokens=False`` when decoding, then this on the result.
    """
    if not text:
        return text
    s = _RE_GEMMA4_THOUGHT_BLOCK.sub("", text)
    s = _RE_ORPHAN_THOUGHT_LINE.sub("", s)
    s = _dedupe_repeated_content_lines(s)
    s = _dedupe_consecutive_duplicate_lines(s)
    s = _collapse_duplicate_stanzas(s)
    for _ in range(8):
        before = s
        s = re.sub(r"<\s*turn\s*\|>\s*?", "", s, flags=re.IGNORECASE)
        s = re.sub(r"<\s*\|turn\|>\s*?", "", s, flags=re.IGNORECASE)
        s = re.sub(r"<\s*\|?\s*channel[^>]*\|>\s*?", "", s, flags=re.IGNORECASE)
        s = re.sub(r"<\s*/?channel[^>]*\|>\s*?", "", s, flags=re.IGNORECASE)
        s = re.sub(r"<\s*end_of_turn\s*>\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"<\s*pad[^>]*>\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"<\s*eos[^>]*>\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"``\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"``\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"<\s*/?s>\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"``+", "", s)
        if s == before:
            break
    s = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", s)
    s = re.sub(r"`\s*$", "", s)  # lone HF/placeholder backtick at end
    s = s.strip()
    s = _collapse_duplicate_stanzas(s)
    return s.strip()


def strip_gemma4_thought_channel(text: str) -> str:
    """Back-compat alias: full Gemma 4 *display* cleanup (thought block + turn/channel junk)."""
    return strip_gemma4_display(text)


def trim_trailing_gemma4_layout_global_indices(
    tokenizer: Any, input_ids_row, global_indices: list[int] | list,
) -> list[int]:
    """Remove trailing *layout-only* token positions (e.g. ``<turn|>`` after the final-answer text)."""
    if not global_indices:
        return list(global_indices)

    out: list[int] = list(global_indices)
    # Try layout-token detection first, then *clean* equality trim.
    while len(out) > 0:
        tids = [int(input_ids_row[i].item()) for i in out]
        last = tokenizer.decode([tids[-1]], skip_special_tokens=False)
        if is_gemma4_layout_or_control_piece(last):
            out.pop()
            continue
        if len(out) < 2:
            break
        tids1 = tids
        tids0 = tids[:-1]
        full = tokenizer.decode(tids1, skip_special_tokens=False)
        prev = tokenizer.decode(tids0, skip_special_tokens=False) if tids0 else ""
        c_full = strip_gemma4_display(full)
        c_prev = strip_gemma4_display(prev)
        if c_full and c_full == c_prev:
            out.pop()
            continue
        break
    return out


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
