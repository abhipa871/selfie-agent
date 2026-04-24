import re

# Qwen3 thinking: <|think|> ... <|/think|>
_b = "<" + "|" + "think" + "|>"
_c = "<" + "|" + "/think" + "|>"
# redacted_thinking-style blocks (Gemma, etc.): see _d_open / _d_close
_d_open = "<" + "redacted_thinking" + ">"
_d_close = "<" + "/redacted_thinking" + ">"

_PATTERNS = (
    re.compile(re.escape(_b) + r".*?" + re.escape(_c), flags=re.DOTALL),
    re.compile(re.escape(_d_open) + r".*?" + re.escape(_d_close), flags=re.DOTALL),
    re.compile(re.escape(_c), flags=re.DOTALL),
)


def clean_thinking(text: str) -> str:
    """Remove common *thinking* or chain-of-thought blocks from decoded model text."""
    s = text
    for rx in _PATTERNS:
        s = rx.sub("", s)
    return s.strip()
