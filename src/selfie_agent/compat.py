from __future__ import annotations

import inspect
from typing import Any, Literal, Tuple

import torch
import torch.nn as nn

# Chat-template-friendly: placeholders + suffix are wrapped by tokenizer.apply_chat_template
# (Gemma, Llama-3, Mistral, etc.). "llama_instruct" keeps the legacy [INST] user-string pattern.
# Styles in ``CHATML_LIKE_STYLES`` only differ by name: layout is
#   user placeholders (+ optional user suffix) | assistant prefill via ``InterpretationPrompt.assistant_prefill`` —
#   same for Meta Llama 2 (via template), Llama 3, Gemma 2/3/4, Qwen, etc.
InterpretationStyle = Literal[
    "universal",
    "llama3",
    "llama_instruct",
    "gemma",
    "gemma2",
    "gemma3",
    "gemma4",
    "qwen",
]

# All equivalent for the built-in sequence: ``(0,…,0)`` and optional ``\n{suffix}`` in user, or
# SelfIE split with ``user_message_only_placeholders=True`` + ``assistant_prefill`` on :class:`InterpretationPrompt`.
CHATML_LIKE_STYLES: frozenset[str] = frozenset(
    {
        "universal",
        "llama3",
        "gemma",
        "gemma2",
        "gemma3",
        "gemma4",
        "qwen",
    }
)


def interpretation_user_prompt_sequence(
    num_placeholders: int,
    suffix: str,
    style: InterpretationStyle = "universal",
    *,
    user_message_only_placeholders: bool = False,
) -> Tuple[object, ...]:
    """Build the user-turn ``(string | 0 | …)`` sequence for :class:`selfie_agent.prompts.InterpretationPrompt`.

    Each ``0`` becomes a ``placeholder`` when building the user text. If
    ``user_message_only_placeholders`` is ``True``, *suffix* is omitted here and should be passed as
    ``assistant_prefill`` (SelfIE-style: only placeholders in the user turn, task text as assistant
    prefill; see the upstream SelfIE repo). If ``False`` (default), the suffix is appended in the
    **user** string (older single-user-turn behavior).

    For **Meta Llama 2 / 3**, **Gemma 2 / 3 / 4**, **Qwen**, and other ``apply_chat_template`` models,
    any style in :data:`CHATML_LIKE_STYLES` produces the same placeholder layout; the tokenizer's
    template (and ``enable_thinking`` on :class:`InterpretationPrompt` for thinking models) does the
    rest.
    """
    s = str(style)
    if s in CHATML_LIKE_STYLES:
        if user_message_only_placeholders:
            return tuple([0] * num_placeholders)
        return tuple([0] * num_placeholders + [f"\n{suffix}"])
    if s == "llama_instruct":
        if user_message_only_placeholders:
            return tuple(["[INST]"] + [0] * num_placeholders + ["[/INST]"])
        return tuple(["[INST]"] + [0] * num_placeholders + [f"[/INST] {suffix}"])
    raise ValueError(
        f"Unknown interpretation style {style!r}. "
        "Use one of CHATML_LIKE_STYLES (e.g. 'universal', 'llama3', 'gemma2', 'gemma3', 'gemma4', 'qwen') "
        "or 'llama_instruct' (legacy Llama-2-Chat only)."
    )


def _apply_chat_accepts_thinking_kw(tokenizer) -> bool:
    try:
        params = inspect.signature(tokenizer.apply_chat_template).parameters
    except (TypeError, ValueError):
        return False
    if "enable_thinking" in params:
        return True
    return any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())


def apply_chat_template_with_thinking(
    tokenizer,
    conversation,
    *,
    enable_thinking: bool = False,
    **kwargs: Any,
) -> Any:
    """Call ``apply_chat_template``, passing ``enable_thinking`` when the method can accept it."""
    call_kw = dict(kwargs)
    try:
        if _apply_chat_accepts_thinking_kw(tokenizer):
            call_kw["enable_thinking"] = enable_thinking
        return tokenizer.apply_chat_template(conversation, **call_kw)
    except TypeError:
        call_kw.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(conversation, **call_kw)


def get_decoder_layers(model) -> nn.Module:
    """Return the decoder block stack (``nn.ModuleList``-like) for common Hugging Face causal LMs."""
    if hasattr(model, "model"):
        inner = model.model
        if hasattr(inner, "layers"):
            return inner.layers
        if hasattr(inner, "language_model") and hasattr(inner.language_model, "layers"):
            return inner.language_model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    if hasattr(model, "transformer") and hasattr(model.transformer, "layers"):
        return model.transformer.layers
    if hasattr(model, "layers"):
        return model.layers
    raise ValueError("Could not find decoder layers on this model.")


def resolve_model_device(model) -> torch.device:
    d = getattr(model, "device", None)
    if d is not None and isinstance(d, torch.device) and d.type != "meta":
        return d
    first = next(model.parameters(), None)
    if first is not None:
        return first.device
    return torch.device("cpu")
