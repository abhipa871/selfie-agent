from __future__ import annotations

import inspect
from typing import Any, Literal, Tuple

import torch
import torch.nn as nn

# Chat-template-friendly: placeholders + suffix are wrapped by tokenizer.apply_chat_template
# (Gemma, Qwen, Llama-3, Mistral, etc.). "llama_instruct" keeps the legacy [INST] user-string pattern.
InterpretationStyle = Literal["universal", "llama_instruct", "gemma", "qwen"]


def interpretation_user_prompt_sequence(
    num_placeholders: int,
    suffix: str,
    style: InterpretationStyle = "universal",
) -> Tuple[object, ...]:
    s = str(style)
    if s in ("universal", "gemma", "qwen"):
        return tuple([0] * num_placeholders + [f"\n{suffix}"])
    if s == "llama_instruct":
        return tuple(["[INST]"] + [0] * num_placeholders + [f"[/INST] {suffix}"])
    raise ValueError(
        f"Unknown interpretation style {style!r}. "
        "Use 'universal' (default; Gemma, Qwen, most chat LMs) or 'llama_instruct' (legacy Llama-2-Chat FRAMING)."
    )


def apply_chat_template_with_thinking(
    tokenizer,
    conversation,
    *,
    enable_thinking: bool = False,
    **kwargs: Any,
) -> Any:
    """Call ``apply_chat_template``, passing ``enable_thinking`` only if the tokenizer supports it (Qwen3 / Qwen3.5)."""
    try:
        params = inspect.signature(tokenizer.apply_chat_template).parameters
    except (TypeError, ValueError):
        return tokenizer.apply_chat_template(conversation, **kwargs)
    call_kw = dict(kwargs)
    if "enable_thinking" in params:
        call_kw["enable_thinking"] = enable_thinking
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
