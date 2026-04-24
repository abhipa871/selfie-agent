"""Sampling presets and helpers for ``model.generate`` (Qwen 3.5 instruct vs reasoning)."""

from __future__ import annotations

from typing import Any, Dict, Mapping, MutableMapping

import torch
from transformers import LogitsProcessor

# Recommended Qwen 3.5 chat settings (Hugging Face ``generate`` + local presence penalty).
QWEN35_INSTRUCT_GENERATION_KWARGS: Dict[str, Any] = {
    "do_sample": True,
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 1.5,
    "repetition_penalty": 1.0,
}

QWEN35_REASONING_GENERATION_KWARGS: Dict[str, Any] = {
    "do_sample": True,
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 1.5,
    "repetition_penalty": 1.0,
}


class PresencePenaltyLogitsProcessor(LogitsProcessor):
    """Penalize logits for tokens proportional to how often they appear in ``input_ids`` (OpenAI-style)."""

    def __init__(self, penalty: float) -> None:
        self.penalty = float(penalty)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.penalty == 0.0:
            return scores
        scores = scores.clone()
        for row in range(input_ids.shape[0]):
            ids = input_ids[row]
            uniq, counts = torch.unique(ids, return_counts=True)
            scores[row, uniq] -= self.penalty * counts.to(scores.dtype)
        return scores


def qwen35_generation_kwargs(enable_thinking: bool) -> Dict[str, Any]:
    """Return a copy of the Qwen 3.5 preset for instruct (``enable_thinking=False``) or reasoning."""
    base = QWEN35_REASONING_GENERATION_KWARGS if enable_thinking else QWEN35_INSTRUCT_GENERATION_KWARGS
    return dict(base)


def _attach_presence_penalty(kwargs: MutableMapping[str, Any]) -> None:
    penalty = kwargs.pop("presence_penalty", None)
    if penalty is None or float(penalty) == 0.0:
        return
    from transformers import LogitsProcessorList

    proc = PresencePenaltyLogitsProcessor(float(penalty))
    existing = kwargs.get("logits_processor")
    if existing is None:
        kwargs["logits_processor"] = LogitsProcessorList([proc])
        return
    if isinstance(existing, LogitsProcessorList):
        existing.append(proc)
        kwargs["logits_processor"] = existing
        return
    kwargs["logits_processor"] = LogitsProcessorList([*existing, proc])


def prepare_generation_kwargs(kwargs: Mapping[str, Any]) -> Dict[str, Any]:
    """Copy kwargs, map ``presence_penalty`` to a logits processor, drop ``min_p`` when 0 (HF treats as active)."""
    out: Dict[str, Any] = dict(kwargs)
    mp = out.get("min_p", None)
    if mp is None or float(mp) == 0.0:
        out.pop("min_p", None)
    _attach_presence_penalty(out)
    return out


def merge_generation_kwargs(
    *,
    max_new_tokens: int,
    eos_token_id: Any,
    pad_token_id: int,
    use_qwen35_sampling: bool,
    enable_thinking: bool,
    instance_kwargs: Mapping[str, Any] | None,
    call_kwargs: Mapping[str, Any] | None,
) -> Dict[str, Any]:
    """Merge required keys, optional Qwen 3.5 preset, instance defaults, and per-call overrides (later wins)."""
    required: Dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "eos_token_id": eos_token_id,
        "pad_token_id": pad_token_id,
        "return_dict_in_generate": True,
    }
    preset: Dict[str, Any] = {}
    if use_qwen35_sampling:
        preset = qwen35_generation_kwargs(enable_thinking)
    merged: Dict[str, Any] = {**required, **preset, **dict(instance_kwargs or {}), **dict(call_kwargs or {})}
    return prepare_generation_kwargs(merged)
