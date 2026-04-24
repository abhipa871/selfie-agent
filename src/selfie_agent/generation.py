"""Helpers for ``model.generate`` (optional ``presence_penalty``, ``min_p`` normalization)."""

from __future__ import annotations

from typing import Any, Dict, Mapping, MutableMapping

import torch
from transformers import LogitsProcessor


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
    """Copy kwargs, map ``presence_penalty`` to a logits processor, drop ``min_p`` when 0."""
    out: Dict[str, Any] = dict(kwargs)
    mp = out.get("min_p", None)
    if mp is None or float(mp) == 0.0:
        out.pop("min_p", None)
    _attach_presence_penalty(out)
    return out


def build_generation_kwargs(
    *,
    max_new_tokens: int,
    eos_token_id: Any,
    pad_token_id: int,
    instance_kwargs: Mapping[str, Any] | None,
    call_kwargs: Mapping[str, Any] | None,
) -> Dict[str, Any]:
    """Merge required ``generate`` keys with instance and per-call overrides (later wins)."""
    required: Dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "eos_token_id": eos_token_id,
        "pad_token_id": pad_token_id,
        "return_dict_in_generate": True,
    }
    merged: Dict[str, Any] = {**required, **dict(instance_kwargs or {}), **dict(call_kwargs or {})}
    return prepare_generation_kwargs(merged)
