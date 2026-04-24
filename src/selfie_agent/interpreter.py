from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple

import torch

from .compat import (
    InterpretationStyle,
    get_decoder_layers,
    interpretation_user_prompt_sequence,
    resolve_model_device,
)
from .prompts import InterpretationPrompt
from .utils import clean_thinking


def _pad_id(tokenizer) -> int:
    if tokenizer.pad_token_id is not None:
        return int(tokenizer.pad_token_id)
    if tokenizer.eos_token_id is not None:
        return int(tokenizer.eos_token_id)
    raise ValueError("Tokenizer has neither pad_token_id nor eos_token_id")


def _stop_ids_for_answer_span(tokenizer) -> Set[int]:
    """Token ids that end the assistant 'answer' for hidden-state / debug slices (Gemma eot, EOS, Qwen im_end, …)."""
    s: set[int] = set()
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is not None:
        if isinstance(eos, (list, tuple)):
            s.update(int(x) for x in eos)
        else:
            s.add(int(eos))
    for attr in ("eot_id", "im_end_id"):
        tid = getattr(tokenizer, attr, None)
        if tid is not None:
            s.add(int(tid))
    for marker in (
        # Gemma 2, etc. (if encoded as a single id; closes assistant turn)
        "<" + "end" + "of" + "turn" + ">",
    ):
        try:
            ids = tokenizer.encode(marker, add_special_tokens=False)
        except Exception:
            continue
        if len(ids) == 1:
            s.add(int(ids[0]))
    return s


class SelfieInterpreter:
    def __init__(self, model, tokenizer) -> None:
        self.model = model
        self.tokenizer = tokenizer

    def make_interpretation_prompt(
        self,
        num_placeholders: int,
        suffix: str = "Summarize this message in two sentences:",
        style: InterpretationStyle = "universal",
        placeholder: str = "- ",
    ) -> InterpretationPrompt:
        """Build an :class:`InterpretationPrompt` for the loaded tokenizer's chat template.

        Use ``style="gemma"`` or ``"qwen"`` (or default ``"universal"``) for Gemma 2, Qwen 3, Qwen 2.5, etc. —
        only the tokenizer's ``apply_chat_template`` wraps the user text, without Llama-2 ``[INST]`` markers.

        For legacy prompts that match original Llama-2-Chat *user* strings with ``[INST]...[/INST]`` inside
        the user turn, pass ``style="llama_instruct"``.

        ``placeholder`` is appended for each ``0`` in the built sequence; it must add exactly one token
        with this tokenizer and chat template (see :class:`InterpretationPrompt`).
        """
        return InterpretationPrompt(
            self.tokenizer,
            interpretation_user_prompt_sequence(
                num_placeholders,
                suffix,
                style,
            ),
            placeholder=placeholder,
        )

    def get_hidden_states_from_sequences(
        self,
        sequences: torch.LongTensor,
        answer_only: bool = True,
        prompt_len: int | None = None,
    ):
        with torch.no_grad():
            outputs = self.model(
                input_ids=sequences.to(resolve_model_device(self.model)),
                output_hidden_states=True,
                return_dict=True,
            )

        full_hs = tuple(layer[0] for layer in outputs.hidden_states)
        if not answer_only:
            return outputs, full_hs, list(range(sequences.shape[1]))

        if prompt_len is None:
            raise ValueError("prompt_len must be provided when answer_only=True")

        input_ids = sequences[0]
        answer_stop_ids = _stop_ids_for_answer_span(self.tokenizer)
        answer_indices = []
        for i in range(prompt_len, input_ids.shape[0]):
            token_id = input_ids[i].item()
            if self.tokenizer.pad_token_id is not None and token_id == self.tokenizer.pad_token_id:
                continue
            if token_id in answer_stop_ids:
                break
            answer_indices.append(i)

        answer_hs = tuple(layer_hs[answer_indices, :] for layer_hs in full_hs)
        return outputs, answer_hs, answer_indices

    def interpret(
        self,
        original_prompt: str,
        tokens_to_interpret: Sequence[Tuple[int, int]] | str,
        interpretation_prompt: InterpretationPrompt | None = None,
        target_layer: int = 0,
        original_max_new_tokens: int = 20,
        interpreter_max_new_tokens: int = 64,
        replacing_mode: str = "normalized",
        overlay_strength: float = 1.0,
        answer_only: bool = True,
        injection_mode: str = "batch",
        interpretation_suffix: str = "Summarize this message in two sentences:",
        interpretation_style: InterpretationStyle = "universal",
        source_layer: int | None = None,
        placeholder: str = "- ",
    ) -> Dict[str, Any]:
        """Run the original model generate pass, then the interpretation generate pass with injection.

        original_max_new_tokens limits the first completion (source hidden states).
        interpreter_max_new_tokens limits tokens in the interpretation pass after injection.
        When ``tokens_to_interpret`` is ``"all"``, you must set ``source_layer`` to the index into
        ``outputs.hidden_states`` (0 = embeddings, 1 = after first block, …) for which block’s
        states to use for all answer positions; use ``-1`` for the last index.

        For Gemma 2, Qwen 3, Qwen 2.5, and similar, keep ``interpretation_style`` as ``"universal"``
        (or ``"gemma"`` / ``"qwen"``, which are equivalent). Use ``interpretation_style="llama_instruct"``
        for legacy Llama-2-Chat ``[INST]``-style user strings.

        ``placeholder`` is passed to the default :class:`InterpretationPrompt` only; ignored if you pass
        ``interpretation_prompt`` yourself.
        """
        original_input_ids, original_attention_mask = self._encode_chat_prompt(original_prompt)
        original_prompt_len = original_input_ids.shape[1]

        pad_id = _pad_id(self.tokenizer)
        with torch.no_grad():
            original_gen = self.model.generate(
                input_ids=original_input_ids,
                attention_mask=original_attention_mask,
                max_new_tokens=original_max_new_tokens,
                do_sample=False,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=pad_id,
                return_dict_in_generate=True,
            )

        original_sequences = original_gen.sequences
        original_full_text = clean_thinking(
            self.tokenizer.decode(original_sequences[0], skip_special_tokens=True)
        )
        original_answer = clean_thinking(
            self.tokenizer.decode(original_sequences[0][original_prompt_len:], skip_special_tokens=True)
        )

        _, source_hs, answer_indices = self.get_hidden_states_from_sequences(
            sequences=original_sequences,
            answer_only=answer_only,
            prompt_len=original_prompt_len,
        )

        if tokens_to_interpret == "all":
            if source_layer is None:
                raise ValueError(
                    "When tokens_to_interpret is 'all', pass source_layer: hidden_states index to read "
                    "for every answer token (0 .. len-1, or -1 for the last layer in hidden_states)."
                )
            n_hs = len(source_hs)
            layer_idx = n_hs - 1 if source_layer == -1 else source_layer
            if not (0 <= layer_idx < n_hs):
                raise IndexError(
                    f"source_layer={source_layer!r} resolved to {layer_idx}, out of range for "
                    f"{n_hs} hidden state tensors (use 0..{n_hs - 1} or -1)."
                )
            token_count = len(answer_indices) if answer_only else source_hs[0].shape[-2]
            tokens_to_interpret = [(layer_idx, i) for i in range(token_count)]
        elif source_layer is not None:
            raise ValueError("source_layer is only used when tokens_to_interpret is 'all'")

        if interpretation_prompt is None:
            interpretation_prompt = self.make_interpretation_prompt(
                num_placeholders=len(tokens_to_interpret),
                suffix=interpretation_suffix,
                style=interpretation_style,
                placeholder=placeholder,
            )

        if injection_mode == "aligned" and len(tokens_to_interpret) != len(
            interpretation_prompt.insert_locations
        ):
            raise ValueError(
                "aligned mode requires the same number of tokens and placeholder positions"
            )

        dev = resolve_model_device(self.model)
        target_inputs = {
            key: (value.to(dev) if value is not None else None)
            for key, value in interpretation_prompt.interpretation_prompt_model_inputs.items()
        }

        result = self._forward_with_injection(
            source_hidden_states=source_hs,
            input_ids=target_inputs["input_ids"],
            attention_mask=target_inputs.get("attention_mask"),
            target_layer=target_layer,
            tokens_to_interpret=tokens_to_interpret,  # type: ignore[arg-type]
            target_insert_locations=interpretation_prompt.insert_locations,
            max_new_tokens=interpreter_max_new_tokens,
            replacing_mode=replacing_mode,
            overlay_strength=overlay_strength,
            injection_mode=injection_mode,
        )

        interpretation_answers = [clean_thinking(x) for x in result["decoded_texts"]]
        interpretation_full_texts = [clean_thinking(x) for x in result["full_texts"]]

        return {
            "original_full_text": original_full_text,
            "original_answer": original_answer,
            "interpretation_answers": interpretation_answers,
            "interpretation_full_texts": interpretation_full_texts,
            "tokens_to_interpret": tokens_to_interpret,
            "original_outputs": original_gen,
            "interpretation_outputs": result["outputs"],
            "original_prompt_len": original_prompt_len,
            "answer_indices": answer_indices,
            "target_prompt_len": result["target_prompt_len"],
            "batch_insert_infos": result["batch_insert_infos"],
            "injection_mode": result["injection_mode"],
            "interpretation_prompt": interpretation_prompt,
        }

    def show_answer_tokens(self, original_sequences, prompt_len: int):
        toks = original_sequences[0]
        rows = []
        answer_i = 0
        answer_stop_ids = _stop_ids_for_answer_span(self.tokenizer)

        for global_i in range(prompt_len, toks.shape[0]):
            tid = toks[global_i].item()
            if self.tokenizer.pad_token_id is not None and tid == self.tokenizer.pad_token_id:
                continue
            if tid in answer_stop_ids:
                break

            rows.append(
                {
                    "answer_idx": answer_i,
                    "global_idx": global_i,
                    "token_id": tid,
                    "decoded": repr(self.tokenizer.decode([tid])),
                }
            )
            answer_i += 1

        return rows

    def _encode_chat_prompt(self, prompt: str):
        dev = resolve_model_device(self.model)
        messages = [{"role": "user", "content": prompt}]
        encoded = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
        )
        if hasattr(encoded, "input_ids"):
            input_ids = encoded.input_ids.to(dev)
            attention_mask = (
                encoded.attention_mask.to(dev) if getattr(encoded, "attention_mask", None) is not None else None
            )
            return input_ids, attention_mask
        return encoded.to(dev), None

    def _get_layers(self):
        return get_decoder_layers(self.model)

    @staticmethod
    def _make_pre_hook(batch_insert_infos: Iterable[Dict[str, Any]]):
        def pre_hook_fn(module, inputs):
            hidden_states = inputs[0]
            if hidden_states.shape[1] <= 1:
                return inputs

            new_hidden_states = hidden_states.clone()
            batch_size, seq_len, hidden_dim = hidden_states.shape

            for batch_idx, info in enumerate(batch_insert_infos):
                positions = info["insert_locations"]
                replacement = info["replacement"]
                replacing_mode = info.get("replacing_mode", "normalized")
                overlay_strength = info.get("overlay_strength", 1.0)

                if batch_idx >= batch_size:
                    continue

                if not torch.is_tensor(positions):
                    positions = torch.tensor(positions, dtype=torch.long)
                positions = positions.to(hidden_states.device)

                if torch.any(positions >= seq_len):
                    continue

                repl = replacement.to(hidden_states.device, hidden_states.dtype)
                if repl.dim() == 1:
                    repl = repl.view(1, 1, hidden_dim).repeat(1, positions.numel(), 1)
                elif repl.dim() == 2:
                    if repl.shape == (1, hidden_dim):
                        repl = repl.unsqueeze(0).repeat(1, positions.numel(), 1)
                    elif repl.shape == (positions.numel(), hidden_dim):
                        repl = repl.unsqueeze(0)
                    else:
                        raise ValueError(
                            f"Bad replacement shape {tuple(repl.shape)} for {positions.numel()} positions."
                        )
                elif repl.dim() == 3:
                    if repl.shape != (1, positions.numel(), hidden_dim):
                        raise ValueError(
                            f"Bad replacement shape {tuple(repl.shape)}, expected (1, {positions.numel()}, {hidden_dim})"
                        )
                else:
                    raise ValueError("replacement must be 1D, 2D, or 3D")

                if replacing_mode == "addition":
                    new_hidden_states[batch_idx : batch_idx + 1, positions, :] += overlay_strength * repl
                elif replacing_mode == "normalized":
                    new_hidden_states[batch_idx : batch_idx + 1, positions, :] = (
                        overlay_strength * repl
                        + (1.0 - overlay_strength)
                        * new_hidden_states[batch_idx : batch_idx + 1, positions, :]
                    )
                else:
                    raise ValueError("replacing_mode must be 'normalized' or 'addition'")

            return (new_hidden_states,) + inputs[1:]

        return pre_hook_fn

    def _forward_with_injection(
        self,
        source_hidden_states,
        input_ids,
        attention_mask,
        target_layer: int | None,
        tokens_to_interpret: Sequence[Tuple[int, int]],
        target_insert_locations,
        max_new_tokens: int,
        replacing_mode: str,
        overlay_strength: float,
        injection_mode: str,
    ):
        if injection_mode not in {"batch", "aligned"}:
            raise ValueError("injection_mode must be 'batch' or 'aligned'")

        layers = self._get_layers()
        if target_layer is None:
            target_layer = len(layers) - 1
        if not (0 <= target_layer < len(layers)):
            raise IndexError(f"target_layer={target_layer} out of range")

        def get_vec(layer_idx: int, token_idx: int):
            if not (0 <= layer_idx < len(source_hidden_states)):
                raise IndexError(f"retrieve_layer={layer_idx} out of range")
            hs = source_hidden_states[layer_idx]
            if token_idx < 0 or token_idx >= hs.shape[-2]:
                raise IndexError(f"retrieve_token={token_idx} out of range for layer {layer_idx}")
            if hs.dim() == 2:
                return hs[token_idx, :]
            if hs.dim() == 3:
                return hs[0, token_idx, :]
            raise ValueError(f"Unexpected hidden state shape: {tuple(hs.shape)}")

        if injection_mode == "aligned":
            if len(tokens_to_interpret) != len(target_insert_locations):
                raise ValueError("aligned mode requires same token and insert-location counts")
            replacement = torch.stack([get_vec(layer_idx, token_idx) for layer_idx, token_idx in tokens_to_interpret], dim=0)
            batch_insert_infos = [
                {
                    "insert_locations": target_insert_locations,
                    "replacement": replacement,
                    "replacing_mode": replacing_mode,
                    "overlay_strength": overlay_strength,
                }
            ]
            batch_size = 1
        else:
            batch_insert_infos = []
            for retrieve_layer, retrieve_token in tokens_to_interpret:
                batch_insert_infos.append(
                    {
                        "insert_locations": target_insert_locations,
                        "replacement": get_vec(retrieve_layer, retrieve_token),
                        "replacing_mode": replacing_mode,
                        "overlay_strength": overlay_strength,
                    }
                )
            batch_size = len(tokens_to_interpret)

        hook = self._make_pre_hook(batch_insert_infos)
        handle = layers[target_layer].register_forward_pre_hook(hook)

        try:
            expanded_input_ids = input_ids.repeat(batch_size, 1)
            expanded_attention_mask = attention_mask.repeat(batch_size, 1) if attention_mask is not None else None
            prompt_len = expanded_input_ids.shape[1]
            pad_id = _pad_id(self.tokenizer)

            outputs = self.model.generate(
                input_ids=expanded_input_ids,
                attention_mask=expanded_attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=pad_id,
                return_dict_in_generate=True,
            )
            sequences = outputs.sequences
            return {
                "outputs": outputs,
                "sequences": sequences,
                "decoded_texts": self.tokenizer.batch_decode(sequences[:, prompt_len:], skip_special_tokens=True),
                "full_texts": self.tokenizer.batch_decode(sequences, skip_special_tokens=True),
                "batch_insert_infos": batch_insert_infos,
                "target_prompt_len": prompt_len,
                "injection_mode": injection_mode,
            }
        finally:
            handle.remove()
