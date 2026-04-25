from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple

import torch

from .compat import (
    InterpretationStyle,
    apply_chat_template_with_thinking,
    get_decoder_layers,
    interpretation_user_prompt_sequence,
    resolve_model_device,
)
from .generation import build_generation_kwargs
from .gemma4 import first_global_index_after_gemma4_final_channel_prefix
from .prompts import InterpretationPrompt
from .utils import clean_thinking


# Strings that map to a single id for *generate* end-of-sequence and answer-span boundaries.
# Llama 2/3, Gemma 2/3/4, Qwen: template-specific EOT often differs from ``tokenizer.eos_token_id`` alone.
def _add_marker_ids_from_strings(tokenizer, markers: tuple[str, ...], target: set[int]) -> None:
    unk = getattr(tokenizer, "unk_token_id", None)
    for marker in markers:
        try:
            tid = tokenizer.convert_tokens_to_ids(marker)
        except Exception:  # noqa: BLE001
            tid = None
        if tid is not None:
            t = int(tid)
            if t >= 0 and (unk is None or t != unk):
                target.add(t)
        try:
            enc = tokenizer.encode(marker, add_special_tokens=False)
        except Exception:  # noqa: BLE001
            enc = []
        if len(enc) == 1:
            t = int(enc[0])
            if unk is None or t != unk:
                target.add(t)


# Chat turn / end markers: stop ``generate`` and end assistant answer spans. Covers Llama 2/3, Gemma 2/3/4, Qwen.
def _chat_stopping_markers() -> tuple[str, ...]:
    return (
        # Meta Llama 2 (HF chat) + legacy
        "</s>",
        # Gemma 2/3/4, many Gemma / HF templates
        "<end_of_turn>",
        "<|end_of_turn|>",
        "<|turn|>",
        "<turn|>",
        # ChatML / Qwen assistant end
        "<|im_end|>",
        # Meta Llama 3+ Instruct: assistant end — must be in generate ``eos_token_id``
        "<|eot_id|>",
        "<|eot|>",
        "<|eom_id|>",  # some Llama 3.1+ configs
        "<|end_of_text|>",
    )


# Layout / non-content: skip (after checking stops) in :meth:`get_hidden_states_from_sequences` — not for stripping decode.
def _chat_layout_drop_markers() -> tuple[str, ...]:
    return (
        "<bos>",
        "<eos>",
        "<pad>",
        "<start_of_turn>",
        "<end_of_turn>",
        "<end_of_turn>\n",
        "<|begin_of_text|>",
        "<|end_of_text|>",
        "<|eot_id|>",
        "<|eom_id|>",
        "<|im_end|>",
        # Qwen / chatml (if echoed in assistant)
        "<|im_start|>",
        # Meta Llama 3.x chat template header ids (if echoed)
        "<|start_header_id|>",
        "<|end_header_id|>",
        "<|start_header|>",
        "<|end_header|>",
    )


def _merge_eos_string_id(tokenizer, ids: set[int]) -> None:
    """Attach tokenizer.eos string id when it encodes to a single id (Llama-2-Chat </s>, etc.)."""
    s = getattr(tokenizer, "eos_token", None)
    if not s or not isinstance(s, str):
        return
    try:
        enc = tokenizer.encode(s, add_special_tokens=False)
    except Exception:  # noqa: BLE001
        return
    if len(enc) == 1:
        unk = getattr(tokenizer, "unk_token_id", None)
        t = int(enc[0])
        if unk is None or t != unk:
            ids.add(t)


def _validate_tokens_to_interpret(
    source_hs: Sequence[torch.Tensor],
    tokens_to_interpret: Sequence[Tuple[int, int]],
) -> None:
    """Ensure ``(layer, token_idx)`` pairs are in range for the sliced **answer** ``source_hs``."""
    if not source_hs:
        return
    n = int(source_hs[0].shape[-2])
    n_layers = len(source_hs)
    for j, (layer_idx, token_idx) in enumerate(tokens_to_interpret):
        li, ti = int(layer_idx), int(token_idx)
        if not (0 <= li < n_layers):
            raise IndexError(
                f"tokens_to_interpret[{j}] layer {layer_idx} is out of range; "
                f"hidden state tensors are indexed 0..{n_layers - 1}."
            )
        if n == 0 or not (0 <= ti < n):
            raise IndexError(
                f"tokens_to_interpret[{j}] has token index {token_idx} but the answer "
                f"hidden-state slice has {n} position(s) (use 0..{n - 1} when non-empty). "
                f"The second element is a 0-based index along the answer hidden-state span, in the same "
                f"order as result['answer_indices']; it is not a global input_ids index."
            )


def _pad_id(tokenizer) -> int:
    if tokenizer.pad_token_id is not None:
        return int(tokenizer.pad_token_id)
    if tokenizer.eos_token_id is not None:
        return int(tokenizer.eos_token_id)
    raise ValueError("Tokenizer has neither pad_token_id nor eos_token_id")


def _eos_token_ids_for_stopping(tokenizer) -> List[int]:
    """Token ids that should end generation and the visible assistant span.

    Gemma 2 chat often finishes with ``<end_of_turn>``, which is not always the same id as
    ``tokenizer.eos_token_id``; ``generate`` must receive every such id or it will emit EOT
    as a normal token and run until ``max_new_tokens``.

    Meta **Llama 3+** Instruct uses ``<|eot_id|>`` and related ids; **Llama 2** HF chat and **Gemma 2/3/4**
    use ``<end_of_turn>`` / ``</s>`` / etc., often distinct from a single ``eos_token_id`` field.
    """
    ids: set[int] = set()
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is not None:
        if isinstance(eos, (list, tuple)):
            ids.update(int(x) for x in eos)
        else:
            ids.add(int(eos))
    for attr in ("eot_id", "im_end_id"):
        tid = getattr(tokenizer, attr, None)
        if tid is not None:
            ids.add(int(tid))
    _add_marker_ids_from_strings(tokenizer, _chat_stopping_markers(), ids)
    _merge_eos_string_id(tokenizer, ids)
    return sorted(ids)


def _special_token_ids_to_drop(tokenizer) -> set[int]:
    """Ids treated as non-content (layout, specials) when walking generated assistant text."""
    ids: set[int] = set()

    for attr in ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id"):
        tid = getattr(tokenizer, attr, None)
        if tid is None:
            continue
        if isinstance(tid, (list, tuple)):
            ids.update(int(x) for x in tid)
        else:
            ids.add(int(tid))

    _add_marker_ids_from_strings(tokenizer, _chat_layout_drop_markers(), ids)
    return ids
def _prefix_stream_control_token_indices(tokenizer, seq: List[int]) -> Set[int]:
    """Contiguous indices from position 0 that are BOS / stream-start specials (not conversational content)."""
    if not seq:
        return set()
    starter: Set[int] = set()
    bos = getattr(tokenizer, "bos_token_id", None)
    if bos is not None:
        starter.add(int(bos))
    for marker in (
        "<|begin_of_text|>",
        "<|startoftext|>",
        "<s>",
    ):
        try:
            tid = int(tokenizer.convert_tokens_to_ids(marker))
        except Exception:
            continue
        unk = getattr(tokenizer, "unk_token_id", None)
        if unk is not None and tid == unk:
            continue
        if tid >= 0:
            starter.add(tid)
    exclude: Set[int] = set()
    i = 0
    while i < len(seq) and seq[i] in starter:
        exclude.add(i)
        i += 1
    return exclude


def _continuation_leading_layout_indices(
    tokenizer,
    seq: List[int],
    prompt_len: int,
    max_skip: int = 64,
) -> Set[int]:
    """Indices at the start of the generated span that are whitespace / layout-only (no visible text)."""
    exclude: Set[int] = set()
    L = len(seq)
    i = prompt_len
    n = 0
    while i < L and n < max_skip:
        piece = tokenizer.decode([seq[i]], skip_special_tokens=False)
        if piece.strip() == "":
            exclude.add(i)
            i += 1
            n += 1
            continue
        break
    return exclude


def _exclude_indices_for_full_sequence_span(
    tokenizer,
    input_ids_row: torch.Tensor,
    prompt_len: int,
) -> Set[int]:
    """When ``answer_only=False``, drop stream-start specials and leading layout in the completion."""
    seq = input_ids_row.tolist()
    out = _prefix_stream_control_token_indices(tokenizer, seq)
    out.update(_continuation_leading_layout_indices(tokenizer, seq, prompt_len))
    return out


def _eos_token_id_for_generate(tokenizer) -> int | List[int] | None:
    stops = _eos_token_ids_for_stopping(tokenizer)
    if not stops:
        return getattr(tokenizer, "eos_token_id", None)
    if len(stops) == 1:
        return stops[0]
    return stops


def _stop_ids_for_answer_span(tokenizer) -> Set[int]:
    """Token ids that end the assistant 'answer' for hidden-state / debug slices (Gemma eot, EOS, im_end, …)."""
    return set(_eos_token_ids_for_stopping(tokenizer))


def answer_stop_id_set(tokenizer) -> Set[int]:
    """Return the same set of token ids used to delimit the assistant *answer* in this package.

    Use with custom loops over token ids (e.g. mirroring :meth:`show_answer_tokens`) instead of
    checking only ``tokenizer.eos_token_id``, which misses Gemma-style additional stop tokens.
    """
    return _stop_ids_for_answer_span(tokenizer)


class SelfieInterpreter:
    def __init__(
        self,
        model,
        tokenizer,
        *,
        generation_kwargs: Dict[str, Any] | None = None,
        interpreter_generation_kwargs: Dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.generation_kwargs = generation_kwargs
        self.interpreter_generation_kwargs = interpreter_generation_kwargs

    def make_interpretation_prompt(
        self,
        num_placeholders: int,
        suffix: str = "Summarize this message in two sentences:",
        style: InterpretationStyle = "universal",
        placeholder: str = "_ ",
        enable_thinking: bool = False,
        *,
        assistant_prefill_suffix: bool = True,
    ) -> InterpretationPrompt:
        """Build an :class:`InterpretationPrompt` for the loaded tokenizer's chat template.

        Use any style in :data:`selfie_agent.compat.CHATML_LIKE_STYLES` — e.g. ``"universal"``,
        ``"llama3"``, ``"gemma"`` / ``"gemma2"`` / ``"gemma3"`` / ``"gemma4"``, ``"qwen"`` — for
        the same user-placeholder + optional assistant-prefill layout; only the tokenizer's
        ``apply_chat_template`` differs. For **Llama 2**-style *raw* user strings with ``[INST]``,
        use ``"llama_instruct"`` instead of these.

        For legacy prompts that match original Llama-2-Chat *user* strings with ``[INST]...[/INST]`` inside
        the user turn, pass ``style="llama_instruct"`` (do not use for Llama 3+).

        If ``assistant_prefill_suffix`` is ``True`` (default), the *suffix* is the **start of the
        assistant** message (``{"role": "assistant", "content": suffix}``), and the user turn only
        contains the placeholder pattern — matching the original SelfIE / ``[INST]``…``[/INST]`` +
        assistant prefill layout. If ``False``, the suffix is part of a single user message
        (older behavior).

        ``placeholder`` is appended for each ``0`` in the built sequence; it must add exactly one token
        with this tokenizer and chat template (see :class:`InterpretationPrompt`).
        """
        if assistant_prefill_suffix:
            return InterpretationPrompt(
                self.tokenizer,
                interpretation_user_prompt_sequence(
                    num_placeholders,
                    suffix,
                    style,
                    user_message_only_placeholders=True,
                ),
                placeholder=placeholder,
                enable_thinking=enable_thinking,
                assistant_prefill=suffix,
            )
        return InterpretationPrompt(
            self.tokenizer,
            interpretation_user_prompt_sequence(
                num_placeholders,
                suffix,
                style,
            ),
            placeholder=placeholder,
            enable_thinking=enable_thinking,
        )

    def get_hidden_states_from_sequences(
        self,
        sequences: torch.LongTensor,
        answer_only: bool = True,
        prompt_len: int | None = None,
        *,
        gemma4_final_answer_tokens_only: bool = False,
    ):
        """Return hidden states sliced to relevant token positions.

        When ``answer_only=False``, ``prompt_len`` is required. Indices exclude stream-start specials
        (e.g. BOS / ``<|begin_of_text|>``) and leading whitespace-only tokens at the start of the
        generated continuation (so layout / sentence-initial spacing is not treated as content).

        If ``gemma4_final_answer_tokens_only`` is ``True`` and ``answer_only`` is ``True``, only
        tokens **after** a detected Gemma 4 *final* channel header are kept (for models that emit
        an empty *thought* channel with thinking disabled; see :mod:`selfie_agent.gemma4`). If no
        header matches, the full assistant span is used (unchanged).
        """
        with torch.no_grad():
            outputs = self.model(
                input_ids=sequences.to(resolve_model_device(self.model)),
                output_hidden_states=True,
                return_dict=True,
            )

        full_hs = tuple(layer[0] for layer in outputs.hidden_states)
        if not answer_only:
            if prompt_len is None:
                raise ValueError("prompt_len must be provided when answer_only=False")
            L = sequences.shape[1]
            exclude = _exclude_indices_for_full_sequence_span(self.tokenizer, sequences[0], prompt_len)
            kept = [i for i in range(L) if i not in exclude]
            answer_hs = tuple(layer_hs[kept, :] for layer_hs in full_hs)
            return outputs, answer_hs, kept

        if prompt_len is None:
            raise ValueError("prompt_len must be provided when answer_only=True")

        input_ids = sequences[0]
        drop_ids = _special_token_ids_to_drop(self.tokenizer)
        stop_ids = _stop_ids_for_answer_span(self.tokenizer)

        scan_start = int(prompt_len)
        if gemma4_final_answer_tokens_only:
            sidx = first_global_index_after_gemma4_final_channel_prefix(
                self.tokenizer, input_ids, int(prompt_len)
            )
            if sidx is not None:
                scan_start = sidx

        answer_indices: List[int] = []
        for i in range(scan_start, int(input_ids.shape[0])):
            token_id = int(input_ids[i].item())

            if token_id in stop_ids:
                break

            if token_id in drop_ids:
                continue

            answer_indices.append(i)

        answer_hs = tuple(layer_hs[answer_indices, :] for layer_hs in full_hs)
        return outputs, answer_hs, answer_indices

    # `answer_indices` / sliced `answer_hs` row *i* align with the second int in
    # ``tokens_to_interpret`` = ``(layer, i)`` in :meth:`interpret` (not global positions).

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
        batch_num_placeholders: int | None = None,
        interpretation_suffix: str = "Summarize this message in two sentences:",
        interpretation_style: InterpretationStyle = "universal",
        source_layer: int | None = None,
        placeholder: str = "_ ",
        enable_thinking: bool = False,
        assistant_prefill_suffix: bool = True,
        gemma4_final_answer_tokens_only: bool = False,
        generation_kwargs: Dict[str, Any] | None = None,
        interpreter_generation_kwargs: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Run the original model generate pass, then the interpretation generate pass with injection.

        original_max_new_tokens limits the first completion (source hidden states).
        interpreter_max_new_tokens limits tokens in the interpretation pass after injection.
        When ``tokens_to_interpret`` is ``"all"``, you must set ``source_layer`` to the index into
        ``outputs.hidden_states`` (0 = embeddings, 1 = after first block, …) for which block’s
        states to use for all answer positions; use ``-1`` for the last index.

        For Meta **Llama 3+**, **Gemma 2 / 3**, **Qwen**, and similar models, any style in
        :data:`selfie_agent.compat.CHATML_LIKE_STYLES` (e.g. ``"universal"``, ``"llama3"``,
        ``"gemma2"``, ``"gemma3"``, ``"gemma4"``) is equivalent for *layout*; pick a name to match
        your checkpoint. For **Gemma 3/4 thinking** models, if the tokenizer expects it, set
        ``enable_thinking=True``. Use ``interpretation_style="llama_instruct"`` for **Llama 2**-style
        raw ``[INST]…[/INST]`` *user* strings; use ``"universal"``/``"llama3"``/``"gemma*"`` when
        `apply_chat_template` defines the turn structure.

        If ``assistant_prefill_suffix`` is ``True`` (default), ``interpretation_suffix`` is the first
        **assistant** content (the paper / SelfIE design); placeholders live only in the **user**
        turn. If ``False``, the suffix is embedded in a single user message. Ignored if you pass
        ``interpretation_prompt`` yourself.

        ``placeholder`` is passed to the default :class:`InterpretationPrompt` only; ignored if you pass
        ``interpretation_prompt`` yourself.

        ``batch_num_placeholders`` is used only when the default prompt is built and ``injection_mode`` is
        ``"batch"``. If ``None`` (default), ``num_placeholders = max(5, len(tokens_to_interpret))``. If an
        integer, that value is used as ``num_placeholders`` (must be >= 1). Ignored in ``aligned`` mode or
        when you pass a custom ``interpretation_prompt``.

        ``enable_thinking`` is passed to ``apply_chat_template`` only if the tokenizer defines that
        argument (default ``False``). Ignored otherwise. For a custom :class:`InterpretationPrompt`, set
        ``enable_thinking`` on that object; this parameter applies only when the default prompt is built.

        If ``gemma4_final_answer_tokens_only`` is ``True`` (default ``False``), when reading hidden
        states for the *original* completion, only token positions **after** the *final* channel
        header are used (Gemma 4 with thinking off still emits an empty *thought* channel; E2B/E4B
        differ). See :mod:`selfie_agent.gemma4`.  No effect when ``answer_only`` is ``False``.

        Optional ``generation_kwargs`` / ``interpreter_generation_kwargs`` (and the same on
        ``SelfieInterpreter(...)``) are merged into ``model.generate`` after required keys
        (``max_new_tokens``, ``eos_token_id``, ``pad_token_id``, ``return_dict_in_generate``). Values
        from :func:`selfie_agent.generation.prepare_generation_kwargs` handle ``presence_penalty`` and
        strip ``min_p=0``. If no extra kwargs are passed, generation uses greedy decoding
        (``do_sample=False``).

        **Placeholder / token alignment:** Each entry in ``tokens_to_interpret`` is ``(layer, i)``
        where ``i`` is **not** a global input_ids position. It indexes the **sliced** answer hidden
        states from :meth:`get_hidden_states_from_sequences` (row ``i`` = the ``i``-th
        value in ``result['answer_indices']`` in order). In ``injection_mode="aligned"``,
        ``tokens_to_interpret[j]`` is injected at the *j*-th ``0`` placeholder in the interpretation
        prompt, in the order those placeholders were laid out in the user sequence (the default prompt
        lists placeholders before the summary suffix, left to right). Use ``"all"`` and ``source_layer`` to
        get ``(source_layer, 0)…(source_layer, K-1)`` automatically, which stays 1:1 with placeholders
        when you use the default :class:`InterpretationPrompt` with ``K`` placeholders.
        """
        original_input_ids, original_attention_mask = self._encode_chat_prompt(
            original_prompt, enable_thinking=enable_thinking
        )
        original_prompt_len = original_input_ids.shape[1]

        pad_id = _pad_id(self.tokenizer)
        eos_gen = _eos_token_id_for_generate(self.tokenizer)
        orig_gen_kw = build_generation_kwargs(
            max_new_tokens=original_max_new_tokens,
            eos_token_id=eos_gen,
            pad_token_id=pad_id,
            instance_kwargs=self.generation_kwargs,
            call_kwargs=generation_kwargs,
        )
        if not (self.generation_kwargs or generation_kwargs):
            orig_gen_kw["do_sample"] = False

        with torch.no_grad():
            original_gen = self.model.generate(
                input_ids=original_input_ids,
                attention_mask=original_attention_mask,
                **orig_gen_kw,
            )

        original_sequences = original_gen.sequences

        _, source_hs, answer_indices = self.get_hidden_states_from_sequences(
            sequences=original_sequences,
            answer_only=answer_only,
            prompt_len=original_prompt_len,
            gemma4_final_answer_tokens_only=gemma4_final_answer_tokens_only,
        )

        def _postprocess_visible_text(s: str) -> str:
            if enable_thinking:
                return s.strip()
            return clean_thinking(s)

        row = original_sequences[0]
        original_full_text = _postprocess_visible_text(
            self.tokenizer.decode(row, skip_special_tokens=False)
        )
        if answer_only and answer_indices:
            a0, a1 = int(answer_indices[0]), int(answer_indices[-1]) + 1
            original_answer = _postprocess_visible_text(
                self.tokenizer.decode(row[a0:a1], skip_special_tokens=False)
            )
        elif answer_only:
            original_answer = ""
        else:
            original_answer = _postprocess_visible_text(
                self.tokenizer.decode(row[original_prompt_len:], skip_special_tokens=False)
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

        _validate_tokens_to_interpret(source_hs, list(tokens_to_interpret))

        if interpretation_prompt is None:
            n_tok = len(list(tokens_to_interpret))
            if injection_mode == "batch":
                if batch_num_placeholders is not None:
                    if batch_num_placeholders < 1:
                        raise ValueError("batch_num_placeholders must be >= 1 when set")
                    num_placeholders = batch_num_placeholders
                else:
                    num_placeholders = max(5, n_tok)
            elif injection_mode == "aligned":
                num_placeholders = n_tok
            else:
                raise ValueError("injection_mode must be 'batch' or 'aligned'")

            interpretation_prompt = self.make_interpretation_prompt(
                num_placeholders=num_placeholders,
                suffix=interpretation_suffix,
                style=interpretation_style,
                placeholder=placeholder,
                enable_thinking=enable_thinking,
                assistant_prefill_suffix=assistant_prefill_suffix,
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

        interp_gen_kw = build_generation_kwargs(
            max_new_tokens=interpreter_max_new_tokens,
            eos_token_id=eos_gen,
            pad_token_id=pad_id,
            instance_kwargs=self.interpreter_generation_kwargs,
            call_kwargs=interpreter_generation_kwargs,
        )
        if not (self.interpreter_generation_kwargs or interpreter_generation_kwargs):
            interp_gen_kw["do_sample"] = False

        result = self._forward_with_injection(
            source_hidden_states=source_hs,
            input_ids=target_inputs["input_ids"],
            attention_mask=target_inputs.get("attention_mask"),
            target_layer=target_layer,
            tokens_to_interpret=tokens_to_interpret,  # type: ignore[arg-type]
            target_insert_locations=interpretation_prompt.insert_locations,
            generation_kwargs=interp_gen_kw,
            replacing_mode=replacing_mode,
            overlay_strength=overlay_strength,
            injection_mode=injection_mode,
            decode_skip_special_tokens=True,
        )

        interpretation_answers = [_postprocess_visible_text(x) for x in result["decoded_texts"]]
        interpretation_full_texts = [_postprocess_visible_text(x) for x in result["full_texts"]]

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

    def show_answer_tokens(
        self,
        original_sequences: torch.LongTensor,
        prompt_len: int,
        *,
        only_global_indices: Sequence[int] | None = None,
    ):
        toks = original_sequences[0]
        rows = []

        if only_global_indices is not None:
            for answer_i, global_i in enumerate(only_global_indices):
                tid = int(toks[int(global_i)].item())
                rows.append({
                    "answer_idx": answer_i,
                    "global_idx": int(global_i),
                    "token_id": tid,
                    "decoded": repr(self.tokenizer.decode([tid], skip_special_tokens=False)),
                })
            return rows

        drop_ids = _special_token_ids_to_drop(self.tokenizer)
        answer_stop_ids = _stop_ids_for_answer_span(self.tokenizer)

        answer_i = 0
        for global_i in range(prompt_len, toks.shape[0]):
            tid = int(toks[global_i].item())

            if tid in answer_stop_ids:
                break

            if tid in drop_ids:
                continue

            rows.append({
                "answer_idx": answer_i,
                "global_idx": global_i,
                "token_id": tid,
                "decoded": repr(self.tokenizer.decode([tid], skip_special_tokens=False)),
            })
            answer_i += 1

        return rows

    def _encode_chat_prompt(self, prompt: str, enable_thinking: bool = False):
        dev = resolve_model_device(self.model)
        messages = [{"role": "user", "content": prompt}]
        encoded = apply_chat_template_with_thinking(
            self.tokenizer,
            messages,
            enable_thinking=enable_thinking,
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
        generation_kwargs: Dict[str, Any],
        replacing_mode: str,
        overlay_strength: float,
        injection_mode: str,
        *,
        decode_skip_special_tokens: bool = False,
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

            outputs = self.model.generate(
                input_ids=expanded_input_ids,
                attention_mask=expanded_attention_mask,
                **generation_kwargs,
            )
            sequences = outputs.sequences
            return {
                "outputs": outputs,
                "sequences": sequences,
                "decoded_texts": self.tokenizer.batch_decode(
                    sequences[:, prompt_len:],
                    skip_special_tokens=decode_skip_special_tokens,
                ),
                "full_texts": self.tokenizer.batch_decode(
                    sequences, skip_special_tokens=decode_skip_special_tokens
                ),
                "batch_insert_infos": batch_insert_infos,
                "target_prompt_len": prompt_len,
                "injection_mode": injection_mode,
            }
        finally:
            handle.remove()
