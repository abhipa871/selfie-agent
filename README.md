# selfie-agent

Small, class-based utilities for running hidden-state injection and interpretation experiments on Hugging Face causal language models.

## Install from Git

```bash
pip install "git+https://github.com/abhipa871/selfie-agent.git"
```

## Quick example

```python
from selfie_agent import ModelLoader, SelfieInterpreter

model, tokenizer, _ = ModelLoader().load(
    "meta-llama/Llama-2-70b-chat-hf",
    four_bit_quant=True,
)

agent = SelfieInterpreter(model=model, tokenizer=tokenizer)
result = agent.interpret(
    original_prompt="What's the highest mountain in the world? Answer in 10 words.",
    tokens_to_interpret="all",
    source_layer=-1,  # hidden_states index for every answer token; -1 = last tensor
    target_layer=0,
    original_max_new_tokens=32,
    interpreter_max_new_tokens=120,
    injection_mode="aligned",
    interpretation_style="universal",  # match CHATML_LIKE_STYLES to checkpoint; Llama-2-Chat is fine
)

print(result["original_answer"])
print(result["interpretation_answers"][0])  # single string in aligned mode
```

**Chat templates:** For **Gemma 2 / 3 / 4**, **Meta Llama 2 (HF chat) / Llama 3+**, **Qwen**, and similar models, the built-in user-placeholder + assistant-prefill layout is the same; use any name in `CHATML_LIKE_STYLES` (e.g. `interpretation_style="gemma4"`, `"llama3"`, or `"universal"`)—only the checkpoint’s `apply_chat_template` differs (Llama-2-Chat’s `[INST]…[/INST]` and newer chat formats are all applied by the tokenizer, not a separate `interpretation_style`). **Gemma 3/4 thinking** models may need `enable_thinking=True` on `interpret()`.

**SelfIE-style user vs assistant (default):** with `assistant_prefill_suffix=True` (default on `interpret()` and `make_interpretation_prompt`), `interpretation_suffix` is the **start of the assistant** turn, and placeholders sit only in the **user** turn (same idea as the original SelfIE / `[INST] _ … [/INST]` + assistant prefill, but via `apply_chat_template` for modern chat models). The template call uses `add_generation_prompt=False` when an assistant message is already present (so you do not get a duplicate assistant header), and passes `continue_final_message=True` when the tokenizer supports it. Set `assistant_prefill_suffix=False` to put the suffix in a single user message (older behavior).

If a checkpoint needs custom modeling code, use `ModelLoader().load("namespace/model", trust_remote_code=True)`.

---

## `SelfieInterpreter.interpret()` — options

The pipeline is always: (1) `generate` on `original_prompt`, (2) read hidden states from that completion, (3) `generate` on the interpretation prompt while injecting those states at `target_layer`.

### `tokens_to_interpret`

| Form | Meaning |
|------|--------|
| `"all"` | Use every kept position from the original answer (see `answer_only`). You **must** pass `source_layer`: an index into `outputs.hidden_states` (`0` = embeddings, `1` = after first block, …). Use `-1` for the **last** index. The interpreter expands this to `[(layer_idx, 0), (layer_idx, 1), …, (layer_idx, n-1)]`. |
| `[(L0, t0), (L1, t1), …]` | Explicit list. Each pair is `(hidden_state_layer_index, token_index)` into the **sliced** hidden-state tensors (`layer` must satisfy `0 <= layer < len(hidden_states)`, i.e. **not** Python `-1` here). Token index `0` is the first answer token when `answer_only=True`. **Do not** pass `source_layer` with this form (it is only valid with `"all"`). |

If the original completion has fewer tokens than an explicit list expects, you will get an index error.

### `injection_mode`: `"batch"` vs `"aligned"`

- **`"aligned"`** (the quick example uses this; the `interpret` default is actually `"batch"`): **one** interpretation `generate` run. The hidden vector for `tokens_to_interpret[i]` is injected **only** at the *i*-th placeholder in the interpretation prompt. Requires **`len(tokens_to_interpret) == len(interpretation_prompt.insert_locations)`**. Result: **`interpretation_answers` has length 1** — one decoded string for the whole run.

- **`"batch"`**: **one batched** `generate` with batch size `len(tokens_to_interpret)`. Row *k* uses the hidden state from `tokens_to_interpret[k]` and injects that **same** vector at **every** placeholder position for that row. Result: **`interpretation_answers` has length `len(tokens_to_interpret)`** — one string per source token experiment.

`target_layer` is which **decoder block** the forward **hook** is registered on (injection site), independent of the layer indices inside `tokens_to_interpret` (where hidden states are **read** from).

### Default interpretation prompt vs custom `InterpretationPrompt`

If `interpretation_prompt` is omitted, the library builds one with `make_interpretation_prompt`:

- `num_placeholders`: in **`batch`** mode, `interpret(..., batch_num_placeholders=…)` when set, else `max(5, len(tokens_to_interpret))` (after `"all"` is expanded). In **`aligned`**, `len(tokens_to_interpret)`.
- `interpretation_suffix` (default: summarize instruction), as **assistant** prefill when `assistant_prefill_suffix=True` (default).
- `interpretation_style` / `placeholder` / `enable_thinking`.

**Custom prompt:** build `InterpretationPrompt(tokenizer, user_prompt_sequence, placeholder="…", enable_thinking=False, assistant_prefill="…"?)` where `user_prompt_sequence` is a mix of **strings** and **`0`** markers. Pass **`assistant_prefill=…`** to put the task text in the **assistant** turn; omit it to keep one user turn only. Each `0` must become **exactly one** new token in the templated model input. Example (Llama 3 / universal chat — placeholders in user, task as assistant prefill):

```python
from selfie_agent import InterpretationPrompt

interpretation_prompt = InterpretationPrompt(
    tokenizer,
    (0, 0, 0),  # user turn: three placeholder slots
    placeholder="- ",
    enable_thinking=False,
    assistant_prefill="Summarize this message in two sentences:",
)

result = agent.interpret(
    original_prompt="What is 2+2? One sentence.",
    # Either tokens_to_interpret="all", source_layer=-1  OR explicit (layer, idx) with layer >= 0
    tokens_to_interpret="all",
    source_layer=-1,
    interpretation_prompt=interpretation_prompt,
    injection_mode="aligned",
    target_layer=0,
)
```

In **`aligned`** mode, the number of **`0`** placeholders in your custom prompt must equal **`len(tokens_to_interpret)`** after expansion (e.g. if `"all"` yields 12 answer tokens, you need 12 placeholders—or omit `interpretation_prompt` so the library builds a matching prompt). **`batch`** mode still requires every `token_idx` in `tokens_to_interpret` to exist in the answer.

When `interpretation_prompt` is provided, `interpretation_suffix`, `interpretation_style`, and `interpret`’s `placeholder` are ignored for building that prompt (but `enable_thinking` on the `InterpretationPrompt` object still matters).

You can also build sequences with `interpretation_user_prompt_sequence(num_placeholders, suffix, style)` from `selfie_agent.compat` and pass that tuple into `InterpretationPrompt`.

**Batch mode** with `"all"` (one interpretation string per answer token; same placeholders flooded with each source vector):

```python
result = agent.interpret(
    original_prompt="What is the capital of France? One word.",
    tokens_to_interpret="all",
    source_layer=-1,
    injection_mode="batch",
    target_layer=0,
)
for i, text in enumerate(result["interpretation_answers"]):
    print(i, text)
```

### Other useful parameters

| Parameter | Role |
|-----------|------|
| `answer_only` | If `True` (default), hidden states use only the generated answer span (after `original_prompt_len`). If `False`, more positions are kept (see docstring on `get_hidden_states_from_sequences`); `prompt_len` is required internally. |
| `gemma4_final_answer_tokens_only` | If `True`, for the **original** completion only, keep hidden states only for tokens **after** the detected *final* channel header (Gemma 4 with thinking disabled often emits an empty *thought* channel first; use with `interpretation_style="gemma4"` / Gemma 4 checkpoints). If the header is not found, the full answer span is used. |
| `replacing_mode` | `"normalized"` (default) or `"addition"` for how injected vectors mix with activations at hook sites. |
| `overlay_strength` | Scalar weight for injection (default `1.0`). |
| `generation_kwargs` / `interpreter_generation_kwargs` | Extra `model.generate` arguments merged after required keys; constructor can set defaults via `SelfieInterpreter(..., generation_kwargs=…, interpreter_generation_kwargs=…)`. Use `prepare_generation_kwargs` if you pass `presence_penalty` or `min_p=0`. If no extras are given, decoding is greedy (`do_sample=False`). |

### Return dict (selected keys)

- `original_answer`, `original_full_text` — first completion.
- `interpretation_answers` — list of decoded interpretation strings (length `1` for aligned, or `len(tokens_to_interpret)` for batch).
- `interpretation_full_texts` — full decoded sequences for each interpretation row.
- `tokens_to_interpret` — final list of `(layer, token_idx)` pairs (after `"all"` expansion).
- `original_outputs`, `interpretation_outputs` — raw `generate` outputs.
- `original_prompt_len`, `answer_indices`, `target_prompt_len`, `interpretation_prompt`, `injection_mode`.

### LangGraph

Optional extra: `pip install "selfie-agent[langgraph]"` (brings in `langgraph`).

Use the same keys as `interpret()` on your graph state: `state_to_interpret_kwargs` builds the keyword dict (only keys present in the state are forwarded, so defaults match `interpret`). `tokens_to_interpret` may be `"all"`, a list of `(layer, idx)` pairs, or JSON-style `[[layer, idx], ...]`. `interpretation_prompt` can be a live `InterpretationPrompt` in memory.

- `run_selfie_interpret_state(state, agent)` — calls `agent.interpret` and, by default, **merges** the return dict into the state (needed because `StateGraph(dict)` replaces state unless the node returns the full mapping).
- `make_selfie_interpret_node(agent)` — node callable for a custom graph.
- `compile_selfie_interpret_graph(agent)` — minimal `START → selfie_interpret → END` graph.
- `SelfieInterpretGraphState` — `TypedDict` of all `interpret` parameters plus return fields (all optional for partial state).

```python
from selfie_agent import ModelLoader, SelfieInterpreter
from selfie_agent import compile_selfie_interpret_graph

model, tokenizer, _ = ModelLoader().load("meta-llama/Llama-2-7b-chat-hf", four_bit_quant=True)
agent = SelfieInterpreter(model=model, tokenizer=tokenizer)
graph = compile_selfie_interpret_graph(agent)

out = graph.invoke(
    {
        "original_prompt": "What is 2+2? One token.",
        "tokens_to_interpret": "all",
        "source_layer": -1,
        "injection_mode": "aligned",
        "interpretation_style": "universal",
    }
)
print(out["interpretation_answers"])
```

---

## Package layout

- `selfie_agent.loader`: model/tokenizer loading
- `selfie_agent.compat`: interpretation prompt styles, decoder layer lookup, device resolution, `apply_chat_template_with_thinking`
- `selfie_agent.prompts`: `InterpretationPrompt`
- `selfie_agent.interpreter`: `SelfieInterpreter`, hidden-state extraction, injection, `interpret`
- `selfie_agent.gemma4`: optional *final*-channel detection for Gemma 4 answer spans
- `selfie_agent.utils`: shared helpers (e.g. thinking-strip)
- `selfie_agent.generation`: `prepare_generation_kwargs`, `PresencePenaltyLogitsProcessor`
- `selfie_agent.langgraph_state`: LangGraph state ↔ `interpret()` (`compile_selfie_interpret_graph`, `state_to_interpret_kwargs`, …)
