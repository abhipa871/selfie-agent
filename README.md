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
    interpretation_style="llama_instruct",  # Llama-2-Chat user-string framing (see below)
)

print(result["original_answer"])
print(result["interpretation_answers"][0])  # single string in aligned mode
```

**Chat templates:** Gemma 2 and most Hugging Face instruct models use the default framing (`interpretation_style="universal"`, or `"gemma"` / `"qwen"`, which are equivalent). Use `interpretation_style="llama_instruct"` only when the *user* text itself uses legacy `[INST]…[/INST]` wrapping.

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

- `num_placeholders=len(tokens_to_interpret)` (after `"all"` is expanded).
- `interpretation_suffix` (default: summarize instruction).
- `interpretation_style` / `placeholder` / `enable_thinking`.

**Custom prompt:** build `InterpretationPrompt(tokenizer, user_prompt_sequence, placeholder="…", enable_thinking=False)` where `user_prompt_sequence` is a mix of **strings** (literal user text) and **`0`** markers (each `0` becomes one `placeholder` in the user message, in order). The tokenizer’s chat template must turn each appended `placeholder` into **exactly one new token** in context (validated at construction). Example for a generic chat model:

```python
from selfie_agent import InterpretationPrompt

interpretation_prompt = InterpretationPrompt(
    tokenizer,
    (
        0,
        0,
        0,
        "\nSummarize this message in two sentences:",
    ),
    placeholder="- ",
    enable_thinking=False,
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

---

## Package layout

- `selfie_agent.loader`: model/tokenizer loading
- `selfie_agent.compat`: interpretation prompt styles, decoder layer lookup, device resolution, `apply_chat_template_with_thinking`
- `selfie_agent.prompts`: `InterpretationPrompt`
- `selfie_agent.interpreter`: `SelfieInterpreter`, hidden-state extraction, injection, `interpret`
- `selfie_agent.utils`: shared helpers (e.g. thinking-strip)
- `selfie_agent.generation`: `prepare_generation_kwargs`, `PresencePenaltyLogitsProcessor`
