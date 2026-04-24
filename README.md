# selfie-agent

Small, class-based utilities for running hidden-state injection and interpretation experiments on Hugging Face causal language models.

## Install from Git

```bash
pip install "git+https://github.com/abhipa871/selfie-agent.git"
```

## Quick Example

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
    source_layer=-1,  # which hidden_states index to read for every answer token; -1 = last
    target_layer=0,
    original_max_new_tokens=32,
    interpreter_max_new_tokens=120,
    injection_mode="aligned",
    # For Llama-2-70b-chat, use the legacy in-user-turn INST framing (see below).
    interpretation_style="llama_instruct",
    # optional: override default placeholder="- " (each must be 1 token for the tokenizer)
    # placeholder="_",
)

print(result["original_answer"])
print(result["interpretation_answers"][0])
```

**Gemma 2 and Qwen 3 / 2.5** use the same chat templates as in Hugging Face; use the default interpretation framing (or pass `interpretation_style="gemma"` or `"qwen"`, which are equivalent to `"universal"`). Pass `style="llama_instruct"` to `make_interpretation_prompt` (or `interpretation_style="llama_instruct"` to `interpret`) only for older Llama-2-Chat–style user strings that embed INST open/close markers in the user turn.

If a checkpoint requires custom modeling code, use `ModelLoader().load("namespace/model", trust_remote_code=True)`.

## Package Layout

- `selfie_agent.loader`: model/tokenizer loading
- `selfie_agent.compat`: interpretation prompt styles, decoder layer lookup, device resolution
- `selfie_agent.prompts`: interpretation prompt construction
- `selfie_agent.interpreter`: hidden-state extraction, injection, and interpretation pipeline
- `selfie_agent.utils`: small shared helpers
