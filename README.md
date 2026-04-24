# selfie-agent

Small, class-based utilities for running hidden-state injection and interpretation experiments on Hugging Face causal language models.

## Install from Git

```bash
pip install "git+https://github.com/<your-username>/selfie-agent.git"
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
    target_layer=0,
    original_max_new_tokens=32,
    interpreter_max_new_tokens=120,
    injection_mode="aligned",
)

print(result["original_answer"])
print(result["interpretation_answers"][0])
```

## Package Layout

- `selfie_agent.loader`: model/tokenizer loading
- `selfie_agent.prompts`: interpretation prompt construction
- `selfie_agent.interpreter`: hidden-state extraction, injection, and interpretation pipeline
- `selfie_agent.utils`: small shared helpers
