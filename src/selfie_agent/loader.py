from __future__ import annotations

from typing import Tuple

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


class ModelLoader:
    def load(
        self,
        model_path: str,
        four_bit_quant: bool = False,
        trust_remote_code: bool = False,
    ) -> Tuple[AutoModelForCausalLM, AutoTokenizer, AutoConfig]:
        bnb_config = None
        if four_bit_quant:
            try:
                import bitsandbytes  # noqa: F401
            except ImportError as e:
                raise ImportError(
                    "four_bit_quant=True requires bitsandbytes. Reinstall the package or run: "
                    "pip install 'bitsandbytes==0.49.2' (or a compatible version)"
                ) from e
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

        loader_kw = {"trust_remote_code": trust_remote_code}
        tokenizer = AutoTokenizer.from_pretrained(model_path, **loader_kw)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model_kwargs = {**loader_kw, "device_map": "auto"}
        if bnb_config is not None:
            model_kwargs["quantization_config"] = bnb_config
        else:
            model_kwargs["torch_dtype"] = torch.bfloat16

        model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        config = AutoConfig.from_pretrained(model_path, **loader_kw)

        return model, tokenizer, config
