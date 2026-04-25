"""Gemma 4 final-channel prefix detection (token subsequence)."""
import unittest

import torch

from selfie_agent.gemma4 import (
    first_global_index_after_gemma4_final_channel_prefix,
)
from selfie_agent.gemma4 import _find_subsequence


class _Tok:
    """Minimal tokenizer: fixed encodings for tests."""

    def encode(self, s: str, add_special_tokens: bool = False) -> list[int]:
        if s == "<|channel|>final":
            return [200, 201]
        if s == "<|channel|> final":
            return [200, 202, 203]
        raise ValueError(s)


class TestGemma4Channel(unittest.TestCase):
    def test_find_subsequence(self) -> None:
        self.assertEqual(_find_subsequence([1, 2, 3, 4, 5], [3, 4]), 2)
        self.assertIsNone(_find_subsequence([1, 2], [1, 2, 3]))

    def test_final_index_after_prefix(self) -> None:
        tok = _Tok()
        # prompt_len=1: gen = [10, 200, 201, 7, 8, 9] — marker [200,201] at gen[1:3];
        # first index after marker: 1 + 1 + 2 = 4
        row = torch.tensor([0, 10, 200, 201, 7, 8, 9], dtype=torch.long)
        idx = first_global_index_after_gemma4_final_channel_prefix(tok, row, prompt_len=1)
        self.assertEqual(idx, 4)

    def test_no_match_returns_none(self) -> None:
        tok = _Tok()
        row = torch.tensor([0, 1, 2, 3], dtype=torch.long)
        self.assertIsNone(
            first_global_index_after_gemma4_final_channel_prefix(tok, row, prompt_len=1)
        )


if __name__ == "__main__":
    unittest.main()
