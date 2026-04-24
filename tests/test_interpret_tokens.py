import unittest

import torch

from selfie_agent.interpreter import _validate_tokens_to_interpret


class TestInterpretTokensToInterpret(unittest.TestCase):
    def test_valid_indices_pass(self) -> None:
        source_hs = (torch.zeros(3, 8), torch.zeros(3, 8))
        _validate_tokens_to_interpret(source_hs, [(0, 0), (1, 2)])

    def test_token_index_out_of_range(self) -> None:
        source_hs = (torch.zeros(2, 4),)
        with self.assertRaisesRegex(IndexError, "not a global input_ids index"):
            _validate_tokens_to_interpret(source_hs, [(0, 0), (0, 2)])

    def test_layer_index_out_of_range(self) -> None:
        source_hs = (torch.zeros(1, 4),)
        with self.assertRaisesRegex(IndexError, "hidden state tensors are indexed"):
            _validate_tokens_to_interpret(source_hs, [(1, 0)])

    def test_empty_slice_rejects_any_token(self) -> None:
        source_hs = (torch.zeros(0, 4),)
        with self.assertRaisesRegex(IndexError, "0 position"):
            _validate_tokens_to_interpret(source_hs, [(0, 0)])


if __name__ == "__main__":
    unittest.main()
