"""CHATML_LIKE_STYLES share the same interpretation sequence layout (Llama 3, Gemma 2/3, Qwen, …)."""
import unittest

from selfie_agent.compat import CHATML_LIKE_STYLES, interpretation_user_prompt_sequence


class TestChatmlLikeStyles(unittest.TestCase):
    def test_gemma2_gemma3_llama3_match_universal_layout(self) -> None:
        suffix = "Task here."
        for st in (
            "universal",
            "llama3",
            "gemma",
            "gemma2",
            "gemma3",
            "gemma4",
            "qwen",
        ):
            with self.subTest(style=st):
                a = interpretation_user_prompt_sequence(4, suffix, st, user_message_only_placeholders=True)
                b = interpretation_user_prompt_sequence(4, suffix, "universal", user_message_only_placeholders=True)
                self.assertEqual(a, b)
                self.assertEqual(a, (0, 0, 0, 0))

    def test_frozen_set_is_complete_for_literal_stems(self) -> None:
        for st in ("universal", "llama3", "gemma", "gemma2", "gemma3", "gemma4", "qwen"):
            self.assertIn(st, CHATML_LIKE_STYLES)


if __name__ == "__main__":
    unittest.main()
