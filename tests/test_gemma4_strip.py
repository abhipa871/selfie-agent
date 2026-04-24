import unittest

from selfie_agent.gemma4 import _collapse_duplicate_stanzas, strip_gemma4_display


class TestGemma4Strip(unittest.TestCase):
    def test_collapse_duplicate_stanzas_two_paragraphs(self) -> None:
        a = "The highest mountain in the world is Mount Everest."
        s = a + "\n\n" + a
        r = _collapse_duplicate_stanzas(s)
        self.assertEqual(r.strip(), a)

    def test_collapse_ignores_different_stanzas(self) -> None:
        s = "First paragraph.\n\nSecond paragraph is different."
        self.assertEqual(_collapse_duplicate_stanzas(s), s)

    def test_strip_gemma4_display_dup_paragraph(self) -> None:
        a = "The answer is forty-two."
        s = a + "\n\n" + a
        r = strip_gemma4_display(s)
        self.assertIn("forty-two", r)
        self.assertEqual(r.count("forty-two"), 1)

    def test_strip_empty_thought_block_then_dedup(self) -> None:
        block = "<|channel>thought\n<channel|>"
        a = "Short final."
        s = block + a + "\n\n" + a
        r = strip_gemma4_display(s)
        self.assertIn("Short final", r)
        self.assertEqual(r.count("Short final"), 1)

    def test_answer_stop_id_set_exported(self) -> None:
        from selfie_agent import answer_stop_id_set

        self.assertTrue(callable(answer_stop_id_set))


if __name__ == "__main__":
    unittest.main()
