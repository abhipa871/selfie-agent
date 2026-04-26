"""Tests for LangGraph state ↔ :meth:`SelfieInterpreter.interpret` mapping (no model required)."""
import inspect
import unittest

from selfie_agent.interpreter import SelfieInterpreter
from selfie_agent.langgraph_state import (
    INTERPRET_PARAM_NAMES,
    normalize_tokens_to_interpret,
    state_to_interpret_kwargs,
)


class TestLanggraphInterpretKwargs(unittest.TestCase):
    def test_param_names_match_interpret_signature(self) -> None:
        sig = inspect.signature(SelfieInterpreter.interpret)
        from_method = {p for p in sig.parameters if p != "self"}
        self.assertEqual(from_method, INTERPRET_PARAM_NAMES)

    def test_normalize_all(self) -> None:
        self.assertEqual(normalize_tokens_to_interpret("all"), "all")

    def test_normalize_pairs(self) -> None:
        self.assertEqual(
            normalize_tokens_to_interpret([(0, 1), (2, 3)]),
            [(0, 1), (2, 3)],
        )
        self.assertEqual(
            normalize_tokens_to_interpret([[0, 1], [2, 3]]),
            [(0, 1), (2, 3)],
        )

    def test_state_to_kwargs_subsets_and_drops_unknown(self) -> None:
        s = {
            "original_prompt": "hello",
            "tokens_to_interpret": "all",
            "source_layer": -1,
            "enable_thinking": True,
            "not_a_param": 123,
        }
        k = state_to_interpret_kwargs(s)
        self.assertNotIn("not_a_param", k)
        self.assertEqual(k["original_prompt"], "hello")
        self.assertEqual(k["tokens_to_interpret"], "all")
        self.assertEqual(k["source_layer"], -1)
        self.assertTrue(k["enable_thinking"])


class TestCompileGraphOptional(unittest.TestCase):
    def test_compile_selfie_interpret_graph_imports(self) -> None:
        try:
            from selfie_agent.langgraph_state import compile_selfie_interpret_graph
        except ImportError:
            self.skipTest("langgraph not installed")
        # Graph compiles without running interpret (agent unused if never invoked — we only test compile)
        class _Dummy:
            pass

        g = compile_selfie_interpret_graph(_Dummy())  # type: ignore[arg-type]
        self.assertIsNotNone(g)


if __name__ == "__main__":
    unittest.main()
