"""
`LangGraph <https://github.com/langchain-ai/langgraph>`_ helpers: graph state lines up 1:1 with
:meth:`SelfieInterpreter.interpret` parameters, plus the keys that method returns.

Install: ``pip install selfie-agent[langgraph]`` (or ``pip install langgraph``).

**Usage**

* Build kwargs from a state mapping: :func:`state_to_interpret_kwargs`
* Run interpret and merge results into state: :func:`run_selfie_interpret_state`
* Single-node graph: :func:`compile_selfie_interpret_graph` (requires ``langgraph``)
"""
from __future__ import annotations

import inspect
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, TypedDict, Union

from .interpreter import SelfieInterpreter

# Keep in sync with :meth:`SelfieInterpreter.interpret` (parameters only, not ``self``).
INTERPRET_PARAM_NAMES: frozenset[str] = frozenset(
    {
        "original_prompt",
        "tokens_to_interpret",
        "interpretation_prompt",
        "target_layer",
        "original_max_new_tokens",
        "interpreter_max_new_tokens",
        "replacing_mode",
        "overlay_strength",
        "answer_only",
        "injection_mode",
        "batch_num_placeholders",
        "interpretation_suffix",
        "interpretation_style",
        "source_layer",
        "placeholder",
        "enable_thinking",
        "assistant_prefill_suffix",
        "gemma4_final_answer_tokens_only",
        "generation_kwargs",
        "interpreter_generation_kwargs",
    }
)

# Return keys from :meth:`SelfieInterpreter.interpret` (for typing / docs).
INTERPRET_RESULT_KEYS: frozenset[str] = frozenset(
    {
        "original_full_text",
        "original_answer",
        "interpretation_answers",
        "interpretation_full_texts",
        "tokens_to_interpret",
        "original_outputs",
        "interpretation_outputs",
        "original_prompt_len",
        "answer_indices",
        "target_prompt_len",
        "batch_insert_infos",
        "injection_mode",
        "interpretation_prompt",
    }
)


def _verify_interpret_params() -> None:
    sig = inspect.signature(SelfieInterpreter.interpret)
    from_method = {p for p in sig.parameters if p != "self"}
    if from_method != INTERPRET_PARAM_NAMES:
        raise RuntimeError(
            f"langgraph_state.INTERPRET_PARAM_NAMES out of sync with SelfieInterpreter.interpret: "
            f"missing {from_method - INTERPRET_PARAM_NAMES}, extra {INTERPRET_PARAM_NAMES - from_method}"
        )


_verify_interpret_params()


def normalize_tokens_to_interpret(
    value: Any,
) -> Union[str, Sequence[Tuple[int, int]]]:
    """Accept ``\"all\"``, a sequence of ``(layer, idx)`` pairs, or JSON-style ``[[L,i], ...]``."""
    if isinstance(value, str):
        if value == "all":
            return "all"
        raise ValueError(f'tokens_to_interpret string must be "all", got {value!r}')
    if not isinstance(value, (list, tuple)):
        raise TypeError(
            f"tokens_to_interpret must be 'all' or a list of (layer, token_idx) pairs, got {type(value)}"
        )
    out: List[Tuple[int, int]] = []
    for j, item in enumerate(value):
        if isinstance(item, (list, tuple)) and len(item) == 2:
            out.append((int(item[0]), int(item[1])))
            continue
        raise ValueError(
            f"tokens_to_interpret[{j}] must be a pair [layer, idx], got {item!r}"
        )
    return out


def state_to_interpret_kwargs(state: Mapping[str, Any]) -> Dict[str, Any]:
    """Extract keyword arguments for :meth:`SelfieInterpreter.interpret` from a state mapping.

    Only keys listed in :data:`INTERPRET_PARAM_NAMES` that are **present** in ``state`` are passed
    (so defaults from ``interpret`` apply to omitted keys). ``tokens_to_interpret`` is normalized
    via :func:`normalize_tokens_to_interpret` when present.
    """
    kwargs: Dict[str, Any] = {}
    for name in INTERPRET_PARAM_NAMES:
        if name not in state:
            continue
        if name == "tokens_to_interpret":
            kwargs[name] = normalize_tokens_to_interpret(state[name])
        else:
            kwargs[name] = state[name]
    return kwargs


def run_selfie_interpret_state(
    state: Mapping[str, Any],
    agent: SelfieInterpreter,
    *,
    merge_input: bool = True,
) -> Dict[str, Any]:
    """Call ``agent.interpret`` using :func:`state_to_interpret_kwargs`. Returns interpret output, optionally merged
    on top of ``state`` (needed for ``langgraph`` ``StateGraph(dict)`` so prior keys are not dropped).
    """
    kwargs = state_to_interpret_kwargs(state)
    out = agent.interpret(**kwargs)
    if not merge_input:
        return dict(out)
    return {**dict(state), **out}


def make_selfie_interpret_node(
    agent: SelfieInterpreter,
    *,
    merge_input: bool = True,
):
    """Return a LangGraph node `` callable(state) -> state_update `` that runs :meth:`SelfieInterpreter.interpret`."""

    def _node(state: Mapping[str, Any]) -> Dict[str, Any]:
        return run_selfie_interpret_state(state, agent, merge_input=merge_input)

    return _node


def compile_selfie_interpret_graph(
    agent: SelfieInterpreter,
    *,
    node_name: str = "selfie_interpret",
    merge_input: bool = True,
):
    """
    Build a one-node graph: ``START -> selfie_interpret -> END``.

    Requires the ``langgraph`` package. Raises ``ImportError`` if not installed.
    """
    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "langgraph is required. Install with: pip install 'selfie-agent[langgraph]'"
        ) from e

    g = StateGraph(dict)
    g.add_node(node_name, make_selfie_interpret_node(agent, merge_input=merge_input))
    g.add_edge(START, node_name)
    g.add_edge(node_name, END)
    return g.compile()


class SelfieInterpretGraphState(TypedDict, total=False):
    """All :meth:`SelfieInterpreter.interpret` parameters (optional) plus its return fields (after a run)."""

    original_prompt: str
    tokens_to_interpret: Union[str, List[List[int]], List[Tuple[int, int]]]
    interpretation_prompt: Any
    target_layer: int
    original_max_new_tokens: int
    interpreter_max_new_tokens: int
    replacing_mode: str
    overlay_strength: float
    answer_only: bool
    injection_mode: str
    batch_num_placeholders: Optional[int]
    interpretation_suffix: str
    interpretation_style: str
    source_layer: Optional[int]
    placeholder: str
    enable_thinking: bool
    assistant_prefill_suffix: bool
    gemma4_final_answer_tokens_only: bool
    generation_kwargs: Optional[Mapping[str, Any]]
    interpreter_generation_kwargs: Optional[Mapping[str, Any]]
    original_full_text: str
    original_answer: str
    interpretation_answers: List[str]
    interpretation_full_texts: List[str]
    original_outputs: Any
    interpretation_outputs: Any
    original_prompt_len: int
    answer_indices: List[int]
    target_prompt_len: int
    batch_insert_infos: Any
