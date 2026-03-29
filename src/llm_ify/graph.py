"""LangGraph StateGraph wiring for the LLM-IFY pipeline.

Orchestrates the four NERFIFY-inspired stages as a directed graph:

    summarizer → citation_crawler → got_coder → critique
                                                   │
                                                   ▼
                                               (conditional)
                                             pass? → END
                                             fail? → got_coder (repair loop)

The graph uses ``PipelineState`` as its shared state schema and
delegates actual work to the four agent node functions.
"""

from __future__ import annotations

from typing import Any, Dict

from langgraph.graph import END, StateGraph

from llm_ify.agents.citation_crawler import citation_crawler_node
from llm_ify.agents.critique import critique_node
from llm_ify.agents.got_coder import got_coder_node
from llm_ify.agents.summarizer import summarizer_node
from llm_ify.state import PipelineState

# ---------------------------------------------------------------------------
# Max repair iterations before forced termination (Table 5 ablation)
# ---------------------------------------------------------------------------
MAX_REFINEMENT_LOOPS = 5


# ---------------------------------------------------------------------------
# Conditional edge: should we loop back for repairs?
# ---------------------------------------------------------------------------

def _should_refine(state: PipelineState) -> str:
    """Route after critique: loop back or finish.

    Termination criteria (§3.2, Stage 4):
    1. smoke_test_passed is True  →  END
    2. refinement_iteration ≥ MAX_REFINEMENT_LOOPS  →  END
    3. Otherwise  →  loop back to got_coder for repair
    """
    if state.get("smoke_test_passed", False):
        return "end"
    if state.get("refinement_iteration", 0) >= MAX_REFINEMENT_LOOPS:
        return "end"
    return "repair"


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    """Construct and compile the LLM-IFY LangGraph pipeline.

    Returns
    -------
    compiled : CompiledGraph
        Ready to ``.invoke()`` with an initial ``PipelineState``.
    """
    graph = StateGraph(PipelineState)

    # ── Register nodes ──────────────────────────────────────────────────
    graph.add_node("summarizer", summarizer_node)
    graph.add_node("citation_crawler", citation_crawler_node)
    graph.add_node("got_coder", got_coder_node)
    graph.add_node("critique", critique_node)

    # ── Linear edges: Stage 1 → 2 → 3 → 4 ─────────────────────────────
    graph.set_entry_point("summarizer")
    graph.add_edge("summarizer", "citation_crawler")
    graph.add_edge("citation_crawler", "got_coder")
    graph.add_edge("got_coder", "critique")

    # ── Conditional edge: critique → END or → got_coder (repair) ───────
    graph.add_conditional_edges(
        "critique",
        _should_refine,
        {
            "end": END,
            "repair": "got_coder",
        },
    )

    return graph.compile()


# ---------------------------------------------------------------------------
# Convenience runner
# ---------------------------------------------------------------------------

def run_pipeline(pdf_path: str, cfg_rules: str = "") -> Dict[str, Any]:
    """One-shot pipeline execution.

    Parameters
    ----------
    pdf_path : str
        Absolute path to the research paper PDF.
    cfg_rules : str
        Contents of ``.agent/rules/hf_cfg.md`` to inject as constraints.

    Returns
    -------
    final_state : dict
        The terminal ``PipelineState`` after all stages complete.
    """
    app = build_graph()
    initial_state: PipelineState = {
        "pdf_path": pdf_path,
        "cfg_rules": cfg_rules,
        "errors": [],
        "messages": [],
    }
    return app.invoke(initial_state)
