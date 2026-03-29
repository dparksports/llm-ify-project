"""Stage 2 — Citation Crawler Agent.

Scans ``state["cleaned_markdown"]`` for components that reference an
external citation but lack the exact mathematical formulation
(e.g., "We adopt the distortion loss from [3]").

Resolution strategy (in priority order):
1. Check the local knowledge base (``.agent/skills/dependencies.md``)
   for pre-resolved math / PyTorch snippets.
2. Ask GPT-4o (which has web-search capability) to retrieve the
   mathematical formulation and canonical PyTorch implementation
   for the cited component.

Returns ``{"resolved_components": findings_dict}`` where each key is
the component name and the value is the math + code snippet.

References:
    Paper §3.2, Stage 2  — Compositional Dependency Resolution
    Paper Figure 3       — Citation dependency graphs
    Table 5 ablation     — Citation recovery: HIGH priority (1.0→0.65)
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from llm_ify.state import PipelineState


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_LOCAL_KB_PATH = (
    Path(__file__).resolve().parents[3] / ".agent" / "skills" / "dependencies.md"
)

_MODEL_NAME = "gpt-4o"
_TEMPERATURE = 0.1
_MAX_TOKENS = 8192


# ---------------------------------------------------------------------------
# Pydantic schemas for structured LLM output
# ---------------------------------------------------------------------------

class UnresolvedCitation(BaseModel):
    """A single component whose math is missing and needs web resolution."""

    component_name: str = Field(
        ..., description="Name of the component (e.g. 'distortion loss', 'RMSNorm')"
    )
    source_reference: str = Field(
        ..., description="The citation reference (e.g. '[3]', 'Mip-NeRF 360')"
    )
    context: str = Field(
        "", description="Verbatim sentence from the paper mentioning this component"
    )
    category: str = Field(
        "",
        description="One of: 'loss_function', 'normalization', 'attention', "
        "'encoding', 'architecture', 'training', 'activation', 'other'",
    )


class CitationAnalysis(BaseModel):
    """Result of scanning the paper for unresolved citation dependencies."""

    unresolved: List[UnresolvedCitation] = Field(
        default_factory=list,
        description="Components that reference a citation but lack the math",
    )
    already_defined: List[str] = Field(
        default_factory=list,
        description="Components whose math IS already present in the paper",
    )


class ResolvedComponent(BaseModel):
    """A resolved component with math and PyTorch code."""

    component_name: str = Field(..., description="Component name")
    math_formulation: str = Field(
        ...,
        description=(
            "The exact mathematical formulation in LaTeX.  Preserve all "
            "$$...$$ and \\begin{equation} blocks verbatim."
        ),
    )
    pytorch_snippet: str = Field(
        ...,
        description="A concise, working PyTorch implementation (nn.Module or function)",
    )
    source_paper: str = Field(
        "", description="The paper this originates from"
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_llm(temperature: float = _TEMPERATURE) -> ChatOpenAI:
    return ChatOpenAI(
        model=_MODEL_NAME,
        temperature=temperature,
        max_tokens=_MAX_TOKENS,
    )


def _call_llm(system: str, user: str) -> str:
    llm = _get_llm()
    response = llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=user),
    ])
    return response.content.strip()


def _load_local_knowledge_base() -> str:
    """Read .agent/skills/dependencies.md if it exists."""
    try:
        return _LOCAL_KB_PATH.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""


def _search_local_kb(component_name: str, kb_text: str) -> Optional[str]:
    """Check if a component is already resolved in the local KB.

    Returns the full section text if found, else None.
    """
    if not kb_text:
        return None

    # Try to match a section heading containing the component name
    # Section format: ## N. Component Name (...)
    pattern = re.compile(
        rf"##\s+\d+\.\s+.*?{re.escape(component_name)}.*?\n(.*?)(?=\n##\s|\Z)",
        re.DOTALL | re.IGNORECASE,
    )
    match = pattern.search(kb_text)
    if match:
        return match.group(0).strip()

    # Broader fuzzy match: check if the component name words appear
    words = component_name.lower().split()
    if len(words) >= 2:
        for section in re.split(r"\n(?=## )", kb_text):
            section_lower = section.lower()
            if all(w in section_lower for w in words):
                return section.strip()

    return None


def _resolve_via_web_search(component: UnresolvedCitation) -> Optional[str]:
    """Use GPT-4o to search for and synthesise the math + PyTorch code.

    GPT-4o has web-search capability; we ask it to find the canonical
    formulation and return structured JSON.
    """
    system = (
        "You are an expert ML researcher.  For the component described below, "
        "provide:\n"
        "1. The EXACT mathematical formulation (in LaTeX, preserve $$...$$ blocks)\n"
        "2. A concise, working PyTorch implementation\n"
        "3. The source paper\n\n"
        "Search the web if needed to find the correct formulation.\n"
        "Return ONLY a JSON object with keys: "
        '"math_formulation", "pytorch_snippet", "source_paper".'
    )
    user = (
        f"Component: {component.component_name}\n"
        f"Referenced as: {component.source_reference}\n"
        f"Category: {component.category}\n"
        f"Context from paper: {component.context}\n\n"
        "Find the mathematical formulation and provide a PyTorch implementation."
    )

    try:
        response = _call_llm(system, user)
        # Strip code fences if present
        cleaned = re.sub(r"^```(?:json)?\s*\n?", "", response.strip())
        cleaned = re.sub(r"\n?```\s*$", "", cleaned)
        # Extract JSON
        json_match = re.search(r"\{[\s\S]+\}", cleaned)
        if json_match:
            data = json.loads(json_match.group())
            # Format as a readable resolution block
            math = data.get("math_formulation", "")
            code = data.get("pytorch_snippet", "")
            source = data.get("source_paper", component.source_reference)
            return (
                f"### {component.component_name} (from {source})\n\n"
                f"**Mathematical Formulation:**\n{math}\n\n"
                f"**PyTorch Implementation:**\n```python\n{code}\n```"
            )
    except Exception:
        pass

    return None


# ---------------------------------------------------------------------------
# Core engine
# ---------------------------------------------------------------------------

def run_citation_crawler(state: PipelineState) -> dict:
    """Scan for unresolved citation dependencies and resolve them.

    Steps
    -----
    1. Review ``state["cleaned_markdown"]`` for components that
       reference an external citation but lack the exact math.
    2. For each missing dependency:
       a. Check the local knowledge base (``.agent/skills/dependencies.md``)
       b. If not found locally, use GPT-4o web search to retrieve
          the math + PyTorch implementation.
    3. Aggregate findings and return
       ``{"resolved_components": findings_dict}``.

    Parameters
    ----------
    state : PipelineState

    Returns
    -------
    dict
        ``{"resolved_components": {component_name: resolution_text}}``
    """
    cleaned_markdown: str = state.get("cleaned_markdown", "")

    if not cleaned_markdown:
        return {"resolved_components": {}}

    # ── Step 1: Detect unresolved citations via structured LLM output ───
    llm = _get_llm()
    analysis_llm = llm.with_structured_output(CitationAnalysis)

    analysis_messages = [
        SystemMessage(content=(
            "You are an expert at analyzing ML research papers.  "
            "Identify components that reference an external citation but "
            "whose mathematical formulation is NOT present in the paper text.\n\n"
            "Examples of unresolved references:\n"
            '- "We adopt the distortion loss from [3]"\n'
            '- "Following [17], we use RMSNorm"\n'
            '- "The hash encoding of [33] is used"\n\n'
            "Do NOT flag components whose math IS already defined in the text."
        )),
        HumanMessage(content=(
            "Analyze this paper for unresolved citation dependencies:\n\n"
            f"---\n\n{cleaned_markdown[:20000]}"
        )),
    ]

    try:
        analysis: CitationAnalysis = analysis_llm.invoke(analysis_messages)
        unresolved = analysis.unresolved
    except Exception:
        # Fallback: no unresolved components detected
        unresolved = []

    if not unresolved:
        return {"resolved_components": {}}

    # ── Step 2: Resolve each missing dependency ─────────────────────────
    local_kb = _load_local_knowledge_base()
    resolved_components: Dict[str, str] = {}

    for citation in unresolved:
        name = citation.component_name

        # 2a. Check local knowledge base first
        local_hit = _search_local_kb(name, local_kb)
        if local_hit:
            resolved_components[name] = local_hit
            continue

        # 2b. Fall back to GPT-4o web search
        web_result = _resolve_via_web_search(citation)
        if web_result:
            resolved_components[name] = web_result

    # ── Return state update ─────────────────────────────────────────────
    return {"resolved_components": resolved_components}


# ---------------------------------------------------------------------------
# Alias for graph.py compatibility
# ---------------------------------------------------------------------------

def citation_crawler_node(state: PipelineState) -> Dict[str, Any]:
    """LangGraph node wrapper around :func:`run_citation_crawler`.

    Adds message logging and error handling.
    """
    messages = list(state.get("messages", []))
    errors = list(state.get("errors", []))

    messages.append("[Stage 2 - Citation Crawler] Scanning for unresolved dependencies...")

    try:
        result = run_citation_crawler(state)

        resolved = result.get("resolved_components", {})
        messages.append(
            f"[Stage 2 - Citation Crawler] ✅ Resolved {len(resolved)} dependencies"
        )
        if resolved:
            for name in resolved:
                messages.append(f"[Stage 2]   • {name}")

        result["messages"] = messages
        result["errors"] = errors
        return result

    except Exception as exc:
        errors.append(f"[Stage 2] Citation crawler failed: {exc}")
        messages.append(f"[Stage 2 - Citation Crawler] ❌ Failed: {exc}")
        return {
            "resolved_components": {},
            "messages": messages,
            "errors": errors,
        }
