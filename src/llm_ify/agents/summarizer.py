"""Stage 1 — Paper Summarizer Agent.

Parses a research paper PDF via PyMuPDF (``fitz``) and uses GPT-4o with
``with_structured_output()`` to extract architecture metadata, novel
components (with LaTeX math preserved verbatim), hyperparameters, and
the Hugging Face file dependency DAG.

The file DAG is then fed to ``BuildRepoDAG`` (pipeline/dag.py) to
compute a validated topological order for downstream code generation.

Key responsibilities:
- Extract text preserving inline LaTeX ($$...$$, \\begin{equation})
- Produce structured output via Pydantic schema + LLM
- Compute topo_order from the extracted file DAG

References:
    Paper §3.2, Stage 1  — CFG Formalization and In-Context Learning
    Paper §3.1, Eqs. 2-3 — Paper representation E(P)
    .agent/rules/hf_cfg.md — Output code constraints
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import fitz  # PyMuPDF
import os
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from llm_ify.pipeline.dag import build_repo_dag
from llm_ify.state import PipelineState


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Fallback markdown file when no PDF is provided
_FALLBACK_MD = Path(__file__).resolve().parents[3] / "docs" / "paper_parsed.md"

# LLM settings
_MODEL_NAME = "gemini-1.5-pro"
_TEMPERATURE = 0.0
_MAX_TOKENS = 8192


# ---------------------------------------------------------------------------
# Pydantic schema for structured LLM output
# ---------------------------------------------------------------------------

class NovelComponent(BaseModel):
    """A single novel component extracted from the paper.

    All LaTeX math equations and pseudocode must be preserved
    **verbatim** — do not simplify, translate, or paraphrase them.
    """

    name: str = Field(
        ...,
        description="Component name (e.g. 'Multi-Head Latent Attention', 'SwiGLU FFN')",
    )
    description: str = Field(
        ...,
        description=(
            "Detailed description including the EXACT LaTeX math equations "
            "($$...$$ or \\begin{equation}...\\end{equation}) and pseudocode "
            "copied verbatim from the paper.  Do NOT simplify or omit any math."
        ),
    )
    category: str = Field(
        "",
        description=(
            "One of: 'attention', 'normalization', 'embedding', 'ffn', "
            "'loss', 'architecture', 'training', 'other'"
        ),
    )


class PaperStructuredOutput(BaseModel):
    """Structured extraction from a research paper.

    This schema is used with ``ChatGoogleGenerativeAI.with_structured_output()``
    so the LLM returns validated, typed fields.
    """

    architecture_name: str = Field(
        ...,
        description=(
            "Short snake_case name for the architecture "
            "(e.g. 'deepseek_v3', 'zip_nerf').  Used as the suffix in "
            "file names like configuration_<name>.py."
        ),
    )

    novel_components: List[NovelComponent] = Field(
        default_factory=list,
        description=(
            "Every novel architectural component described in the paper.  "
            "Each entry MUST preserve the exact LaTeX math equations and "
            "pseudocode verbatim — no paraphrasing."
        ),
    )

    hyperparameters: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "All hyperparameters mentioned in the paper.  Keys are the "
            "parameter names (snake_case), values are defaults from the paper "
            "(e.g. {'hidden_size': 4096, 'num_layers': 32, 'vocab_size': 102400})."
        ),
    )

    file_dag: Dict[str, List[str]] = Field(
        default_factory=dict,
        description=(
            "Mapping of each required Hugging Face repository file to a list "
            "of files it depends on.  Must follow the HF naming convention:\n"
            "  configuration_<name>.py  — PretrainedConfig (no dependencies)\n"
            "  modeling_<name>.py       — PreTrainedModel (depends on config)\n"
            "  __init__.py              — re-exports (depends on all others)\n"
            "Example:\n"
            '  {"configuration_custom.py": [], '
            '"modeling_custom.py": ["configuration_custom.py"], '
            '"__init__.py": ["configuration_custom.py", "modeling_custom.py"]}'
        ),
    )


# ---------------------------------------------------------------------------
# PDF text extraction (LaTeX-preserving)
# ---------------------------------------------------------------------------

def _extract_pdf_text(pdf_path: str) -> str:
    """Extract text from a PDF, preserving mathematical notation.

    Uses PyMuPDF's dictionary-based extraction with whitespace
    preservation so LaTeX equation formatting ($$...$$,
    \\begin{equation}, etc.) is kept as plain text.
    """
    doc = fitz.open(pdf_path)
    pages: List[str] = []

    for page in doc:
        blocks = page.get_text(
            "dict", flags=fitz.TEXT_PRESERVE_WHITESPACE
        )["blocks"]
        page_lines: List[str] = []

        for block in blocks:
            if block["type"] == 0:  # text block
                for line in block["lines"]:
                    line_text = "".join(
                        span["text"] for span in line["spans"]
                    )
                    page_lines.append(line_text)

        pages.append("\n".join(page_lines))

    doc.close()
    return "\n\n".join(pages)


def _read_fallback_markdown() -> str:
    """Read the pre-parsed markdown from docs/paper_parsed.md."""
    if _FALLBACK_MD.exists():
        return _FALLBACK_MD.read_text(encoding="utf-8")
    raise FileNotFoundError(
        f"No PDF provided and fallback not found at {_FALLBACK_MD}"
    )


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

def _get_llm(temperature: float = _TEMPERATURE) -> ChatGoogleGenerativeAI:
    """Create a ChatGoogleGenerativeAI instance."""
    return ChatGoogleGenerativeAI(
        model=_MODEL_NAME,
        temperature=temperature,
        max_tokens=_MAX_TOKENS,
        max_retries=3,
    )


_SYSTEM_PROMPT = """\
You are a research paper analysis expert specializing in deep learning
architectures and Hugging Face transformers.

Your task is to extract structured information from a research paper.

CRITICAL RULES:
1. **Preserve ALL LaTeX math verbatim**.  Copy every equation exactly as
   written — $$...$$, \\begin{{equation}}, inline $...$ — with no
   simplification, no translation to prose, and no omission.
2. **Preserve ALL pseudocode** verbatim.
3. For file_dag: use the Hugging Face naming convention:
   - configuration_<name>.py  (inherits PretrainedConfig, no deps)
   - modeling_<name>.py       (inherits PreTrainedModel, depends on config)
   - __init__.py              (re-exports, depends on all other files)
4. architecture_name must be short snake_case (e.g. 'deepseek_v3').
5. hyperparameters should include every numeric default mentioned in the
   paper (hidden_size, num_layers, num_attention_heads, vocab_size, etc.).
"""


# ---------------------------------------------------------------------------
# Core engine
# ---------------------------------------------------------------------------

def run_summarizer(state: PipelineState) -> dict:
    """Parse the research paper and extract structured metadata via GPT-4o.

    Steps
    -----
    1. Read raw text from ``state["pdf_path"]`` via PyMuPDF.
       Falls back to ``docs/paper_parsed.md`` if no PDF is given.
    2. Call GPT-4o with ``with_structured_output(PaperStructuredOutput)``
       to get architecture_name, novel_components, hyperparameters,
       and file_dag.
    3. Pass file_dag to ``build_repo_dag`` to compute the topo_order.
    4. Return the state update.

    Parameters
    ----------
    state : PipelineState

    Returns
    -------
    dict
        ``{"extracted_paper": ..., "cleaned_markdown": ...,
          "repo_dag": ..., "topo_order": ...}``
    """
    pdf_path = state.get("pdf_path", "")

    # ── Step 1: Extract raw text ────────────────────────────────────────
    if pdf_path and Path(pdf_path).exists():
        raw_text = _extract_pdf_text(pdf_path)
    else:
        raw_text = _read_fallback_markdown()

    # ── Step 2: Structured extraction via GPT-4o ────────────────────────
    llm = _get_llm()
    structured_llm = llm.with_structured_output(PaperStructuredOutput)

    # Truncate to ~30k chars to fit token limits while keeping math
    paper_excerpt = raw_text[:30000]
    if len(raw_text) > 30000:
        paper_excerpt += "\n\n... (text truncated for token budget)"

    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=(
            "Analyze the following research paper and extract the requested "
            "structured information.  Remember: preserve ALL LaTeX math and "
            "pseudocode VERBATIM.\n\n"
            f"---\n\n{paper_excerpt}"
        )),
    ]

    structured: PaperStructuredOutput = structured_llm.invoke(messages)

    # ── Step 3: Compute topo_order via BuildRepoDAG ─────────────────────
    file_dag = structured.file_dag

    # Ensure the DAG has at least the minimal HF structure
    if not file_dag:
        name = structured.architecture_name or "custom_model"
        file_dag = {
            f"configuration_{name}.py": [],
            f"modeling_{name}.py": [f"configuration_{name}.py"],
            "__init__.py": [
                f"configuration_{name}.py",
                f"modeling_{name}.py",
            ],
        }

    # Validate: all referenced files must be in the file set
    all_files = list(file_dag.keys())
    for fname, deps in list(file_dag.items()):
        file_dag[fname] = [d for d in deps if d in all_files]

    # Ensure every file has an entry
    for f in all_files:
        if f not in file_dag:
            file_dag[f] = []

    _adjacency, topo_order = build_repo_dag(all_files, file_dag)

    # ── Step 4: Build cleaned_markdown as structured string ─────────────
    # Serialise the structured output into a readable string for
    # downstream prompts (GoT coder injects this as context).
    components_str = ""
    for comp in structured.novel_components:
        components_str += (
            f"\n### {comp.name} ({comp.category})\n"
            f"{comp.description}\n"
        )

    hyperparams_str = "\n".join(
        f"  - {k}: {v}" for k, v in structured.hyperparameters.items()
    )

    cleaned_markdown = (
        f"# Architecture: {structured.architecture_name}\n\n"
        f"## Novel Components\n{components_str}\n"
        f"## Hyperparameters\n{hyperparams_str}\n\n"
        f"## File DAG\n"
        + "\n".join(
            f"  - {f} → depends on {deps}"
            for f, deps in file_dag.items()
        )
        + f"\n\n## Topological Order\n{topo_order}\n\n"
        f"## Raw Paper Excerpt\n{paper_excerpt[:8000]}\n"
    )

    # ── Return state update ─────────────────────────────────────────────
    return {
        "extracted_paper": raw_text,
        "cleaned_markdown": cleaned_markdown,
        "repo_dag": file_dag,
        "topo_order": topo_order,
    }


# ---------------------------------------------------------------------------
# Alias for graph.py compatibility
# ---------------------------------------------------------------------------
# graph.py imports ``summarizer_node`` — route through run_summarizer
# with logging and error handling.

def summarizer_node(state: PipelineState) -> Dict[str, Any]:
    """LangGraph node wrapper around :func:`run_summarizer`.

    Adds message logging and error handling expected by the pipeline
    orchestrator.
    """
    messages = list(state.get("messages", []))
    errors = list(state.get("errors", []))
    pdf_path = state.get("pdf_path", "")

    messages.append(f"[Stage 1 - Summarizer] Processing: {pdf_path or 'docs/paper_parsed.md'}")

    try:
        result = run_summarizer(state)

        topo_order = result.get("topo_order", [])
        messages.append(
            f"[Stage 1 - Summarizer] ✅ Extracted {len(topo_order)} files "
            f"in topo order: {topo_order}"
        )
        messages.append(
            f"[Stage 1 - Summarizer] cleaned_markdown: "
            f"{len(result.get('cleaned_markdown', '')):,} chars"
        )

        result["messages"] = messages
        result["errors"] = errors
        return result

    except Exception as exc:
        errors.append(f"[Stage 1] Summarizer failed: {exc}")
        messages.append(f"[Stage 1 - Summarizer] ❌ Failed: {exc}")
        return {
            "messages": messages,
            "errors": errors,
        }
