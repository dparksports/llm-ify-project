"""Graph state definition for the LLM-IFY LangGraph orchestrator.

The ``PipelineState`` TypedDict flows through all four stages of the
NERFIFY-inspired pipeline.  Each agent reads from and writes to the
fields it owns, enabling LangGraph to track deltas and support
checkpointing / human-in-the-loop review.

References
----------
- Paper §3.1, Eqs. 1-3  (repository & paper representations)
- Paper §3.2, Stage 1-4 (pipeline stages)
- .agent/rules/hf_cfg.md (output code constraints)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from pydantic import BaseModel, Field
from typing_extensions import TypedDict


# ---------------------------------------------------------------------------
# Structured sub-models (used inside the TypedDict via Pydantic)
# ---------------------------------------------------------------------------

class Citation(BaseModel):
    """A single bibliographic reference extracted from the paper."""

    ref_id: str = Field(..., description="In-paper citation key, e.g. '[32]'")
    title: str = ""
    authors: List[str] = Field(default_factory=list)
    year: Optional[int] = None
    arxiv_id: Optional[str] = None
    url: Optional[str] = None


class ExtractedPaper(BaseModel):
    """Structured representation E(P) = ⟨T(P), I(P), Q(P), B(P)⟩ (Eq. 2).

    T(P) = ⟨H, {pᵢ}, {aℓ}, {cₖ}, {rₘ}⟩  (Eq. 3)
    """

    # --- T(P): textual components (Eq. 3) ---
    headings: List[str] = Field(default_factory=list, description="Section headings H")
    paragraphs: List[str] = Field(default_factory=list, description="Body paragraphs {pᵢ}")
    algorithms: List[str] = Field(default_factory=list, description="Pseudocode blocks {aℓ}")
    captions: List[str] = Field(default_factory=list, description="Figure captions {cₖ}")
    references: List[Citation] = Field(default_factory=list, description="References {rₘ}")

    # --- I(P): visual content ---
    figure_paths: List[str] = Field(default_factory=list)

    # --- Q(P): mathematical content ---
    equations: List[str] = Field(
        default_factory=list,
        description="LaTeX equations preserved verbatim ($$...$$ and \\begin{equation})",
    )

    # --- B(P): bibliographic metadata ---
    raw_bibtex: str = ""

    # --- convenience ---
    raw_markdown: str = Field("", description="Full MinerU / PyMuPDF markdown output")
    title: str = ""
    abstract: str = ""


class GeneratedFile(BaseModel):
    """A single code file produced by the GoT coder (Stage 3)."""

    filename: str = Field(..., description="Relative path, e.g. 'configuration_zipnerf.py'")
    content: str = ""
    stub_only: bool = Field(False, description="True if only interface stubs exist")
    validated: bool = Field(False, description="True after smoke-test passes")


class DiagnosticReport(BaseModel):
    """Structured critique output from Stage 4 (Figure 2 JSON schema)."""

    detect: str = ""
    diagnose: str = ""
    recover: str = ""
    patch: str = ""
    action: str = ""
    confidence: float = 0.0


# ---------------------------------------------------------------------------
# Top-level graph state
# ---------------------------------------------------------------------------

class PipelineState(TypedDict, total=False):
    """Flows through the LangGraph ``StateGraph``.

    Each key is owned/written by the stage indicated in the comment.
    LangGraph accumulates updates via dict-merge semantics.
    """

    # ── Inputs ──────────────────────────────────────────────────────────
    pdf_path: str                              # User-supplied path to the PDF
    cfg_rules: str                             # Contents of .agent/rules/hf_cfg.md

    # ── Stage 1: Summarizer ─────────────────────────────────────────────
    extracted_paper: Dict[str, Any]            # Serialised ExtractedPaper
    cleaned_markdown: str                      # Refined markdown (irrelevant sections removed)

    # ── Stage 2: Citation Crawler ───────────────────────────────────────
    citation_graph: Dict[str, List[str]]       # Adjacency list:  paper_id → [dependency_ids]
    resolved_components: Dict[str, str]        # component_name → extracted code/math snippet
    crawled_papers: Dict[str, Dict[str, Any]]  # paper_id → serialised ExtractedPaper

    # ── Stage 3: GoT Coder ──────────────────────────────────────────────
    repo_dag: Dict[str, List[str]]             # file → [files it depends on]
    topo_order: List[str]                      # Topologically sorted file list
    generated_files: Dict[str, Dict[str, Any]] # filename → serialised GeneratedFile
    generation_phase: str                      # Current GoT phase: dag|freeze|implement|integrate

    # ── Stage 4: Critique ───────────────────────────────────────────────
    smoke_test_passed: bool
    diagnostics: List[Dict[str, Any]]          # List of serialised DiagnosticReport
    refinement_iteration: int                  # Current iteration counter
    final_repository: Dict[str, str]           # filename → validated source code

    # ── Shared / control ────────────────────────────────────────────────
    errors: List[str]                          # Accumulated error messages
    messages: List[str]                        # Human-readable status log
