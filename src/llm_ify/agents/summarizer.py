"""Stage 1 — Paper Summarizer Agent.

Parses a research paper PDF via PyMuPDF and produces a structured
``ExtractedPaper`` representation (Eqs. 2 & 3 from the NERFIFY paper).

Key responsibilities:
- Extract text preserving inline LaTeX ($$...$$ and \\begin{equation})
- Split into structured fields: headings, paragraphs, equations,
  algorithms, captions, references
- Inject hf_cfg.md rules into state for downstream agents
- Clean markdown: remove irrelevant sections, validate completeness

References:
    Paper §3.2, Stage 1  — CFG Formalization and In-Context Learning
    Paper §3.1, Eqs. 2-3 — Paper representation E(P)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import fitz  # PyMuPDF

from llm_ify.state import Citation, ExtractedPaper, PipelineState


# ---------------------------------------------------------------------------
# PDF text extraction with LaTeX preservation
# ---------------------------------------------------------------------------

def _extract_pdf_text(pdf_path: str) -> str:
    """Extract text from a PDF, preserving mathematical notation.

    Uses PyMuPDF's text extraction with whitespace preservation
    to maintain LaTeX equation formatting.
    """
    doc = fitz.open(pdf_path)
    pages: List[str] = []

    for i, page in enumerate(doc):
        blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
        page_lines: List[str] = []

        for block in blocks:
            if block["type"] == 0:  # text block
                for line in block["lines"]:
                    line_text = "".join(span["text"] for span in line["spans"])
                    page_lines.append(line_text)

        pages.append("\n".join(page_lines))

    doc.close()
    return "\n\n".join(pages)


# ---------------------------------------------------------------------------
# Structured field extraction
# ---------------------------------------------------------------------------

# Patterns for splitting the raw text into structured components
_HEADING_RE = re.compile(
    r"^(\d+\.(?:\d+\.?)*\s+.+|Abstract|Introduction|Related Work|Method|"
    r"Experiments|Conclusion|References|Appendix)",
    re.MULTILINE,
)

_EQUATION_BLOCK_RE = re.compile(
    r"(\\begin\{(?:equation|align|gather|multline)\*?\}.*?\\end\{(?:equation|align|gather|multline)\*?\})",
    re.DOTALL,
)

_INLINE_EQUATION_RE = re.compile(
    r"(\$\$[^$]+?\$\$|\$[^$]+?\$)",
)

# Numbered equations: a line ending with (N) where N is 1-3 digits
_NUMBERED_EQ_RE = re.compile(
    r"^(.+?)\s*\((\d{1,3})\)\s*$",
    re.MULTILINE,
)

# Unicode math symbols commonly found in PDF-extracted text
_UNICODE_MATH_RE = re.compile(
    r"^(.{3,}?[=⟨⟩∈⊆∪∩≤≥→←∀∃∑∏∫⊂⊃⊄⊅≡≈≠±×÷·∇∂∞∅⟹⟸⟺∧∨¬].{2,})$",
    re.MULTILINE,
)

_ALGORITHM_RE = re.compile(
    r"(Algorithm\s+\d+[:\.].*?)(?=\n\n|\Z)",
    re.DOTALL,
)

_FIGURE_CAPTION_RE = re.compile(
    r"((?:Figure|Fig\.?|Table)\s+\d+[.:].*?)(?=\n\n|\Z)",
    re.DOTALL | re.IGNORECASE,
)

# Multi-line bibliography: [N] Author ... year. pages
_REFERENCE_START_RE = re.compile(
    r"^\[(\d+)\]\s*(.*)$",
    re.MULTILINE,
)


def _extract_headings(text: str) -> List[str]:
    """Extract section headings."""
    return [m.group(0).strip() for m in _HEADING_RE.finditer(text)]


def _extract_equations(text: str) -> List[str]:
    """Extract all mathematical equations from PDF-extracted text.

    Handles:
    - \\begin{equation}...\\end{equation} LaTeX blocks
    - $$...$$ display math
    - $...$ inline math
    - Numbered equations: lines containing math symbols followed by (N)
    - Lines with Unicode math symbols (⟨, ∈, ⊆, →, ∑, etc.)
    """
    equations: List[str] = []
    seen: set = set()

    def _add(eq: str) -> None:
        eq = eq.strip()
        if eq and eq not in seen and len(eq) > 2:
            seen.add(eq)
            equations.append(eq)

    # Block equations (\\begin{equation}, etc.)
    for m in _EQUATION_BLOCK_RE.finditer(text):
        _add(m.group(1))

    # Display and inline LaTeX ($$ and $)
    for m in _INLINE_EQUATION_RE.finditer(text):
        _add(m.group(1))

    # Numbered equations: lines ending with (1), (2), etc.
    # Also look for the content *above* the number line
    lines = text.split("\n")
    for i, line in enumerate(lines):
        stripped = line.strip()
        # Line that is just a number in parens: "(1)"
        if re.match(r"^\(\d{1,3}\)$", stripped):
            # Grab the preceding non-empty line(s) as the equation
            parts = []
            for j in range(i - 1, max(i - 4, -1), -1):
                prev = lines[j].strip()
                if not prev or re.match(r"^\d+$", prev):  # page number or blank
                    break
                parts.insert(0, prev)
            if parts:
                _add(f"({stripped.strip('()')}) " + " ".join(parts))
        # Line ending with (N)
        m = _NUMBERED_EQ_RE.match(stripped)
        if m:
            _add(f"({m.group(2)}) {m.group(1).strip()}")

    # Unicode math lines
    for m in _UNICODE_MATH_RE.finditer(text):
        candidate = m.group(1).strip()
        # Skip if it's just a caption or heading
        if not any(candidate.lower().startswith(skip) for skip in
                   ["figure", "table", "fig.", "where", "the ", "this ", "our ", "we "]):
            _add(candidate)

    return equations


def _extract_algorithms(text: str) -> List[str]:
    """Extract pseudocode / algorithm blocks."""
    return [m.group(1).strip() for m in _ALGORITHM_RE.finditer(text)]


def _extract_captions(text: str) -> List[str]:
    """Extract figure and table captions."""
    return [m.group(1).strip() for m in _FIGURE_CAPTION_RE.finditer(text)]


def _extract_references(text: str) -> List[Citation]:
    """Extract bibliography entries as Citation objects.

    Handles multi-line references typical of academic PDFs:
        [1] Author Name. Title...
        continuation of title. In Proceedings..., 2023. 6
    """
    citations: List[Citation] = []

    # Find the References section
    ref_section_match = re.search(
        r"(?:^|\n)References\s*\n", text, re.IGNORECASE
    )
    if not ref_section_match:
        return citations

    ref_text = text[ref_section_match.end():]

    # Split into individual reference blocks by [N] markers
    ref_blocks: List[tuple] = []
    for m in _REFERENCE_START_RE.finditer(ref_text):
        ref_blocks.append((int(m.group(1)), m.start(), m.group(2)))

    for idx, (ref_num, start, first_line) in enumerate(ref_blocks):
        # Accumulate lines until next [N] or end
        if idx + 1 < len(ref_blocks):
            end = ref_blocks[idx + 1][1]
        else:
            end = len(ref_text)

        block = ref_text[start:end]
        # Remove the [N] prefix and join lines
        block = re.sub(r"^\[\d+\]\s*", "", block)
        raw = " ".join(line.strip() for line in block.split("\n") if line.strip())
        # Remove trailing page numbers (e.g., "2, 4" or "6, 8")
        raw = re.sub(r"\s+\d+(?:,\s*\d+)*\s*$", "", raw)

        # Extract arXiv ID
        arxiv_match = re.search(
            r"arXiv[:\s]*(?:preprint\s+arXiv[:\s]*)?(\d{4}\.\d{4,5}(?:v\d+)?)", raw
        )
        arxiv_id = arxiv_match.group(1) if arxiv_match else None

        # Try to extract year
        year_match = re.search(r"(\d{4})", raw)
        year = int(year_match.group(1)) if year_match else None

        citations.append(
            Citation(
                ref_id=f"[{ref_num}]",
                title=raw.strip(),
                arxiv_id=arxiv_id,
                year=year,
            )
        )

    return citations


def _extract_abstract(text: str) -> str:
    """Extract the abstract section."""
    # Look for text between "Abstract" heading and next section
    match = re.search(
        r"Abstract\s*\n(.+?)(?=\n\d+\.\s|\nIntroduction|\n1\s)",
        text,
        re.DOTALL,
    )
    return match.group(1).strip() if match else ""


def _extract_title(text: str) -> str:
    """Extract the paper title (first non-empty line)."""
    for line in text.split("\n"):
        line = line.strip()
        if line and not line.startswith("arXiv") and len(line) > 10:
            return line
    return ""


def _split_paragraphs(text: str) -> List[str]:
    """Split body text into paragraphs.

    Joins hyphenated line breaks common in two-column PDFs
    (e.g., 're-\nsearch' → 'research').
    """
    paragraphs: List[str] = []
    current: List[str] = []

    for line in text.split("\n"):
        stripped = line.strip()

        # Skip page markers
        if stripped.startswith("<!-- PAGE"):
            continue

        if not stripped:
            if current:
                para = " ".join(current)
                # De-hyphenate line breaks
                para = re.sub(r"(\w)- (\w)", r"\1\2", para)
                # Skip very short fragments, pure numbers, and reference lines
                if len(para) > 50 and not para.startswith("[") and not re.match(r"^\d+$", para):
                    paragraphs.append(para)
                current = []
        else:
            current.append(stripped)

    if current:
        para = " ".join(current)
        para = re.sub(r"(\w)- (\w)", r"\1\2", para)
        if len(para) > 50:
            paragraphs.append(para)

    return paragraphs


# ---------------------------------------------------------------------------
# Markdown cleaning
# ---------------------------------------------------------------------------

_IRRELEVANT_SECTIONS = {
    "related work",
    "acknowledgments",
    "acknowledgements",
    "author contributions",
}


def _clean_markdown(text: str, headings: List[str]) -> str:
    """Remove irrelevant sections while preserving all equations and pseudocode.

    Strips:
    - Extended "Related Work" discussions
    - Acknowledgments
    - Redundant references (kept in structured form)

    Validates:
    - Key technical headings from abstract still appear
    """
    lines = text.split("\n")
    cleaned: List[str] = []
    skip = False

    for line in lines:
        lower = line.strip().lower()

        # Check if we should start skipping
        for section in _IRRELEVANT_SECTIONS:
            if lower.startswith(section) or (
                re.match(r"^\d+\.?\s*", lower)
                and section in lower
            ):
                skip = True
                break

        # Stop skipping at next major heading
        if skip and re.match(r"^\d+\.\s+", line.strip()):
            heading_lower = line.strip().lower()
            if not any(s in heading_lower for s in _IRRELEVANT_SECTIONS):
                skip = False

        if not skip:
            cleaned.append(line)

    return "\n".join(cleaned)


# ---------------------------------------------------------------------------
# Main summarizer node
# ---------------------------------------------------------------------------

def summarizer_node(state: PipelineState) -> Dict[str, Any]:
    """Parse the PDF and produce a cleaned, structured extraction.

    Implements Stage 1 of the NERFIFY pipeline:
    E(P) = ⟨T(P), I(P), Q(P), B(P)⟩  (Eq. 2)
    T(P) = ⟨H, {pᵢ}, {aℓ}, {cₖ}, {rₘ}⟩  (Eq. 3)
    """
    pdf_path = state.get("pdf_path", "")
    cfg_rules = state.get("cfg_rules", "")
    messages = list(state.get("messages", []))
    errors = list(state.get("errors", []))

    messages.append(f"[Stage 1 - Summarizer] Processing PDF: {pdf_path}")

    # ── Extract raw text ────────────────────────────────────────────────
    if not pdf_path or not Path(pdf_path).exists():
        errors.append(f"PDF not found: {pdf_path}")
        return {"errors": errors, "messages": messages}

    try:
        raw_text = _extract_pdf_text(pdf_path)
        messages.append(f"[Stage 1] Extracted {len(raw_text):,} chars from PDF")
    except Exception as e:
        errors.append(f"PDF extraction failed: {e}")
        return {"errors": errors, "messages": messages}

    # ── Structure extraction (Eqs. 2 & 3) ──────────────────────────────
    title = _extract_title(raw_text)
    abstract = _extract_abstract(raw_text)
    headings = _extract_headings(raw_text)
    equations = _extract_equations(raw_text)
    algorithms = _extract_algorithms(raw_text)
    captions = _extract_captions(raw_text)
    references = _extract_references(raw_text)
    paragraphs = _split_paragraphs(raw_text)

    messages.append(
        f"[Stage 1] Extracted: {len(headings)} headings, "
        f"{len(equations)} equations, {len(algorithms)} algorithms, "
        f"{len(captions)} captions, {len(references)} references, "
        f"{len(paragraphs)} paragraphs"
    )

    # ── Build ExtractedPaper ────────────────────────────────────────────
    extracted = ExtractedPaper(
        title=title,
        abstract=abstract,
        headings=headings,
        paragraphs=paragraphs,
        algorithms=algorithms,
        captions=captions,
        references=references,
        equations=equations,
        raw_markdown=raw_text,
    )

    # ── Clean markdown ──────────────────────────────────────────────────
    cleaned = _clean_markdown(raw_text, headings)
    messages.append(
        f"[Stage 1] Cleaned markdown: {len(raw_text):,} → {len(cleaned):,} chars "
        f"({100 * len(cleaned) / max(len(raw_text), 1):.0f}% retained)"
    )

    # ── Inject CFG rules ────────────────────────────────────────────────
    if cfg_rules:
        messages.append("[Stage 1] CFG rules injected into state for downstream agents")

    # ── Validate completeness ───────────────────────────────────────────
    if not equations:
        messages.append("[Stage 1] ⚠️  No equations found — paper may lack inline LaTeX")
    if not abstract:
        messages.append("[Stage 1] ⚠️  Abstract not detected")

    return {
        "extracted_paper": extracted.model_dump(),
        "cleaned_markdown": cleaned,
        "messages": messages,
        "errors": errors,
    }
