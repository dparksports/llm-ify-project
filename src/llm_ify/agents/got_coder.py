"""Stage 3 — Graph-of-Thought (GoT) Coder Agent.

Generates Hugging Face-compliant PyTorch code in topological DAG order,
following Phase 3 (Implementation) of the GoT pipeline from Figure 4.

Core loop:
    For each file in ``state["topo_order"]``:
        1. Build a strict prompt injecting hf_cfg.md rules, paper architecture,
           resolved component math, and upstream "Interface Freeze" code.
        2. If state["errors"] exist, append the smoke-test stack-trace so the
           LLM can self-repair.
        3. Call GPT-4o via langchain-openai, parse the ```python``` block.
        4. Store result in state["generated_files"] and write to output/.

All output must satisfy .agent/rules/hf_cfg.md:
    - Config inherits PretrainedConfig
    - Model inherits PreTrainedModel
    - forward(input_ids, attention_mask) → CausalLMOutputWithPast

References:
    Paper §3.2, Stage 3  — Grammar-Guided Repository Generation
    Paper Figure 4       — GoT Multi-Agent Code Synthesis
    Table 5 ablation     — GoT is CRITICAL (score drops 0.98→0.45 without it)
"""

from __future__ import annotations

import os
import re
import textwrap
from pathlib import Path
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from llm_ify.state import PipelineState


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Path to the HF CFG rules file (relative to project root)
_HF_CFG_PATH = Path(__file__).resolve().parents[3] / ".agent" / "rules" / "hf_cfg.md"

# Output directory for physically written files (relative to project root)
_OUTPUT_DIR = Path(__file__).resolve().parents[3] / "output"

# LLM parameters
_MODEL_NAME = "gemini-1.5-pro"
_TEMPERATURE = 0.0
_MAX_TOKENS = 16384


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_hf_cfg_rules() -> str:
    """Read .agent/rules/hf_cfg.md from disk and return its full contents.

    Falls back to a minimal constraint string if the file is missing.
    """
    try:
        return _HF_CFG_PATH.read_text(encoding="utf-8")
    except FileNotFoundError:
        return (
            "# HF CFG Rules (file not found — using fallback)\n"
            "- Config MUST inherit transformers.PretrainedConfig\n"
            "- Model MUST inherit transformers.PreTrainedModel\n"
            "- forward() MUST return CausalLMOutputWithPast\n"
        )


def _get_llm(temperature: float = _TEMPERATURE) -> ChatGoogleGenerativeAI:
    """Create a ChatGoogleGenerativeAI instance for code generation."""
    return ChatGoogleGenerativeAI(
        model=_MODEL_NAME,
        temperature=temperature,
        max_tokens=_MAX_TOKENS,
        max_retries=3,
    )


def _call_llm(system: str, user: str, temperature: float = _TEMPERATURE) -> str:
    """Send a system+user message pair and return the assistant text."""
    llm = _get_llm(temperature)
    response = llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=user),
    ])
    return response.content.strip()


def _parse_python_block(text: str) -> str:
    """Extract the first ```python ... ``` fenced block from *text*.

    If no fenced block is found, returns the raw text with any leading/
    trailing markdown fences stripped as a best-effort fallback.
    """
    # Try to match a fenced python block
    match = re.search(
        r"```python\s*\n(.*?)```",
        text,
        re.DOTALL,
    )
    if match:
        return match.group(1).strip()

    # Fallback: strip any generic fences
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:python)?\s*\n?", "", cleaned)
    cleaned = re.sub(r"\n?```\s*$", "", cleaned)
    return cleaned.strip()


def _build_dependency_code_section(
    current_file: str,
    topo_order: List[str],
    generated_files: Dict[str, str],
) -> str:
    """Build the 'Interface Freeze' section: exact source of all upstream files.

    Only includes files that appear *before* ``current_file`` in the
    topological order **and** have already been generated.
    """
    sections: List[str] = []
    for f in topo_order:
        if f == current_file:
            break
        code = generated_files.get(f)
        if code:
            sections.append(
                f"# ═══════ {f} ═══════\n"
                f"```python\n{code}\n```"
            )
    if not sections:
        return "(No upstream files generated yet.)"
    return "\n\n".join(sections)


def _build_resolved_components_section(
    resolved_components: Dict[str, str],
) -> str:
    """Format resolved_components as a prompt injection block."""
    if not resolved_components:
        return "(No resolved components available.)"
    parts: List[str] = []
    for name, snippet in resolved_components.items():
        parts.append(f"### {name}\n{snippet}")
    return "\n\n".join(parts)


def _write_output_files(generated_files: Dict[str, str]) -> Path:
    """Write all generated files to the ``output/`` directory on disk.

    Creates ``output/__init__.py`` if not already present in the file set.
    Returns the output directory path.
    """
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for filename, code in generated_files.items():
        filepath = _OUTPUT_DIR / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_text(code, encoding="utf-8")

    # Ensure __init__.py always exists
    init_path = _OUTPUT_DIR / "__init__.py"
    if not init_path.exists():
        init_path.write_text(
            '"""Auto-generated model package."""\n',
            encoding="utf-8",
        )

    return _OUTPUT_DIR


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = textwrap.dedent("""\
    You are a PyTorch code generation expert specializing in Hugging Face
    transformers.  You MUST strictly follow the rules below — no exceptions.

    ═══════════════════════════════════════════
    STRICT RULES  (from .agent/rules/hf_cfg.md)
    ═══════════════════════════════════════════
    {cfg_rules}
    ═══════════════════════════════════════════

    Additional Hard Constraints:
    • Every configuration class MUST inherit from `transformers.PretrainedConfig`.
    • Every model class MUST inherit from `transformers.PreTrainedModel`.
    • The model's `forward()` MUST accept `input_ids` and `attention_mask`.
    • The model's `forward()` MUST return
      `transformers.modeling_outputs.CausalLMOutputWithPast`.
    • All hyperparameters MUST live in the Config class — never hard-coded.
    • All sub-modules MUST be registered as `nn.Module` children.
    • Use `safetensors` format for serialization.

    Return your code inside a single ```python ... ``` fenced block.
    Do NOT include any text outside the fenced block.
""")


_USER_PROMPT = textwrap.dedent("""\
    ## Architecture (from paper)
    {cleaned_markdown}

    ## Resolved Mathematical Components
    {resolved_components}

    ## Previously Generated Files (Interface Freeze)
    The files below have already been generated and their interfaces are
    FROZEN.  You must import from them exactly as written.

    {dependency_code}

    {refinement_section}

    ## Task
    Write the complete, production-ready PyTorch code for `{current_file}`.

    Requirements:
    1. Implement every method completely — no placeholders, no `pass`, no `...`
    2. All tensor operations must have correct shapes.
    3. When labels are provided to forward(), compute cross-entropy loss
       (left-shifted teacher forcing).
    4. Support KV-cache via past_key_values / use_cache for incremental decoding.
    5. All hyperparameters come from the Config object.
    6. Produce ONLY the code inside a ```python``` block.
""")


# ---------------------------------------------------------------------------
# Core engine
# ---------------------------------------------------------------------------

def run_got_coder(state: PipelineState) -> dict:
    """Generate HF-compliant code for each file in topological order.

    This function is the **implementation phase** of the GoT pipeline.
    It iterates over ``state["topo_order"]``, constructs a strict prompt
    for GPT-4o including the HF CFG rules, paper architecture, resolved
    component math, and all upstream "Interface Freeze" code, then parses
    the ``python`` fenced block from the response.

    On refinement loops (when ``state["errors"]`` is non-empty), the
    smoke-test diagnostics are injected so the LLM can self-repair.

    Parameters
    ----------
    state : PipelineState
        Must contain at minimum ``topo_order`` and ``cleaned_markdown``.

    Returns
    -------
    dict
        ``{"generated_files": ..., "generation_phase": "implement"}``
    """
    # ── Mark phase ──────────────────────────────────────────────────────
    state["generation_phase"] = "implement"

    # ── Initialise generated_files if absent ────────────────────────────
    if "generated_files" not in state or state["generated_files"] is None:
        state["generated_files"] = {}
    generated_files: Dict[str, str] = state["generated_files"]

    # ── Read inputs ─────────────────────────────────────────────────────
    topo_order: List[str] = state.get("topo_order", [])
    cleaned_markdown: str = state.get("cleaned_markdown", "")
    resolved_components: Dict[str, str] = state.get("resolved_components", {})
    errors: List[str] = state.get("errors", [])
    diagnostics = state.get("diagnostics", [])

    # ── Read CFG rules from disk (injected verbatim) ────────────────────
    cfg_rules = _read_hf_cfg_rules()

    system_prompt = _SYSTEM_PROMPT.format(cfg_rules=cfg_rules)

    # ── Build refinement section (only if previous errors exist) ────────
    refinement_section = ""
    if errors:
        diag_text = ""
        if diagnostics:
            # Format diagnostics into a readable block
            diag_parts = []
            for i, d in enumerate(diagnostics, 1):
                if isinstance(d, dict):
                    detect = d.get("detect", "")
                    diagnose = d.get("diagnose", "")
                    diag_parts.append(f"  {i}. {detect}\n     Diagnosis: {diagnose}")
                else:
                    diag_parts.append(f"  {i}. {d}")
            diag_text = "\n".join(diag_parts)
        else:
            diag_text = "\n".join(f"  - {e}" for e in errors)

        refinement_section = (
            "## ⚠️ REFINEMENT LOOP — Previous Code Failed Smoke Test\n"
            "The previous code failed the smoke test. Fix the following "
            "stack trace / diagnostics:\n\n"
            f"```\n{diag_text}\n```\n\n"
            "You MUST fix all issues listed above while keeping the rest of "
            "the implementation correct."
        )

    # ── Resolved components block ───────────────────────────────────────
    resolved_section = _build_resolved_components_section(resolved_components)

    # ── Generate each file in topological order ─────────────────────────
    for current_file in topo_order:
        # Build upstream dependency code section (Interface Freeze)
        dependency_code = _build_dependency_code_section(
            current_file, topo_order, generated_files,
        )

        # Truncate paper markdown to fit token budget (~14k chars)
        paper_excerpt = cleaned_markdown[:14000]
        if len(cleaned_markdown) > 14000:
            paper_excerpt += "\n\n... (paper text truncated for token budget)"

        user_prompt = _USER_PROMPT.format(
            cleaned_markdown=paper_excerpt,
            resolved_components=resolved_section,
            dependency_code=dependency_code,
            refinement_section=refinement_section,
            current_file=current_file,
        )

        # ── Call GPT-4o ─────────────────────────────────────────────────
        raw_response = _call_llm(system_prompt, user_prompt)

        # ── Parse the ```python block ───────────────────────────────────
        code = _parse_python_block(raw_response)

        # ── Store in state ──────────────────────────────────────────────
        generated_files[current_file] = code

    # ── Write files to output/ on disk ──────────────────────────────────
    output_dir = _write_output_files(generated_files)

    # ── Return state update ─────────────────────────────────────────────
    return {
        "generated_files": state["generated_files"],
        "generation_phase": "implement",
    }


# ---------------------------------------------------------------------------
# Alias for graph.py compatibility
# ---------------------------------------------------------------------------
# graph.py imports ``got_coder_node`` — route it through run_got_coder.

def got_coder_node(state: PipelineState) -> Dict[str, Any]:
    """LangGraph node wrapper around :func:`run_got_coder`.

    Adds message logging and error handling expected by the pipeline.
    """
    messages = list(state.get("messages", []))
    current_errors = list(state.get("errors", []))

    messages.append("[Stage 3 - GoT Coder] Starting code generation (implement phase)...")

    try:
        result = run_got_coder(state)

        n_files = len(result.get("generated_files", {}))
        messages.append(
            f"[Stage 3 - GoT Coder] ✅ Generated {n_files} files → output/"
        )

        # Merge messages/errors into the result
        result["messages"] = messages
        result["errors"] = current_errors
        return result

    except Exception as exc:
        current_errors.append(f"[Stage 3] Code generation failed: {exc}")
        messages.append(f"[Stage 3 - GoT Coder] ❌ Failed: {exc}")
        return {
            "messages": messages,
            "errors": current_errors,
        }
