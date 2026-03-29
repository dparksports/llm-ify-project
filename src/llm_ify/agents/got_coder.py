"""Stage 3 — Graph-of-Thought (GoT) Coder Agent.

Generates Hugging Face-compliant PyTorch code in topological DAG order,
following the four GoT phases from Figure 4 of the NERFIFY paper:

    Phase 1 — DAG Construction:   Map paper → HF component dependency graph
    Phase 2 — Interface Freeze:   Generate stub files with signatures only
    Phase 3 — Implementation:     GPT-4o fills implementations with paper math
    Phase 4 — Integration Test:   Validate imports resolve, shapes match

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

import json
import textwrap
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from llm_ify.pipeline.dag import build_repo_dag
from llm_ify.state import GeneratedFile, PipelineState


# ---------------------------------------------------------------------------
# System prompt fragments
# ---------------------------------------------------------------------------

_CFG_SYSTEM_PROMPT = """\
You are a PyTorch code generation expert specializing in Hugging Face transformers.

## STRICT RULES (hf_cfg.md)
{cfg_rules}

## Additional Constraints
- Every configuration class MUST inherit from `transformers.PretrainedConfig`
- Every model class MUST inherit from `transformers.PreTrainedModel`
- The model's `forward()` MUST accept `input_ids` and `attention_mask`
- The model's `forward()` MUST return `transformers.modeling_outputs.CausalLMOutputWithPast`
- All hyperparameters MUST live in the Config class, never hard-coded
- All sub-modules MUST be registered as `nn.Module` children
- Use `safetensors` format for serialization
"""

_PHASE1_DAG_PROMPT = """\
Analyze the following research paper and determine the Hugging Face repository
file structure following this convention:

    configuration_<name>.py  — PretrainedConfig subclass
    modeling_<name>.py       — PreTrainedModel subclass with all sub-modules
    __init__.py              — re-exports

Return ONLY a valid JSON object with this schema:
{{
  "model_name": "<short_snake_case_name>",
  "files": ["configuration_<name>.py", "modeling_<name>.py", "__init__.py"],
  "dependencies": {{
    "configuration_<name>.py": [],
    "modeling_<name>.py": ["configuration_<name>.py"],
    "__init__.py": ["configuration_<name>.py", "modeling_<name>.py"]
  }},
  "key_components": {{
    "config_attributes": ["attr1: type = default", ...],
    "model_architecture": "brief description",
    "core_equations": ["equation from paper", ...]
  }}
}}

## Paper Content:
{paper_text}
"""

_PHASE2_STUB_PROMPT = """\
Generate a Python stub file for `{filename}` in a Hugging Face-compatible
repository for the model "{model_name}".

This is Phase 2 (Interface Freeze): generate ONLY the class definition,
constructor signature, and method signatures with docstrings.
Do NOT implement any method body — use `...` or `pass` as placeholder.

Dependencies available (already generated): {available_deps}

## Key components from paper:
{key_components}

## RULES:
{cfg_rules_short}

Return ONLY the Python source code, no markdown fences or explanation.
"""

_PHASE3_IMPL_PROMPT = """\
Implement the full working code for `{filename}` in a Hugging Face-compatible
repository for the model "{model_name}".

This is Phase 3 (Implementation): fill in ALL method bodies with working
PyTorch code that faithfully implements the paper's architecture.

## Previously generated stubs for this file:
```python
{stub_code}
```

## Paper equations and architecture:
{key_components}

## Dependency files already implemented:
{dep_code}

## RULES:
{cfg_rules_short}

Requirements:
1. Implement every method completely — no placeholders, no `pass`, no `...`
2. All tensor operations must have correct shapes
3. When labels are provided to forward(), compute cross-entropy loss
4. Support KV-cache via past_key_values for incremental decoding
5. All hyperparameters come from the Config object

Return ONLY the Python source code, no markdown fences or explanation.
"""


# ---------------------------------------------------------------------------
# LLM helper
# ---------------------------------------------------------------------------

def _get_llm(temperature: float = 0.2) -> ChatOpenAI:
    """Create a ChatOpenAI instance for code generation."""
    return ChatOpenAI(
        model="gpt-4o",
        temperature=temperature,
        max_tokens=8192,
    )


def _call_llm(system: str, user: str, temperature: float = 0.2) -> str:
    """Send a system+user message pair and return the assistant's text."""
    llm = _get_llm(temperature)
    messages = [
        SystemMessage(content=system),
        HumanMessage(content=user),
    ]
    response = llm.invoke(messages)
    return response.content.strip()


def _strip_code_fences(text: str) -> str:
    """Remove ```python ... ``` or ``` ... ``` fences if present."""
    import re
    text = text.strip()
    # Remove opening fence
    text = re.sub(r"^```(?:python)?\s*\n?", "", text)
    # Remove closing fence
    text = re.sub(r"\n?```\s*$", "", text)
    return text


def _short_cfg_rules(cfg_rules: str) -> str:
    """Extract just the critical rules for prompts (keep prompts shorter)."""
    # Take only the first 2000 chars of the CFG rules to stay under token limits
    if len(cfg_rules) > 2000:
        return cfg_rules[:2000] + "\n... (truncated)"
    return cfg_rules


# ---------------------------------------------------------------------------
# Phase implementations
# ---------------------------------------------------------------------------

def _phase1_dag_construction(
    cleaned_markdown: str,
    cfg_rules: str,
    messages: List[str],
) -> Dict[str, Any]:
    """Phase 1: Analyze paper and build the repository DAG.

    Returns dict with model_name, files, dependencies, key_components.
    """
    messages.append("[Stage 3 - GoT Phase 1] Analyzing paper for DAG construction...")

    system = _CFG_SYSTEM_PROMPT.format(cfg_rules=_short_cfg_rules(cfg_rules))
    user = _PHASE1_DAG_PROMPT.format(
        paper_text=cleaned_markdown[:12000]  # First ~12k chars for context
    )

    response = _call_llm(system, user, temperature=0.1)

    # Parse JSON from response
    try:
        # Try to find JSON in the response
        import re
        json_match = re.search(r"\{[\s\S]+\}", response)
        if json_match:
            dag_spec = json.loads(json_match.group())
        else:
            raise ValueError("No JSON object found in LLM response")
    except (json.JSONDecodeError, ValueError) as e:
        messages.append(f"[Stage 3 - GoT Phase 1] ⚠️ JSON parse failed: {e}")
        messages.append("[Stage 3 - GoT Phase 1] Using default HF structure")
        dag_spec = _default_dag_spec()

    # Validate and fix the DAG spec
    dag_spec = _validate_dag_spec(dag_spec)

    # Build and validate the actual DAG
    adjacency, topo_order = build_repo_dag(
        dag_spec["files"],
        dag_spec["dependencies"],
    )

    messages.append(
        f"[Stage 3 - GoT Phase 1] DAG built: {len(dag_spec['files'])} files, "
        f"topo order: {topo_order}"
    )

    return {
        "dag_spec": dag_spec,
        "topo_order": topo_order,
    }


def _phase2_interface_freeze(
    dag_spec: Dict[str, Any],
    topo_order: List[str],
    cfg_rules: str,
    messages: List[str],
) -> Dict[str, str]:
    """Phase 2: Generate interface stubs for each file in topological order.

    Returns dict of filename → stub source code.
    """
    messages.append("[Stage 3 - GoT Phase 2] Freezing interfaces...")
    stubs: Dict[str, str] = {}

    system = _CFG_SYSTEM_PROMPT.format(cfg_rules=_short_cfg_rules(cfg_rules))
    key_components = json.dumps(dag_spec.get("key_components", {}), indent=2)

    for filename in topo_order:
        deps = dag_spec["dependencies"].get(filename, [])
        available = [f for f in deps if f in stubs]

        user = _PHASE2_STUB_PROMPT.format(
            filename=filename,
            model_name=dag_spec.get("model_name", "custom"),
            available_deps=", ".join(available) if available else "none",
            key_components=key_components,
            cfg_rules_short=_short_cfg_rules(cfg_rules),
        )

        stub_code = _call_llm(system, user, temperature=0.1)
        stub_code = _strip_code_fences(stub_code)
        stubs[filename] = stub_code

        messages.append(
            f"[Stage 3 - GoT Phase 2] Stub generated: {filename} "
            f"({len(stub_code)} chars)"
        )

    return stubs


def _phase3_implementation(
    dag_spec: Dict[str, Any],
    topo_order: List[str],
    stubs: Dict[str, str],
    cfg_rules: str,
    messages: List[str],
) -> Dict[str, str]:
    """Phase 3: Implement each file fully, in topological order.

    Returns dict of filename → full implementation source code.
    """
    messages.append("[Stage 3 - GoT Phase 3] Implementing files...")
    implementations: Dict[str, str] = {}

    system = _CFG_SYSTEM_PROMPT.format(cfg_rules=_short_cfg_rules(cfg_rules))
    key_components = json.dumps(dag_spec.get("key_components", {}), indent=2)

    for filename in topo_order:
        deps = dag_spec["dependencies"].get(filename, [])

        # Gather dependency code
        dep_code_parts = []
        for dep in deps:
            if dep in implementations:
                dep_code_parts.append(
                    f"# === {dep} ===\n{implementations[dep]}"
                )

        user = _PHASE3_IMPL_PROMPT.format(
            filename=filename,
            model_name=dag_spec.get("model_name", "custom"),
            stub_code=stubs.get(filename, "# no stub available"),
            key_components=key_components,
            dep_code="\n\n".join(dep_code_parts) if dep_code_parts else "# none",
            cfg_rules_short=_short_cfg_rules(cfg_rules),
        )

        impl_code = _call_llm(system, user, temperature=0.2)
        impl_code = _strip_code_fences(impl_code)
        implementations[filename] = impl_code

        messages.append(
            f"[Stage 3 - GoT Phase 3] Implemented: {filename} "
            f"({len(impl_code)} chars)"
        )

    return implementations


# ---------------------------------------------------------------------------
# Defaults and validation
# ---------------------------------------------------------------------------

def _default_dag_spec() -> Dict[str, Any]:
    """Fallback DAG spec when LLM fails to produce valid JSON."""
    return {
        "model_name": "custom_model",
        "files": [
            "configuration_custom_model.py",
            "modeling_custom_model.py",
            "__init__.py",
        ],
        "dependencies": {
            "configuration_custom_model.py": [],
            "modeling_custom_model.py": ["configuration_custom_model.py"],
            "__init__.py": [
                "configuration_custom_model.py",
                "modeling_custom_model.py",
            ],
        },
        "key_components": {
            "config_attributes": [],
            "model_architecture": "Unknown — using default transformer",
            "core_equations": [],
        },
    }


def _validate_dag_spec(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and fix a DAG spec to ensure it's well-formed."""
    # Ensure required keys
    if "model_name" not in spec:
        spec["model_name"] = "custom_model"
    if "files" not in spec:
        spec = _default_dag_spec()
        return spec
    if "dependencies" not in spec:
        spec["dependencies"] = {f: [] for f in spec["files"]}

    # Ensure __init__.py exists
    if "__init__.py" not in spec["files"]:
        spec["files"].append("__init__.py")
        spec["dependencies"]["__init__.py"] = [
            f for f in spec["files"] if f != "__init__.py"
        ]

    # Ensure all dependency targets are in files
    all_files = set(spec["files"])
    for fname, deps in list(spec["dependencies"].items()):
        spec["dependencies"][fname] = [d for d in deps if d in all_files]

    # Ensure all files have a dependencies entry
    for f in spec["files"]:
        if f not in spec["dependencies"]:
            spec["dependencies"][f] = []

    return spec


# ---------------------------------------------------------------------------
# Main GoT coder node
# ---------------------------------------------------------------------------

def got_coder_node(state: PipelineState) -> Dict[str, Any]:
    """Generate HF-compatible code files in dependency order.

    Executes GoT Phases 1-3 (Phase 4 integration testing is handled
    by the critique agent in Stage 4).
    """
    cleaned_markdown = state.get("cleaned_markdown", "")
    cfg_rules = state.get("cfg_rules", "")
    messages = list(state.get("messages", []))
    errors = list(state.get("errors", []))

    messages.append("[Stage 3 - GoT Coder] Starting code generation...")

    if not cleaned_markdown:
        errors.append("[Stage 3] No cleaned markdown available — run Stage 1 first")
        return {"errors": errors, "messages": messages}

    try:
        # Phase 1: DAG Construction
        phase1 = _phase1_dag_construction(cleaned_markdown, cfg_rules, messages)
        dag_spec = phase1["dag_spec"]
        topo_order = phase1["topo_order"]

        # Phase 2: Interface Freeze
        stubs = _phase2_interface_freeze(dag_spec, topo_order, cfg_rules, messages)

        # Phase 3: Implementation
        implementations = _phase3_implementation(
            dag_spec, topo_order, stubs, cfg_rules, messages
        )

        # Build GeneratedFile objects
        generated_files = {}
        for filename in topo_order:
            gf = GeneratedFile(
                filename=filename,
                content=implementations.get(filename, stubs.get(filename, "")),
                stub_only=filename not in implementations,
                validated=False,
            )
            generated_files[filename] = gf.model_dump()

        messages.append(
            f"[Stage 3 - GoT Coder] ✅ Generated {len(generated_files)} files"
        )

        return {
            "repo_dag": dag_spec["dependencies"],
            "topo_order": topo_order,
            "generated_files": generated_files,
            "generation_phase": "implement",
            "messages": messages,
            "errors": errors,
        }

    except Exception as e:
        errors.append(f"[Stage 3] Code generation failed: {e}")
        messages.append(f"[Stage 3 - GoT Coder] ❌ Failed: {e}")
        return {
            "messages": messages,
            "errors": errors,
        }
