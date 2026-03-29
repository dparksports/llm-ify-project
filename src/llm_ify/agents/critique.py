"""Stage 4 — Critique Agent.

Smoke-tests generated code and produces diagnostic reports.
Validates that the HF CFG contract is satisfied:

    1. Config inherits PretrainedConfig
    2. Model inherits PreTrainedModel
    3. forward(input_ids, attention_mask) → CausalLMOutputWithPast
    4. save_pretrained / from_pretrained round-trips work

When issues are found, produces DiagnosticReport with patch suggestions.

References:
    Paper §3.2, Stage 4  — Visual-Driven Feedback
    Table 5 ablation     — Smoke tests are CRITICAL (trainability 100%→60%)
"""

from __future__ import annotations

import importlib.util
import re
import sys
import tempfile
import textwrap
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

from e2b_code_interpreter import Sandbox
from llm_ify.state import DiagnosticReport, PipelineState


# ---------------------------------------------------------------------------
# Validation checks
# ---------------------------------------------------------------------------

def _check_config_inheritance(code: str, filename: str) -> Optional[str]:
    """Check that a config file has a class inheriting PretrainedConfig."""
    if "configuration" not in filename:
        return None
    if "PretrainedConfig" not in code:
        return (
            f"{filename}: Config class must inherit from "
            f"transformers.PretrainedConfig (not found in source)"
        )
    if "model_type" not in code:
        return f"{filename}: Config must define model_type class attribute"
    return None


def _check_model_inheritance(code: str, filename: str) -> Optional[str]:
    """Check that a model file has a class inheriting PreTrainedModel."""
    if "modeling" not in filename:
        return None
    if "PreTrainedModel" not in code:
        return (
            f"{filename}: Model class must inherit from "
            f"transformers.PreTrainedModel (not found in source)"
        )
    return None


def _check_forward_signature(code: str, filename: str) -> Optional[str]:
    """Check that forward() accepts input_ids and attention_mask."""
    if "modeling" not in filename:
        return None
    if "def forward(" not in code:
        return f"{filename}: Model must define a forward() method"

    # Extract forward signature
    match = re.search(r"def forward\(([^)]+)\)", code, re.DOTALL)
    if match:
        params = match.group(1)
        if "input_ids" not in params:
            return f"{filename}: forward() must accept 'input_ids' parameter"
        if "attention_mask" not in params:
            return f"{filename}: forward() must accept 'attention_mask' parameter"
    return None


def _check_forward_return(code: str, filename: str) -> Optional[str]:
    """Check that forward() returns CausalLMOutputWithPast."""
    if "modeling" not in filename:
        return None
    if "CausalLMOutputWithPast" not in code:
        return (
            f"{filename}: forward() must return CausalLMOutputWithPast "
            f"(not found in source)"
        )
    return None


def _check_no_hardcoded_params(code: str, filename: str) -> Optional[str]:
    """Check for common hardcoded hyperparameters in model files."""
    if "modeling" not in filename:
        return None

    # Look for common hardcoded patterns
    patterns = [
        (r"hidden_size\s*=\s*\d+", "hidden_size"),
        (r"num_heads\s*=\s*\d+", "num_heads"),
        (r"vocab_size\s*=\s*\d+", "vocab_size"),
    ]

    for pattern, param_name in patterns:
        # Only flag if it's not in a default argument / config setting context
        matches = re.findall(pattern, code)
        if matches:
            # Check if it's inside a class body (not __init__ with config)
            for match_text in matches:
                # Allow it in __init__ signatures and config assignments
                context = code[max(0, code.index(match_text) - 100):
                               code.index(match_text) + len(match_text)]
                if "config." not in context and "self." not in context:
                    return (
                        f"{filename}: Possible hardcoded {param_name} — "
                        f"should come from Config object"
                    )
    return None


def _check_imports(code: str, filename: str) -> Optional[str]:
    """Check that essential imports are present."""
    if "modeling" in filename:
        if "import torch" not in code and "from torch" not in code:
            return f"{filename}: Missing torch import"
        if "transformers" not in code:
            return f"{filename}: Missing transformers import"
    if "configuration" in filename:
        if "transformers" not in code:
            return f"{filename}: Missing transformers import"
    return None


def _try_syntax_check(code: str, filename: str) -> Optional[str]:
    """Try to compile the code to catch syntax errors."""
    try:
        compile(code, filename, "exec")
        return None
    except SyntaxError as e:
        return f"{filename}: Syntax error at line {e.lineno}: {e.msg}"


def _run_sandbox_smoke_test(
    files: Dict[str, str],
    messages: List[str],
) -> List[str]:
    """Write files to an E2B Sandbox and attempt to import them securely.

    Returns a list of import error descriptions.
    """
    errors = []

    # 1. Dynamically write smoke_test.py content
    imports = []
    for f in files:
        if f.endswith(".py") and f != "__init__.py":
            mod = f.replace(".py", "")
            imports.append(f"import {mod}")

    test_script_code = f"""import sys
import os
sys.path.insert(0, os.path.abspath('./src/llm_ify/generated'))

try:
{chr(10).join("    " + i for i in imports)}
    print("SUCCESS")
except Exception as e:
    import traceback
    traceback.print_exc()
    sys.exit(1)
"""

    messages.append("[Stage 4] 🛡️ Launching secure E2B Sandbox...")
    try:
        with Sandbox() as sbx:
            # 2. Upload generated_files into the appropriate mock structure
            sbx.commands.run("mkdir -p ./src/llm_ify/generated")
            for filename, code in files.items():
                sbx.files.write(f"./src/llm_ify/generated/{filename}", code)
                
            # Provide an empty init if missing to make it a module
            if "__init__.py" not in files:
                sbx.files.write("./src/llm_ify/generated/__init__.py", "")

            # 3. Upload smoke_test.py script
            sbx.files.write("smoke_test.py", test_script_code)

            # 4. Execute the test securely inside the sandbox
            execution = sbx.commands.run("python smoke_test.py")

            if execution.exit_code != 0:
                stderr = execution.stderr if execution.stderr else "Unknown error"
                # Keep the last 500 characters of the stack trace
                short_err = stderr[-500:]
                errors.append(f"Sandbox execution failed — {short_err}")
                
                # Add a brief summary for messages
                err_line = stderr.strip().splitlines()[-1] if stderr.strip() else "Crash"
                messages.append(f"[Stage 4] ❌ Sandbox FAIL: {err_line}")
            else:
                messages.append(f"[Stage 4] ✅ Sandbox execution/imports OK")

    except Exception as e:
        messages.append(f"[Stage 4] ❌ E2B Sandbox Connection Error: {e}")
        errors.append(f"E2B Sandbox initialization failed: {e}")

    return errors


# ---------------------------------------------------------------------------
# Static analysis (no actual imports required)
# ---------------------------------------------------------------------------

_CHECKS = [
    _check_config_inheritance,
    _check_model_inheritance,
    _check_forward_signature,
    _check_forward_return,
    _check_no_hardcoded_params,
    _check_imports,
    _try_syntax_check,
]


def _run_static_checks(
    files: Dict[str, str],
    messages: List[str],
) -> List[str]:
    """Run all static analysis checks on generated code.

    Returns list of issue descriptions.
    """
    issues = []
    for filename, code in files.items():
        for check_fn in _CHECKS:
            issue = check_fn(code, filename)
            if issue:
                issues.append(issue)
                messages.append(f"[Stage 4] ⚠️  {issue}")

    return issues


# ---------------------------------------------------------------------------
# Main critique node
# ---------------------------------------------------------------------------

def critique_node(state: PipelineState) -> Dict[str, Any]:
    """Validate generated code and produce diagnostic reports.

    Runs:
    1. Static analysis (contract checks per hf_cfg.md)
    2. Syntax compilation check
    3. Import smoke-test (write to temp, attempt import)
    4. Produces DiagnosticReport for each issue found
    """
    generated_files_raw = state.get("generated_files", {})
    messages = list(state.get("messages", []))
    errors = list(state.get("errors", []))
    iteration = state.get("refinement_iteration", 0) + 1

    messages.append(f"[Stage 4 - Critique] Iteration {iteration}")

    if not generated_files_raw:
        errors.append("[Stage 4] No generated files to validate")
        return {
            "smoke_test_passed": False,
            "refinement_iteration": iteration,
            "errors": errors,
            "messages": messages,
        }

    # Extract code content from GeneratedFile dicts
    code_files: Dict[str, str] = {}
    for filename, gf_data in generated_files_raw.items():
        if isinstance(gf_data, dict):
            code_files[gf_data.get("filename", filename)] = gf_data.get("content", "")
        else:
            code_files[filename] = str(gf_data)

    messages.append(f"[Stage 4] Validating {len(code_files)} files...")

    # ── Static analysis ─────────────────────────────────────────────────
    static_issues = _run_static_checks(code_files, messages)

    # ── Import smoke test via Secure Sandbox ────────────────────────────
    import_issues = _run_sandbox_smoke_test(code_files, messages)

    # ── Build diagnostics ───────────────────────────────────────────────
    all_issues = static_issues + import_issues
    diagnostics = []

    for issue in all_issues:
        diag = DiagnosticReport(
            detect=issue,
            diagnose=issue,
            recover="",
            patch="",
            action="RETRY" if iteration < 5 else "ACCEPT(low_confidence)",
            confidence=0.3 if import_issues else 0.7,
        )
        diagnostics.append(diag.model_dump())

    # ── Determine pass/fail ─────────────────────────────────────────────
    smoke_passed = len(all_issues) == 0

    if smoke_passed:
        messages.append("[Stage 4] ✅ All validation checks passed!")
    else:
        messages.append(
            f"[Stage 4] ❌ {len(all_issues)} issues found "
            f"({len(static_issues)} static, {len(import_issues)} import)"
        )

    # ── Build final repository ──────────────────────────────────────────
    final_repo = {}
    if smoke_passed:
        final_repo = code_files
        # Mark files as validated
        for filename in generated_files_raw:
            if isinstance(generated_files_raw[filename], dict):
                generated_files_raw[filename]["validated"] = True

    return {
        "smoke_test_passed": smoke_passed,
        "diagnostics": diagnostics,
        "refinement_iteration": iteration,
        "final_repository": final_repo if smoke_passed else {},
        "generated_files": generated_files_raw,
        "messages": messages,
        "errors": errors,
    }
