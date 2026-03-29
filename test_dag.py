#!/usr/bin/env python3
"""Test suite for BuildRepoDAG (Kahn's algorithm).

Validates:
1. Correct topological sort on a valid DAG
2. Cycle detection raises CyclicDependencyError
3. Edge cases: single file, diamond dependencies, unknown file reference
"""

from __future__ import annotations

import sys, os

# Ensure src/ is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from llm_ify.pipeline.dag import CyclicDependencyError, build_repo_dag


def test_basic_topological_sort():
    """Mock repo: config.py → modules.py → modeling.py"""
    files = ["config.py", "modules.py", "modeling.py"]
    deps = {
        "config.py": [],                             # depends on nothing
        "modules.py": ["config.py"],                 # depends on config
        "modeling.py": ["modules.py", "config.py"],  # depends on both
    }

    adjacency, topo_order = build_repo_dag(files, deps)

    print("=" * 60)
    print("TEST 1: Basic Topological Sort")
    print(f"  Files:       {files}")
    print(f"  Dependencies: {deps}")
    print(f"  Topo order:  {topo_order}")
    print(f"  Adjacency:   {adjacency}")

    # config.py must come before modules.py and modeling.py
    assert topo_order.index("config.py") < topo_order.index("modules.py"), \
        "config.py must precede modules.py"
    assert topo_order.index("config.py") < topo_order.index("modeling.py"), \
        "config.py must precede modeling.py"
    # modules.py must come before modeling.py
    assert topo_order.index("modules.py") < topo_order.index("modeling.py"), \
        "modules.py must precede modeling.py"

    print("  ✅ PASSED\n")


def test_diamond_dependency():
    """Diamond: config → {model, tokenizer} → pipeline"""
    files = ["config.py", "model.py", "tokenizer.py", "pipeline.py"]
    deps = {
        "config.py": [],
        "model.py": ["config.py"],
        "tokenizer.py": ["config.py"],
        "pipeline.py": ["model.py", "tokenizer.py"],
    }

    adjacency, topo_order = build_repo_dag(files, deps)

    print("=" * 60)
    print("TEST 2: Diamond Dependency")
    print(f"  Topo order: {topo_order}")

    assert topo_order.index("config.py") < topo_order.index("model.py")
    assert topo_order.index("config.py") < topo_order.index("tokenizer.py")
    assert topo_order.index("model.py") < topo_order.index("pipeline.py")
    assert topo_order.index("tokenizer.py") < topo_order.index("pipeline.py")

    print("  ✅ PASSED\n")


def test_single_file():
    """Single file with no dependencies."""
    files = ["standalone.py"]
    deps = {"standalone.py": []}

    adjacency, topo_order = build_repo_dag(files, deps)

    print("=" * 60)
    print("TEST 3: Single File")
    print(f"  Topo order: {topo_order}")
    assert topo_order == ["standalone.py"]
    print("  ✅ PASSED\n")


def test_cycle_detection():
    """Circular: A → B → C → A must raise CyclicDependencyError."""
    files = ["a.py", "b.py", "c.py"]
    deps = {
        "a.py": ["c.py"],
        "b.py": ["a.py"],
        "c.py": ["b.py"],
    }

    print("=" * 60)
    print("TEST 4: Circular Dependency Detection")
    print(f"  Dependencies: {deps}")

    try:
        build_repo_dag(files, deps)
        print("  ❌ FAILED — no exception raised!")
        sys.exit(1)
    except CyclicDependencyError as e:
        print(f"  Caught CyclicDependencyError: {e}")
        print(f"  Cycle members: {e.cycle_members}")
        assert set(e.cycle_members) == {"a.py", "b.py", "c.py"}
        print("  ✅ PASSED\n")


def test_unknown_dependency():
    """Referencing a file not in the file set must raise ValueError."""
    files = ["config.py", "model.py"]
    deps = {
        "config.py": [],
        "model.py": ["nonexistent.py"],  # bad ref
    }

    print("=" * 60)
    print("TEST 5: Unknown Dependency Reference")

    try:
        build_repo_dag(files, deps)
        print("  ❌ FAILED — no exception raised!")
        sys.exit(1)
    except ValueError as e:
        print(f"  Caught ValueError: {e}")
        print("  ✅ PASSED\n")


def test_hf_repo_layout():
    """Realistic HF repository layout matching the CFG from hf_cfg.md."""
    files = [
        "configuration_zipnerf.py",
        "modeling_zipnerf.py",
        "tokenization_zipnerf.py",
        "__init__.py",
    ]
    deps = {
        "configuration_zipnerf.py": [],
        "modeling_zipnerf.py": ["configuration_zipnerf.py"],
        "tokenization_zipnerf.py": [],
        "__init__.py": [
            "configuration_zipnerf.py",
            "modeling_zipnerf.py",
            "tokenization_zipnerf.py",
        ],
    }

    adjacency, topo_order = build_repo_dag(files, deps)

    print("=" * 60)
    print("TEST 6: Realistic HF Repo Layout")
    print(f"  Topo order: {topo_order}")

    # Config must come before modeling
    assert topo_order.index("configuration_zipnerf.py") < topo_order.index("modeling_zipnerf.py")
    # __init__ must be last (depends on everything)
    assert topo_order[-1] == "__init__.py"
    print("  ✅ PASSED\n")


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n🧪 LLM-IFY DAG Test Suite\n")
    test_basic_topological_sort()
    test_diamond_dependency()
    test_single_file()
    test_cycle_detection()
    test_unknown_dependency()
    test_hf_repo_layout()

    print("=" * 60)
    print("🎉 ALL 6 TESTS PASSED")
    print("=" * 60)
