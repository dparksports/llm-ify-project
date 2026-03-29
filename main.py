#!/usr/bin/env python3
"""CLI entry-point for the LLM-IFY pipeline.

Usage
-----
    python main.py synthesize --pdf path/to/paper.pdf --name deepseek_v3

The ``synthesize`` command loads environment variables from ``.env``,
seeds the pipeline state, and streams execution updates to the console.
Generated code is written to ``output/<name>/``.
"""

import os
import sys
from pathlib import Path

import click
from dotenv import load_dotenv

# Ensure the project's src/ is importable
_PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))


@click.group()
def cli():
    """LLM-IFY — autonomous research-to-code synthesis."""


@cli.command()
@click.option(
    "--pdf",
    required=True,
    type=click.Path(exists=True, dir_okay=False, resolve_path=True),
    help="Path to the research paper PDF.",
)
@click.option(
    "--name",
    required=True,
    type=str,
    help="Architecture / model name (snake_case). Used as the output sub-directory.",
)
def synthesize(pdf: str, name: str):
    """Run the full LLM-IFY pipeline on a research paper PDF.

    Reads the paper, extracts architecture metadata, generates
    Hugging Face-compliant PyTorch code, and validates it via
    smoke tests — all in one shot.

    Generated files land in ``output/<name>/``.
    """
    # ── Load .env (API keys, etc.) ──────────────────────────────────────
    load_dotenv(_PROJECT_ROOT / ".env")

    if not os.environ.get("GOOGLE_API_KEY"):
        click.secho(
            "⚠  GOOGLE_API_KEY is not set. The pipeline will fail if the LLM "
            "cannot be reached.",
            fg="yellow",
            err=True,
        )

    # ── Read HF CFG rules ──────────────────────────────────────────────
    cfg_rules_path = _PROJECT_ROOT / ".agent" / "rules" / "hf_cfg.md"
    cfg_rules = ""
    if cfg_rules_path.exists():
        cfg_rules = cfg_rules_path.read_text(encoding="utf-8")
    else:
        click.secho(
            f"⚠  HF CFG rules not found at {cfg_rules_path}; using defaults.",
            fg="yellow",
            err=True,
        )

    # ── Import pipeline (deferred so --help is fast) ────────────────────
    from llm_ify.graph import build_graph
    from llm_ify.state import PipelineState

    click.echo("🚀 Initializing LLM-IFY LangGraph Pipeline...")
    app = build_graph()

    initial_state: PipelineState = {
        "pdf_path": pdf,
        "cfg_rules": cfg_rules,
        "architecture_name": name,
        "errors": [],
        "messages": [],
        "refinement_iteration": 0,
    }

    click.echo("=" * 60)
    click.echo(f"Paper : {pdf}")
    click.echo(f"Model : {name}")
    click.echo(f"Output: output/{name}/")
    click.echo("=" * 60)

    # ── Stream execution ────────────────────────────────────────────────
    seen_messages: set[str] = set()

    for event in app.stream(initial_state):
        for node_name, state_update in event.items():
            click.echo(f"\n[>>> NODE: {node_name.upper()} <<<]")
            click.echo(f"Updated keys: {', '.join(state_update.keys())}")

            for msg in state_update.get("messages", []):
                if msg not in seen_messages:
                    click.echo(f"  {msg}")
                    seen_messages.add(msg)

            errors = state_update.get("errors", [])
            if errors:
                click.secho(f"  ❌ {errors[-1]}", fg="red")

            if state_update.get("diagnostics"):
                click.echo(
                    f"  Found {len(state_update['diagnostics'])} diagnostics to patch."
                )

    click.echo("\n" + "=" * 60)
    click.secho("Pipeline execution completed.", fg="green", bold=True)
    click.echo(f"Check  output/{name}/  for generated code.")
    click.echo("=" * 60)


if __name__ == "__main__":
    cli()
