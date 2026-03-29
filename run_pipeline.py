#!/usr/bin/env python3
"""Execute the LLM-IFY LangGraph pipeline.

Initializes the graph, seeds the state with the HF CFG rules and the parsed paper,
and streams the node execution updates to the console.
"""

import os
import sys
from pathlib import Path

from llm_ify.graph import build_graph
from llm_ify.state import PipelineState

def main():
    root = Path(__file__).resolve().parent
    docs_path = root / "docs" / "paper_parsed.md"
    cfg_rules_path = root / ".agent" / "rules" / "hf_cfg.md"

    if not cfg_rules_path.exists():
        print(f"Error: Could not find HF CFG rules at {cfg_rules_path}")
        sys.exit(1)

    cfg_rules = cfg_rules_path.read_text(encoding="utf-8")

    # The pipeline uses Gemini-1.5-Pro, which requires GOOGLE_API_KEY
    if not os.environ.get("GOOGLE_API_KEY"):
        print("Warning: GOOGLE_API_KEY environment variable is not set.")
        print("The pipeline will fail if the LLM cannot be reached.")

    print("🚀 Initializing LLM-IFY LangGraph Pipeline...")
    app = build_graph()

    # Create initial state. We pass empty pdf_path since the summarizer node 
    # will fall back to reading docs_path (which contains paper_parsed.md).
    initial_state: PipelineState = {
        "pdf_path": "",
        "cfg_rules": cfg_rules,
        "errors": [],
        "messages": [],
        "refinement_iteration": 0,
    }

    print("==========================================================")
    print("Starting pipeline execution (streaming state updates)...")
    print("==========================================================")

    # Track seen messages to avoid duplicate printing during streaming
    seen_messages = set()
    
    # Stream the graph execution updates
    for event in app.stream(initial_state):
        for node_name, state_update in event.items():
            print(f"\n[>>> NODE: {node_name.upper()} <<<]")
            
            # Print state keys that were updated
            keys_updated = list(state_update.keys())
            print(f"Updated keys: {', '.join(keys_updated)}")
            
            # Specifically print new messages for progress tracking
            current_messages = state_update.get("messages", [])
            new_messages = [msg for msg in current_messages if msg not in seen_messages]
            
            for msg in new_messages:
                print(f"  {msg}")
                seen_messages.add(msg)
                
            errors = state_update.get("errors", [])
            if errors:
                print("  --- Errors ---")
                # Print only the last error or new errors to avoid spam
                print(f"  {errors[-1]}")
                
            # If critique generated diagnostics, print a summary
            if "diagnostics" in state_update and state_update["diagnostics"]:
                print(f"  Found {len(state_update['diagnostics'])} diagnostics to patch.")

    print("\n==========================================================")
    print("Pipeline execution completed.")
    print("Check the 'output/' directory for generated Code.")
    print("==========================================================")

if __name__ == "__main__":
    main()
