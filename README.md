# LLM-IFY

A multi-agent framework for turning research papers into Hugging Face-compatible PyTorch repositories.

Inspired by the NERFIFY paper ([arXiv:2603.00805v1](https://arxiv.org/abs/2603.00805v1)), LLM-IFY adapts the domain-specific paper-to-code paradigm from Nerfstudio/NeRF to the Hugging Face transformers ecosystem.

## Architecture

LLM-IFY is a **LangGraph orchestrator** with four pipeline stages:

```
PDF → [Summarizer] → [Citation Crawler] → [GoT Coder] → [Critique] → HF Repository
            │                │                  │              │
         Stage 1          Stage 2           Stage 3        Stage 4
       PyMuPDF parse   Dependency BFS    GPT-4o code gen  Smoke-test
```

### Stage 1 — Summarizer
Parses research paper PDFs via PyMuPDF, extracting structured representations: headings, equations (LaTeX preserved), algorithms, figure captions, and bibliography.

### Stage 2 — Citation Crawler
Resolves implicit dependencies via recursive multi-hop retrieval over the citation graph. Extracts architectural modules, loss functions, and training protocols.

### Stage 3 — GoT Coder (Graph-of-Thought)
Generates HF-compliant PyTorch code in topological DAG order using three phases:
1. **DAG Construction** — maps paper to HF component dependency graph
2. **Interface Freeze** — generates stub files with signatures only
3. **Implementation** — fills method bodies with paper math → PyTorch

### Stage 4 — Critique
Validates generated code with 7 static analysis checks and import smoke-tests. Triggers repair loops (max 5 iterations) when issues are found.

## Generated Code Contract (CFG)

All output code strictly follows the Hugging Face transformers API:
- Configuration classes inherit from `PretrainedConfig`
- Model classes inherit from `PreTrainedModel`
- `forward()` accepts `input_ids` and `attention_mask`, returns `CausalLMOutputWithPast`
- Serialization via `save_pretrained()` / `from_pretrained()`

See [`.agent/rules/hf_cfg.md`](.agent/rules/hf_cfg.md) for the full grammar specification.

## Project Structure

```
src/llm_ify/
  graph.py                  # LangGraph StateGraph orchestration
  state.py                  # TypedDict defining the graph state
  agents/
    summarizer.py           # Stage 1: PyMuPDF parsing + CFG injection
    citation_crawler.py     # Stage 2: Dependency resolution via GPT-4o
    got_coder.py            # Stage 3: GoT code generation in DAG order
    critique.py             # Stage 4: Smoke-tests + diagnostic patching
  pipeline/
    dag.py                  # Kahn's algorithm for topological sorting
```

## Quickstart

```bash
# Install dependencies
pip install langchain langgraph langchain-openai pymupdf networkx pydantic

# Set API key
export OPENAI_API_KEY="sk-..."

# Run pipeline
python -c "
from llm_ify.graph import run_pipeline
result = run_pipeline('path/to/paper.pdf', open('.agent/rules/hf_cfg.md').read())
print(result['final_repository'])
"
```

## License

[Apache License 2.0](LICENSE)
