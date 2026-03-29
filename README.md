# LLM-IFY 🚀

Automating the translation of Deep Learning research papers into production-ready Hugging Face PyTorch repositories.

## 🤔 Why was it created?

Implementing complex AI models from research papers often involves days of untangling dense mathematical notation, tracking down implicit dependencies from older papers, and ensuring strict compliance with frameworks like Hugging Face `transformers`. 

**LLM-IFY** was created to automate this painstaking process. Inspired by the **NERFIFY** paper ([arXiv:2603.00805v1](https://arxiv.org/abs/2603.00805v1)), it tackles the structural and semantic gaps between academic PDFs and functional code. By introducing formal grammar constraints and closed-loop testing, LLM-IFY reliably generates state-of-the-art model implementations that pass structural shape checks right out of the box.

## 🤖 What is it?

LLM-IFY is a multi-agent LangGraph orchestrator powered by GPT-4o. It features a four-stage pipeline:

1. **📄 Paper Summarizer**: Parses standard academic PDFs (via `fitz`/PyMuPDF), extracting exact mathematical formulations, pseudo-algorithms, and architectural novelties without losing structural integrity.
2. **🕸️ Citation Crawler**: Detects implicit math dependencies (e.g., "We use the memory-efficient attention from [14]") and iteratively resolves them using web search or a localized Knowledge Base.
3. **🧠 Graph-of-Thought (GoT) Coder**: Generates complex multi-file repositories in topological order. It strictly obeys a Context-Free Grammar (CFG) enforcing that classes inherit `PreTrainedModel` or `PretrainedConfig`, and that `forward()` signatures follow the Hugging Face spec.
4. **🕵️ Critique & Self-Healing**: Automatically executes an import and runtime smoke-test on the generated model (e.g., running 50 dummy training steps). It feeds any OOM, shape mismatch, or syntax stack traces back to the coder for continuous refinement until convergence.

## 🛠️ How to use it

### Prerequisites

Ensure you have Python 3.9+ and the required packages:

```bash
pip install langchain langchain-openai langgraph pydantic pymupdf
```

You must also have an OpenAI api key set:

```bash
export OPENAI_API_KEY="sk-..."
```

### Running the Pipeline

You can kick off the entire end-to-end framework using the root runner script. 

```bash
python run_pipeline.py
```

This will:
1. Spin up the LangGraph application.
2. Read the initial research text (`docs/paper_parsed.md` or a provided PDF).
3. Inject the `hf_cfg.md` rules.
4. Stream the real-time agent state, allowing you to watch the DAG construction, interface freezes, and smoke tests negotiate with each other.

Once the critique agent gives the green light, the generated Hugging Face-compliant files will be deposited into the `output/` directory, ready to be imported and trained! 
