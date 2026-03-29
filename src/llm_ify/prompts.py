"""Centralised prompt templates for the LLM-IFY multi-agent pipeline.

This module clearly separates prompt design from execution logic, providing
transparent heuristics and full context injection templates for each stage.
"""

import textwrap

# =============================================================================
# Stage 1: Summarizer
# =============================================================================

SUMMARIZER_PROMPT = textwrap.dedent("""\
    You are a research paper analysis expert specializing in deep learning
    architectures and Hugging Face transformers.

    Your task is to extract structured information from a research paper.

    CRITICAL RULES:
    1. **Preserve ALL LaTeX math verbatim**.  Copy every equation exactly as
       written — $$...$$, \\begin{equation}, inline $...$ — with no
       simplification, no translation to prose, and no omission.
    2. **Preserve ALL pseudocode** verbatim.
    3. For file_dag: use the Hugging Face naming convention:
       - configuration_<name>.py  (inherits PretrainedConfig, no deps)
       - modeling_<name>.py       (inherits PreTrainedModel, depends on config)
       - __init__.py              (re-exports, depends on all other files)
    4. architecture_name must be short snake_case (e.g. 'deepseek_v3').
    5. hyperparameters should include every numeric default mentioned in the
       paper (hidden_size, num_layers, num_attention_heads, vocab_size, etc.).
""")

CITATION_DETECTION_PROMPT = textwrap.dedent("""\
    You are an expert at analyzing ML research papers.  Identify components
    that reference an external citation but whose mathematical formulation 
    is NOT present in the paper text.

    Examples of unresolved references:
    - "We adopt the distortion loss from [3]"
    - "Following [17], we use RMSNorm"
    - "The hash encoding of [33] is used"

    Do NOT flag components whose math IS already defined in the text.
""")

# =============================================================================
# Stage 2: Citation Crawler
# =============================================================================

CITATION_ROUTING_PROMPT = textwrap.dedent("""\
    You are an expert ML architectures router.
    Analyze the following missing architectural component cited in a paper.
    
    You must classify whether this component is:
    1. A "STANDARD_LIBRARY": A common operation that needs no retrieval because 
       it's natively available or trivially known (e.g., standard LayerNorm, 
       RMSNorm, ReLU, CrossEntropyLoss, basic Multi-Head Attention).
    2. A "NOVELTY": A complex, custom, or novel mechanism requiring explicit 
       mathematical formulation and retrieval (e.g., distortion loss from Mip-NeRF, 
       Multi-Head Latent Attention from DeepSeek-V3).

    Respond with ONLY a JSON object containing a single key "classification" 
    with the exact string value "STANDARD_LIBRARY" or "NOVELTY".
""")


CITATION_RESOLUTION_PROMPT = textwrap.dedent("""\
    You are an expert ML researcher. For the component described below, provide:
    1. The EXACT mathematical formulation (in LaTeX, preserve $$...$$ blocks)
    2. A concise, working PyTorch implementation
    3. The source paper

    Search the web if needed to find the correct formulation.
    Return ONLY a JSON object with keys: 
    "math_formulation", "pytorch_snippet", "source_paper".
""")

# =============================================================================
# Stage 3: GoT Coder
# =============================================================================

GOT_CODER_PROMPT = textwrap.dedent("""\
    You are a PyTorch code generation expert specializing in Hugging Face
    transformers.  You MUST strictly follow the rules below — no exceptions.

    ═══════════════════════════════════════════
    STRICT RULES (Hugging Face CFG Contract)
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

GOT_CODER_USER_PROMPT = textwrap.dedent("""\
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


# =============================================================================
# Stage 4: Critique
# =============================================================================

CRITIQUE_PROMPT = textwrap.dedent("""\
    You are a Senior PyTorch Debugger and Machine Learning Systems Engineer.
    I have generated PyTorch code for an ML architecture, but it failed the
    local execution/smoke tests. 
    
    Here is the failure context from the Sandbox execution stdout/stderr:

    ```
    {error_trace}
    ```

    Analyze the stack trace and static analysis issues. You must diagnose 
    the failure and provide a structured JSON diagnosis.

    Return ONLY a JSON object matching this schema:
    {
        "detect": "Brief 1-sentence summary of the error.",
        "diagnose": "Detailed explanation of WHY this failed mathematically or systemically.",
        "recover": "General strategy to fix this issue without breaking other components.",
        "patch": "Specific code fix or instruction for the GoT Coder to implement.",
        "action": "RETRY",
        "confidence": 0.0 to 1.0 float representing how likely your patch will fix it
    }
""")
