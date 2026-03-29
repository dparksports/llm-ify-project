# Hugging Face Transformers — Context-Free Grammar (CFG)

> **Scope:** All generated PyTorch code in this repository MUST strictly adhere to
> the rules defined below. No exceptions.

---

## 1. Model Configuration

All model configurations **must** inherit from `transformers.PretrainedConfig`.

```
ConfigClass → class <Name>Config(PretrainedConfig):
    model_type: str                          # unique identifier, e.g. "my_model"
    attribute_list: (attribute)*             # all hyper-parameters as typed attrs
    attribute → name: type = default_value
```

### Required attributes (minimum set)

| Attribute             | Type    | Description                             |
|-----------------------|---------|-----------------------------------------|
| `vocab_size`          | `int`   | Vocabulary size of the tokenizer        |
| `hidden_size`         | `int`   | Dimensionality of hidden representations|
| `num_hidden_layers`   | `int`   | Number of transformer layers            |
| `num_attention_heads` | `int`   | Number of attention heads               |
| `intermediate_size`   | `int`   | Feed-forward inner dimension            |
| `max_position_embeddings` | `int` | Maximum sequence length                |

---

## 2. Model Architecture

All model classes **must** inherit from `transformers.PreTrainedModel`.

```
ModelClass → class <Name>ForCausalLM(PreTrainedModel):
    config_class  = <Name>Config
    base_model_prefix = "<name>"

    __init__(config: <Name>Config) → void
    forward(
        input_ids: Optional[torch.LongTensor],
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.LongTensor]       = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor]      = None,
        labels: Optional[torch.LongTensor]              = None,
        use_cache: Optional[bool]                       = None,
        output_attentions: Optional[bool]               = None,
        output_hidden_states: Optional[bool]            = None,
        return_dict: Optional[bool]                     = None,
    ) → CausalLMOutputWithPast
```

### `forward()` Contract

| Rule | Detail |
|------|--------|
| **Accept** | `input_ids` (`LongTensor [B, T]`) and `attention_mask` (`Tensor [B, T]`) as the two primary positional/keyword arguments. |
| **Return** | A `transformers.modeling_outputs.CausalLMOutputWithPast` named-tuple containing at minimum: `loss`, `logits`, `past_key_values`, `hidden_states`, `attentions`. |
| **Labels** | When `labels` is provided, compute the language-modeling loss (cross-entropy with left-shift) inside `forward()` and populate `loss` in the output. |
| **KV-Cache** | Support incremental decoding via `past_key_values` / `use_cache`. |

---

## 3. Sub-Module Grammar

```
TransformerBlock → Attention  LayerNorm  FeedForward  LayerNorm
Attention        → QKV_Projection  ScaledDotProduct  Output_Projection
FeedForward      → Linear  Activation  Linear
Activation       → GELU | SiLU | ReLU          # paper-specific choice
LayerNorm        → RMSNorm | nn.LayerNorm      # paper-specific choice
```

All sub-modules must be registered as `nn.Module` children so they are visible to
`model.parameters()`, `model.state_dict()`, and HF's serialization utilities
(`save_pretrained` / `from_pretrained`).

---

## 4. File Layout Convention

```
src/
  modeling_<name>.py      # PreTrainedModel + all sub-modules
  configuration_<name>.py # PretrainedConfig
  tokenization_<name>.py  # (optional) custom tokenizer
  __init__.py              # re-exports
```

---

## 5. Serialization & Interoperability

- Models must be saveable/loadable via `model.save_pretrained()` / `Model.from_pretrained()`.
- Weights must be stored in `safetensors` format by default.
- A `config.json` produced by `config.save_pretrained()` must round-trip without data loss.

---

## 6. Prohibited Patterns

| ❌ Do NOT | Why |
|-----------|-----|
| Use raw `nn.Module` as the top-level model class | Breaks `from_pretrained`, `push_to_hub`, Trainer integration. |
| Omit `input_ids` or `attention_mask` from `forward()` | Violates the HF generate / pipeline contract. |
| Return plain tensors from `forward()` | Must return `CausalLMOutputWithPast` for compatibility with `generate()`. |
| Hard-code hyperparameters in module bodies | All tunables must live in the `Config` object. |
| Use custom serialization (pickle, torch.save) | Must rely on HF `save_pretrained` / `from_pretrained`. |
