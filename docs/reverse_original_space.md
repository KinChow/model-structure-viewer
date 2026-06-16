# Original Space Reverse Notes

Target: `https://huggingface.co/spaces/maomao88/model_structure_viewer`

The original Space is a Docker SDK app with a FastAPI backend and React frontend. The important backend API is:

```text
GET /api/model?name=<hf_model_id>&model_type=<optional>
```

The backend implementation calls `AutoConfig.from_pretrained(..., trust_remote_code=True)` and then creates an empty-weight model through one of the `AutoModel*` classes under `accelerate.init_empty_weights()`. It recursively walks `module.named_children()` and `module.named_parameters(recurse=False)` to produce a tree that resembles `print(model)`.

This approach is useful for old or standard Transformer models, but it is fragile for new models:

- It requires the corresponding `transformers` architecture class or remote code to load.
- It may need optional runtime dependencies such as custom attention kernels.
- It still constructs a Python module graph, so remote code and framework compatibility matter.
- VLM, MoE, sparse-attention, and custom multimodal wrappers can fail before any structure can be returned.
- The UI error collapses many failures into a generic "Failed to fetch model structure" message.

This repository uses a different MVP strategy:

- Read `config.json` and small metadata only.
- Do not instantiate models.
- Do not download weights.
- Do not execute remote model code.
- Generate an explainable structure tree from architecture templates and generic fallbacks.

For MiniMax-M3 this is the right tradeoff: the model is custom multimodal MoE with sparse attention, but its `config.json` contains enough structural information for a useful diagram.
