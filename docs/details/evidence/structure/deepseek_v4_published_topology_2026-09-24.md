# DeepSeek V4 published compressor/indexer topology

- Date: 2026-09-24
- Scope: `DeepSeek-V4-Flash`, `DeepSeek-V4-Flash-0731`, `DeepSeek-V4-Flash-Vision-Exp`, `DeepSeek-V4-Pro`, `DeepSeek-V4-Pro-0813`
- Change: model structure viewer only; Graph IR remains v2.

## Evidence

The production source-ref manifests under `models/deepseek-ai/DeepSeek-V4-Flash/source-ref.json`
(and the corresponding V4 variants) expose the following paths:

```text
layers.N.self_attn.compressor.kv_proj
layers.N.self_attn.compressor.gate_proj
layers.N.self_attn.compressor.kv_norm
layers.N.self_attn.compressor.rotary_emb
layers.N.self_attn.compressor.indexer.kv_proj
layers.N.self_attn.compressor.indexer.gate_proj
layers.N.self_attn.compressor.indexer.kv_norm
layers.N.self_attn.compressor.indexer.q_b_proj
layers.N.self_attn.compressor.indexer.rotary_emb
layers.N.self_attn.compressor.indexer.scorer.weights_proj
```

The pinned Transformers implementation used for the audit is
`transformers v5.16.1`, `modeling_deepseek_v4.py`.  Its `DeepseekV4CSACompressor`
and `DeepseekV4HCACompressor` define the two projections, position bias,
RMSNorm and compression RoPE; `DeepseekV4CSACompressor` owns the nested
`DeepseekV4Indexer`, whose scorer owns `weights_proj`.

## Implementation decision

The previous graph put the published children behind pseudo nodes such as
`indexer.q_proj`, `indexer.compressor.wkv_gate`, and `compressor.norm`.  That
made source-ref binding incomplete and made the diagram assert an implementation
path that is not present in the published module tree.

The V4 builder now represents the exact nested topology:

```text
self_attn
└── compressor (billing composite)
    ├── kv_proj
    ├── gate_proj
    ├── position_bias
    ├── window_reduce
    ├── kv_norm
    ├── rotary_emb
    └── indexer
        ├── kv_proj
        ├── gate_proj
        ├── position_bias
        ├── window_reduce
        ├── kv_norm
        ├── q_b_proj
        ├── rotary_emb
        └── scorer
            └── weights_proj
```

The parent composite retains the existing `mla_kv_compress` billing rule.  Real
checkpoint-shaped linear/norm/bias parameters are owned by the corresponding
children, so the parent no longer duplicates their residency.  Internal
control/fused-in edges are explicitly marked in the shape audit rather than
being treated as same-width activation edges.

`window_reduce` is a non-checkpoint semantic node.  It represents the published
softmax gate plus position-bias weighted reduction over complete compression
windows (including the CSA overlap layout).  It prevents the diagram from
claiming that `position_bias` is an input to `gate_proj`, or that the packed
projection feeds RMSNorm without the window reduction.

DeepSeek V4.1 remains on its existing flat approximation because its current
production source-ref manifest does not expose the V4 nested compressor paths.
No V4 APE parameter is introduced for V4.1.

## Verification

- V4 config-only and production-artifact mechanism tests: 10/10.
- DeepSeek V4 compressor/source-ref/cost tests: 13/13.
- Built-in model and architecture tests covering this change: passed.
- Weight declaration and activation-shape identity audits: passed.
- `ops-edge.golden.json` and `ops-spec-tree.golden.json` regenerated only for the six DeepSeek V4/V4.1 entries affected by the intentional topology change.
