import { moduleSpec, withShapeDims } from "./base.js";
import { operatorSpec, sdpaAttentionModule } from "../operators/ops/index.js";
import { shapeFlow, tensorShapes } from "../operators/shapes.js";
import { visionDimensions } from "../config/visionDims.js";
import { foldedLayerName } from "./foldedLayerName.js";
import { hfVisionAttr } from "../archs/index.js";

// DeepSeek V4/V4.1 released inference/vision.py: ViT(Linear patch embed,
// Block pre-norm attention + residual, pre-norm SwiGLU + residual, final RMSNorm).
// Checkpoint names are vision.patch_embed.proj / vision.blocks.N.* / vision.norm.
// Keep this family-specific topology separate from Qwen/Kimi generic visual blocks.
export function nativeVitTowerModule(normalized) {
  const id = hfVisionAttr(normalized);
  const d = visionDimensions(normalized);
  const shape = `[batch, visual tokens, vision hidden size=${d.hidden}]`;
  const qkv = `[batch, visual tokens, fused qkv=${3 * d.hidden}]`;
  const q = `[batch, visual tokens, vision heads=${d.heads}, head dimension=${d.headDim}]`;
  const scores = `[batch, vision heads=${d.heads}, query visual tokens, key visual tokens]`;
  const doubled = `[batch, visual tokens, gate and up=${2 * d.intermediate}]`;
  const intermediate = `[batch, visual tokens, intermediate=${d.intermediate}]`;
  const blockId = `${id}.blocks.0`;
  const op = (suffix, name, kind, input, output, attributes = {}) =>
    operatorSpec(`${blockId}.${suffix}`, name, kind, {
      ...shapeFlow(input[0], output[0]), modality: "vision", ...attributes,
    }, { input: input[1], output: output[1] });
  const visual = [shape, d.visual];
  const head = [q, d.q];
  const children = [
    op("block_input", "vision block input", "identity", visual, visual,
      { checkpoint_module: false }),
    op("norm1", "vision attention RMSNorm", "rmsnorm", visual, visual),
    op("attn.wqkv", "vision fused QKV projection", "linear", visual, [qkv, d.qkv],
      { bias: true, semantic_role: "vision_q_k_v_projection" }),
    op("attn.qkv_split", "vision QKV split", "attention_qkv_split", [qkv, d.qkv], head,
      { checkpoint_module: false, split_sizes: [d.hidden, d.hidden, d.hidden] }),
    op("attn.rope", "vision 2D rotary position embedding", "rope", head, head, {
      checkpoint_module: false, position_encoding: "rope_2d",
      partial_rotary_factor: 1, semantic_role: "vision_query_key_rotation",
    }),
    sdpaAttentionModule(`${blockId}.attn`, {
      attentionQuery: q, attentionKey: q, attentionValue: q,
      attentionScores: scores, attentionProbabilities: scores, attentionContext: q,
    }, {
      attentionQuery: d.q, attentionKey: d.q, attentionValue: d.q,
      attentionScores: d.scores, attentionProbabilities: d.scores, attentionContext: d.context,
    }, { scores: { attention_kind: "vision" }, context: { attention_kind: "vision" },
      modality: "vision", attention_mask_kind: "bidirectional" }),
    op("attn.context_merge", "merge vision attention heads", "identity",
      [q, d.context], visual, { checkpoint_module: false, view_transform: "transpose_reshape" }),
    op("attn.wo", "vision output projection", "linear", visual, visual,
      { bias: true, semantic_role: "vision_attention_output_projection" }),
    op("residual_attn", "vision attention residual add", "residual_add", visual, visual,
      { checkpoint_module: false }),
    op("norm2", "vision MLP RMSNorm", "rmsnorm", visual, visual),
    op("mlp.w1", "vision gate and up projection", "linear", visual,
      [doubled, [-1, -1, 2 * d.intermediate]]),
    op("mlp.gate_up", "vision SwiGLU", "swiglu",
      [doubled, [-1, -1, 2 * d.intermediate]], [intermediate, d.intermediateShape],
      { checkpoint_module: false, split_sizes: [d.intermediate, d.intermediate] }),
    op("mlp.w2", "vision output MLP projection", "linear",
      [intermediate, d.intermediateShape], visual),
    op("residual_mlp", "vision MLP residual add", "residual_add", visual, visual,
      { checkpoint_module: false }),
  ];
  const block = withShapeDims(moduleSpec(blockId,
    foldedLayerName(0, normalized.visionLayers - 1, "Vision Block"),
    "vision-block-group", {
      class: "Block", modality: "vision", range: `0..${normalized.visionLayers - 1}`,
      dataflow_edges: [
        ["block_input", "norm1"], ["norm1", "attn.wqkv"], ["attn.wqkv", "attn.qkv_split"],
        ["attn.qkv_split", "attn.rope"], ["attn.qkv_split", "attn.sdpa"],
        ["attn.rope", "attn.sdpa"], ["attn.sdpa", "attn.context_merge"],
        ["attn.context_merge", "attn.wo"],
        ["attn.wo", "residual_attn"], ["block_input", "residual_attn"],
        ["residual_attn", "norm2"], ["norm2", "mlp.w1"],
        ["mlp.w1", "mlp.gate_up"], ["mlp.gate_up", "mlp.w2"],
        ["mlp.w2", "residual_mlp"], ["residual_attn", "residual_mlp"],
      ],
      dataflow_edge_relations: [
        { from: "attn.qkv_split", to: "attn.rope", label: "Q, K" },
        { from: "attn.qkv_split", to: "attn.sdpa", label: "V (unrotated)" },
        { from: "block_input", to: "residual_attn", label: "residual input (before norm)" },
        { from: "residual_attn", to: "residual_mlp", label: "residual input" },
      ],
    }, children, normalized.visionLayers), d.visual, d.visual);
  const patch = operatorSpec(`${id}.patch_embed.proj`, "vision patch embedding", "linear", {
    ...shapeFlow(tensorShapes(normalized).visionInput, shape),
    modality: "vision", bias: true, semantic_role: "patch_embedding",
    patch_size: d.patch,
  }, { input: d.patchInput, output: d.visual });
  const finalNorm = operatorSpec(`${id}.norm`, "vision final RMSNorm", "rmsnorm", {
    ...shapeFlow(shape, shape), modality: "vision",
  }, { input: d.visual, output: d.visual });
  return withShapeDims(moduleSpec(id, "Vision Tower", "vision-encoder", {
    class: "ViT", modality: "vision", num_hidden_layers: normalized.visionLayers,
    num_attention_heads: d.heads, hidden_size: d.hidden,
    dataflow_edges: [["patch_embed.proj", "blocks.0"], ["blocks.0", "norm"]],
    ...shapeFlow(tensorShapes(normalized).visionInput, shape),
  }, [patch, block, finalNorm], normalized.visionLayers), [-1, -1, -1, -1, -1], d.visual);
}
