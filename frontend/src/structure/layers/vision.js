import { moduleSpec, withShapeDims } from "./base.js";
import { operatorSpec, sdpaAttentionModule } from "../operators/ops/index.js";
import { shapeFlow, tensorShapes } from "../operators/shapes.js";
import { visionDimensions } from "../config/visionDims.js";
import { hfNamedClass, hfVisionAttr, recipeFlag, recipeVisionInternalMerger } from "../archs/index.js";
import { foldedLayerName } from "./foldedLayerName.js";
import { nativeVitTowerModule } from "./nativeVit.js";

function visionLayerModule(id, normalized) {
  const d = visionDimensions(normalized);
  const rope2d = recipeFlag(normalized, "visionRope2d");
  const visual = `[batch, visual tokens, vision hidden size=${d.hidden}]`;
  const qkv = `[batch, visual tokens, fused qkv=${3 * d.qkvHiddenSize}]`;
  const q = `[batch, visual tokens, vision heads=${d.heads}, head dimension=${d.headDim}]`;
  const scores = `[batch, vision heads=${d.heads}, query visual tokens, key visual tokens]`;
  const context = `[batch, visual tokens, vision heads=${d.heads}, head dimension=${d.headDim}]`;
  const intermediate = `[batch, visual tokens, vision intermediate=${d.intermediate}]`;
  const attentionChildren = [
    operatorSpec(`${id}.input_norm`, "vision input RMSNorm", "rmsnorm", { ...shapeFlow(visual, visual), modality: "vision" }, { input: d.visual, output: d.visual }),
    operatorSpec(`${id}.qkv_proj`, "vision QKV projection", "linear", {
      ...shapeFlow(visual, qkv), modality: "vision", semantic_role: "vision_q_k_v_projection",
    }, { input: d.visual, output: d.qkv }),
    operatorSpec(`${id}.qkv_split`, "vision QKV split", "attention_qkv_split", {
      ...shapeFlow(qkv, `${q}, ${q}, ${q}`),
      split_sizes: [d.heads * d.headDim, d.heads * d.headDim, d.heads * d.headDim],
      modality: "vision",
    }, { input: d.qkv, output: d.q }),
    ...(rope2d ? [operatorSpec(`${id}.rope`, "vision 2D rotary position embedding", "rope", {
      ...shapeFlow(`${q}, ${q}`, `${q}, ${q}`),
      modality: "vision", position_encoding: "rope_2d", checkpoint_module: false,
      partial_rotary_factor: 1,
      semantic_role: "vision_query_key_rotation",
    }, { input: d.q, output: d.q })] : []),
    sdpaAttentionModule(id, {
      attentionQuery: q,
      attentionKey: q,
      attentionValue: q,
      attentionScores: scores,
      attentionProbabilities: scores,
      attentionContext: context,
    }, {
      attentionQuery: d.q,
      attentionKey: d.q,
      attentionValue: d.q,
      attentionScores: d.scores,
      attentionProbabilities: d.scores,
      attentionContext: d.context,
    }, {
      scoresName: "vision attention scores",
      scores: { attention_kind: "vision" },
      context: { attention_kind: "vision" },
      modality: "vision", attention_mask_kind: "bidirectional",
    }),
    operatorSpec(`${id}.out_proj`, "vision output projection", "linear", {
      ...shapeFlow(context, visual), modality: "vision", semantic_role: "vision_attention_output_projection",
    }, { input: d.context, output: d.visual }),
    operatorSpec(`${id}.post_norm`, "vision post-attention RMSNorm", "rmsnorm", { ...shapeFlow(visual, visual), modality: "vision" }, { input: d.visual, output: d.visual }),
  ];
  const mlpChildren = d.gatedMlp
    ? [
      operatorSpec(`${id}.gate_proj`, "vision gate projection", "linear", { ...shapeFlow(visual, intermediate), modality: "vision" }, { input: d.visual, output: d.intermediateShape }),
      operatorSpec(`${id}.up_proj`, "vision up projection", "linear", { ...shapeFlow(visual, intermediate), modality: "vision" }, { input: d.visual, output: d.intermediateShape }),
      operatorSpec(`${id}.activation`, "vision SwiGLU activation", "swiglu", { ...shapeFlow(`${intermediate}, ${intermediate}`, intermediate), modality: "vision" }, { input: d.intermediateShape, output: d.intermediateShape }),
      operatorSpec(`${id}.down_proj`, "vision down projection", "linear", { ...shapeFlow(intermediate, visual), modality: "vision" }, { input: d.intermediateShape, output: d.visual }),
    ]
    : [
      operatorSpec(`${id}.fc1`, "vision feed-forward projection", "linear", { ...shapeFlow(visual, intermediate), modality: "vision" }, { input: d.visual, output: d.intermediateShape }),
      operatorSpec(`${id}.activation`, "vision activation", "vision_activation", { ...shapeFlow(intermediate, intermediate), modality: "vision", activation: normalized.visionConfig?.hidden_act }, { input: d.intermediateShape, output: d.intermediateShape }),
      operatorSpec(`${id}.fc2`, "vision feed-forward output projection", "linear", { ...shapeFlow(intermediate, visual), modality: "vision" }, { input: d.intermediateShape, output: d.visual }),
    ];
  const children = [...attentionChildren, ...mlpChildren];
  children.forEach((child) => { child.attributes.vision_stage = "encoder"; });
  const mlpEdges = d.gatedMlp
    ? [["post_norm", "gate_proj"], ["post_norm", "up_proj"], ["gate_proj", "activation"], ["up_proj", "activation"], ["activation", "down_proj"]]
    : [["post_norm", "fc1"], ["fc1", "activation"], ["activation", "fc2"]];
  return withShapeDims(moduleSpec(id, "VisionLayer", "vision-block-group", {
    class: hfNamedClass(normalized, "visionBlockClass", "VisionBlock"),
    hidden_size: d.hidden,
    num_attention_heads: d.heads,
    intermediate_size: d.intermediate,
    modality: "vision",
    dataflow_edges: [
      ["input_norm", "qkv_proj"], ["qkv_proj", "qkv_split"],
      ...(rope2d ? [["qkv_split", "rope"], ["rope", "sdpa"], ["qkv_split", "sdpa"]] : [["qkv_split", "sdpa"]]),
      ["sdpa", "out_proj"],
      ["out_proj", "post_norm"], ...mlpEdges,
    ],
    ...(rope2d ? { dataflow_edge_relations: [
      { from: "qkv_split", to: "rope", label: "Q, K" },
      { from: "qkv_split", to: "sdpa", label: "V (unrotated)" },
    ] } : {}),
  }, children), d.visual, d.visual);
}

function visionMergerModule(id, normalized) {
  const d = visionDimensions(normalized);
  const inputShape = `[batch, patch tokens, vision hidden size=${d.hidden}]`;
  const mergedShape = `[batch, visual tokens, merged width=${d.mergedWidth}]`;
  const outputShape = `[batch, visual tokens, vision hidden size=${normalized.visionOutputSize || d.hidden}]`;
  const children = [
    operatorSpec(`${id}.patch_merge`, "vision patch merge", "vision_merge", {
      ...shapeFlow(inputShape, mergedShape), modality: "vision", vision_stage: "merger", merge_size: d.mergeSize,
    }, { input: d.visual, output: [-1, -1, d.mergedWidth] }),
    operatorSpec(`${id}.norm`, "vision merger norm", "rmsnorm", {
      ...shapeFlow(mergedShape, mergedShape), modality: "vision", vision_stage: "merger",
    }, { input: [-1, -1, d.mergedWidth], output: [-1, -1, d.mergedWidth] }),
  ];
  if (recipeFlag(normalized, "visionMergerMlp")) {
    const intermediate = normalized.visionMergerIntermediateSize || d.intermediate;
    children.push(
      operatorSpec(`${id}.proj`, "vision merger projection", "linear", { ...shapeFlow(mergedShape, outputShape), modality: "vision", vision_stage: "merger" }, { input: [-1, -1, d.mergedWidth], output: d.mergedVisual }),
      operatorSpec(`${id}.post_norm`, "vision merger post norm", "rmsnorm", { ...shapeFlow(outputShape, outputShape), modality: "vision", vision_stage: "merger" }, { input: d.mergedVisual, output: d.mergedVisual }),
      operatorSpec(`${id}.gate_proj`, "vision merger gate projection", "linear", { ...shapeFlow(outputShape, `[batch, visual tokens, merger intermediate=${intermediate}]`), modality: "vision", vision_stage: "merger" }, { input: d.mergedVisual, output: [-1, -1, intermediate] }),
      operatorSpec(`${id}.up_proj`, "vision merger up projection", "linear", { ...shapeFlow(outputShape, `[batch, visual tokens, merger intermediate=${intermediate}]`), modality: "vision", vision_stage: "merger" }, { input: d.mergedVisual, output: [-1, -1, intermediate] }),
      operatorSpec(`${id}.activation`, "vision merger SwiGLU activation", "swiglu", { ...shapeFlow("[batch, visual tokens, merger intermediate]", "[batch, visual tokens, merger intermediate]"), modality: "vision", vision_stage: "merger" }, { input: [-1, -1, intermediate], output: [-1, -1, intermediate] }),
      operatorSpec(`${id}.down_proj`, "vision merger down projection", "linear", { ...shapeFlow("[batch, visual tokens, merger intermediate]", outputShape), modality: "vision", vision_stage: "merger" }, { input: [-1, -1, intermediate], output: d.mergedVisual }),
    );
  } else {
    children.push(
      operatorSpec(`${id}.fc1`, "vision merger projection", "linear", { ...shapeFlow(mergedShape, mergedShape), modality: "vision", vision_stage: "merger" }, { input: [-1, -1, d.mergedWidth], output: [-1, -1, d.mergedWidth] }),
      operatorSpec(`${id}.activation`, "vision merger activation", "vision_activation", { ...shapeFlow(mergedShape, mergedShape), modality: "vision", vision_stage: "merger" }, { input: [-1, -1, d.mergedWidth], output: [-1, -1, d.mergedWidth] }),
      operatorSpec(`${id}.fc2`, "vision merger output projection", "linear", { ...shapeFlow(mergedShape, outputShape), modality: "vision", vision_stage: "merger" }, { input: [-1, -1, d.mergedWidth], output: d.mergedVisual }),
    );
  }
  const edges = recipeFlag(normalized, "visionMergerMlp")
    ? [["patch_merge", "norm"], ["norm", "proj"], ["proj", "post_norm"], ["post_norm", "gate_proj"], ["post_norm", "up_proj"], ["gate_proj", "activation"], ["up_proj", "activation"], ["activation", "down_proj"]]
    : [["patch_merge", "norm"], ["norm", "fc1"], ["fc1", "activation"], ["activation", "fc2"]];
  return withShapeDims(moduleSpec(id, "Vision Merger", "vision-merger", {
    class: hfNamedClass(normalized, "patchMergerClass", "VisionPatchMerger"),
    modality: "vision",
    merge_size: d.mergeSize,
    dataflow_edges: edges,
  }, children), d.visual, d.mergedVisual);
}

export function visionTowerModule(normalized) {
  if (recipeFlag(normalized, "visionNativeVit")) {
    return nativeVitTowerModule(normalized);
  }
  if (recipeFlag(normalized, "visionNativeMiniMax")) {
    return miniMaxNativeVitTowerModule(normalized);
  }
  const shapes = tensorShapes(normalized);
  const d = visionDimensions(normalized);
  const layers = normalized.visionLayers || 0;
  const id = hfVisionAttr(normalized);
  const layer = layers > 0 ? visionLayerModule(`${id}.0`, normalized) : null;
  if (layer) {
    layer.repeat = layers;
    layer.attributes.range = `0..${layers - 1}`;
    layer.name = foldedLayerName(0, layers - 1, "VisionLayer");
  }
  const children = [
    operatorSpec(`${id}.patch_embed`, "vision patch embedding", "linear", {
      ...shapeFlow(shapes.visionInput, `[batch, visual tokens, vision hidden size=${d.hidden}]`),
      modality: "vision", semantic_role: "patch_embedding", patch_size: d.patch, temporal_patch_size: d.temporalPatch,
    }, { input: d.patchInput, output: d.visual }),
    operatorSpec(`${id}.position`, "vision position embedding", "vision_position", {
      ...shapeFlow(`[batch, visual tokens, vision hidden size=${d.hidden}]`, `[batch, visual tokens, vision hidden size=${d.hidden}]`),
      modality: "vision",
    }, { input: d.visual, output: d.visual }),
    ...(layer ? [layer] : []),
    ...(recipeVisionInternalMerger(normalized) ? [visionMergerModule(`${id}.merger`, normalized)] : []),
  ];
  return withShapeDims(moduleSpec(
    id,
    "Vision Tower",
    "vision-encoder",
    {
      class: hfNamedClass(normalized, "visionModelClass", "VisionModel"),
      hidden_size: normalized.visionHiddenSize,
      output_hidden_size: normalized.visionOutputSize,
      num_hidden_layers: normalized.visionLayers,
      num_attention_heads: d.heads,
      intermediate_size: d.intermediate,
      vision_tokens: normalized.visionTokens,
      dataflow_edges: [
        ["patch_embed", "position"],
        ...(layer ? [["position", "0"]] : []),
        ...(recipeVisionInternalMerger(normalized) ? [[layer ? "0" : "position", "merger"]] : []),
      ],
      ...shapeFlow(shapes.visionInput, shapes.visionOutput),
    },
    children,
    layers || undefined,
  ), [-1, -1, -1, -1, -1], d.mergedVisual);
}

// MiniMax-M3 VL uses a CLIP-style tower whose published implementation is
// structurally different from the generic fused-QKV vision recipe:
// Conv3d patch embedding, a tower-level LayerNorm, separate q/k/v projections,
// axial 3D RoPE, LayerNorm pre-norm residual blocks, and a two-stage projector.
// Keep the exact checkpoint paths here instead of making the generic builder
// guess whether a visual attention implementation is fused.
function miniMaxNativeVitTowerModule(normalized) {
  const id = hfVisionAttr(normalized);
  const d = visionDimensions(normalized);
  const visual = `[batch, visual tokens, vision hidden size=${d.hidden}]`;
  const q = `[batch, visual tokens, vision heads=${d.heads}, head dimension=${d.headDim}]`;
  const scores = `[batch, vision heads=${d.heads}, query visual tokens, key visual tokens]`;
  const context = `[batch, visual tokens, vision heads=${d.heads}, head dimension=${d.headDim}]`;
  const intermediate = `[batch, visual tokens, vision intermediate=${d.intermediate}]`;
  const blockId = `${id}.layers.0`;
  const visualShape = [-1, -1, d.hidden];
  const headShape = [-1, -1, d.heads, d.headDim];
  const op = (suffix, name, kind, input, output, attributes = {}) =>
    operatorSpec(`${blockId}.${suffix}`, name, kind, {
      ...shapeFlow(input[0], output[0]), modality: "vision", ...attributes,
    }, { input: input[1], output: output[1] });
  const children = [
    op("layers_input", "vision encoder layer input", "identity", [visual, visualShape],
      [visual, visualShape], { checkpoint_module: false }),
    op("layer_norm1", "vision attention LayerNorm", "rmsnorm",
      [visual, visualShape], [visual, visualShape], { affine_bias: true }),
    op("self_attn.q_proj", "vision query projection", "linear", [visual, visualShape],
      [visualShape, visualShape], { bias: true, semantic_role: "vision_query_projection" }),
    op("self_attn.k_proj", "vision key projection", "linear", [visual, visualShape],
      [visualShape, visualShape], { bias: true, semantic_role: "vision_key_projection" }),
    op("self_attn.v_proj", "vision value projection", "linear", [visual, visualShape],
      [visualShape, visualShape], { bias: true, semantic_role: "vision_value_projection" }),
    op("self_attn.q_reshape", "reshape query into heads", "identity", [visual, visualShape],
      [q, headShape], { checkpoint_module: false, view_transform: "reshape_heads" }),
    op("self_attn.k_reshape", "reshape key into heads", "identity", [visual, visualShape],
      [q, headShape], { checkpoint_module: false, view_transform: "reshape_heads" }),
    op("self_attn.v_reshape", "reshape value into heads", "identity", [visual, visualShape],
      [q, headShape], { checkpoint_module: false, view_transform: "reshape_heads" }),
    op("self_attn.rope", "vision axial 3D rotary position embedding", "rope",
      [q, headShape], [q, headShape], {
        checkpoint_module: false, position_encoding: "rope_axial_3d",
        partial_rotary_factor: 1, semantic_role: "vision_query_key_rotation",
      }),
    sdpaAttentionModule(`${blockId}.self_attn`, {
      attentionQuery: q, attentionKey: q, attentionValue: q,
      attentionScores: scores, attentionProbabilities: scores, attentionContext: context,
    }, {
      attentionQuery: d.q, attentionKey: d.q, attentionValue: d.q,
      attentionScores: d.scores, attentionProbabilities: d.scores, attentionContext: d.context,
    }, {
      scores: { attention_kind: "vision" }, context: { attention_kind: "vision" },
      modality: "vision", attention_mask_kind: "bidirectional",
    }),
    op("self_attn.context_merge", "merge vision attention heads", "identity",
      [context, d.context], [visual, visualShape], {
        checkpoint_module: false, view_transform: "transpose_reshape",
      }),
    op("self_attn.out_proj", "vision output projection", "linear",
      [visual, visualShape], [visual, visualShape], { bias: true, semantic_role: "vision_attention_output_projection" }),
    op("layer_norm2", "vision MLP LayerNorm", "rmsnorm",
      [visual, visualShape], [visual, visualShape], { affine_bias: true }),
    op("mlp.fc1", "vision feed-forward projection", "linear",
      [visual, visualShape], [intermediate, [-1, -1, d.intermediate]], { bias: true }),
    op("mlp.activation", "vision GELU activation", "vision_activation",
      [intermediate, [-1, -1, d.intermediate]], [intermediate, [-1, -1, d.intermediate]]),
    op("mlp.fc2", "vision feed-forward output projection", "linear",
      [intermediate, [-1, -1, d.intermediate]], [visual, visualShape], { bias: true }),
    op("residual_attn", "vision attention residual add", "residual_add",
      [visual, visualShape], [visual, visualShape], { checkpoint_module: false }),
    op("residual_mlp", "vision MLP residual add", "residual_add",
      [visual, visualShape], [visual, visualShape], { checkpoint_module: false }),
  ];
  const block = withShapeDims(moduleSpec(blockId,
    foldedLayerName(0, normalized.visionLayers - 1, "Vision Encoder Layer"),
    "vision-block-group", {
      class: hfNamedClass(normalized, "visionBlockClass", "VisionEncoderLayer"),
      modality: "vision", range: `0..${normalized.visionLayers - 1}`,
      dataflow_edges: [
        ["layers_input", "layer_norm1"],
        ["layer_norm1", "self_attn.q_proj"], ["layer_norm1", "self_attn.k_proj"],
        ["layer_norm1", "self_attn.v_proj"],
        ["self_attn.q_proj", "self_attn.q_reshape"], ["self_attn.k_proj", "self_attn.k_reshape"],
        ["self_attn.v_proj", "self_attn.v_reshape"],
        ["self_attn.q_reshape", "self_attn.rope"], ["self_attn.k_reshape", "self_attn.rope"],
        ["self_attn.v_reshape", "self_attn.sdpa"], ["self_attn.rope", "self_attn.sdpa"],
        ["self_attn.sdpa", "self_attn.context_merge"],
        ["self_attn.context_merge", "self_attn.out_proj"],
        ["self_attn.out_proj", "residual_attn"], ["layers_input", "residual_attn"],
        ["residual_attn", "layer_norm2"], ["layer_norm2", "mlp.fc1"],
        ["mlp.fc1", "mlp.activation"], ["mlp.activation", "mlp.fc2"],
        ["mlp.fc2", "residual_mlp"], ["residual_attn", "residual_mlp"],
      ],
      dataflow_edge_relations: [
        { from: "self_attn.q_reshape", to: "self_attn.rope", label: "Q" },
        { from: "self_attn.k_reshape", to: "self_attn.rope", label: "K" },
        { from: "self_attn.v_reshape", to: "self_attn.sdpa", label: "V (unrotated)" },
        { from: "layers_input", to: "residual_attn", label: "residual input" },
        { from: "residual_attn", to: "residual_mlp", label: "residual input" },
      ],
    }, children, normalized.visionLayers), visualShape, visualShape);
  const patch = operatorSpec(`${id}.embeddings.proj`, "vision Conv3D patch embedding", "linear", {
    ...shapeFlow(tensorShapes(normalized).visionInput, visual),
    modality: "vision", semantic_role: "patch_embedding", bias: false,
  }, { input: d.patchInput, output: d.visual });
  const preNorm = operatorSpec(`${id}.pre_layrnorm`, "vision tower LayerNorm", "rmsnorm", {
    ...shapeFlow(visual, visual), modality: "vision", affine_bias: true,
  }, { input: d.visual, output: d.visual });
  // This object produces the position angles; the per-layer rope leaf owns
  // the actual Q/K rotation. Billing both as "rope" would double the work.
  const rotary = operatorSpec(`${id}.rotary_emb`, "vision axial position angles", "identity", {
    ...shapeFlow(visual, visual), modality: "vision", checkpoint_module: false,
    position_encoding: "rope_axial_3d", semantic_role: "vision_position_provider",
    position_provider: true,
  }, { input: d.visual, output: d.visual });
  return withShapeDims(moduleSpec(id, "Vision Tower", "vision-encoder", {
    class: hfNamedClass(normalized, "visionModelClass", "VisionModel"),
    modality: "vision", num_hidden_layers: normalized.visionLayers,
    num_attention_heads: d.heads, hidden_size: d.hidden,
    dataflow_edges: [
      ["embeddings.proj", "pre_layrnorm"],
      ["pre_layrnorm", "layers.0"],
      ["rotary_emb", "layers.0"],
    ],
    dataflow_edge_relations: [
      { from: "rotary_emb", to: "layers.0",
        relation: "index-control", label: "axial position angles" },
    ],
    ...shapeFlow(tensorShapes(normalized).visionInput, visual),
  }, [patch, preNorm, rotary, block], normalized.visionLayers),
    [-1, -1, -1, -1, -1], d.visual);
}
