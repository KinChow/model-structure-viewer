import { moduleSpec, withShapeDims } from "./base.js";
import { operatorSpec } from "../ops/index.js";
import { shapeFlow, tensorShapes } from "../shapes.js";

function visionDimensions(normalized) {
  const hidden = normalized.visionHiddenSize || 0;
  const heads = normalized.visionAttentionHeads || 0;
  const headDim = normalized.visionHeadDim || (heads ? hidden / heads : 0);
  const intermediate = normalized.visionIntermediateSize || 0;
  const channels = normalized.visionChannels || 3;
  const patch = normalized.visionPatchSize || 0;
  const temporalPatch = normalized.visionTemporalPatchSize || 1;
  // The visual-token counts are workload assumptions, not matrix widths.
  // Keep both dimensions dynamic so linear MACs do not multiply them twice.
  const tokens = -1;
  const mergedTokens = -1;
  const mergeSize = normalized.visionMergeSize || 1;
  return {
    hidden, heads, headDim, intermediate, channels, patch, temporalPatch, tokens,
    visual: [-1, tokens, hidden],
    patchInput: [-1, tokens, channels, temporalPatch * patch * patch],
    qkv: [-1, tokens, 3 * heads * headDim],
    q: [-1, tokens, heads, headDim],
    scores: [-1, heads, tokens, tokens],
    context: [-1, tokens, heads, headDim],
    intermediateShape: [-1, tokens, intermediate],
    gatedMlp: Boolean(normalized.visionMlpGated),
    mergedVisual: [-1, mergedTokens, normalized.visionOutputSize || hidden],
    mergedWidth: mergeSize * mergeSize * hidden,
    mergeSize,
  };
}

function visionLayerModule(id, normalized) {
  const d = visionDimensions(normalized);
  const visual = `[batch, visual tokens, vision hidden size=${d.hidden}]`;
  const qkv = `[batch, visual tokens, fused qkv=${3 * d.heads * d.headDim}]`;
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
    operatorSpec(`${id}.scores`, "vision attention scores", "matmul", {
      ...shapeFlow(`${q}, ${q}`, scores), formula: "S = Q K^T / sqrt(d)", attention_kind: "vision", modality: "vision",
    }, { input: d.q, output: d.scores }),
    operatorSpec(`${id}.softmax`, "vision attention probabilities", "softmax", { ...shapeFlow(scores, scores), modality: "vision" }, { input: d.scores, output: d.scores }),
    operatorSpec(`${id}.context`, "vision weighted value", "matmul", {
      ...shapeFlow(`${scores}, ${q}`, context), formula: "O = P V", attention_kind: "vision", modality: "vision",
    }, { input: d.scores, output: d.context }),
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
  return withShapeDims(moduleSpec(id, "0 (VisionLayer)", "vision-block-group", {
    class: "VisionLayer",
    hidden_size: d.hidden,
    num_attention_heads: d.heads,
    intermediate_size: d.intermediate,
    modality: "vision",
    dataflow_edges: [
      ["input_norm", "qkv_proj"], ["qkv_proj", "qkv_split"], ["qkv_split", "scores"],
      ["scores", "softmax"], ["softmax", "context"], ["context", "out_proj"],
      ["out_proj", "post_norm"], ...mlpEdges,
    ],
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
  if (normalized.modelType === "glm5_next") {
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
  const edges = normalized.modelType === "glm5_next"
    ? [["patch_merge", "norm"], ["norm", "proj"], ["proj", "post_norm"], ["post_norm", "gate_proj"], ["post_norm", "up_proj"], ["gate_proj", "activation"], ["up_proj", "activation"], ["activation", "down_proj"]]
    : [["patch_merge", "norm"], ["norm", "fc1"], ["fc1", "activation"], ["activation", "fc2"]];
  return withShapeDims(moduleSpec(id, "Vision Merger", "vision-merger", {
    class: normalized.modelType === "glm5_next" ? "Glm5NextPatchMerger" : "Qwen3VisionPatchMerger",
    modality: "vision",
    merge_size: d.mergeSize,
    dataflow_edges: edges,
  }, children), d.visual, d.mergedVisual);
}

export function visionTowerModule(normalized) {
  const shapes = tensorShapes(normalized);
  const d = visionDimensions(normalized);
  const layers = normalized.visionLayers || 0;
  const layer = layers > 0 ? visionLayerModule("vision_tower.0", normalized) : null;
  if (layer) {
    layer.repeat = layers;
    layer.attributes.range = `0..${layers - 1}`;
  }
  const children = [
    operatorSpec("vision_tower.patch_embed", "vision patch embedding", "linear", {
      ...shapeFlow(shapes.visionInput, `[batch, visual tokens, vision hidden size=${d.hidden}]`),
      modality: "vision", semantic_role: "patch_embedding", patch_size: d.patch, temporal_patch_size: d.temporalPatch,
    }, { input: d.patchInput, output: d.visual }),
    operatorSpec("vision_tower.position", "vision position embedding", "vision_position", {
      ...shapeFlow(`[batch, visual tokens, vision hidden size=${d.hidden}]`, `[batch, visual tokens, vision hidden size=${d.hidden}]`),
      modality: "vision",
    }, { input: d.visual, output: d.visual }),
    ...(layer ? [layer] : []),
    ...(normalized.visionInternalMerger ? [visionMergerModule("vision_tower.merger", normalized)] : []),
  ];
  return withShapeDims(moduleSpec(
    "vision_tower",
    "Vision Tower",
    "vision-encoder",
    {
      class: "VisionTower",
      hidden_size: normalized.visionHiddenSize,
      output_hidden_size: normalized.visionOutputSize,
      num_hidden_layers: normalized.visionLayers,
      num_attention_heads: d.heads,
      intermediate_size: d.intermediate,
      vision_tokens: normalized.visionTokens,
      dataflow_edges: [
        ["patch_embed", "position"],
        ...(layer ? [["position", "0"]] : []),
        ...(normalized.visionInternalMerger ? [[layer ? "0" : "position", "merger"]] : []),
      ],
      ...shapeFlow(shapes.visionInput, shapes.visionOutput),
    },
    children,
    layers || undefined,
  ), [-1, -1, -1, -1, -1], d.mergedVisual);
}
