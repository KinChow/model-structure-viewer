import { operatorSpec } from "../operators/ops/index.js";
import { visionTowerModule } from "./vision.js";
import { projectorModule } from "./projector.js";
import { embeddingModule } from "./embedding.js";
import { recipeVisionInternalMerger, recipeVisionFusion } from "../archs/index.js";
import { tensorDims } from "../config/dims.js";

// This is semantic dataflow, not a new checkpoint module or a runtime kernel.
// The workload sequence is already the LM's post-fusion length; a Kimi
// placeholder expansion must not add visual tokens to that length again.
export function multimodalEntry(normalized) {
  const dims = tensorDims(normalized);
  const vision = visionTowerModule(normalized);
  const projectorSpec = normalized.hasVisionProjector && !recipeVisionInternalMerger(normalized)
    ? projectorModule(normalized) : null;
  const projectors = projectorSpec ? (Array.isArray(projectorSpec) ? projectorSpec : [projectorSpec]) : [];
  const visualOutput = projectors.at(-1) || vision;
  const image = operatorSpec("image_input", "image / video input", "identity", {
    external_input: true, checkpoint_module: false, activation_materialization: "unknown",
  }, { input: [-1, -1, -1, -1, -1], output: [-1, -1, -1, -1, -1] });
  const text = operatorSpec("text_input", "text / placeholder token IDs", "identity", {
    external_input: true, checkpoint_module: false, data_dtype: "integer",
  }, { input: dims.tokenIds, output: dims.tokenIds });
  const embed = embeddingModule("embed_tokens", normalized);
  const fusion = operatorSpec("multimodal_fusion", "text / vision fusion", "multimodal_fusion", {
    semantic_role: "multimodal_fusion", checkpoint_module: false,
    fusion_semantics: recipeVisionFusion(normalized),
    sequence_policy: "workload_is_post_fusion",
    execution_phase: "image_prefill",
    traffic_status: "unknown",
    traffic_reason: "image occupancy, padding, mask and in-place/materialized implementation not supplied",
    activation_materialization: "unknown",
    input_shape: "[text embeddings, projected visual features, placeholder positions]",
    output_shape: `[batch, post-fusion sequence, hidden=${normalized.hiddenSize}]`,
  }, { input: dims.hidden, output: dims.hidden });
  const children = [image, vision, ...projectors, text, embed, fusion];
  const visualChain = [vision, ...projectors];
  const edges = [[image.id, vision.id],
    ...visualChain.slice(1).map((stage, i) => [visualChain[i].id, stage.id]),
    [text.id, embed.id], [embed.id, fusion.id], [visualOutput.id, fusion.id],
    [text.id, fusion.id]];
  const relations = [
    { from: embed.id, to: fusion.id, label: "text embeddings" },
    { from: visualOutput.id, to: fusion.id, label: "visual features" },
    { from: text.id, to: fusion.id, relation: "index-control", label: "placeholder positions" },
  ];
  return { children, edges, relations, branchIds: new Set(children.filter(child => child !== fusion).map(child => child.id)) };
}
