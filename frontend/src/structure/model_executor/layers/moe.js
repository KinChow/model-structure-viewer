import { moduleSpec, withShapeDims } from "./base.js";
import { moeOperatorSpecs } from "../ops/index.js";
import { shapeFlow, tensorShapes } from "../shapes.js";
import { tensorDims } from "../dims.js";

export function moeModule(id, normalized) {
  const shapes = tensorShapes(normalized);
  const dims = tensorDims(normalized);
  return withShapeDims(moduleSpec(
    id,
    "Routed MoE",
    "moe",
    {
      class: "RoutedMoE",
      hidden_size: normalized.hiddenSize,
      moe_intermediate_size: normalized.moeIntermediateSize,
      num_experts: normalized.experts,
      num_experts_per_tok: normalized.expertsPerToken,
      ...shapeFlow(shapes.hidden, shapes.hidden, {
        router_logits_shape: shapes.routerLogits,
        selected_experts_shape: shapes.topExperts,
      }),
    },
    moeOperatorSpecs(id, normalized),
  ), dims.hidden, dims.hidden);
}
