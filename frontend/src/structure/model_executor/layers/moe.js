import { moduleSpec } from "./base.js";
import { moeOperatorSpecs } from "../ops/index.js";
import { shapeFlow, tensorShapes } from "../shapes.js";

export function moeModule(id, normalized) {
  const shapes = tensorShapes(normalized);
  return moduleSpec(
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
  );
}
