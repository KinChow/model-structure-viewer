import { moduleSpec } from "./base.js";
import { moeOperatorSpecs } from "../ops/index.js";

export function moeModule(id, normalized) {
  return moduleSpec(
    id,
    "Routed MoE",
    "moe",
    {
      class: "RoutedMoE",
      hidden_size: normalized.hiddenSize,
      num_experts: normalized.experts,
      num_experts_per_tok: normalized.expertsPerToken,
    },
    moeOperatorSpecs(id),
  );
}
