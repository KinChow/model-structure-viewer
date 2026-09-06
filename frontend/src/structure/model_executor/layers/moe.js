import { moduleSpec, withShapeDims } from "./base.js";
import { kimiK3MoeOperatorSpecs, moeOperatorSpecs } from "../ops/index.js";
import { shapeFlow, tensorShapes } from "../shapes.js";
import { tensorDims } from "../dims.js";
import { mlpModule } from "./mlp.js";
import { sharedExpertGateModule } from "./hybrid.js";

export function moeModule(id, normalized) {
  const shapes = tensorShapes(normalized);
  const dims = tensorDims(normalized);
  const isKimiK3 = normalized.modelType === "kimi_k3";
  const operatorSpecs = isKimiK3 ? kimiK3MoeOperatorSpecs(id, normalized) : moeOperatorSpecs(id, normalized);
  const sharedExpert = normalized.sharedExperts
    ? mlpModule(`${id}.shared_experts`, {
      ...normalized,
      intermediateSize: normalized.sharedExpertIntermediateSize || normalized.moeIntermediateSize,
    })
    : null;
  if (sharedExpert && isKimiK3) sharedExpert.name = "Shared Expert MLP";
  return withShapeDims(moduleSpec(
    id,
    isKimiK3 ? "Kimi K3 Latent Routed MoE" : "Routed MoE",
    "moe",
    {
      class: isKimiK3 ? "KimiK3LatentMoE" : "RoutedMoE",
      hidden_size: normalized.hiddenSize,
      moe_intermediate_size: normalized.moeIntermediateSize,
      num_experts: normalized.experts,
      num_experts_per_tok: normalized.expertsPerToken,
      routed_expert_hidden_size: normalized.routedExpertHiddenSize,
      shared_expert_intermediate_size: normalized.sharedExpertIntermediateSize,
      ...shapeFlow(shapes.hidden, shapes.hidden, {
        router_logits_shape: shapes.routerLogits,
        selected_experts_shape: shapes.topExperts,
      }),
    },
    [
      ...operatorSpecs,
      ...(sharedExpert ? [sharedExpert] : []),
      ...(normalized.sharedExpertGate ? [sharedExpertGateModule(`${id}.shared_expert_gate`, normalized)] : []),
    ],
  ), dims.hidden, dims.hidden);
}
