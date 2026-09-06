import { moduleSpec, withShapeDims } from "./base.js";
import { deepseekV4MoeOperatorSpecs, kimiK3MoeOperatorSpecs, moeOperatorSpecs, operatorSpec } from "../ops/index.js";
import { shapeFlow, tensorShapes } from "../shapes.js";
import { tensorDims } from "../dims.js";
import { mlpModule } from "./mlp.js";
import { sharedExpertGateModule } from "./hybrid.js";

export function moeModule(id, normalized, { layerIndex = 0 } = {}) {
  const shapes = tensorShapes(normalized);
  const dims = tensorDims(normalized);
  const isKimiK3 = normalized.modelType === "kimi_k3";
  const isDeepseekV4 = normalized.modelType === "deepseek_v4";
  const isHashMoe = isDeepseekV4 && layerIndex < (normalized.numHashLayers || 0);
  const operatorSpecs = isKimiK3
    ? kimiK3MoeOperatorSpecs(id, normalized)
    : isDeepseekV4
      ? deepseekV4MoeOperatorSpecs(id, normalized, isHashMoe)
      : moeOperatorSpecs(id, normalized);
  const sharedExpert = normalized.sharedExperts
    ? mlpModule(`${id}.shared_experts`, {
      ...normalized,
      intermediateSize: normalized.sharedExpertIntermediateSize || normalized.moeIntermediateSize,
    })
    : null;
  if (sharedExpert && isKimiK3) sharedExpert.name = "Shared Expert MLP";
  const children = [
    ...operatorSpecs,
    ...(sharedExpert ? [sharedExpert] : []),
    ...(normalized.sharedExpertGate ? [sharedExpertGateModule(`${id}.shared_expert_gate`, normalized)] : []),
  ];
  if (normalized.sharedExperts && (isDeepseekV4 || (normalized.sharedExpertGate && !isKimiK3))) {
    children.push(operatorSpec(`${id}.shared_expert_add`, "shared expert branch add", "moe_add", {
      ...shapeFlow(`${shapes.hidden}, ${shapes.hidden}`, shapes.hidden),
      shared_experts: normalized.sharedExperts,
      implementation: ["vLLM.shared_experts fused or serial", "SGLang.shared_experts"],
    }, { input: dims.hidden, output: dims.hidden }));
  }
  return withShapeDims(moduleSpec(
    id,
    isDeepseekV4 ? (isHashMoe ? "DeepSeek V4 Hash Routed MoE" : "DeepSeek V4 Routed MoE") : isKimiK3 ? "Kimi K3 Latent Routed MoE" : "Routed MoE",
    "moe",
    {
      class: isKimiK3 ? "KimiK3LatentMoE" : "RoutedMoE",
      hidden_size: normalized.hiddenSize,
      moe_intermediate_size: normalized.moeIntermediateSize,
      num_experts: normalized.experts,
      num_experts_per_tok: normalized.expertsPerToken,
      routed_expert_hidden_size: normalized.routedExpertHiddenSize,
      shared_expert_intermediate_size: normalized.sharedExpertIntermediateSize,
      hash_moe: isHashMoe,
      hash_layer_index: isHashMoe ? layerIndex : undefined,
      implementation: isDeepseekV4 ? ["vLLM.DeepseekV4MoE", "SGLang.DeepSeekV4 MoE"] : undefined,
      ...shapeFlow(shapes.hidden, shapes.hidden, {
        router_logits_shape: shapes.routerLogits,
        selected_experts_shape: shapes.topExperts,
      }),
    },
    children,
  ), dims.hidden, dims.hidden);
}
