import { moduleSpec, withShapeDims } from "./base.js";
import { deepseekV4MoeOperatorSpecs, kimiK3MoeOperatorSpecs, moeOperatorSpecs, operatorSpec } from "../ops/index.js";
import { shapeFlow, tensorShapes } from "../shapes.js";
import { tensorDims } from "../../config/dims.js";
import { mlpModule } from "./mlp.js";
import { sharedExpertGateModule } from "./hybrid.js";
import { hfNamedClass } from "../../archs/index.js";

export function moeModule(id, normalized, { layerIndex = 0 } = {}) {
  const shapes = tensorShapes(normalized);
  const dims = tensorDims(normalized);
  const isKimiK3 = normalized.modelType === "kimi_k3";
  const isDeepseekV4 = normalized.modelType === "deepseek_v4";
  const isMiniMaxM3 = normalized.modelType === "minimax_m3_vl";
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
    }, { roleScope: "shexp" })
    : null;
  if (sharedExpert && isKimiK3) sharedExpert.name = "Shared Expert MLP";
  const children = [
    ...operatorSpecs,
    ...(sharedExpert ? [sharedExpert] : []),
    ...(normalized.sharedExpertGate ? [sharedExpertGateModule(`${id}.shared_expert_gate`, normalized)] : []),
  ];
  if (normalized.sharedExperts && (isDeepseekV4 || isMiniMaxM3 || (!normalized.sharedExpertGate && !isKimiK3) || normalized.sharedExpertGate)) {
    children.push(operatorSpec(`${id}.shared_expert_add`, "shared expert branch add", "moe_add", {
      ...shapeFlow(`${shapes.hidden}, ${shapes.hidden}`, shapes.hidden),
      shared_experts: normalized.sharedExperts,
      implementation: ["vLLM.shared_experts fused or serial", "SGLang.shared_experts"],
    }, { input: dims.hidden, output: dims.hidden }));
  }
  const declaredEdges = [
    ["router", "topk"],
    ["topk", "dispatch"],
    ["hash_router", "dispatch"],
    ["routed_expert_down_proj", "dispatch"],
    ["dispatch", "expert_mlp"],
    ["expert_mlp", "combine"],
    ["combine", "routed_expert_norm"],
    ["routed_expert_norm", "routed_expert_up_proj"],
    ["shared_experts", "shared_expert_add"],
    ["shared_expert_gate", "shared_expert_add"],
  ];
  const childSuffixes = new Set(children.map((child) => String(child.id || "").split(".").at(-1)));
  // 路由分支汇入 shared_expert_add 的那条边：K3 的潜空间 MoE 在 combine 之后还有
  // routed_expert_norm → routed_expert_up_proj（3584 → 7168），必须从 up_proj 出边，
  // 否则 combine 的 3584 直接对上 add 的 7168，末维不连续（形状连续性检查抓出）。
  declaredEdges.push([childSuffixes.has("routed_expert_up_proj") ? "routed_expert_up_proj" : "combine", "shared_expert_add"]);
  const filteredEdges = declaredEdges.filter(([source, target]) => childSuffixes.has(source) && childSuffixes.has(target));
  return withShapeDims(moduleSpec(
    id,
    isDeepseekV4 ? (isHashMoe ? "DeepSeek V4 Hash Routed MoE" : "DeepSeek V4 Routed MoE") : isKimiK3 ? "Kimi K3 Latent Routed MoE" : "Routed MoE",
    "moe",
    {
      class: hfNamedClass(normalized, "moeStem", "MoE"),
      hidden_size: normalized.hiddenSize,
      moe_intermediate_size: normalized.moeIntermediateSize,
      num_experts: normalized.experts,
      num_experts_per_tok: normalized.expertsPerToken,
      routed_expert_hidden_size: normalized.routedExpertHiddenSize,
      shared_expert_intermediate_size: normalized.sharedExpertIntermediateSize,
      hash_moe: isHashMoe,
      hash_layer_index: isHashMoe ? layerIndex : undefined,
      implementation: isDeepseekV4 ? ["vLLM.DeepseekV4MoE", "SGLang.DeepSeekV4 MoE"] : undefined,
      dataflow_edges: filteredEdges,
      ...shapeFlow(shapes.hidden, shapes.hidden, {
        router_logits_shape: shapes.routerLogits,
        selected_experts_shape: shapes.topExperts,
      }),
    },
    children,
  ), dims.hidden, dims.hidden);
}
