import assert from "node:assert/strict";
import fs from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { normalizeConfig } from "./config/normalize.js";
import { resolveArchitecture } from "./registry/resolveArchitecture.js";
import { buildNetwork } from "./model_executor/models/index.js";
import { createStructureIr } from "./ir/createStructureIr.js";
import { materializeModelStructure } from "./materializers/toStructureNode.js";
import { formulaForOperator } from "./formulas/index.js";
import { derivedWeightParameters } from "../cost/derivedWeights.js";
import { kvBytesPerToken, linearStateBytesPerSequence } from "../cost/memory.js";
import { aggregateCost } from "../cost/aggregate.js";
import { deriveBuildPlan } from "./config/plan.js";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../../..");

// P7（步骤 7）：legacy structure.root 遍历退役——断言改走 Graph IR 节点
// （顶层模块 = root_id 直接子节点；语义 id = canonical_id）。
function childrenOf(graph, parentId) {
  return graph.nodes
    .filter((node) => node.parent_id === parentId)
    .sort((left, right) => (left.order || 0) - (right.order || 0) || left.id.localeCompare(right.id));
}

test("all built-in models have modules, formulas, and finite cost inputs", () => {
  const catalog = JSON.parse(fs.readFileSync(path.join(repoRoot, "models/catalog.json"), "utf8"));
  assert.equal(catalog.models.length, 59);
  for (const entry of catalog.models) {
    const config = JSON.parse(fs.readFileSync(path.join(repoRoot, "models", entry.config_path), "utf8"));
    const normalized = normalizeConfig(config);
    const resolved = resolveArchitecture(normalized, { modelId: entry.model_id });
    const structure = materializeModelStructure(createStructureIr({
      network: buildNetwork(resolved, normalized),
      normalized,
      resolved,
    }));
    const canonicalIds = structure.graph.nodes.map((node) => node.canonical_id);
    assert.equal(new Set(canonicalIds).size, canonicalIds.length, `${entry.model_id}: duplicate canonical graph ids`);
    assert.ok(structure.graph.edges.every((edge) => edge.source_canonical_id && edge.target_canonical_id), `${entry.model_id}: graph edge missing canonical endpoints`);
    const topLevel = childrenOf(structure.graph, structure.graph.root_id);
    assert.ok(topLevel.length > 0, `${entry.model_id}: empty module graph`);
    if (normalized.hasVision) {
      const vision = topLevel.find((node) => node.type === "vision-encoder");
      assert.ok(vision, `${entry.model_id}: vision config is missing from the structure`);
      const visionChildren = childrenOf(structure.graph, vision.id);
      assert.ok(visionChildren.length >= 2, `${entry.model_id}: vision tower has no detail modules`);
      const visionLayer0 = visionChildren.find((node) => (node.canonical_id || "").endsWith(".0"));
      assert.ok(childrenOf(structure.graph, visionLayer0?.id).length >= 8, `${entry.model_id}: vision layer has no operator detail`);
      assert.ok(structure.graph.edges.some((edge) => edge.source_canonical_id === "vision_tower.patch_embed"), `${entry.model_id}: vision patch edge missing`);
    }
    for (const node of structure.graph.nodes) {
      if (node.type !== "operator") continue;
      assert.ok(node.attributes.formula, `${entry.model_id}: missing formula for ${node.attributes.operator_id}`);
      assert.ok(formulaForOperator(node.attributes.formula_id), `${entry.model_id}: unknown formula id ${node.attributes.formula_id}`);
      assert.ok(Array.isArray(node.input_shape), `${entry.model_id}: ${node.canonical_id} missing numeric input shape`);
      assert.ok(Array.isArray(node.output_shape), `${entry.model_id}: ${node.canonical_id} missing numeric output shape`);
    }
    const cost = aggregateCost({ graph: structure.graph, config: normalized, phase: "prefill", batch: 1, sequence: 128 });
    assert.equal(cost.computeComplete, true, `${entry.model_id}: ${cost.unknownComputePaths.join(", ")}`);
    assert.ok(Number.isFinite(derivedWeightParameters(normalized)), `${entry.model_id}: invalid derived weights`);
    assert.ok(Number.isFinite(kvBytesPerToken(normalized)), `${entry.model_id}: invalid KV cost`);
    assert.ok(Number.isFinite(linearStateBytesPerSequence(normalized)), `${entry.model_id}: invalid recurrent state cost`);
    if (entry.model_id === "moonshotai/Kimi-K3") {
      assert.equal(resolved.canonicalArchitecture, "hybrid-multimodal-moe-decoder");
      assert.deepEqual(deriveBuildPlan(normalized.raw ?? normalized).attentionSchedule.filter((kind) => kind === "linear").length, 69);
      assert.deepEqual(deriveBuildPlan(normalized.raw ?? normalized).attentionSchedule.filter((kind) => kind === "mla").length, 24);
      assert.equal(normalized.sharedExperts, 2);
      assert.equal(normalized.attnResBlockSize, 12);
      const nodes = structure.graph.nodes;
      assert.ok(nodes.some((node) => node.name === "Attention Residual"), "Kimi-K3: missing per-layer AttnRes module");
      assert.ok(nodes.some((node) => node.name === "Output Attention Residual"), "Kimi-K3: missing output AttnRes module");
      assert.ok(nodes.some((node) => node.name === "MLA output gate"), "Kimi-K3: missing MLA output gate");
      assert.ok(nodes.some((node) => node.canonical_id.endsWith(".moe.shared_experts")), "Kimi-K3: missing shared experts");
      // P3 fused shared expert：checkpoint 取证（models/moonshotai/Kimi-K3/k3-index.json
      // 每个 MoE 层只有 shared_experts.{gate,up,down}_proj.weight 各一个，共 92×3=276 个
      // 张量；modeling_kimi_linear.py:797-801 `intermediate_size =
      // moe_intermediate_size * num_shared_experts` 后实例化**单个** KimiMLP）——
      // 融合形态就是"一个更宽的 MLP"，不是 n 份专家，也不涉及 ep 亲和。
      const sharedGate = nodes.find((node) => node.canonical_id.endsWith(".moe.shared_experts.gate_proj"));
      assert.ok(sharedGate, "Kimi-K3: missing fused shared expert gate projection");
      assert.equal(normalized.sharedExpertIntermediateSize, normalized.moeIntermediateSize * normalized.sharedExperts);
      assert.deepEqual(
        sharedGate.attributes.weightMatrices,
        [{
          class: "tp",
          out: normalized.sharedExpertIntermediateSize,
          in: normalized.hiddenSize,
          count: 1,
          matrices: 1,
          shape: [normalized.sharedExpertIntermediateSize, normalized.hiddenSize],
          split: "output",
        }],
        "Kimi-K3: fused shared expert 声明应为单组 tp（模块宽 = moeI×n_shared，count=1，gate 沿 output 切）",
      );
      assert.equal(deriveBuildPlan(normalized.raw ?? normalized).sharedExpertsAreFused, true);
    }
  }
});
