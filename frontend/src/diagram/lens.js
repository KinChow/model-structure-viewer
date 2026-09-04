import { computeNodeCosts } from "../cost/compute.js";
import { nodeCommunicationBytes } from "../cost/comm.js";
import { activationTensorBytes } from "../cost/memory.js";
import { nodeCostPerCard, validatePlan } from "../cost/parallel.js";
import { classifyRoofline } from "../cost/roofline.js";
import { normalizeConfig } from "../structure/config/normalize.js";

/** 构建一组给定芯片和并行计划下的逐节点理论 roofline 结果。 */
export function buildNodeLens(structure, chip, {
  phase = "prefill",
  batch = 1,
  sequence = 2048,
  bytesPerElement = 2,
  plan = {},
  efficiency,
  dtype = "bf16",
} = {}) {
  if (!structure?.root || !structure.extra_config || !chip) {
    return { ok: false, errors: ["缺少结构、模型配置或芯片规格"], nodes: {} };
  }
  const config = normalizeConfig(structure.extra_config);
  const checked = validatePlan(plan, config);
  if (!checked.ok) return { ok: false, errors: checked.errors, nodes: {} };

  const rows = computeNodeCosts(structure.root, config, { batch, sequence, phase });
  const tokens = phase === "decode" ? 1 : sequence;
  const shapeOptions = { batch, sequence, phase, attentionHeads: config.attentionHeads };
  const nodes = Object.fromEntries(rows.map((row) => {
    const perCardCost = nodeCostPerCard({
      macs: row.macs,
      weightBytes: row.weightBytes,
      actInBytes: activationTensorBytes(row.node.input_shape, shapeOptions, bytesPerElement) * row.multiplier,
      actOutBytes: activationTensorBytes(row.node.output_shape, shapeOptions, bytesPerElement) * row.multiplier,
    }, row.node, checked.plan);
    perCardCost.commBytes = nodeCommunicationBytes(
      row.node,
      config,
      checked.plan,
      { batch, tokens, bytesPerElement },
    ) * row.multiplier;
    return [row.path, classifyRoofline(perCardCost, chip, { dtype, efficiency })];
  }));
  return { ok: true, errors: [], nodes, plan: checked.plan };
}
