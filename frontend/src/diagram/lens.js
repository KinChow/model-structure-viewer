import { aggregateNodeCosts, computeNodeCosts } from "../cost/compute.js";
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
  if (!structure?.graph || !structure.extra_config || !chip) {
    return { ok: false, errors: ["缺少结构、模型配置或芯片规格"], nodes: {} };
  }
  const config = normalizeConfig(structure.extra_config);
  const checked = validatePlan(plan, config);
  if (!checked.ok) return { ok: false, errors: checked.errors, nodes: {} };

  const rows = aggregateNodeCosts(computeNodeCosts(null, config, { batch, sequence, phase, graph: structure.graph }));
  const tokens = phase === "decode" ? 1 : sequence;
  const forwardTokens = batch * tokens;
  const shapeOptions = { batch, sequence, phase, attentionHeads: config.attentionHeads };
  const nodes = Object.fromEntries(rows.map((row) => {
    const nodeShapeOptions = {
      ...shapeOptions,
      vision: row.node.attributes?.modality === "vision",
      visionTokens: config.visionTokens || 1,
    };
    const perCardCost = nodeCostPerCard({
      macs: row.aggregate_macs,
      weightBytes: row.aggregate_weightBytes,
      actInBytes: activationTensorBytes(row.node.input_shape, nodeShapeOptions, bytesPerElement) * row.multiplier,
      actOutBytes: activationTensorBytes(row.node.output_shape, nodeShapeOptions, bytesPerElement) * row.multiplier,
    }, row.node, checked.plan);
    perCardCost.commBytes = nodeCommunicationBytes(
      row.node,
      config,
      checked.plan,
      { batch, tokens, bytesPerElement },
    ) * row.multiplier;
    const roofline = classifyRoofline(perCardCost, chip, { dtype, efficiency });
    return [row.path, {
      ...roofline,
      metrics: {
        macs: perCardCost.macs,
        macsPerToken: Number.isFinite(perCardCost.macs) && forwardTokens > 0 ? perCardCost.macs / forwardTokens : null,
        flops: Number.isFinite(perCardCost.macs) ? perCardCost.macs * 2 : null,
        flopsPerToken: Number.isFinite(perCardCost.macs) && forwardTokens > 0 ? (perCardCost.macs * 2) / forwardTokens : null,
        macsSource: row.macs_source,
        computeSeconds: roofline.times.compute,
        memorySeconds: roofline.times.memory,
        communicationSeconds: roofline.times.comm,
        vramBytes: (perCardCost.weightBytes || 0) + (perCardCost.actInBytes || 0) + (perCardCost.actOutBytes || 0),
        memoryBytes: (perCardCost.actInBytes || 0) + (perCardCost.actOutBytes || 0),
      },
    }];
  }));
  return { ok: true, errors: [], nodes, plan: checked.plan };
}
