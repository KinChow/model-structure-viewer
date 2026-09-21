import { aggregateNodeCosts, computeNodeCosts } from "../cost/compute.js";
import { nodeCommunicationBytes } from "../cost/comm.js";
import { activationTensorBytes } from "../cost/memory.js";
import { nodeCostPerCard, validatePlan } from "../cost/parallel.js";
import { classifyRoofline } from "../cost/roofline.js";
import { normalizeConfig } from "../structure/config/normalize.js";
import { resolveFrameworkPlan } from "../cost/sharding.js";

/** 构建一组给定芯片和并行计划下的逐节点理论 roofline 结果。 */
export function buildNodeLens(structure, chip, {
  phase = "prefill",
  batch = 1,
  sequence = 2048,
  bytesPerElement = 2,
  plan = {},
  efficiency,
  dtype = "bf16",
  frameworkProfile = "neutral",
} = {}) {
  if (!structure?.graph || !structure.extra_config || !chip) {
    return { ok: false, errors: [{ code: "lens.missingInputs" }], nodes: {} };
  }
  const config = normalizeConfig(structure.extra_config);
  const checked = validatePlan(resolveFrameworkPlan(plan, frameworkProfile, config), config);
  if (!checked.ok) return { ok: false, errors: checked.errors, nodes: {} };

  // 切分是叶的事：对本行 own 权重 nodeCostPerCard 一次，再 aggregate。
  // 禁止把已切的 aggregate_weightBytes 再送进 nodeCostPerCard（会 /TP²）。
  const shapeOptions = { batch, sequence, phase, attentionHeads: config.attentionHeads };
  const ownPerCard = computeNodeCosts(structure.graph, config, { batch, sequence, phase, frameworkProfile }).map((row) => {
    const nodeShapeOptions = {
      ...shapeOptions,
      vision: row.node.attributes?.modality === "vision",
      visionTokens: config.visionTokens || 1,
    };
    const shapeActIn = activationTensorBytes(row.node.input_shape, nodeShapeOptions, bytesPerElement) * row.multiplier;
    const shapeActOut = activationTensorBytes(row.node.output_shape, nodeShapeOptions, bytesPerElement) * row.multiplier;
    const actInBytes = row.actions?.bytes?.actIn ?? shapeActIn;
    const actOutBytes = row.actions?.bytes?.actOut ?? shapeActOut;
    const perCard = nodeCostPerCard({
      macs: row.compute_macs,
      weightBytes: row.weightBytes,
      actions: row.actions,
      actInBytes,
      actOutBytes,
    }, row.node, checked.plan);
    return {
      ...row,
      compute_macs: perCard.macs,
      weightBytes: perCard.weightBytes,
      actions: perCard.actions,
      actInBytes: perCard.actInBytes,
      actOutBytes: perCard.actOutBytes,
    };
  });
  const rows = aggregateNodeCosts(ownPerCard);
  const tokens = phase === "decode" ? 1 : sequence;
  const forwardTokens = batch * tokens;
  const nodes = Object.fromEntries(rows.map((row) => {
    const perCardCost = {
      macs: row.aggregate_macs,
      weightBytes: row.aggregate_weightBytes,
      actions: row.aggregate_actions,
      actInBytes: row.actInBytes,
      actOutBytes: row.actOutBytes,
    };
    perCardCost.commBytes = nodeCommunicationBytes(
      row.node,
      config,
      checked.plan,
      { batch, tokens, bytesPerElement, frameworkProfile },
    ) * row.multiplier;
    if (perCardCost.actions) perCardCost.actions.commBytes = perCardCost.commBytes;
    const roofline = classifyRoofline(perCardCost, chip, { dtype, efficiency });
    return [row.path, {
      ...roofline,
      metrics: {
        macs: perCardCost.macs,
        macsPerToken: Number.isFinite(perCardCost.macs) && forwardTokens > 0 ? perCardCost.macs / forwardTokens : null,
        flops: Number.isFinite(perCardCost.macs) ? perCardCost.macs * 2 : null,
        flopsPerToken: Number.isFinite(perCardCost.macs) && forwardTokens > 0 ? (perCardCost.macs * 2) / forwardTokens : null,
        macsSource: row.macs_source,
        // M11-P0-4：W5-2 更名后 times.compute 键已不存在，此行长期渲染 "-"
        computeSeconds: roofline.times.matrix,
        memorySeconds: roofline.times.memory,
        communicationSeconds: roofline.times.comm,
        vramBytes: (row.aggregate_weightBytes || 0) + (row.actInBytes || 0) + (row.actOutBytes || 0),
        memoryBytes: (row.actInBytes || 0) + (row.actOutBytes || 0),
      },
    }];
  }));
  return { ok: true, errors: [], nodes, plan: checked.plan };
}
