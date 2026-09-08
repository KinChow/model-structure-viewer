// M11-P0-5 前置：内存侧基线库。P0-5 把 counts.bytes 接入访存侧将改变全部
// 模型的访存数值与部分 bound——恒等式/结构基线全部锚在 matrix 侧，测不到
// 这些变化，本基线是该变更的唯一护栏（照抄 ops-spec-tree.golden 的
// regenerate + 人工审阅 diff 流程）。
// 数值经 12 位有效数字归一，消除浮点尾差；语义与 CostSummary.jsx 聚合
// 调用形状一致（prefill、batch=1、sequence=512、首张公开卡）。
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { normalizeConfig } from "../../structure/config/normalize.js";
import { resolveArchitecture } from "../../structure/registry/resolveArchitecture.js";
import { buildNetwork } from "../../structure/model_executor/models/index.js";
import { createStructureIr } from "../../structure/ir/createStructureIr.js";
import { materializeModelStructure } from "../../structure/materializers/toStructureNode.js";
import { deriveBuildPlan } from "../../structure/model_executor/plan.js";
import { aggregateCost } from "../aggregate.js";
import { planCommunicationBytes } from "../comm.js";
import { classifyRoofline } from "../roofline.js";
import { PUBLIC_CHIPS } from "../chips/public.js";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../../../..");

const sig = (value) => (value == null ? null : Number(value.toPrecision(12)));

export function computeModelChain(rawConfig, modelId) {
  const normalized = normalizeConfig(rawConfig);
  const resolved = resolveArchitecture(normalized, { modelId });
  const structure = materializeModelStructure(createStructureIr({
    network: buildNetwork(resolved, normalized),
    normalized,
    resolved,
  }));
  const cost = aggregateCost({ graph: structure.graph, config: normalized, phase: "prefill", batch: 1, sequence: 512 });
  const plan = deriveBuildPlan(normalized.raw ?? normalized);
  const communication = planCommunicationBytes({ graph: structure.graph, config: normalized, plan, batch: 1, tokens: 512 });
  // M11-P0-4/P0-5：与 CostSummary 同步——actions 通道 + counts.bytes 流量
  // （weights 仍以 memory 侧为权威保 what-if 语义）
  const roofline = classifyRoofline({
    actions: {
      ...cost.actions,
      bytes: {
        weights: cost.memory.weightBytes,
        actIn: cost.actions.actIn,
        actOut: cost.actions.actOut,
      },
      commBytes: communication?.totalBytes || 0,
    },
  }, PUBLIC_CHIPS[0], { dtype: "bf16", efficiency: {} });
  return { cost, communication, roofline };
}

export function buildMemoryActionsMap() {
  const catalog = JSON.parse(fs.readFileSync(path.join(repoRoot, "models/catalog.json"), "utf8"));
  const map = {};
  for (const entry of catalog.models) {
    const rawConfig = JSON.parse(fs.readFileSync(path.join(repoRoot, "models", entry.config_path), "utf8"));
    const { cost, communication, roofline } = computeModelChain(rawConfig, entry.model_id);
    map[entry.model_id] = {
      actions: cost.actions && {
        matrix: sig(cost.actions.matrix),
        vector: sig(cost.actions.vector),
        sfu: sig(cost.actions.sfu),
        weights: sig(cost.actions.weights),
        actIn: sig(cost.actions.actIn),
        actOut: sig(cost.actions.actOut),
      },
      memory: {
        weightBytes: sig(cost.memory.weightBytes),
        activationBytes: sig(cost.memory.activationBytes),
        kvBytes: sig(cost.memory.kvBytes),
        stateBytes: sig(cost.memory.stateBytes),
      },
      commBytes: sig(communication?.totalBytes ?? 0),
      roofline: {
        bound: roofline.bound,
        bytesMoved: sig(roofline.bytesMoved),
        times: {
          matrix: sig(roofline.times.matrix),
          vector: sig(roofline.times.vector),
          sfu: sig(roofline.times.sfu),
          memory: sig(roofline.times.memory),
          comm: sig(roofline.times.comm),
        },
      },
    };
  }
  return map;
}
