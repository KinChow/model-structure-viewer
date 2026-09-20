// cost/operator_cost.mjs —— dump 逐算子动作向量（前端 cost lens 真值）。回填 docs/details/evidence/cost/operator_cost.md。
// 复用生产链路：buildStructureFromConfig → computeNodeCosts(graph, normalized, {batch,sequence,phase})
//   → aggregateNodeCosts（与 diagram/lens.js、cost/aggregate.js 同调用口径）。
// 产物 frontend_ops.json 供 operator_reconcile.py 逐通道对 FlopCounterMode / ncu 真值。
// 用法：MSV_PROBE_MODEL=<Qwen3-0.6B 目录> node scripts/evidence/cost/operator_cost.mjs [--config <cfg>] [--out <dir>]
import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { buildStructureFromConfig } from "../../../frontend/src/structure/buildStructure.js";
import { normalizeConfig } from "../../../frontend/src/structure/config/normalize.js";
import { computeNodeCosts, aggregateNodeCosts } from "../../../frontend/src/cost/compute.js";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../../..");
const argVal = (flag, def) => {
  const i = process.argv.indexOf(flag);
  return i !== -1 ? process.argv[i + 1] : def;
};
const defaultConfig = process.env.MSV_PROBE_MODEL
  ? path.join(process.env.MSV_PROBE_MODEL, "config.json")
  : "<path-to>/Qwen3-0.6B/config.json";
const configPath = argVal("--config", defaultConfig);
const outDir = path.resolve(argVal("--out", process.env.MSV_EVIDENCE_OUT || path.join(repoRoot, "_evidence_out/cost")));
const seqPrefill = Number(argVal("--seq-prefill", "512"));
const seqDecode = Number(argVal("--seq-decode", "576"));
await fs.mkdir(outDir, { recursive: true });

const rawConfig = JSON.parse(await fs.readFile(configPath, "utf8"));
const normalized = normalizeConfig(rawConfig);
const structure = buildStructureFromConfig(rawConfig, { modelId: "Qwen/Qwen3-0.6B", source: "nv3-operator-cost" });
const graph = structure.graph;

function dumpPhase(phase, sequence) {
  const rows = computeNodeCosts(graph, normalized, { batch: 1, sequence, phase });
  const agg = aggregateNodeCosts(rows);
  const operators = rows
    .filter((r) => r.actions)
    .map((r) => ({
      path: r.path,
      id: r.node?.id ?? null,
      type: r.node?.type ?? null,
      operator_id: r.node?.attributes?.operator_id ?? null,
      multiplier: r.multiplier,
      compute_macs: r.compute_macs,
      actions: r.actions, // {matrix, vector, sfu, bytes:{weights,actIn,actOut,kvRead,indexRead}} —— 已乘 multiplier
    }));
  const root = agg.find((r) => r.path === "root");
  return {
    phase,
    sequence,
    tokens: phase === "decode" ? 1 : sequence,
    operator_count: operators.length,
    aggregate_actions: root?.aggregate_actions ?? null,
    aggregate_macs: root?.aggregate_macs ?? null,
    operators,
  };
}

const result = {
  model: "Qwen/Qwen3-0.6B",
  config_summary: {
    hiddenSize: normalized.hiddenSize,
    layers: normalized.layers,
    attentionHeads: normalized.attentionHeads,
    kvHeads: normalized.kvHeads,
    headDim: normalized.headDim,
    intermediateSize: normalized.intermediateSize,
    vocabSize: normalized.vocabSize,
  },
  caliber: "MSV compulsory traffic (read-once+write-once, no tiling re-read); matrix=MACs (FLOPs=2x)",
  prefill: dumpPhase("prefill", seqPrefill),
  decode: dumpPhase("decode", seqDecode),
};
await fs.writeFile(path.join(outDir, "frontend_ops.json"), JSON.stringify(result, null, 2));
console.log(`wrote ${path.join(outDir, "frontend_ops.json")}`);
console.log(`prefill ops=${result.prefill.operator_count} agg_macs=${result.prefill.aggregate_macs}`);
console.log(`decode  ops=${result.decode.operator_count} agg_macs=${result.decode.aggregate_macs}`);
