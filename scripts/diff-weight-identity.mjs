#!/usr/bin/env node
// diff-weight-identity.mjs —— 权重字节恒等式的**逐层归因**诊断器。
//
// 为什么要它：恒等式两侧是**两套独立实现**（期望侧 cost/derivedWeights.js 的闭式
// 参数量公式 vs 实际侧结构树逐叶 bytes.weights 求和）。这是它作为 oracle 的价值来源，
// 也意味着「差 0.3%」这种残差只能靠逐项对齐消掉。本脚本把差额直接摊到
// 「第几层 / 哪个算子」，替代之前靠代数反推 + 单层变体二分的手工流程。
//
// 用法：
//   node scripts/diff-weight-identity.mjs                      # 16 个结构类代表模型
//   node scripts/diff-weight-identity.mjs <modelId> [<modelId>...]
//   node scripts/diff-weight-identity.mjs --phase decode <modelId>
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { normalizeConfig } from "../frontend/src/structure/config/normalize.js";
import { resolveArchitecture } from "../frontend/src/structure/registry/resolveArchitecture.js";
import { buildNetwork } from "../frontend/src/structure/model_executor/models/index.js";
import { createStructureIr } from "../frontend/src/structure/ir/createStructureIr.js";
import { materializeModelStructure } from "../frontend/src/structure/materializers/modelStructure.js";
import { graphRoot } from "../frontend/src/structure/graph/selectors.js";
import { countsForNode } from "../frontend/src/structure/formulas/extractor.js";
import { childRepeatMultiplier } from "../frontend/src/cost/traverse.js";
import {
  derivedDecoderLayerBreakdown, derivedVisionParameters, derivedMtpParameters,
} from "../frontend/src/cost/derivedWeights.js";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const B = 2;
const DEFAULTS = [
  "Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-35B-A3B", "zai-org/GLM-5",
  "moonshotai/Kimi-K2-Instruct", "deepseek-ai/DeepSeek-V4-Pro", "moonshotai/Kimi-K2.5",
  "deepseek-ai/DeepSeek-V3.1", "zai-org/GLM-5.3-Flash", "MiniMaxAI/MiniMax-M3",
  "Qwen/Qwen3.8-2.4T-A95B", "Qwen/Qwen3.8-Flash-Next", "deepseek-ai/DeepSeek-V3.2",
  "deepseek-ai/DeepSeek-V4-Flash-Vision-Exp", "zai-org/GLM-4.7",
  "moonshotai/Kimi-K3", "MiniMaxAI/MiniMax-M2.7",
];

const argv = process.argv.slice(2);
const phaseIndex = argv.indexOf("--phase");
const phaseName = phaseIndex >= 0 ? argv[phaseIndex + 1] : "decode";
const models = argv.filter((a, i) => !a.startsWith("--") && i !== phaseIndex + 1);
const targets = models.length > 0 ? models : DEFAULTS;
const num = (n) => n.toLocaleString("en-US");

/** 结构树逐叶 bytes.weights → 按「层号」与「算子」两级聚合（单位：参数个数）。 */
function leafWeightBreakdown(normalized, structure, phase) {
  const perLayer = new Map(); // layerIndex|section -> Map(op -> params)
  // P7（步骤 7）：structure.root 消费退役——遍历起点换成 graphRoot 图视图。
  const stack = [{ node: graphRoot(structure.graph), multiplier: 1 }];
  while (stack.length > 0) {
    const { node, multiplier } = stack.pop();
    const kids = node?.children || [];
    if (kids.length > 0) {
      const m = childRepeatMultiplier(node, multiplier);
      for (const c of kids) stack.push({ node: c, multiplier: m });
      continue;
    }
    const id = String(node?.canonical_id ?? node?.id ?? "");
    const inVision = id.includes("vision") || id.includes("merger") || id.startsWith("projector");
    const options = inVision
      ? { batch: 1, sequence: normalized.visionTokens || 1, phase, vision: true, visionTokens: normalized.visionTokens || 1 }
      : { batch: 1, sequence: phase === "decode" ? 4096 : 128, phase };
    const a = countsForNode(node, { config: normalized, options, path: id, bytesPerElement: B });
    if (!a) continue;
    const params = ((a.bytes?.weights || 0) * multiplier) / B;
    if (params <= 0) continue;
    const layerMatch = id.match(/^(?:decoder|text_decoder)\.(\d+)\./);
    const section = inVision ? "vision"
      : id.startsWith("mtp") ? "mtp"
        : id.startsWith("lm_head") ? "lm_head"
          : layerMatch ? `layer.${layerMatch[1]}` : "other";
    if (!perLayer.has(section)) perLayer.set(section, new Map());
    const ops = perLayer.get(section);
    const op = String(node?.attributes?.operator_id || node?.type || "?");
    ops.set(op, (ops.get(op) || 0) + params);
  }
  return perLayer;
}

for (const modelId of targets) {
  const catalog = JSON.parse(fs.readFileSync(path.join(repoRoot, "models/catalog.json"), "utf8"));
  const entry = catalog.models.find((m) => m.model_id === modelId);
  if (!entry) { console.error(`跳过（不在 catalog）：${modelId}`); continue; }
  const raw = JSON.parse(fs.readFileSync(path.join(repoRoot, "models", entry.config_path), "utf8"));
  const normalized = normalizeConfig(raw);
  const resolved = resolveArchitecture(normalized, { modelId });
  const structure = materializeModelStructure(createStructureIr({
    network: buildNetwork(resolved, normalized), normalized, resolved,
  }));

  const leaf = leafWeightBreakdown(normalized, structure, phaseName);
  const { total: derivedDecoder, perLayer } = derivedDecoderLayerBreakdown(normalized);
  const derivedVision = derivedVisionParameters(normalized);
  const derivedMtp = derivedMtpParameters(normalized);

  const sum = (m) => [...m.values()].reduce((s, x) => s + x, 0);
  const leafVision = leaf.has("vision") ? sum(leaf.get("vision")) : 0;
  const leafHead = leaf.has("lm_head") ? sum(leaf.get("lm_head")) : 0;
  const leafMtp = leaf.has("mtp") ? sum(leaf.get("mtp")) : 0;
  const leafOther = leaf.has("other") ? sum(leaf.get("other")) : 0;
  let leafLayers = 0;
  const leafByLayer = new Map();
  for (const [section, ops] of leaf.entries()) {
    const m = section.match(/^layer\.(\d+)$/);
    if (!m) continue;
    leafByLayer.set(Number(m[1]), sum(ops));
    leafLayers += sum(ops);
  }

  console.log(`\n=== ${modelId} · ${phaseName} ===`);
  console.log(`  decoder 层： 叶=${num(leafLayers)}  期望=${num(derivedDecoder)}  差=${num(leafLayers - derivedDecoder)}`);
  console.log(`  vision：     叶=${num(leafVision)}  期望=${num(derivedVision)}  差=${num(leafVision - derivedVision)}`);
  console.log(`  mtp：        叶=${num(leafMtp)}  期望=${num(derivedMtp)}（期望侧在恒等式里被减掉，叶侧 repeat=0）`);
  console.log(`  lm_head=${num(leafHead)}  其它（final norm 等）=${num(leafOther)}`);

  // 逐层对照：结构树的层组会把连续同构层折叠成一个代表节点（multiplier>1），
  // 所以按「代表层 → 该组覆盖的期望层区间」比总量，而不是逐 index 硬配。
  // routed 专家两侧同口径缩放：decode 只读 min(k·T,E)/E 份。
  const tokens = phaseName === "decode" ? 1 : 128;
  const experts = normalized.experts || 0;
  const activeFraction = experts ? Math.min((normalized.expertsPerToken || 0) * tokens, experts) / experts : 1;
  const derivedByIndex = new Map(perLayer.map((r) => [r.index, r]));
  const derivedScaledTotal = perLayer.reduce((s, r) => s + (r.total - r.routed + r.routed * activeFraction), 0);
  console.log(`  decoder（routed 按 ${phaseName} 缩放 ${activeFraction.toFixed(6)}）：`
    + ` 叶=${num(leafLayers)} 期望=${num(Math.round(derivedScaledTotal))} 差=${num(Math.round(leafLayers - derivedScaledTotal))}`);
  const rows = [...leafByLayer.entries()].sort((a, b) => a[0] - b[0]);
  if (rows.length > 0) {
    console.log("  ---- 代表层对照（叶 = 该组全部实例之和；期望单层已按相位缩放 routed）----");
    for (const [index, leafParams] of rows) {
      const d = derivedByIndex.get(index);
      if (!d) { console.log(`   层 ${index}: 叶=${num(leafParams)} 期望侧无此层`); continue; }
      const scaled = d.total - d.routed + d.routed * activeFraction;
      const groupSize = scaled > 0 ? Math.max(1, Math.round(leafParams / scaled)) : 1;
      console.log(`   层 ${index} [${d.kind}/${d.attentionKind}] 组内≈${groupSize}: 叶单层=${num(Math.round(leafParams / groupSize))}`
        + ` 期望单层=${num(Math.round(scaled))} 差=${num(Math.round(leafParams / groupSize - scaled))}`
        + `  (attn=${num(d.attention)} norms=${num(d.norms)} mhc=${num(d.mhc)} hc=${num(d.hc)} routed=${num(d.routed)})`);
    }
  }
  for (const [section, ops] of [...leaf.entries()].sort()) {
    if (!/^layer\./.test(section)) continue;
    const detail = [...ops.entries()].sort((a, b) => b[1] - a[1]).map(([op, p]) => `${op}=${num(p)}`).join(" ");
    console.log(`   ${section} 算子明细: ${detail}`);
  }
}
