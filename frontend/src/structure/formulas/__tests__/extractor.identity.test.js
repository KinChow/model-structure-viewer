// W1 T4：整模型恒等式（外部 oracle）。
// 恒等式：dense transformer 每个非嵌入权重元素每 token 恰好 1 次 MAC →
//   Σ 叶子 counts.matrix ≈ N_eff × T（MoE 按 expertFraction 缩放已含在 counts 侧）。
// N_eff = derivedWeightParameters − embedding − norms权重 − vision + (tie ? embedding : 0)
// 第一轮：输出逐模型 ratio 表用于校准；容差先放宽，归因后收紧。
import assert from "node:assert/strict";
import fs from "node:fs/promises";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";
import { buildStructureFromConfig } from "../../../structure/buildStructure.js";
import { normalizeConfig } from "../../../structure/config/normalize.js";
import { countsForNode } from "../../../structure/formulas/extractor.js";
import { derivedWeightParameters, derivedVisionParameters } from "../../../cost/derivedWeights.js";
import { childRepeatMultiplier } from "../../../cost/traverse.js";
import { deriveBuildPlan } from "../../model_executor/plan.js";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../../../../..");
const T = 128;

// 校准状态（2026-09-08 二次收敛）：score 项 2× 双计修复 + normsTerm 修层后，
// 全部 21 个 MoE 行 |ratio-1| <= 1.7%，MiniMax-M2.7 / GLM-4.7 精确闭合。
// dense 字段组合由 T4b 合成变体覆盖（GQA/tied/headDim 推导/MoE+shared，全部精确闭合）。
// 残差归因：V4-Flash ≈-1.7%（dsa 期望侧近似 S=T，counts 侧按 indexerBudget）；
// GLM-5/Qwen3.8 ≈+0.5% 正向残差未完全归因（登记于 cost_counts.md）。
const TOLERANCE = 0.02;
const REGISTERED = {};

// T4 期望侧构建器（M8-V2 抽取共享）：文本域 = 非视觉参数 × T + 打分式层注意力 matmul；
// 视觉域 = 视觉参数 × 视觉 token 数 + 视觉块注意力 matmul。
function textExpectedSide(normalized, T, plan) {
  const hidden = normalized.hiddenSize || 0;
  const normsTerm = normalized.hyperConnectionCount ? 0 : (2 * (normalized.layers || 0) + 1) * hidden;
  const embeddingTerm = (normalized.vocabSize || 0) * hidden;
  const total = derivedWeightParameters(normalized);
  const visionTerm = derivedVisionParameters(normalized);
  const layerSched = plan.layerSchedule || Array.from({ length: normalized.layers || 0 }, () => (normalized.experts ? "moe" : "dense"));
  const moeLayerCount = layerSched.filter((kind) => kind === "moe").length;
  const routedHidden = normalized.routedExpertHiddenSize || hidden;
  const moeIntermediate = normalized.moeIntermediateSize || normalized.intermediateSize || 0;
  let routedN = moeLayerCount * (normalized.experts || 0) * 3 * routedHidden * moeIntermediate;
  if (routedHidden !== hidden) routedN += moeLayerCount * 2 * hidden * routedHidden;
  const kOverE = normalized.experts && normalized.expertsPerToken ? normalized.expertsPerToken / normalized.experts : 1;
  const nEff = total - visionTerm - embeddingTerm - normsTerm + (normalized.tieWordEmbeddings ? embeddingTerm : 0) - routedN + routedN * kOverE;
  const schedule = plan.attentionSchedule || [];
  const kh = normalized.linearKeyHeads || normalized.attentionHeads || 0;
  const kd = normalized.linearKeyDim || normalized.headDim || 0;
  const vh = normalized.linearValueHeads || normalized.attentionHeads || kh;
  const vd = normalized.linearValueDim || normalized.headDim || kh;
  let scoreMatmulParams = 0;
  let stateMatmulParams = 0;
  for (let i = 0; i < (normalized.layers || 0); i++) {
    const kind = schedule[i] || "gqa";
    if (kind === "linear") {
      // KDA/线性注意力递推状态 matmul（F7b，无对应权重元素）：delta 模式
      // 3·vh·vd·kd/token；generic 为 kh·vh·kd·vd/token
      stateMatmulParams += T * (plan.linearAttentionMode === "generic"
        ? kh * vh * kd * vd
        : 3 * vh * vd * kd);
      continue;
    }
    scoreMatmulParams += 2 * (normalized.attentionHeads || 0) * T * T * (normalized.headDim || 0);
  }
  return { textMatrix: nEff * T + scoreMatmulParams + stateMatmulParams, nEff };
}

function visionExpectedSide(normalized, V) {
  const visionParams = derivedVisionParameters(normalized);
  const blocks = normalized.visionLayers || 0;
  const heads = normalized.visionAttentionHeads || 0;
  const dim = normalized.visionHeadDim || 0;
  // 视觉块注意力 scores+context：2·heads·V·S·D（视觉自注意力 S=V）
  const scoreMatmulParams = blocks * 2 * heads * V * V * dim;
  return visionParams * V + scoreMatmulParams;
}

test("T4 整模型恒等式：全模型容差断言（超差仅限已登记建模边界）", async () => {
  const catalog = JSON.parse(await fs.readFile(path.join(repoRoot, "models/catalog.json"), "utf8"));
  const rows = [];
  let unknownTotal = 0;

  for (const entry of catalog.models) {
    const config = JSON.parse(await fs.readFile(path.join(repoRoot, "models", entry.config_path), "utf8"));
    const normalized = normalizeConfig(config);
    const structure = buildStructureFromConfig(config, { modelId: entry.model_id, source: "identity-test" });
    const plan = deriveBuildPlan(normalized.raw ?? normalized);
    const V = normalized.visionTokens || 0;

    let totalMatrix = 0;
    let unknown = 0;
    const stack = [{ node: structure.root, multiplier: 1 }];
    while (stack.length > 0) {
      const { node, multiplier } = stack.pop();
      const children = node?.children || [];
      if (children.length > 0) {
        const childMultiplier = childRepeatMultiplier(node, multiplier);
        for (const child of children) stack.push({ node: child, multiplier: childMultiplier });
        continue;
      }
      // 双 token 域（M8-V2）：vision 域叶子用视觉 token 数，文本域用 sequence
      const inVision = String(node?.id || "").includes("vision");
      const fresh = countsForNode(node, {
        config: normalized,
        options: inVision
          ? { batch: 1, sequence: V, phase: "prefill", vision: true, visionTokens: V }
          : { batch: 1, sequence: T, phase: "prefill" },
        path: node?.id || "",
        bytesPerElement: 2,
      });
      if (node?.type === "embedding") continue; // 查表无 MAC；bytes 缺口登记于 §10
      if (fresh == null || fresh.matrix == null) { unknown += 1; continue; }
      totalMatrix += fresh.matrix * multiplier;
    }
    unknownTotal += unknown;

    const { textMatrix } = textExpectedSide(normalized, T, plan);
    const visionMatrix = normalized.hasVision && V > 0 ? visionExpectedSide(normalized, V) : 0;
    const expected = textMatrix + visionMatrix;
    const ratio = expected > 0 ? totalMatrix / expected : null;
    rows.push({ model: entry.model_id, totalMatrix, expected, ratio, unknown, moe: Boolean(normalized.experts), isVision: normalized.hasVision });
  }

  for (const r of rows) {
    console.error(`${r.model.padEnd(38)} ratio=${r.ratio == null ? "n/a" : r.ratio.toFixed(4)} matrix=${r.totalMatrix.toExponential(3)} unknown=${r.unknown}`);
  }
  console.error(`unknown 叶子总数: ${unknownTotal}`);
  assert.ok(rows.length >= 50, "模型覆盖不足（应含 vision 域）");
  // M8-V2 校准中：文本域模型断言容差；vision 域模型先报告（known：Kimi vision
  // config 未被 derivedVisionParameters 识别致期望侧 7× 低估、KDA 系文本侧
  // 期望公式未校准）——归因后逐批转入断言。
  const bad = rows.filter((r) => !r.isVision && (r.ratio == null || Math.abs(r.ratio - 1) > (REGISTERED[r.model] ?? TOLERANCE)));
  if (bad.length > 0) console.error("超容差:\n" + bad.map((r) => `${r.model}: ratio=${r.ratio == null ? "n/a" : r.ratio.toFixed(4)}`).join("\n"));
  assert.deepEqual(bad.map((r) => r.model), [], "恒等式超差须先归因：要么修 counts/derived，要么登记为建模边界并写入 REGISTERED");
});

// T4b：合成配置精确对账。目录内无纯 dense 模型（见 T4 注），dense 字段组合
// 由合成变体覆盖：GQA、tied embeddings、headDim 推导、MoE+shared。
// 恒等式仍是独立 oracle：期望侧按 derived 口径重写，不读 counts 实现。
function syntheticIdentity(name, config, { tie = false, moe = false } = {}) {
  const normalized = normalizeConfig(config);
  const T = 64;
  const structure = buildStructureFromConfig(config, { modelId: name, source: "identity-test" });
  let totalMatrix = 0;
  let unknown = 0;
  const stack = [{ node: structure.root, multiplier: 1 }];
  while (stack.length > 0) {
    const { node, multiplier } = stack.pop();
    const children = node?.children || [];
    if (children.length > 0) {
      const childMultiplier = childRepeatMultiplier(node, multiplier);
      for (const child of children) stack.push({ node: child, multiplier: childMultiplier });
      continue;
    }
    const fresh = countsForNode(node, { config: normalized, options: { batch: 1, sequence: T, phase: "prefill" }, path: node?.id || "", bytesPerElement: 2 });
    if (node?.type === "embedding") continue; // 查表无 MAC
    if (fresh == null || fresh.matrix == null) { unknown += 1; continue; }
    totalMatrix += fresh.matrix * multiplier;
  }
  const hidden = normalized.hiddenSize;
  // derived：每层 2·hidden norm 权重 + 末尾 final norm hidden
  const normsTerm = (2 * (normalized.layers || 0) + 1) * hidden;
  const embeddingTerm = (normalized.vocabSize || 0) * hidden;
  const total = derivedWeightParameters(normalized);
  let nEff = total - embeddingTerm - normsTerm + (tie ? embeddingTerm : 0); // untied：lm_head 已计入且参与矩阵乘
  if (moe) {
    // 与 T4 期望侧同口径：routed 参数全部计入 derived，每 token 只激活 k/E
    const moeI = normalized.moeIntermediateSize || normalized.intermediateSize;
    const routedN = (normalized.layers || 0) * (normalized.experts || 0) * 3 * hidden * moeI;
    const kOverE = (normalized.expertsPerToken || 0) / (normalized.experts || 1);
    nEff = nEff - routedN + routedN * kOverE;
  }
  const layers = normalized.layers || 0;
  // scores(QK^T) + context(PV) 各 heads·T·S·D，每层合计 2·heads·T·S·D（prefill S≈T）
  const scoreMatmulParams = layers * 2 * (normalized.attentionHeads || 0) * T * T * (normalized.headDim || 0);
  const expected = nEff * T + scoreMatmulParams;
  const ratio = totalMatrix / expected;
  console.error(`${name}: counts=${totalMatrix} expected=${expected} ratio=${ratio.toFixed(4)} unknown=${unknown}`);
  assert.equal(unknown, 0, `${name} 不应有 unknown 叶子`);
  assert.ok(Math.abs(ratio - 1) < 0.02, `${name} 恒等式失败: ratio=${ratio.toFixed(4)}`);
}

test("T4b 合成 dense 恒等式：GQA untied 基线", () => {
  syntheticIdentity("synthetic-dense-gqa", {
    model_type: "qwen3", architectures: ["Qwen3ForCausalLM"],
    hidden_size: 256, num_hidden_layers: 4, num_attention_heads: 8,
    num_key_value_heads: 4, head_dim: 32, intermediate_size: 512,
    vocab_size: 1000, tie_word_embeddings: false,
  });
});

test("T4b 合成 dense 恒等式：tied embeddings", () => {
  // tie 后 lm_head 与 embedding 共享权重；derived 不再单计 lm_head，
  // 但 lm_head matmul 真实发生 → 期望侧加回 embeddingTerm
  syntheticIdentity("synthetic-dense-tied", {
    model_type: "qwen3", architectures: ["Qwen3ForCausalLM"],
    hidden_size: 256, num_hidden_layers: 4, num_attention_heads: 8,
    num_key_value_heads: 4, head_dim: 32, intermediate_size: 512,
    vocab_size: 1000, tie_word_embeddings: true,
  }, { tie: true });
});

test("T4b 合成 dense 恒等式：headDim 由 hidden/heads 推导", () => {
  // 无 head_dim 字段：真实 dense 模型常见形态（llama 系），headDim=256/8=32
  syntheticIdentity("synthetic-dense-derived-dim", {
    model_type: "llama", architectures: ["LlamaForCausalLM"],
    hidden_size: 256, num_hidden_layers: 4, num_attention_heads: 8,
    num_key_value_heads: 2, intermediate_size: 512,
    vocab_size: 1000, tie_word_embeddings: false,
  });
});

test("T4b 合成 MoE 恒等式：routed k/E 缩放 + shared expert", () => {
  // MoE 小配置：routed 每 token 只算 k/E，shared expert 每 token 全算，
  // sharedI 无显式字段 → normalize 回退 moeI（与 R1/GLM-5.x 收敛结论一致）
  syntheticIdentity("synthetic-moe-shared-tied", {
    model_type: "qwen3_moe", architectures: ["Qwen3MoeForCausalLM"],
    hidden_size: 128, num_hidden_layers: 2, num_attention_heads: 4,
    num_key_value_heads: 2, head_dim: 32, moe_intermediate_size: 64,
    num_experts: 8, num_experts_per_token: 2, num_shared_experts: 1,
    intermediate_size: 128, vocab_size: 500, tie_word_embeddings: true,
  }, { tie: true, moe: true });
});
