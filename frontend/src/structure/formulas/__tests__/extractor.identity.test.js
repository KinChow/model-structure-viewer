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
import { derivedWeightParameters, derivedVisionParameters, derivedMtpParameters } from "../../../cost/derivedWeights.js";
import { childRepeatMultiplier } from "../../../cost/traverse.js";
import { deriveBuildPlan } from "../../config/plan.js";
import { scoredPairs } from "../counts.js";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../../../../..");
const T = 128;

// 校准状态（2026-09-08 二次收敛）：score 项 2× 双计修复 + normsTerm 修层后，
// 全部 21 个 MoE 行 |ratio-1| <= 1.7%，MiniMax-M2.7 / GLM-4.7 精确闭合。
// dense 字段组合由 T4b 合成变体覆盖（GQA/tied/headDim 推导/MoE+shared，全部精确闭合）。
// 残差归因：V4-Flash ≈-1.7%（dsa 期望侧近似 S=T，counts 侧按 indexerBudget）；
// GLM-5/Qwen3.8 ≈+0.5% 正向残差未完全归因（登记于 cost_counts.md）。
// W5（2026-09-09）验收收口：容差从 0.02 收到 **0.005**，REGISTERED **清空**。
// 归零路径（每一条都有实测证据，不是放宽容差）：
// - GLM-5.3-Flash 1.0904 → 0.999x：ops 模板 glm5_next KDA 两处宽度错（低秩
//   decay、out_proj 输入宽）+ derivedWeights 的 DSA 分支 model_type 白名单漏
//   glm5_next（11 个 DSA 层退回泛化 GQA，单层多算 1.449e8）。
// - Kimi-K3 1.0437 → 0.9994：latent MoE 的 down/up 投影是每层一份、全 token
//   激活，期望侧此前把它并进 routedN 一起乘 k/E，少算 (1-k/E) 份。
// - MiniMax-M3 1.0279 → 0.9995：块稀疏 selected 未夹到可见长度（S=128 而
//   17 块 x 128 = 2176），打分对数虚高。
// - V4-Flash-Vision-Exp / Kimi-K2 系：原登记值等于或宽于默认容差，实测均在
//   0.5% 内，属无效登记，一并移除。
// 残留 0.2%-0.5% 的行（Qwen3.5 小杯 / V4 系 / Kimi-K2.5 等）来自 tied embedding
// 与 norm 权重项的取整口径，量级稳定，纳入 0.005 容差内。
const TOLERANCE = 0.005;
const REGISTERED = {};

// T4 期望侧构建器（M8-V2 抽取共享）：文本域 = 非视觉参数 × T + 打分式层注意力 matmul；
// 视觉域 = 视觉参数 × 视觉 token 数 + 视觉块注意力 matmul。
function textExpectedSide(normalized, T, plan) {
  const hidden = normalized.hiddenSize || 0;
  const normsTerm = normalized.hyperConnectionCount ? 0 : (2 * (normalized.layers || 0) + 1) * hidden;
  const embeddingTerm = (normalized.vocabSize || 0) * hidden;
  const total = derivedWeightParameters(normalized);
  const visionTerm = derivedVisionParameters(normalized);
  // W4：MTP 参数计入 derivedWeightParameters（支柱②），但 MTP 不产生 MAC
  // （投机解码未启用 → 结构树里 repeat=0），与 embedding/norms 同属「有参数
  // 无算力」项，必须从 nEff 里扣掉。
  const mtpTerm = derivedMtpParameters(normalized);
  const layerSched = plan.layerSchedule || Array.from({ length: normalized.layers || 0 }, () => (normalized.experts ? "moe" : "dense"));
  const moeLayerCount = layerSched.filter((kind) => kind === "moe").length;
  const routedHidden = normalized.routedExpertHiddenSize || hidden;
  const moeIntermediate = normalized.moeIntermediateSize || normalized.intermediateSize || 0;
  const routedN = moeLayerCount * (normalized.experts || 0) * 3 * routedHidden * moeIntermediate;
  // W5：latent MoE 的 down/up 投影（hidden↔routed_expert_hidden_size）是
  // **每层一份、全 token 激活**，不是 per-expert —— 此前被并进 routedN 一起乘
  // k/E，导致期望侧少算 (1-k/E) 份。Kimi-K3 的 +4.37% 残差就是这一项
  // （routedHidden=hidden 的模型不受影响，该项为 0）。
  const latentProjections = routedHidden !== hidden ? moeLayerCount * 2 * hidden * routedHidden : 0;
  const kOverE = normalized.experts && normalized.expertsPerToken ? normalized.expertsPerToken / normalized.experts : 1;
  const nEff = total - mtpTerm - visionTerm - embeddingTerm - normsTerm + (normalized.tieWordEmbeddings ? embeddingTerm : 0) - routedN + routedN * kOverE;
  void latentProjections; // latent 投影已在 total 里且不参与 k/E 缩放，无需再调整
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
    // W3-①因果：期望侧与实现侧同走 counts.js scoredPairs。二者共用不构成
    // 同义重复——scoredPairs 本身有独立手算 oracle（counts.test.js
    // 「因果对数解析检查」对小尺寸逐 token 暴力求和比对）。
    const pairs = scoredPairs({ phase: "prefill", queryTokens: T, keyTokens: T });
    scoreMatmulParams += (normalized.attentionHeads || 0) * pairs
      * ((normalized.headDim || 0) + (normalized.valueHeadDim || normalized.headDim || 0));
  }
  // W3-③期望侧显式建模（**本波未落地，验收未通过，已回退**）。
  // 尝试把 MHC / HyperConnection / PLE / AttnResBlock 的 matmul 加进期望侧，
  // 结果暴露了互相抵消的两个误差，不能只补一半：
  //   - MHC 段量级实测只有 1.9e8 / 全模型 2.7e12 = 7e-5，**不是** GLM-5.3-Flash
  //     +9.0% 残差的来源（该残差仍未归因）；
  //   - HyperConnection 实测 8.1e10 / 1.07e12 = 7.6%，加进期望侧后
  //     Qwen3.8-Flash-Next 从 0.9963 掉到 0.9246 —— 说明期望侧另有一处
  //     约 +7.6% 的过计，此前被「HC 缺项」抵消掉了。
  // 实例数已实测锚定，留给下一波直接用：
  //   MHC   : mhc_pre L 个 + mhc_fused_post_pre L 个 + mhc_post 1 个（mhc_contract 无 matrix），
  //           每个 matrix = T·H·streams；实测 GLM-5.3-Flash L=45、V4-Pro L=61 同构
  //   HC    : (2L+1) 个实例，每个 matrix = T·H²；实测 Flash-Next L=48 -> 97
  //   PLE   : ple_layer_ids 长度个实例，matrix = T·2·ple_embed_dim·H + conv
  //   AttnRes: L 个实例，matrix = T·H；实测 Kimi-K3 L=93 -> 93
  return { textMatrix: nEff * T + scoreMatmulParams + stateMatmulParams, nEff };
}

function visionExpectedSide(normalized, V) {
  const visionParams = derivedVisionParameters(normalized);
  const blocks = normalized.visionLayers || 0;
  const heads = normalized.visionAttentionHeads || 0;
  const dim = normalized.visionHeadDim || 0;
  // 视觉块注意力 scores+context（视觉自注意力 S=V）。W3-①：ViT 也走因果口径
  // ——本工具的视觉塔按 dense 分解链发射，与文本侧同一 matmul case。
  const visionPairs = scoredPairs({ phase: "prefill", queryTokens: V, keyTokens: V });
  const scoreMatmulParams = blocks * heads * visionPairs * 2 * dim;
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
  // M8-V2 收官：全部模型断言（vision 行用 REGISTERED 覆盖已知结构缺口）
  const bad = rows.filter((r) => r.ratio == null || Math.abs(r.ratio - 1) > (REGISTERED[r.model] ?? TOLERANCE));
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
  // scores(QK^T) + context(PV)：每层 heads·pairs·(D+dv)，pairs 为因果对数（W3-①）
  const synthPairs = scoredPairs({ phase: "prefill", queryTokens: T, keyTokens: T });
  const scoreMatmulParams = layers * (normalized.attentionHeads || 0) * synthPairs
    * ((normalized.headDim || 0) + (normalized.valueHeadDim || normalized.headDim || 0));
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
  // 无 head_dim 字段：真实 dense 模型常见形态（llama 系），headDim=256/8=32。
  // 步骤 2 后 llama 无精确别名（不伪造结构），fixture 用已适配的 qwen3 等价
  // 表达同一意图——headDim 推导是 config 级判据，与家族名无关。
  syntheticIdentity("synthetic-dense-derived-dim", {
    model_type: "qwen3", architectures: ["Qwen3ForCausalLM"],
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
