// 模型级对账报表（W1，warn 模式）：权重字节恒等式 + bound 期望断言。
//
// 判据（W4 转 error）：
//   权重字节：Σ 叶子 bytes.weights === 该相位下应读一遍的权重字节
//             （MoE routed 按相位：prefill 全 E、decode min(k·T, E)/E）
//   bound：逐算子逐相位的瓶颈分类须符合期望（attention prefill=compute、
//          attention decode=memory、MoE decode=memory、逐元素类恒 memory）
// W1 只断言报表可跑通，差额以清单输出，作为 W3/W4 的待修项。
import assert from "node:assert/strict";
import fs from "node:fs";
import test from "node:test";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { normalizeConfig } from "../../config/normalize.js";
import { resolveArchitecture } from "../../registry/resolveArchitecture.js";
import { buildNetwork } from "../../model_executor/models/index.js";
import { createStructureIr } from "../../ir/createStructureIr.js";
import { materializeModelStructure } from "../../materializers/toStructureNode.js";
import { countsForNode } from "../extractor.js";
import { childRepeatMultiplier } from "../../../cost/traverse.js";
import { derivedWeightParameters, derivedVisionParameters, derivedMtpParameters, derivedDecoderLayerBreakdown } from "../../../cost/derivedWeights.js";
import { kvBytesPerToken, kvBytesPerTokenBreakdown } from "../../../cost/memory.js";
import { classifyRoofline } from "../../../cost/roofline.js";
import { deriveBuildPlan } from "../../config/plan.js";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../../../../..");
const B = 2;

const REPRESENTATIVES = [
  ["S01", "Qwen/Qwen3.5-0.8B"], ["S02", "Qwen/Qwen3.5-35B-A3B"], ["S03", "zai-org/GLM-5"],
  ["S04", "moonshotai/Kimi-K2-Instruct"], ["S05", "deepseek-ai/DeepSeek-V4-Pro"], ["S06", "moonshotai/Kimi-K2.5"],
  ["S07", "deepseek-ai/DeepSeek-V3.1"], ["S08", "zai-org/GLM-5.3-Flash"], ["S09", "MiniMaxAI/MiniMax-M3"],
  ["S10", "Qwen/Qwen3.8-2.4T-A95B"], ["S11", "Qwen/Qwen3.8-Flash-Next"], ["S12", "deepseek-ai/DeepSeek-V3.2"],
  ["S13", "deepseek-ai/DeepSeek-V4-Flash-Vision-Exp"], ["S14", "zai-org/GLM-4.7"],
  ["S15", "moonshotai/Kimi-K3"], ["S16", "MiniMaxAI/MiniMax-M2.7"],
];

const PHASES = [
  { name: "prefill", tokens: 128, sequence: 128 },
  { name: "decode", tokens: 1, sequence: 4096 },
];

// 逐算子逐相位的 bound 期望（plan §四）。未列出的算子不断言。
// **必须声明工作点**：bound 是 arithmetic intensity 与 ridge point 的比较结果，
// T 太小时投影类也会落在访存侧。W1 实测：T=128 的 prefill 下 linear/matmul 全部
// memory-bound（AI 未过 A100 的 ridge≈306），所以 bound 断言另用一组"真实
// prefill 工作点"（T=S=2048），且只保留判据无歧义的算子。
const BOUND_PHASES = [
  { name: "prefill", tokens: 2048, sequence: 2048 },
  { name: "decode", tokens: 1, sequence: 4096 },
];

// W5：bound 期望改成**模块级**。用户的论断「attention prefill 算力瓶颈、decode
// 访存瓶颈」说的是模块，不是叶 —— 叶级的低秩小投影（q_a 1536 宽、indexer q/k）
// 在任何相位都是访存侧，按叶断言只会得到一堆假阳性（实测 linear|prefill 在
// 11/16 个结构类"违反"，其实是判据太粗）。这里按 attention / mlp / moe 三类
// 模块的子树聚合后再分类。
const MODULE_BOUND_EXPECTATION = {
  attention: { prefill: "matrix", decode: "memory" },
  mlp: { prefill: "matrix", decode: "memory" },
  // MoE 两相位都是访存侧，这是算出来的不是猜的：prefill 下每个专家的权重要读
  // 一遍（3·E·EH·EI），而只做 T·k·3·EH·EI 次 MAC → arithmetic intensity
  // = 2·T·k/E。以 E=288/k=8/T=2048 计得 113.8 FLOP/B，A100 的 ridge 约 306，
  // 仍在访存侧；要过 ridge 需要 T·k/E ≳ 153（即 T ≳ 5500）。
  // 这正是「MoE 的 prefill 也常被专家权重带宽卡住」的量化表述。
  moe: { prefill: "memory", decode: "memory" },
};


const CHIP = {
  id: "identity-probe", memory_bytes: 80e9, memory_bandwidth: 2.039e12,
  peak_flops: { bf16: 312e12 }, vector_flops: 19.5e12, sfu_ops: 4.875e12,
};

function buildStructure(raw, modelId) {
  const normalized = normalizeConfig(raw);
  const resolved = resolveArchitecture(normalized, { modelId });
  return {
    normalized,
    structure: materializeModelStructure(createStructureIr({ network: buildNetwork(resolved, normalized), normalized, resolved })),
  };
}

// 权重字节恒等式的已归因残差（W3.5），记结论避免下一波重复排查：
// - Qwen3.5-0.8B 曾恒 1.4242：**已修**，100% 来自 tied lm_head 未加回期望侧
//   （lm_head 权重 = vocab·H = 0.509 GB，与 0.506 GB 缺口逐位对上）。
// - GLM-5.3-Flash decode 1.0958 已排除项：routed 专家权重（实测
//   Σ swiglu weights = 1.6911e10，与 3·min(k·T,E)·H·moeI·b x 42 个 MoE 层
//   逐位吻合）；MoE 层数（schedule 实测 42 moe + 3 dense，与
//   first_k_dense_replace=3 一致）。prefill 侧同模型仅 +0.57%。

/** 期望侧：该相位下应被读一遍的权重字节（不含 embedding 表，gather 不计权重读；
 *  也不含 tid2eid 等 buffer —— 它们是常驻数据，容量由 derivedBufferBytes 单独计）。 */
function expectedWeightBytes(normalized, phase, tokens, plan) {
  const hidden = normalized.hiddenSize || 0;
  const total = derivedWeightParameters(normalized);
  const vision = derivedVisionParameters(normalized);
  const embed = (normalized.vocabSize || 0) * hidden;
  const experts = normalized.experts || 0;
  const topk = normalized.expertsPerToken || 0;
  const moeI = normalized.moeIntermediateSize || normalized.intermediateSize || 0;
  const routedHidden = normalized.routedExpertHiddenSize || hidden;
  // routed 专家只存在于 MoE 层。此前按全部 layers 算，dense 前缀层
  // （first_k_dense_replace）也被计成 MoE → routedN 过大、dense 项被压小，
  // decode 侧 expected 偏小（S08 甚至为负）。改走 plan 的 layerSchedule。
  const schedule = plan?.layerSchedule
    || Array.from({ length: normalized.layers || 0 }, () => (experts ? "moe" : "dense"));
  const moeLayers = schedule.filter((kind) => kind === "moe").length;
  const routedN = experts ? moeLayers * experts * 3 * routedHidden * moeI : 0;
  // routed 专家：prefill 大 T 下全部被激活；decode 只触达 min(k·T, E) 份
  const activeExperts = experts ? Math.min(topk * tokens, experts) : 0;
  const routedActive = experts ? routedN * (activeExperts / experts) : 0;
  // tied embeddings：lm_head 与嵌入表共享同一张量，但 lm_head 的 GEMM 仍要从
  // HBM 读一遍权重 —— 叶子侧计了，期望侧必须加回来。与 identity 的 nEff
  // 「+ (tie ? embeddingTerm : 0)」同一处理。实测 Qwen3.5-0.8B 的 1.4242 偏差
  // 100% 来自这一项（lm_head 权重 = vocab·H = 0.509 GB / 缺口 0.506 GB）。
  const tiedHead = normalized.tieWordEmbeddings ? embed : 0;
  // W4：MTP 参数在 total 里（支柱②），但每次前向不读它的权重（repeat=0），
  // 叶子侧因此为 0 —— 期望侧同步扣掉。
  const mtp = derivedMtpParameters(normalized);
  const dense = total - mtp - vision - embed - routedN;
  // fp32 参数（paramDtypes 登记的 dt_bias/A_log、mHC base/scale）按 4B 计，
  // 其余按 B。fp32 元素数只数主干层（MTP 的期望侧本来就被整体减掉）。
  const fp32 = derivedDecoderLayerBreakdown(normalized).perLayer
    .reduce((sum, row) => sum + (row.fp32Elements || 0), 0);
  return (dense - fp32 + tiedHead + routedActive + vision) * B + fp32 * 4;
}

function walkLeaves(root, visit) {
  const stack = [{ node: root, multiplier: 1 }];
  while (stack.length > 0) {
    const { node, multiplier } = stack.pop();
    const children = node?.children || [];
    if (children.length > 0) {
      const childMultiplier = childRepeatMultiplier(node, multiplier);
      for (const child of children) stack.push({ node: child, multiplier: childMultiplier });
      continue;
    }
    visit(node, multiplier);
  }
}

const fmt = (n) => (Number.isFinite(n) ? n.toExponential(3) : String(n));

test("W1 报表：权重字节恒等式 + bound 期望", () => {
  const catalog = JSON.parse(fs.readFileSync(path.join(repoRoot, "models/catalog.json"), "utf8"));
  const byId = new Map(catalog.models.map((m) => [m.model_id, m]));
  const weightRows = [];
  const boundViolations = new Map();

  for (const [cls, modelId] of REPRESENTATIVES) {
    const entry = byId.get(modelId);
    assert.ok(entry, `代表模型缺失: ${modelId}`);
    const raw = JSON.parse(fs.readFileSync(path.join(repoRoot, "models", entry.config_path), "utf8"));
    const { normalized, structure } = buildStructure(raw, modelId);

    for (const ph of PHASES) {
      let weights = 0;
      walkLeaves(structure.root, (node, multiplier) => {
        const inVision = String(node?.id || "").includes("vision");
        const options = inVision
          ? { batch: 1, sequence: normalized.visionTokens || 1, phase: ph.name, vision: true, visionTokens: normalized.visionTokens || 1 }
          : { batch: 1, sequence: ph.sequence, phase: ph.name };
        const actions = countsForNode(node, { config: normalized, options, path: node?.id || "", bytesPerElement: B });
        if (!actions) return;
        weights += (actions.bytes?.weights || 0) * multiplier;
      });
      const expected = expectedWeightBytes(normalized, ph.name, ph.tokens, deriveBuildPlan(normalized.raw ?? normalized));
      weightRows.push({ cls, modelId, phase: ph.name, actual: weights, expected, ratio: expected > 0 ? weights / expected : null });
    }

    for (const ph of BOUND_PHASES) {
      // 按模块子树聚合：遇到 attention/mlp/moe 模块就把它整棵子树的动作向量求和
      const stack = [{ node: structure.root, multiplier: 1 }];
      while (stack.length > 0) {
        const { node, multiplier } = stack.pop();
        const type = String(node?.type || "");
        const want = MODULE_BOUND_EXPECTATION[type]?.[ph.name];
        if (want && !String(node?.id || "").includes("vision")) {
          let agg = { matrix: 0, vector: 0, sfu: 0, bytes: { weights: 0, actIn: 0, actOut: 0 } };
          walkLeaves(node, (leaf, m) => {
            const a = countsForNode(leaf, {
              config: normalized,
              options: { batch: 1, sequence: ph.sequence, phase: ph.name },
              path: leaf?.id || "",
              bytesPerElement: B,
            });
            if (!a) return;
            agg = {
              matrix: agg.matrix + a.matrix * m,
              vector: agg.vector + a.vector * m,
              sfu: agg.sfu + a.sfu * m,
              bytes: {
                weights: agg.bytes.weights + (a.bytes?.weights || 0) * m,
                actIn: agg.bytes.actIn + (a.bytes?.actIn || 0) * m,
                actOut: agg.bytes.actOut + (a.bytes?.actOut || 0) * m,
              },
            };
          });
          const { bound } = classifyRoofline({ actions: agg }, CHIP, { efficiency: { flops: 1, hbm: 1 } });
          if (bound !== want) {
            const key = `${type}|${ph.name}|want=${want}|got=${bound}`;
            if (!boundViolations.has(key)) boundViolations.set(key, new Set());
            boundViolations.get(key).add(cls);
          }
          continue; // 子树已整体判定，不再下钻
        }
        for (const child of node?.children || []) stack.push({ node: child, multiplier });
      }
    }
  }


  console.error("\n=== W1 权重字节恒等式（Σ 叶子 bytes.weights vs 应读一遍的权重字节）===");
  for (const r of weightRows) {
    console.error(`  ${r.cls} ${r.phase.padEnd(7)} ${r.modelId.padEnd(42)} actual=${fmt(r.actual)} expected=${fmt(r.expected)} ratio=${r.ratio == null ? "n/a" : r.ratio.toFixed(4)}`);
  }
  console.error("\n=== W1 bound 期望违反（逐算子逐相位，prefill 工作点 T=S=2048）===");
  if (boundViolations.size === 0) console.error("  （无）");
  for (const [key, classes] of [...boundViolations.entries()].sort()) {
    console.error(`  ${key}  (${classes.size} 个结构类: ${[...classes].sort().join(",")})`);
  }
  console.error("");

  // W5：bound 期望从 warn 升级为**断言**（模块级，工作点 prefill T=S=2048 /
  // decode T=1 S=4096）。这条把用户那句「attn prefill 算力瓶颈、decode 访存
  // 瓶颈」变成 CI 可拦的契约；MoE 两相位访存侧的判据见 MODULE_BOUND_EXPECTATION
  // 的推导注释。
  assert.deepEqual(
    [...boundViolations.keys()].sort(),
    [],
    "模块级 bound 与期望不符：先核 arithmetic intensity 与 ridge point，再决定是改公式还是改期望",
  );

  assert.equal(weightRows.length, REPRESENTATIVES.length * PHASES.length, "权重字节报表覆盖不全");

  // W4 → W6：权重字节恒等式收到 **容差 0（逐字节相等）**，登记表空。
  // 两侧是两套独立实现（期望侧 = cost/derivedWeights.js 的闭式参数量公式，
  // 实际侧 = 结构树逐叶 bytes.weights 求和），能逐字节对上才说明两边都对。
  // 归零路径（每一条都是**公式修正**，不是放宽容差；逐层归因工具
  // `node scripts/diff-weight-identity.mjs --phase decode <modelId>`）：
  //   · 视觉 patch embedding 被误当查表（regex `(patch_)?embed`）→ 权重与 MAC 全丢
  //   · 逐头归一化的权重宽度：q_norm/k_norm/GDN 输出门/index_k_norm 一律
  //     `RMSNorm(head_dim)`，不是全宽（normWeightWidth 取最后一维）
  //   · linear 的 bias 也是权重；LayerNorm 有 bias（affineBias）
  //   · MLA 的 q_a_layernorm 在 mla_query_compress 与独立 `q_a_norm` 叶双计
  //   · KDA 的 beta 已在融合 qkvbfg_a/qkvgfab 内，独立 beta 叶是双计；
  //     GLM 还缺 g_b_proj、A_log、o_norm；K3 的 attn_residual 聚合叶双计了
  //     两个 norm 与两个打分投影
  //   · mHC 的混合矩阵是 [mix_hc, hc_mult·hidden]（不是 [H, hc_mult]），
  //     attn_norm/ffn_norm 融进 mhc_pre/fused，最终 hc_post 复用末层权重
  //     （weightsShared，算力照计、字节不重复计）
  //   · 哈希路由层没有 router GEMM，用的是 tid2eid 表
  //   · 扁平 vision 配置也有投影器；Kimi 的 PatchMergerMLP 输出是**文本** hidden；
  //     GLM merger 的 mergeWidth→output 与 downsample 是同一条
  //   · PLE、q/k norm、latent norms、routed_expert_norm 等期望侧缺项逐条补齐
  const WEIGHT_BYTES_TOLERANCE = 0;
  const WEIGHT_BYTES_REGISTERED = {};
  const weightOffenders = weightRows.filter((r) => {
    const key = `${r.modelId}|${r.phase}`;
    const tol = WEIGHT_BYTES_REGISTERED[key] ?? WEIGHT_BYTES_TOLERANCE;
    if (r.actual == null || r.expected == null) return true;
    // 容差 0 时比**整数字节差**，不比浮点比值（比值会被 toFixed 掩盖 1 字节的差）。
    if (tol === 0) return Math.round(r.actual) !== Math.round(r.expected);
    return r.ratio == null || Math.abs(r.ratio - 1) > tol;
  });
  assert.deepEqual(
    weightOffenders.map((r) => `${r.modelId}|${r.phase} actual=${r.actual} expected=${r.expected} 差=${r.actual - r.expected}`),
    [],
    "权重字节恒等式不再逐字节相等：跑 `node scripts/diff-weight-identity.mjs --phase <phase> <modelId>` 定位到层与算子，修公式；不要放宽容差",
  );
});

// 注意力族算子：KV 恒等式的参与者。W2 拆 id 后同步更新（旧的
// qsa_attention / qsa_indexer 已不存在，留着会让这三行统计成 0）。
// 主注意力（读 KV cache 的那些叶）。W5：**indexer 不在内** —— 它读的是自己那份
// 独立的 index-k cache（宽 index_head_dim、单头），与 kvBytesPerToken 无关，
// 混在一起比会让稀疏模型看起来「超读」。indexer 侧单列在 INDEXER_OPS。
const ATTENTION_OPS = new Set([
  "matmul",
  "qsa_sparse_attention", "dsa_sparse_mla", "dsv4_sparse_mla",
  "minimax_sparse_attention", "dsv4_swa_attention", "dsv4_compressed_attention",
]);
const INDEXER_OPS = new Set([
  "qsa_indexer", "dsa_indexer", "dsa_kpool_indexer", "dsv4_indexer", "minimax_sparse_indexer",
]);

// 读**整条 cache** 的注意力形态（每次 decode 必须把全长 S 流一遍）；
// 其余形态（qsa/dsa 的 top-k 选择、minimax 块稀疏、dsv4 的压缩+滑窗）只读一部分。
const FULL_READ_KINDS = new Set(["gqa", "mha", "mla", "qwen35_full"]);

test("W5 恒等式：KV 读量（逐层 cache 容量对账，容差 0）", () => {
  const catalog = JSON.parse(fs.readFileSync(path.join(repoRoot, "models/catalog.json"), "utf8"));
  const byId = new Map(catalog.models.map((m) => [m.model_id, m]));
  const rows = [];

  for (const [cls, modelId] of REPRESENTATIVES) {
    const raw = JSON.parse(fs.readFileSync(path.join(repoRoot, "models", byId.get(modelId).config_path), "utf8"));
    const { normalized, structure } = buildStructure(raw, modelId);
    const S = 4096;
    const options = { batch: 1, sequence: S, phase: "decode" };
    const plan = deriveBuildPlan(normalized.raw ?? normalized);
    const schedule = plan.attentionSchedule || [];
    const { main, index } = kvBytesPerTokenBreakdown(normalized, B);

    // 期望侧：逐层 cache 容量 × 全长 S，按「读全 cache / 只读一部分」分两桶。
    let fullCapacity = 0;
    let selectiveCapacity = 0;
    let indexCapacity = 0;
    for (let i = 0; i < (normalized.layers || 0); i += 1) {
      const kind = schedule[i] || "gqa";
      if (kind === "linear") continue; // KDA 状态是 request 级，不在 KV cache 口径内
      if (FULL_READ_KINDS.has(kind)) fullCapacity += (main[i] || 0) * S;
      else selectiveCapacity += (main[i] || 0) * S;
      indexCapacity += (index[i] || 0) * S;
    }

    // 实际侧：逐叶的 kvRead / indexRead（都是 actIn 的子项，单独声明）。
    let fullRead = 0;
    let selectiveRead = 0;
    let indexRead = 0;
    let attnLeaves = 0;
    walkLeaves(structure.root, (node, multiplier) => {
      const id = String(node?.id || "");
      if (id.includes("vision")) return;
      const actions = countsForNode(node, { config: normalized, options, path: id, bytesPerElement: B });
      if (!actions) return;
      const op = String(node?.attributes?.operator_id || "").toLowerCase();
      if (INDEXER_OPS.has(op)) {
        indexRead += (actions.bytes?.indexRead || 0) * multiplier;
        return;
      }
      if (!ATTENTION_OPS.has(op)) return;
      const kv = (actions.bytes?.kvRead || 0) * multiplier;
      if (kv > 0) attnLeaves += 1;
      const layer = Number((id.match(/^(?:decoder|text_decoder)\.(\d+)\./) || [])[1]);
      const kind = Number.isFinite(layer) ? (schedule[layer] || "gqa") : "gqa";
      if (FULL_READ_KINDS.has(kind)) fullRead += kv;
      else selectiveRead += kv;
    });

    rows.push({
      cls, modelId, fullRead, fullCapacity, selectiveRead, selectiveCapacity, indexRead, indexCapacity, attnLeaves,
    });
  }

  console.error("\n=== W5 KV 读量对账（decode S=4096，单位字节）===");
  for (const r of rows) {
    console.error(`  ${r.cls} ${r.modelId.padEnd(42)}`
      + ` 全读层 ${fmt(r.fullRead)}/${fmt(r.fullCapacity)}`
      + ` · 选择性层 ${fmt(r.selectiveRead)}/${fmt(r.selectiveCapacity)}`
      + ` · indexer ${fmt(r.indexRead)}/${fmt(r.indexCapacity)}`);
  }
  console.error("");

  assert.ok(rows.length === REPRESENTATIVES.length, "KV 报表覆盖不全");

  // W6：从「actIn 总量比 cache 容量，容差 30%」换成**逐层容量的两侧夹逼，容差 0**。
  //
  // 为什么不是单一等式：把 actIn 总量拿来比就必须留松量（里面混着 Q、scores、
  // top-k 索引）。分出 `bytes.kvRead` 之后仍然不能对**所有**层写等式 ——
  // 选择性读的层（top-k / 块稀疏 / 压缩+滑窗）读多少由各家的预算与窗口规则决定，
  // 期望侧要写等式就得把那套规则抄一遍，恒等式退化成同义重复（plan §四明确要
  // 「打破同义重复」）。所以按层分桶，只在判据无歧义处写等式：
  //   ① 读全 cache 的层（gqa/mha/mla/qwen35_full）：kvRead **逐字节等于**
  //      逐层 cache 容量 × S。历史上所有 KV bug 都在这一桶
  //      （按 query 头数读 → V3.1 超读 71x；MLA 的 K/V 同一份 latent 读两遍 → 2.1x）。
  //   ② 选择性读的层：kvRead ≤ 该桶容量 × S（读不可能超过整条 cache），且 > 0。
  //   ③ indexer：indexRead ≤ 自己那份 index-k cache 容量 × S（至多扫一遍）。
  // 三条都是物理不等式/等式，无容差、无登记表。
  const fullOffenders = rows.filter((r) => Math.round(r.fullRead) !== Math.round(r.fullCapacity));
  assert.deepEqual(
    fullOffenders.map((r) => `${r.cls} ${r.modelId} 全读层 ${r.fullRead} vs ${r.fullCapacity} 差=${r.fullRead - r.fullCapacity}`),
    [],
    "读全 cache 的层 KV 读量不等于逐层容量×S：查 kvHeads 共享、latent 读宽、K/V 是否重复读",
  );
  const ceilingOffenders = rows.filter((r) => r.selectiveRead > r.selectiveCapacity + 1e-9);
  assert.deepEqual(
    ceilingOffenders.map((r) => `${r.cls} ${r.modelId} 选择性层 ${r.selectiveRead} > 容量 ${r.selectiveCapacity}`),
    [],
    "选择性读的层读量超过整条 cache：预算或读宽算错了",
  );
  const emptyOffenders = rows.filter((r) => (r.fullCapacity + r.selectiveCapacity) > 0 && r.fullRead + r.selectiveRead <= 0);
  assert.deepEqual(
    emptyOffenders.map((r) => `${r.cls} ${r.modelId}`),
    [],
    "有 KV cache 却没有任何 kvRead：注意力叶子漏了 cache 读",
  );
  const indexOffenders = rows.filter((r) => r.indexRead > r.indexCapacity + 1e-9);
  assert.deepEqual(
    indexOffenders.map((r) => `${r.cls} ${r.modelId} indexer ${r.indexRead} > 容量 ${r.indexCapacity}`),
    [],
    "indexer 读量超过自己那份 index-k cache：查池化粒度与读宽",
  );
});

// ---------------------------------------------------------------------------
// 第四条恒等式：激活流连续性。
//
// W1 最初按**字节**比（actOut(src) === actIn(dst)），实测 62 类不等 —— 框架错了：
// 字节口径把权重/中间量都算进 actIn，且融合节点的 actIn 是整个模块的输入，
// 不是上游那一个张量。改成按**末维形状**比（src.output_shape.at(-1) vs
// dst.input_shape.at(-1)），这才是"上游产出的张量正好是下游要吃的张量"的可判定
// 表述。全 59 模型 33434 条声明边：29180 匹配、59 无形状、4195 落在 31 个
// **语义边**类里（下面登记）。
//
// 语义边不是建模缺口，是"声明边"这一表示的固有多义性 —— 一条边可以表示：
//   control  控制/辅助信号（topk 出的是专家下标、beta/decay 出的是门控标量）
//   slice    上游是融合宽张量，下游只吃其中一片（qkv_split → q_norm）
//   fused-in 子算子汇入融合父节点（indexer.q_proj → indexer，父节点入口是 hidden）
//   concat   多源拼成更宽的入（mtp.enorm + mtp.hnorm → eh_proj 的 2H）
//   entry    下游入口不是特征维（embed_tokens 吃 token id，末维记 -1）
// 登记表按**节点后缀对**索引（不是算子对）—— `linear -> rmsnorm` 这种常见对
// 绝大多数是要匹配的，按算子对登记会把真 bug 一起放过。
//
// 棘轮：登记表只许缩短。出现未登记的类即失败 —— 要么是真 bug（本波就是这样抓出
// K3 潜空间 MoE 的 combine → shared_expert_add 越过了 norm+up_proj），要么归因后
// 追加一行并写清属于上面哪一类。
const SHAPE_EDGE_REGISTERED = new Map(Object.entries({
  // control：上游产出下标 / 门控标量，不是下游要吃的激活
  "topk -> dispatch": "control",
  "hash_router -> dispatch": "control",
  "beta_projection -> state_update": "control",
  "decay_projection -> state_update": "control",
  "f_b_proj -> state_update": "control",
  "indexer -> sparse_attention": "control",
  // slice：上游是融合宽张量，下游只吃一片
  "qkv_split -> q_norm": "slice",
  "qkv_split -> k_norm": "slice",
  // QSA 的融合 qkv（heads·head_dim + 2·kv_heads·head_dim）直接喂逐头 norm
  "qkv_proj -> q_norm": "slice",
  "qkv_proj -> k_norm": "slice",
  "qkv_split -> kv_norm": "slice",
  "qkv_gate_split -> q_norm": "slice",
  "qkv_gate_split -> k_norm": "slice",
  "qkv_index_split -> q_norm": "slice",
  "qkv_index_split -> k_norm": "slice",
  "qkv_index_split -> index_q_norm": "slice",
  "qkv_index_split -> index_k_norm": "slice",
  "kv_b_proj -> rope": "slice",
  "wk_weights_proj -> k_norm": "slice",
  "qkv_projection -> short_conv": "slice",
  "compressor -> attention": "slice",
  // KDA 融合投影里的 f_a / g_a / g（全秩门）切片，宽度都是 head_dim 或 projection
  "qkv_projection -> f_b_proj": "slice",
  "qkv_projection -> g_b_proj": "slice",
  "qkv_projection -> output_gate_norm": "slice",
  // fused-in：子算子汇入融合父节点，父节点入口是整个模块的输入
  "q_proj -> indexer": "fused-in",
  "k_norm -> indexer": "fused-in",
  // concat：多源拼成更宽的入
  "enorm -> eh_proj": "concat",
  "hnorm -> eh_proj": "concat",
  // entry：下游入口不是特征维
  "vision_tower -> embed_tokens": "entry",
  "projector -> embed_tokens": "entry",
  // regroup：同一张量换分组视图（逐头 ↔ 摊平、patch merge 把 merge² 个 token 拼成一行），
  // 元素总数不变、末维按整数倍变化。判据：两端末维互为整数倍。
  "pre_norm -> fc1": "regroup",
  // g_b 输出 projection 宽，逐头门控 norm 按 [heads, valueDim] 看同一张量
  "g_b_proj -> output_gate_norm": "regroup",
}));

function indexNodesByCanonicalId(root, map) {
  const stack = [root];
  while (stack.length > 0) {
    const node = stack.pop();
    if (!node) continue;
    map.set(node.canonical_id ?? node.id, node);
    for (const child of node.children || []) stack.push(child);
  }
}

const lastDim = (shape) => (Array.isArray(shape) && shape.length > 0 ? shape[shape.length - 1] : undefined);

test("W5 恒等式：激活流形状连续性（全 59 模型声明边）", () => {
  const catalog = JSON.parse(fs.readFileSync(path.join(repoRoot, "models/catalog.json"), "utf8"));
  const classes = new Map();
  let total = 0;
  let matched = 0;
  let noShape = 0;

  for (const entry of catalog.models) {
    const raw = JSON.parse(fs.readFileSync(path.join(repoRoot, "models", entry.config_path), "utf8"));
    const { structure } = buildStructure(raw, entry.model_id);
    const nodes = new Map();
    indexNodesByCanonicalId(structure.root, nodes);
    for (const edge of structure.graph?.edges || []) {
      const src = nodes.get(edge.source_canonical_id);
      const dst = nodes.get(edge.target_canonical_id);
      if (!src || !dst) continue;
      total += 1;
      const out = lastDim(src.output_shape);
      const inn = lastDim(dst.input_shape);
      if (out == null || inn == null) { noShape += 1; continue; }
      if (out === inn) { matched += 1; continue; }
      const key = `${String(edge.source_canonical_id).split(".").pop()} -> ${String(edge.target_canonical_id).split(".").pop()}`;
      const rec = classes.get(key) || { count: 0, models: new Set(), out, inn };
      rec.count += 1;
      rec.models.add(entry.model_id);
      classes.set(key, rec);
    }
  }

  const unregistered = [...classes.entries()].filter(([key]) => !SHAPE_EDGE_REGISTERED.has(key));
  // slice 类的数值不变量：上游是融合宽张量，切出来的片必须更窄
  const sliceViolations = [...classes.entries()]
    .filter(([key]) => SHAPE_EDGE_REGISTERED.get(key) === "slice")
    .filter(([, v]) => !(v.out > v.inn));
  // entry 类：下游入口末维必须是非特征维标记 -1
  const entryViolations = [...classes.entries()]
    .filter(([key]) => SHAPE_EDGE_REGISTERED.get(key) === "entry")
    .filter(([, v]) => v.inn !== -1);
  // regroup 类：两端末维必须互为整数倍（元素总数不变，只是换了分组视图）
  const regroupViolations = [...classes.entries()]
    .filter(([key]) => SHAPE_EDGE_REGISTERED.get(key) === "regroup")
    .filter(([, v]) => {
      const [big, small] = v.out >= v.inn ? [v.out, v.inn] : [v.inn, v.out];
      return !(small > 0 && big % small === 0);
    });

  console.error(`\n=== W5 激活流形状连续性：总边=${total} 末维匹配=${matched} 无形状=${noShape} 语义边=${total - matched - noShape}（${classes.size} 类）===`);
  for (const [key, v] of [...classes.entries()].sort((a, b) => b[1].count - a[1].count)) {
    console.error(`  ${String(v.count).padStart(5)}  ${key.padEnd(46)} ${v.out} vs ${v.inn}  [${SHAPE_EDGE_REGISTERED.get(key) ?? "未登记"}]  ${v.models.size} 模型`);
  }
  console.error("");

  assert.deepEqual(
    unregistered.map(([key, v]) => `${key}(${v.out} vs ${v.inn}, ${v.models.size} 模型, 例 ${[...v.models][0]})`),
    [],
    "出现未登记的形状不连续边：先当真 bug 查（上游是否漏了一级投影/归一化），确认是语义边再追加进 SHAPE_EDGE_REGISTERED 并标类别",
  );
  assert.deepEqual(sliceViolations.map(([key, v]) => `${key}(${v.out} vs ${v.inn})`), [], "登记为 slice 的边上游反而更窄：切片语义不成立，重新归因");
  assert.deepEqual(entryViolations.map(([key, v]) => `${key}(inn=${v.inn})`), [], "登记为 entry 的边下游末维不是 -1：入口语义不成立，重新归因");
  assert.deepEqual(regroupViolations.map(([key, v]) => `${key}(${v.out} vs ${v.inn})`), [], "登记为 regroup 的边两端末维不成整数倍：不是换视图，重新归因");
  assert.ok(matched / (total - noShape) > 0.85, `形状连续率 ${(matched / (total - noShape) * 100).toFixed(1)}% 低于 85%：语义边占比异常上升`);
});

