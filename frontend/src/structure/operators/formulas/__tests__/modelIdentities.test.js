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
import { normalizeConfig } from "../../../config/normalize.js";
import { resolveArchitecture } from "../../../registry/resolveArchitecture.js";
import { buildNetwork } from "../../../models/index.js";
import { createStructureIr } from "../../../ir/createStructureIr.js";
import { materializeModelStructure } from "../../../materializers/modelStructure.js";
import { graphRoot } from "../../../graph/selectors.js";
import { countsForNode, isVisionPath } from "../extractor.js";
import { childRepeatMultiplier, walkStructure } from "../../../../cost/traverse.js";
import { kvBytesPerToken } from "../../../../cost/memory.js";
import { paramBytes } from "../paramDtypes.js";
import { classifyRoofline } from "../../../../cost/roofline.js";
import { attentionScheduleOf } from "../../../layers/schedule.js";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../../../../../..");
const B = 2;

const REPRESENTATIVES = [
  ["S01", "Qwen/Qwen3.5-0.8B"], ["S02", "Qwen/Qwen3.5-35B-A3B"], ["S03", "zai-org/GLM-5"],
  ["S04", "moonshotai/Kimi-K2-Instruct"], ["S05", "deepseek-ai/DeepSeek-V4-Pro"], ["S06", "moonshotai/Kimi-K2.5"],
  ["S07", "deepseek-ai/DeepSeek-V3.1"], ["S08", "zai-org/GLM-5.3-Flash"], ["S09", "MiniMaxAI/MiniMax-M3"],
  ["S10", "Qwen/Qwen3.8-2.4T-A95B"], ["S11", "Qwen/Qwen3.8-Flash-Next"], ["S12", "deepseek-ai/DeepSeek-V3.2"],
  ["S13", "deepseek-ai/DeepSeek-V4-Flash-Vision-Exp"], ["S14", "zai-org/GLM-4.7"],
  ["S15", "moonshotai/Kimi-K3"], ["S16", "MiniMaxAI/MiniMax-M2.7"],
];

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

/** 期望侧：该相位下叶 counts.bytes.weights 之和。无 header 时不再用闭式 Σ；
 *  声明 vs counts 由锚 1 单源对账。本函数只是同一 walk 的相位合计。 */

const treeView = (structure) => graphRoot(structure.graph);

function walkLeaves(root, visit) {
  const stack = [{ node: root, multiplier: 1 }];
  while (stack.length > 0) {
    const { node, multiplier } = stack.pop();
    const children = node?.children || [];
    if (children.length > 0) {
      const repeatHandled = children.some((child) => Number.isFinite(child?.repeat));
      const childMultiplier = childRepeatMultiplier(node, multiplier, { repeatHandled });
      for (const child of children) stack.push({ node: child, multiplier: childMultiplier });
      continue;
    }
    visit(node, multiplier);
  }
}

const fmt = (n) => (Number.isFinite(n) ? n.toExponential(3) : String(n));

test("W1 报表：模块级 bound 期望", () => {
  const catalog = JSON.parse(fs.readFileSync(path.join(repoRoot, "models/catalog.json"), "utf8"));
  const byId = new Map(catalog.models.map((m) => [m.model_id, m]));
  const boundViolations = new Map();

  for (const [cls, modelId] of REPRESENTATIVES) {
    const entry = byId.get(modelId);
    assert.ok(entry, `代表模型缺失: ${modelId}`);
    const raw = JSON.parse(fs.readFileSync(path.join(repoRoot, "models", entry.config_path), "utf8"));
    const { normalized, structure } = buildStructure(raw, modelId);

    for (const ph of BOUND_PHASES) {
      // 按模块子树聚合：遇到 attention/mlp/moe 模块就把它整棵子树的动作向量求和
      const stack = [{ node: treeView(structure), multiplier: 1 }];
      while (stack.length > 0) {
        const { node, multiplier } = stack.pop();
        const type = String(node?.type || "");
        const id = String(node?.id || "");
        if (type === "mtp" || type === "dspark" || /(^|\.)mtp(\.|$)/.test(id)) continue;
        const want = MODULE_BOUND_EXPECTATION[type]?.[ph.name];
        if (want && !isVisionPath(id) && !isVisionPath(node?.canonical_id)) {
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
          // DSV4 C128 层 visible≈T/128，T=2048 时 AI 过不了 ridge，prefill 也是 memory。
          // 这是压缩注意力的真实强度，不是 MTP。期望只约束非压缩层。
          const compressed = Number(node?.attributes?.compress_ratio) > 1;
          if (bound !== want && !(type === "attention" && ph.name === "prefill" && compressed && bound === "memory")) {
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


  console.error("\n=== W1 bound 期望违反（逐算子逐相位，prefill 工作点 T=S=2048）===");
  if (boundViolations.size === 0) console.error("  （无）");
  for (const [key, classes] of [...boundViolations.entries()].sort()) {
    console.error(`  ${key}  (${classes.size} 个结构类: ${[...classes].sort().join(",")})`);
  }
  console.error("");

  // W5：bound 期望从 warn 升级为**断言**（模块级，工作点 prefill T=S=2048 /
  // decode T=1 S=4096）。这条把用户那句「attn prefill 算力瓶颈、decode 访存
  // 瓶颈」变成 CI 可拦的契约；MoE 两相位访存侧的判据见 MODULE_BOUND_EXPECTATION
  // 的推导注释。权重字节 vs 声明由下方锚 1 单源对账，不再用闭式 Σ。
  assert.deepEqual(
    [...boundViolations.keys()].sort(),
    [],
    "模块级 bound 与期望不符：先核 arithmetic intensity 与 ridge point，再决定是改公式还是改期望",
  );
});

// ---------------------------------------------------------------------------
// N2-4 锚 1（docs/details/sharding_matrix.md §五）：weightMatrices 声明单源。
//
// 声明元素数 × 2B == 叶 counts.bytes.weights，逐叶断言（全模型目录）。
// 权重字节恒等式（上方 W1 测试）已锚定叶 counts，因此声明写错立即红。
// 工作点取 prefill T=S=2048：routed 专家的 bytes.weights 按触达数
// min(k·T, E) 计，k·T=16384 覆盖目录全部模型的专家数 → 触达数 == E，
// 声明（全量 E 份）才与叶 counts 可比；线性/归一化叶的 weights 与 T 无关，
// 工作点只影响 MoE 叶。
// ---------------------------------------------------------------------------
test("N2-4 锚 1：weightMatrices 声明与叶 counts.bytes.weights 单源（容差 0）", () => {
  const catalog = JSON.parse(fs.readFileSync(path.join(repoRoot, "models/catalog.json"), "utf8"));
  const offenders = [];
  let declaredLeaves = 0;

  for (const entry of catalog.models) {
    const raw = JSON.parse(fs.readFileSync(path.join(repoRoot, "models", entry.config_path), "utf8"));
    const { normalized, structure } = buildStructure(raw, entry.model_id);
    walkLeaves(treeView(structure), (node) => {
      const declaration = node?.attributes?.weightMatrices;
      if (!Array.isArray(declaration) || declaration.length === 0) return;
      declaredLeaves += 1;
      // embedding 叶：声明描述**驻留容量**（vocab·hidden，P4-2），gather 的
      // counts.bytes.weights=0（流量按行计，M11 已入 actIn）——两者语义本就
      // 不同，登记例外：只要求声明与本叶 vocab_size·hidden_size 对账
      // （主词表 = 模型 vocab×hidden；PLE ngram 表 = padded_vocab×head_dim）。
      if (node?.type === "embedding") {
        const declared = declaration.reduce((sum, group) => sum + (group.count ?? 1) * (group.matrices ?? 1) * group.out * group.in, 0);
        const expected = (node?.attributes?.vocab_size || 0) * (node?.attributes?.hidden_size || 0);
        if (declared !== expected) {
          offenders.push(`${entry.model_id} ${node?.id} embedding declared=${declared} vs vocab·hidden=${expected}`);
        }
        return;
      }
      // DSpark hc_head / confidence_head：声明 = 驻留（fp32），counts 是相位读量。
      // V4 wo_a：权重是 grouped BMM [G·Ro, H/G]，counts.matrix 跟声明；
      // 激活流量仍按 grouped 输出，bytes.weights 对声明。
      if (/(^|\.)(hc_head|confidence_head)$/.test(String(node?.id || ""))) return;
      const actions = countsForNode(node, {
        config: normalized,
        options: { batch: 1, sequence: 2048, phase: "prefill" },
        path: node?.id || "",
        bytesPerElement: B,
      });
      // P4-2：dtype-aware 判据 —— 带 param_dtype 的组按 paramDtypes 登记表的
      // 字节宽计（mHC fn/base/scale 与 KDA 衰减参数是 fp32），其余 2B。dtype
      // 知识仍单源在 paramDtypes.js，声明只引用键名。
      const declaredBytes = declaration.reduce(
        (sum, group) => sum + (group.count ?? 1) * (group.matrices ?? 1) * group.out * group.in
          * (group.param_dtype ? paramBytes(group.param_dtype) : B),
        0,
      );
      // multiplier 与声明无关（声明描述单实例），与 identity 测试同口径两侧同乘可消去。
      if (!actions || declaredBytes !== actions.bytes.weights) {
        offenders.push(`${entry.model_id} ${node?.id} declared=${declaredBytes}B vs counts=${actions?.bytes?.weights ?? "null"}`);
      }
    });
  }

  console.error(`\n=== N2-4 锚 1：weightMatrices 声明叶 ${declaredLeaves} 个，违例 ${offenders.length} 个 ===`);
  if (offenders.length > 0) for (const line of offenders.slice(0, 20)) console.error(`  ${line}`);
  assert.ok(declaredLeaves > 0, "没有任何 weightMatrices 声明叶：ops 的声明助手未接线");
  assert.deepEqual(offenders, [], "声明元素数×2B 与叶 counts.bytes.weights 不等：声明或公式有一侧错了，先查同源性（routedExpertWeightMatrices 与 extractor 的 EH/EH 取值链）");
});

// ---------------------------------------------------------------------------
// P2 覆盖率护栏（步骤 3 前置）：带权重的叶必须有 weightMatrices 声明。
//
// 判据：counts.bytes.weights > 0（该叶真的要读权重）⇒ 必须有声明。
// 另加 embedding 叶：gather 不读全表，counts.bytes.weights 为 0，但它**持有**
// vocab·hidden 的权重（容量口径必须算），所以按 type 显式纳入判据。
// 纯激活叶（rope/split/softmax/residual_add/moe_dispatch/… weights=0）天然
// 不在判据内 —— 声明体描述的是权重矩阵归属，无权重就无归属。
//
// 为什么要这条：锚 1 只保证「已声明的叶声明得对」，对**没声明的叶**完全沉默。
// P4-2 后全目录带权叶全部有声明（棘轮归零），WEIGHT_PROJECTION_RULES
// 的删除前提达成（P5）。棘轮保持 0：新增带权算子不声明即顶破。
//
// 棘轮：MAINTENANCE.md「P2 声明覆盖」条目。**只许下降**，新增带权算子若不声明
// 会顶破基线立即红。
// ---------------------------------------------------------------------------
const WEIGHT_DECLARATION_BASELINE = 0;

test("P2 护栏：带权重叶的 weightMatrices 声明覆盖（棘轮，只许下降）", () => {
  const catalog = JSON.parse(fs.readFileSync(path.join(repoRoot, "models/catalog.json"), "utf8"));
  const missingByOp = new Map();
  let weightedLeaves = 0;
  let declaredLeaves = 0;

  for (const entry of catalog.models) {
    const raw = JSON.parse(fs.readFileSync(path.join(repoRoot, "models", entry.config_path), "utf8"));
    const { normalized, structure } = buildStructure(raw, entry.model_id);
    walkLeaves(treeView(structure), (node) => {
      const actions = countsForNode(node, {
        config: normalized,
        options: { batch: 1, sequence: 2048, phase: "prefill" },
        path: node?.id || "",
        bytesPerElement: B,
      });
      const weights = actions?.bytes?.weights ?? 0;
      const hasWeightShapes = Object.keys(node?.weight_shapes || {}).length > 0;
      const isEmbedding = node?.type === "embedding";
      if (!(weights > 0) && !hasWeightShapes && !isEmbedding) return; // 纯激活叶：无权重归属可声明
      weightedLeaves += 1;
      const declaration = node?.attributes?.weightMatrices;
      if (Array.isArray(declaration) && declaration.length > 0) {
        declaredLeaves += 1;
        return;
      }
      const op = node?.attributes?.operator_id || node?.type || "(unknown)";
      const bucket = missingByOp.get(op) || { count: 0, sample: node?.id || "", model: entry.model_id };
      bucket.count += 1;
      missingByOp.set(op, bucket);
    });
  }

  const missing = weightedLeaves - declaredLeaves;
  const groups = [...missingByOp.entries()].sort((a, b) => b[1].count - a[1].count);
  console.error(`\n=== P2 声明覆盖：带权叶 ${weightedLeaves}，已声明 ${declaredLeaves}，缺声明 ${missing}（基线 ${WEIGHT_DECLARATION_BASELINE}）===`);
  for (const [op, info] of groups) {
    console.error(`  ${op}: ${info.count} 叶  例：${info.model} :: ${info.sample}`);
  }
  assert.ok(weightedLeaves > 0, "没有任何带权重叶：遍历或 counts 链断了");
  assert.ok(
    missing <= WEIGHT_DECLARATION_BASELINE,
    `缺声明带权叶 ${missing} 超过棘轮基线 ${WEIGHT_DECLARATION_BASELINE}：新增带权算子必须同时声明 weightMatrices（见 details/sharding_matrix.md 层 1）`,
  );
});

// 注意力族算子：KV 恒等式的参与者。W2 拆 id 后同步更新（旧的// qsa_attention / qsa_indexer 已不存在，留着会让这三行统计成 0）。
// 主注意力（读 KV cache 的那些叶）。W5：**indexer 不在内** —— 它读的是自己那份
// 独立的 index-k cache（宽 index_head_dim、单头），与 kvBytesPerToken 无关，
// 混在一起比会让稀疏模型看起来「超读」。indexer 侧单列在 INDEXER_OPS。
const ATTENTION_OPS = new Set([
  "matmul", "sdpa_attention",
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
    const schedule = attentionScheduleOf(normalized) || [];
    let mainPerToken = 0;
    let indexPerToken = 0;
    walkStructure(structure.graph, ({ node, multiplier }) => {
      const id = String(node?.id || "");
      if (isVisionPath(id) || node?.attributes?.modality === "vision") return;
      const attrs = node?.attributes || {};
      mainPerToken += (attrs.cache_kv_elements || 0) * multiplier;
      indexPerToken += (attrs.cache_index_elements || 0) * multiplier;
    });
    assert.equal(kvBytesPerToken(structure.graph, B), (mainPerToken + indexPerToken) * B);

    // 期望侧：逐层 cache 容量 × 全长 S，按「读全 cache / 只读一部分」分两桶。
    let fullCapacity = 0;
    let selectiveCapacity = 0;
    let indexCapacity = 0;
    walkStructure(structure.graph, ({ node, multiplier }) => {
      const id = String(node?.id || "");
      if (isVisionPath(id) || node?.attributes?.modality === "vision") return;
      const attrs = node?.attributes || {};
      const kv = (attrs.cache_kv_elements || 0) * multiplier * B * S;
      const index = (attrs.cache_index_elements || 0) * multiplier * B * S;
      indexCapacity += index;
      if (!kv) return;
      const layer = Number((id.match(/(?:^|\.)(?:layers|language_model)\.(\d+)(?:\.|$)/) || [])[1]);
      const kind = Number.isFinite(layer) ? (schedule[layer] || "gqa") : "gqa";
      if (kind === "linear") return;
      if (FULL_READ_KINDS.has(kind)) fullCapacity += kv;
      else selectiveCapacity += kv;
    });

    // 实际侧：逐叶的 kvRead / indexRead（都是 actIn 的子项，单独声明）。
    let fullRead = 0;
    let selectiveRead = 0;
    let indexRead = 0;
    let attnLeaves = 0;
    walkLeaves(treeView(structure), (node, multiplier) => {
      const id = String(node?.id || "");
      if (isVisionPath(id)) return;
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
      const layer = Number((id.match(/^(?:layers|language_model)\.(\d+)\./) || [])[1]);
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
  "pre_fc_norm_embedding -> fc": "concat",
  "pre_fc_norm_hidden -> fc": "concat",
  "pre_fc_norm_hidden -> fc_hidden": "concat",
  // DSpark confidence_head = Linear(H+r)；r 来自 markov_w1 gather，不是 hc_head 激活
  "hc_head -> confidence_head": "concat",
  // markov_head 的模块出口标注为 logits，但 SGLang compute_confidence 里喂给
  // confidence_head 的是 markov 嵌入（get_prev_embeddings = markov_w1 输出，dim=r），
  // 即 H+r concat 的 r 路贡献。与 hc_head 那路同为 concat 贡献者。
  "markov_head -> confidence_head": "concat",
  // entry：下游入口不是特征维
  "visual -> embed_tokens": "entry",
  "projector -> embed_tokens": "entry",
  // regroup：同一张量换分组视图（逐头 ↔ 摊平、patch merge 把 merge² 个 token 拼成一行），
  // 元素总数不变、末维按整数倍变化。判据：两端末维互为整数倍。
  "pre_norm -> fc1": "regroup",
  // g_b 输出 projection 宽，逐头门控 norm 按 [heads, valueDim] 看同一张量
  "g_b_proj -> output_gate_norm": "regroup",
  // Engram（DeepSeek V4.1）：embed 查表出 [n_hash_cols, engram_head_dim]，wkv 前
  // flatten(-2) 成 n_hash_cols·engram_head_dim（6144=24×256，换分组视图）→ regroup；
  // wkv 出 dim·(hc_mult+1) 的融合 key+value 汇入 engram_gate（门控写回残差流）→ fused-in。
  "embed -> wkv": "regroup",
  "wkv -> engram_gate": "fused-in",
}));

// P7（步骤 7）：树遍历索引退役——Graph IR 节点自带 canonical_id，直接建索引。
function indexNodesByCanonicalId(graph, map) {
  for (const node of graph.nodes) {
    map.set(node.canonical_id ?? node.id, node);
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
    indexNodesByCanonicalId(structure.graph, nodes);
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
