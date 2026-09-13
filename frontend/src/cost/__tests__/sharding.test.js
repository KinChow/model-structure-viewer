// N2-4 W-B 验收（docs/details/sharding_matrix.md §五/§六）：
// - 组合语义纯函数手算（EP=TP×DP、无 EP 时 DP 切专家、混合 ETP moe_tp×moe_ep）
// - 锚 2：EP 计划下 M2.7 每卡权重 = 专家块÷moe_ep + 其余÷tp，与聚合投影、
//   expertWeightRange 三方一致
// - 锚 3（已退役，P5）：路径规则表已删除，weightMatrices 是唯一入口 ——
//   同义断言失去对象；无声明的带权叶现在返回 axis: "unknown"（诚实缺项）。
import assert from "node:assert/strict";
import fs from "node:fs";
import test from "node:test";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { expertShardDivisor, declaredWeightBytesPerCard, declaredWeightElements, declaredClassDivisor } from "../sharding.js";
import { expertWeightRange, projectNodePlan, validatePlan, weightBytesPerCard } from "../parallel.js";
import { normalizeConfig } from "../../structure/config/normalize.js";
import { resolveArchitecture } from "../../structure/registry/resolveArchitecture.js";
import { buildNetwork } from "../../structure/models/index.js";
import { createStructureIr } from "../../structure/ir/createStructureIr.js";
import { materializeModelStructure } from "../../structure/materializers/modelStructure.js";
import { layerScheduleOf } from "../../structure/layers/schedule.js";
import { aggregateCost } from "../aggregate.js";
import { graphWeightCapacity } from "../memory.js";
import { materializeStructureGraph } from "../../structure/graph/materializeStructureGraph.js";

// P7（步骤 7）：projectNodePlan/aggregateCost 只收 Graph IR——
// 夹具 tree root 统一经 materializeStructureGraph 转图。
const toGraph = (root) => materializeStructureGraph(root);

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../../../..");

// ---------------------------------------------------------------------------
// sharding.js 纯函数：组合语义手算
// ---------------------------------------------------------------------------

test("expertShardDivisor：无 EP 时专家跟 TP 切，DP 也把专家集合 ÷dp 切", () => {
  // 专家 TP 切分语义（每个专家的矩阵 ÷tp）——与规则表旧行为一致
  assert.equal(expertShardDivisor({ tp: 8 }).divisor, 8);
  // vLLM DP-shards-experts：无 EP 时 DP 也切专家（「DP=复制」只对 attention 成立）
  assert.equal(expertShardDivisor({ tp: 8, dp: 2 }).divisor, 16);
  assert.equal(expertShardDivisor({ tp: 8, dp: 2 }).setDegree, 2);
});

test("expertShardDivisor：EP 启用时每卡持完整专家（不再 TP 内切），除数 = epSize", () => {
  // vLLM 单机 EP=TP=8：8 个 rank 即 8 个 EP rank，每卡 E/8 个完整专家
  assert.equal(expertShardDivisor({ tp: 8, ep: 8 }).divisor, 8);
  assert.equal(expertShardDivisor({ tp: 8, ep: 8 }).epSize, 8);
  // DP attention + EP（DeepSeek 标准部署）：ep_size = tp × dp
  assert.equal(expertShardDivisor({ tp: 4, dp: 2, ep: 8, attnMode: "dp" }).divisor, 8);
  assert.equal(expertShardDivisor({ tp: 4, dp: 2, ep: 8, attnMode: "dp" }).setDegree, 8);
});

test("expertShardDivisor：TRT-LLM 混合 ETP（每卡 E/moe_ep 个完整专家、再 ÷moe_tp）", () => {
  assert.equal(expertShardDivisor({ tp: 8, moe_ep: 2, moe_tp: 4 }).divisor, 8);
  assert.equal(expertShardDivisor({ tp: 8, moe_ep: 2, moe_tp: 4 }).epSize, 2);
  assert.equal(expertShardDivisor({ tp: 8, moe_ep: 2, moe_tp: 4 }).moeTp, 4);
  // 显式 moe_ep 单独出现即视为 EP 启用
  assert.equal(expertShardDivisor({ tp: 4, moe_ep: 2 }).divisor, 2);
});

test("validatePlan：协议 Q2 专家域闭合 + Q4 无 EP moe_tp=tp + EP=TP×DP 组合校验（P6）", () => {
  // 无 EP：moe_tp 缺省走 sharding 缺省链；显式给出必须 = tp（Q4）
  assert.equal(validatePlan({ tp: 4 }, { experts: 8 }).ok, true);
  assert.match(validatePlan({ tp: 4, moe_tp: 2 }, { experts: 8 }).errors[0], /moe_tp\(2\) 应等于 tp\(4\)/);
  // EP 启用 + 混合 ETP：专家域闭合 moe_ep × moe_tp = ep × tp（TRT-LLM Hybrid ETP）
  assert.equal(validatePlan({ tp: 4, ep: 2, moe_ep: 2, moe_tp: 4 }, { experts: 8 }).ok, true);
  assert.match(validatePlan({ tp: 4, ep: 2, moe_ep: 2, moe_tp: 2 }, { experts: 8 }).errors[0], /专家域不闭合/);
  // moe_ep 是 ep 的细化：不能超过 ep
  assert.match(validatePlan({ tp: 4, ep: 1, moe_ep: 2 }, { experts: 8 }).errors[0], /moe_ep\(2\) 不能大于 ep\(1\)/);
  // 整除：experts % moe_ep == 0
  assert.match(validatePlan({ tp: 2, ep: 4, moe_ep: 3, moe_tp: 8 }, { experts: 8 }).errors[0], /整除/);
  assert.equal(validatePlan({ tp: 2, ep: 4, moe_ep: 4, moe_tp: 2 }, { experts: 8 }).ok, true);
  // 既有校验保留
  assert.match(validatePlan({ tp: 4, moe_ep: 0 }, {}).errors[0], /moeEp/);
  assert.match(validatePlan({ tp: 4, moe_ep: 16 }, { experts: 8 }).errors[0], /moe_ep 不能大于专家总数/);
  // vLLM：EP_SIZE = TP×DP（DP attention + EP）。ep 与 tp×dp 不一致即拒绝
  assert.equal(validatePlan({ tp: 2, dp: 2, ep: 4, attnMode: "dp" }, {}).ok, true);
  assert.match(validatePlan({ tp: 2, dp: 2, ep: 2, attnMode: "dp" }, {}).errors[0], /TP×DP=4/);
  // 混合 ETP 显式声明 moe_ep 时不做该约束（TRT-LLM 语义自洽）
  assert.equal(validatePlan({ tp: 2, dp: 2, ep: 2, attnMode: "dp", moe_ep: 2, moe_tp: 2 }, {}).ok, true);
});

test("declaredClassDivisor / declaredWeightBytesPerCard：四类 class 的单卡响应", () => {
  const plan = { tp: 4, ep: 8 };
  assert.equal(declaredClassDivisor("ep", plan), 8);
  assert.equal(declaredClassDivisor("tp", plan), 4);
  assert.equal(declaredClassDivisor("vocab", { ...plan, vocabParallel: true }), 4);
  assert.equal(declaredClassDivisor("vocab", { ...plan, vocabParallel: false }), 1);
  assert.equal(declaredClassDivisor("replicated", plan), 1);

  // 专家叶：3×E 矩阵一个 ep 组，totalBytes 按元素占比全额进组
  const expertGroups = [{ class: "ep", out: 1536, in: 3072, count: 256, matrices: 3 }];
  assert.equal(declaredWeightElements(expertGroups), 256 * 3 * 1536 * 3072);
  assert.equal(declaredWeightBytesPerCard(100, expertGroups, { tp: 4, ep: 2 }).bytes, 50);

  // 混合组（shared 融合形态）：ep 组 ÷moe_ep、tp 组 ÷tp，axis 取主导组
  const mixed = [
    { class: "ep", out: 1536, in: 3072, count: 256, matrices: 3 },
    { class: "tp", out: 6144, in: 3072, count: 1, matrices: 3 },
  ];
  const epElements = 256 * 3 * 1536 * 3072;
  const tpElements = 3 * 6144 * 3072;
  const total = 1e12;
  const projected = declaredWeightBytesPerCard(total, mixed, { tp: 4, ep: 2 });
  assert.equal(projected.bytes, total * (epElements / (epElements + tpElements)) / 2 + total * (tpElements / (epElements + tpElements)) / 4);
  assert.equal(projected.axis, "ep");
});

test("锚 3 终态：无声明带权叶返回 unknown（不猜归属），零权重叶按复制", () => {
  // P5：规则表已删。无声明 + 有权重 → unknown（诚实缺项，覆盖率护栏会把
  // 内置模型的这类叶挡在门外）；无权重（纯激活/共享叶）→ 复制、字节 0。
  const bare = weightBytesPerCard(100, { id: "decoder.0.self_attn.q_proj", attributes: {} }, { tp: 4 });
  assert.deepEqual(bare, { bytes: 100, divisor: 1, axis: "unknown" });
  const zero = weightBytesPerCard(0, { id: "decoder.0.mlp.combine", attributes: {} }, { tp: 4 });
  assert.deepEqual(zero, { bytes: 0, divisor: 1, axis: "replicated" });
});

// ---------------------------------------------------------------------------
// 锚 2：M2.7 EP 计划三方一致（聚合投影 = 闭式分解；expertWeightRange 接声明 count）
// ---------------------------------------------------------------------------

function buildMiniMaxM27() {
  const catalog = JSON.parse(fs.readFileSync(path.join(repoRoot, "models/catalog.json"), "utf8"));
  const entry = catalog.models.find((m) => m.model_id === "MiniMaxAI/MiniMax-M2.7");
  const raw = JSON.parse(fs.readFileSync(path.join(repoRoot, "models", entry.config_path), "utf8"));
  const normalized = normalizeConfig(raw);
  const resolved = resolveArchitecture(normalized, { modelId: entry.model_id });
  const structure = materializeModelStructure(createStructureIr({ network: buildNetwork(resolved, normalized), normalized, resolved }));
  return { normalized, structure };
}

/** 路由专家声明字节的闭式独立推导（不读树）：moe 层数 × E × 3 × EH × EI × 2B。
 *  层调度与 expectedWeightBytes（modelIdentities）同口径：layerSchedule 缺省时
 *  均匀 MoE 模型按全 moe 回退。 */
function routedDeclaredBytes(normalized) {
  const experts = normalized.experts || 0;
  const schedule = layerScheduleOf(normalized)
    || Array.from({ length: normalized.layers || 0 }, () => (experts ? "moe" : "dense"));
  const moeLayers = schedule.filter((kind) => kind === "moe").length + (normalized.mtpModules || 0);
  const eh = normalized.routedExpertHiddenSize || normalized.hiddenSize;
  const ei = normalized.moeIntermediateSize || normalized.intermediateSize;
  return moeLayers * experts * 3 * eh * ei * 2;
}

/** MoE 门控（router）与归一化族的复制字节：从投影本身取 replicated 类总量。
 *  P4：router 是 **replicated**（每卡各算一份完整门控，不切）——出处
 *  vLLM `qwen3_moe.py:167` `self.gate = ReplicatedLinear(...)` 与
 *  `fused_moe/router/gate_linear.py:18` `class GateLinear(ReplicatedLinear)`；
 *  norm 的 scale 同为复制。声明覆盖前二者都落在规则表第 4 条（tp>1 即 ÷tp，
 *  norm 靠路径正则第 2 条免除），router 属分片轴归属错误。
 *  这里用 tp=1 与 tp=N 的差额反解复制量，避免把 norm 闭式公式在测试里抄一遍
 *  （同义重复；norm 宽度知识已由锚 1 锁定）。 */
function replicatedBytes(structure, config) {
  const one = projectNodePlan({ graph: structure.graph, config, plan: { tp: 1, ep: 1 } }).stages[0].weightBytes;
  const two = projectNodePlan({ graph: structure.graph, config, plan: { tp: 2, ep: 1 } }).stages[0].weightBytes;
  // tp=2 时非复制类全部减半：two = replicated + (one − replicated)/2 ⇒ replicated = 2·two − one
  return 2 * two - one;
}

test("锚 2：M2.7 EP 计划每卡权重 = 专家块÷moe_ep + 复制类 + 其余÷tp（聚合投影与闭式分解一致）", () => {
  const { normalized, structure } = buildMiniMaxM27();
  const config = { ...normalized, layers: normalized.layers, experts: normalized.experts };
  const base = projectNodePlan({ graph: structure.graph, config, plan: { tp: 1, ep: 1 } });
  const natural = base.stages[0].weightBytes; // tp=1/ep=1 → 全复制，即树的声明驻留总量
  const routed = routedDeclaredBytes(normalized);
  const replicated = replicatedBytes(structure, config); // router + norm 族（P4 声明后不再 ÷tp）
  const rest = natural - routed - replicated;
  assert.ok(replicated > 0 && rest > 0, "复制类与其余类都应为正（分解口径检查）");

  // {tp:4, ep:2}：专家块 ÷moe_ep=2、复制类不切、其余（attention/dense GEMM）÷tp=4
  const projected = projectNodePlan({ graph: structure.graph, config, plan: { tp: 4, ep: 2 } });
  const expected = routed / 2 + replicated + rest / 4;
  assert.ok(Math.abs(projected.stages[0].weightBytes - expected) < 1e-6 * natural,
    `每卡权重 ${projected.stages[0].weightBytes} 与 专家块÷ep+复制类+其余÷tp ${expected} 不一致`);

  // 无 EP 的 DP：专家集合 ÷dp、矩阵 ÷tp（DP attention 复制其余）
  const dpPlan = projectNodePlan({ graph: structure.graph, config, plan: { tp: 2, dp: 2 } });
  const expectedDp = routed / (2 * 2) + replicated + rest / 2;
  assert.ok(Math.abs(dpPlan.stages[0].weightBytes - expectedDp) < 1e-6 * natural);

  // EP + DP attention（vLLM 组合语义 ep_size = tp×dp）：专家 ÷4、attention 复制
  const epDpPlan = projectNodePlan({ graph: structure.graph, config, plan: { tp: 2, dp: 2, ep: 4, attnMode: "dp" } });
  const expectedEpDp = routed / 4 + replicated + rest / 2;
  assert.ok(Math.abs(epDpPlan.stages[0].weightBytes - expectedEpDp) < 1e-6 * natural);
});

test("锚 2：expertWeightRange 接声明的专家数与组合 setDegree（平均/最坏区间一致）", () => {
  const { normalized, structure } = buildMiniMaxM27();
  const config = { ...normalized, layers: normalized.layers, experts: normalized.experts };
  const base = projectNodePlan({ graph: structure.graph, config, plan: { tp: 1, ep: 1 } });
  const natural = base.stages[0].weightBytes;
  const routed = routedDeclaredBytes(normalized);
  const experts = normalized.experts;

  const projected = projectNodePlan({ graph: structure.graph, config, plan: { tp: 1, ep: 3 } });
  // 平均 = 全复制 − 专家全量 + 专家÷3；最坏再换 ceil(E/3) 份专家
  const expectedAverage = natural - routed + routed / 3;
  const expectedWorst = expectedAverage - routed / 3 + (routed / experts) * Math.ceil(experts / 3);
  assert.ok(Math.abs(projected.stages[0].weightAverageBytes - expectedAverage) < 1e-6 * natural);
  assert.ok(Math.abs(projected.stages[0].weightWorstBytes - expectedWorst) < 1e-6 * natural);
  // expertWeightRange 本体与声明的 count/setDegree 同式（三方一致）
  const range = expertWeightRange(routed, experts, 3);
  assert.equal(range.expertsPerRank, Math.ceil(experts / 3));
  assert.equal(range.worstBytes, (routed / experts) * Math.ceil(experts / 3));
});

// ---------------------------------------------------------------------------
// 消费者 1/3：量化枚举消费声明组（专家 3×E 矩阵进枚举，排除留 bf16 基桶）
// ---------------------------------------------------------------------------

test("量化枚举消费 weightMatrices：fp8 下专家矩阵按声明组精确计（含 scale）", () => {
  const quant = { quant_method: "fp8", weight_block_size: [128, 128] };
  const group = { class: "ep", out: 256, in: 128, count: 4, matrices: 3 };
  const root = { id: "model", children: [
    { id: "decoder.0.mlp.expert_mlp", type: "operator", attributes: { operator_id: "fused_moe_mlp", weightMatrices: [group] }, children: [] },
  ] };
  const config = { hiddenSize: 8, layers: 1, vocabSize: 16, intermediateSize: 8, attentionHeads: 1, headDim: 2, kvHeads: 1, quantization_config: quant };
  const graph = toGraph(root);
  const base = graphWeightCapacity(graph).bytes;
  // 每矩阵 = 256·128·1B + ceil(256/128)·ceil(128/128)·4B(scale) = 32776
  const perMatrix = 256 * 128 + 2 * 1 * 4;
  const instances = 4 * 3;
  const expected = base - 256 * 128 * instances * 2 + perMatrix * instances;
  const cost = aggregateCost({ graph, config, batch: 1, sequence: 4, kvBytes: 2 });
  assert.equal(cost.memory.weightBytes, expected);
});

test("量化枚举排除语义对声明组同源：modules_to_not_convert 命中即整叶留 bf16 基桶", () => {
  const quant = { quant_method: "fp8", weight_block_size: [128, 128], modules_to_not_convert: [".*expert_mlp.*"] };
  const group = { class: "ep", out: 256, in: 128, count: 4, matrices: 3 };
  const root = { id: "model", children: [
    { id: "decoder.0.mlp.expert_mlp", type: "operator", attributes: { operator_id: "fused_moe_mlp", weightMatrices: [group] }, children: [] },
  ] };
  const config = { hiddenSize: 8, layers: 1, vocabSize: 16, intermediateSize: 8, attentionHeads: 1, headDim: 2, kvHeads: 1, quantization_config: quant };
  const graph = toGraph(root);
  const cost = aggregateCost({ graph, config, batch: 1, sequence: 4, kvBytes: 2 });
  assert.equal(cost.memory.weightBytes, graphWeightCapacity(graph).bytes);
});

// ---------------------------------------------------------------------------
// P6 接缝测试（MAINTENANCE 纪律 7）：UI 第四消费者（CostSummary PlanFields 产出的
// snake_case 计划）→ validatePlan → 声明分片，全链贯通。两端各有测试不等于链路
// 被覆盖——moe_tp/moe_ep/vocab_parallel 此前协议层支持但 UI 无入口，组合语义在
// 用户路径上触发不了。
// ---------------------------------------------------------------------------
test("P6 接缝：UI 输入的 snake_case 计划贯通 validatePlan 与声明分片", () => {
  // UI 输入形态：moe_tp/moe_ep 为字符串或 undefined（空输入），vocab_parallel 为布尔
  const uiPlan = { tp: "4", pp: "1", ep: "2", dp: "1", moe_tp: "4", moe_ep: "2", vocab_parallel: false };
  const plan = {
    tp: Number(uiPlan.tp), pp: Number(uiPlan.pp), ep: Number(uiPlan.ep), dp: Number(uiPlan.dp),
    moe_tp: uiPlan.moe_tp === undefined ? undefined : Number(uiPlan.moe_tp),
    moe_ep: uiPlan.moe_ep === undefined ? undefined : Number(uiPlan.moe_ep),
    vocab_parallel: uiPlan.vocab_parallel,
  };
  const checked = validatePlan(plan, { experts: 8 });
  assert.equal(checked.ok, true, checked.errors.join("; "));
  // 协议生效：moe_ep=2、moe_tp=4，专家域闭合 2×4 = 2×4
  assert.equal(checked.plan.moeEp, 2);
  assert.equal(checked.plan.moeTp, 4);
  // 声明分片消费该计划：专家叶 ÷(moe_ep×moe_tp)=8；vocabParallel=false 时 lm_head 复制
  const expert = weightBytesPerCard(800, { id: "decoder.0.mlp.expert_mlp", attributes: { weightMatrices: [{ class: "ep", out: 1, in: 1, count: 8, matrices: 3 }] } }, checked.plan);
  assert.equal(expert.bytes, 100);
  assert.equal(expert.axis, "ep");
  const head = weightBytesPerCard(400, { id: "lm_head.linear", attributes: { weightMatrices: [{ class: "vocab", out: 1, in: 1 }] } }, checked.plan);
  assert.equal(head.bytes, 400);
  assert.equal(head.axis, "vocab");
  // 校验失败的计划（专家域不闭合）被拒：UI 显示方案无效而不是静默算错
  const bad = validatePlan({ ...plan, moe_tp: 2 }, { experts: 8 });
  assert.equal(bad.ok, false);
});
