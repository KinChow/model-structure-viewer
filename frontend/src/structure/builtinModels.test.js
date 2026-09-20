import assert from "node:assert/strict";
import fs from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { normalizeConfig } from "./config/normalize.js";
import { resolveArchitecture } from "./registry/resolveArchitecture.js";
import { buildNetwork } from "./models/index.js";
import { createStructureIr } from "./ir/createStructureIr.js";
import { materializeModelStructure } from "./materializers/modelStructure.js";
import { formulaForOperator } from "./operators/formulas/index.js";
import { kvBytesPerToken, linearStateBytesPerSequence } from "../cost/memory.js";
import { aggregateCost } from "../cost/aggregate.js";
import { attentionScheduleOf } from "./layers/schedule.js";
import { recipeSharedExpertsAreFused } from "./archs/index.js";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../../..");

function childrenOf(graph, parentId) {
  return graph.nodes
    .filter((node) => node.parent_id === parentId)
    .sort((left, right) => (left.order || 0) - (right.order || 0) || left.id.localeCompare(right.id));
}

test("all built-in models have modules, formulas, and finite cost inputs", () => {
  const catalog = JSON.parse(fs.readFileSync(path.join(repoRoot, "models/catalog.json"), "utf8"));
  assert.equal(catalog.models.length, 60);
  for (const entry of catalog.models) {
    const config = JSON.parse(fs.readFileSync(path.join(repoRoot, "models", entry.config_path), "utf8"));
    const normalized = normalizeConfig(config);
    const resolved = resolveArchitecture(normalized, { modelId: entry.model_id });
    const structure = materializeModelStructure(createStructureIr({
      network: buildNetwork(resolved, normalized),
      normalized,
      resolved,
    }));
    const canonicalIds = structure.graph.nodes.map((node) => node.canonical_id);
    assert.equal(new Set(canonicalIds).size, canonicalIds.length, `${entry.model_id}: duplicate canonical graph ids`);
    assert.ok(structure.graph.edges.every((edge) => edge.source_canonical_id && edge.target_canonical_id), `${entry.model_id}: graph edge missing canonical endpoints`);
    const topLevel = childrenOf(structure.graph, structure.graph.root_id);
    assert.ok(topLevel.length > 0, `${entry.model_id}: empty module graph`);
    if (normalized.hasVision) {
      const vision = topLevel.find((node) => node.type === "vision-encoder");
      assert.ok(vision, `${entry.model_id}: vision config is missing from the structure`);
      const visionChildren = childrenOf(structure.graph, vision.id);
      assert.ok(visionChildren.length >= 2, `${entry.model_id}: vision tower has no detail modules`);
      const visionLayer0 = visionChildren.find((node) => (node.canonical_id || "").endsWith(".0"));
      assert.ok(childrenOf(structure.graph, visionLayer0?.id).length >= 8, `${entry.model_id}: vision layer has no operator detail`);
      assert.ok(structure.graph.edges.some((edge) => ["visual.patch_embed", "vision_tower.patch_embed"].includes(edge.source_canonical_id)), `${entry.model_id}: vision patch edge missing`);
    }
    for (const node of structure.graph.nodes) {
      if (node.type !== "operator") continue;
      assert.ok(node.attributes.formula, `${entry.model_id}: missing formula for ${node.attributes.operator_id}`);
      assert.ok(formulaForOperator(node.attributes.operator_id), `${entry.model_id}: unknown formula id ${node.attributes.operator_id}`);
      assert.ok(Array.isArray(node.input_shape), `${entry.model_id}: ${node.canonical_id} missing numeric input shape`);
      assert.ok(Array.isArray(node.output_shape), `${entry.model_id}: ${node.canonical_id} missing numeric output shape`);
    }
    const cost = aggregateCost({ graph: structure.graph, config: normalized, phase: "prefill", batch: 1, sequence: 128 });
    assert.equal(cost.computeComplete, true, `${entry.model_id}: ${cost.unknownComputePaths.join(", ")}`);
    assert.ok(Number.isFinite(cost.memory.weightBytes), `${entry.model_id}: invalid graph weight capacity`);
    assert.ok(Number.isFinite(kvBytesPerToken(structure.graph)), `${entry.model_id}: invalid KV cost`);
    assert.ok(Number.isFinite(linearStateBytesPerSequence(structure.graph)), `${entry.model_id}: invalid recurrent state cost`);
    if (entry.model_id === "moonshotai/Kimi-K3") {
      assert.equal(resolved.architecture, "KimiK3ForConditionalGeneration");
      const attention = attentionScheduleOf(normalized) || [];
      assert.deepEqual(attention.filter((kind) => kind === "linear").length, 69);
      assert.deepEqual(attention.filter((kind) => kind === "mla").length, 24);
      assert.equal(normalized.sharedExperts, 2);
      assert.equal(normalized.attnResBlockSize, 12);
      const nodes = structure.graph.nodes;
      assert.ok(nodes.some((node) => node.name === "Attention Residual"), "Kimi-K3: missing per-layer AttnRes module");
      assert.ok(nodes.some((node) => node.name === "Output Attention Residual"), "Kimi-K3: missing output AttnRes module");
      assert.ok(nodes.some((node) => node.name === "MLA output gate"), "Kimi-K3: missing MLA output gate");
      assert.ok(nodes.some((node) => node.canonical_id.endsWith(".block_sparse_moe.shared_experts")), "Kimi-K3: missing shared experts");
      // P3 fused shared expert：checkpoint 取证（models/moonshotai/Kimi-K3/index-summary.json
      // 每个 MoE 层只有 shared_experts.{gate,up,down}_proj.weight 各一个，共 92×3=276 个
      // 张量；modeling_kimi_linear.py:797-801 `intermediate_size =
      // moe_intermediate_size * num_shared_experts` 后实例化**单个** KimiMLP）——
      // 融合形态就是"一个更宽的 MLP"，不是 n 份专家，也不涉及 ep 亲和。
      const sharedGate = nodes.find((node) => node.canonical_id.endsWith(".block_sparse_moe.shared_experts.gate_proj"));
      assert.ok(sharedGate, "Kimi-K3: missing fused shared expert gate projection");
      assert.equal(normalized.sharedExpertIntermediateSize, normalized.moeIntermediateSize * normalized.sharedExperts);
      assert.deepEqual(
        sharedGate.attributes.weightMatrices,
        [{
          class: "tp",
          out: normalized.sharedExpertIntermediateSize,
          in: normalized.hiddenSize,
          count: 1,
          matrices: 1,
          shape: [normalized.sharedExpertIntermediateSize, normalized.hiddenSize],
          split: "output",
        }],
        "Kimi-K3: fused shared expert 声明应为单组 tp（模块宽 = moeI×n_shared，count=1，gate 沿 output 切）",
      );
      assert.equal(recipeSharedExpertsAreFused(normalized), true);
    }
  }
});

// P4 守卫：MoE 分组受限 top-k 与上游 config 的 n_group/topk_group 保持一致。
// 锁住 grouped_topk 建模——防止"折叠 builder + 漏建区分字段"重新出现（如 K2.5 vs V3）。
test("grouped top-k routing conforms to upstream n_group/topk_group", () => {
  const catalog = JSON.parse(fs.readFileSync(path.join(repoRoot, "models/catalog.json"), "utf8"));
  for (const entry of catalog.models) {
    const config = JSON.parse(fs.readFileSync(path.join(repoRoot, "models", entry.config_path), "utf8"));
    const tc = config.text_config || config;
    const nGroup = tc.n_group ?? config.n_group ?? tc.num_expert_group ?? config.num_expert_group;
    const topkGroup = tc.topk_group ?? config.topk_group;
    const normalized = normalizeConfig(config);
    const resolved = resolveArchitecture(normalized, { modelId: entry.model_id });
    const structure = materializeModelStructure(createStructureIr({
      network: buildNetwork(resolved, normalized), normalized, resolved,
    }));
    const topkNodes = structure.graph.nodes.filter(
      (n) => n.type === "operator" && n.attributes.operator_id === "topk",
    );
    if (topkNodes.length === 0) continue; // dense / non-MoE model
    const grouped = topkNodes.filter((n) => n.attributes.topk_method === "group_limited_topk");
    if (nGroup > 1) {
      assert.ok(grouped.length > 0, `${entry.model_id}: n_group=${nGroup}>1 但 topk 未标 group_limited_topk`);
      for (const n of grouped) {
        assert.equal(n.attributes.num_expert_group, nGroup, `${entry.model_id}: num_expert_group 应=${nGroup}`);
        assert.equal(n.attributes.topk_group, topkGroup, `${entry.model_id}: topk_group 应=${topkGroup}`);
      }
    } else {
      assert.equal(grouped.length, 0, `${entry.model_id}: n_group=${nGroup} 不应标 group_limited_topk`);
    }
  }
});

// P3-A 守卫：MoE 模型（config 有 experts）的解码层必须真的含 MoE 层，不能被 builder 的
// defaultLayerKind 兜底误渲染成全 dense（glm5_next/kimi_k3 曾硬编码 "dense" 的隐患）。
// defaultLayerKind 已收敛为 decoderStack 的 config 推导（experts?"moe":"dense"），本守卫锁死该行为。
test("MoE models render MoE layers (defaultLayerKind not silently dense)", () => {
  const catalog = JSON.parse(fs.readFileSync(path.join(repoRoot, "models/catalog.json"), "utf8"));
  for (const entry of catalog.models) {
    const config = JSON.parse(fs.readFileSync(path.join(repoRoot, "models", entry.config_path), "utf8"));
    const normalized = normalizeConfig(config);
    if (!(normalized.experts > 0)) continue; // 非 MoE 模型跳过
    const resolved = resolveArchitecture(normalized, { modelId: entry.model_id });
    const structure = materializeModelStructure(createStructureIr({
      network: buildNetwork(resolved, normalized), normalized, resolved,
    }));
    const hasMoe = structure.graph.nodes.some(
      (n) => n.type === "operator" && n.attributes.operator_id === "fused_moe_mlp",
    );
    assert.ok(hasMoe, `${entry.model_id}: 有 ${normalized.experts} experts 却未渲染出任何 MoE 层（疑似 defaultLayerKind 兜底成全 dense）`);
  }
});

// checkpoint 张量对账守卫（deepseek-gate-bias-attn-sink）：把三个此前 golden 与实跑
// 都守不住的叶（attn_sink、e_score_correction_bias、bias_vl）钉成「命中即必现、未命中即
// 不得现」。判据源自 SGLang/checkpoint：attn_sink ⇔ dsv4 注意力（V4/V4.1，deepseek_v4.py:708）；
// e_score_correction_bias ⇔ topk_method==noaux_tc 或 sigmoid 路由（deepseek_v2.py:492）；
// bias_vl ⇔ 路由修正 bias × 视觉塔 × 存在 compress_ratios（config 派生 routerBiasVl，SGLang 源码无此符号）。
test("checkpoint 叶对账：attn_sink / e_score_correction_bias / bias_vl 命中即必现、未命中不得现", () => {
  const catalog = JSON.parse(fs.readFileSync(path.join(repoRoot, "models/catalog.json"), "utf8"));
  const DSV4_ATTN = new Set(["dsv4_sparse_mla", "dsv4_compressed_attention", "dsv4_swa_attention"]);
  const hasParam = (nodes, pd) => nodes.some(
    (n) => n.type === "operator" && (n.attributes.weightMatrices || []).some((m) => m.param_dtype === pd),
  );
  for (const entry of catalog.models) {
    const config = JSON.parse(fs.readFileSync(path.join(repoRoot, "models", entry.config_path), "utf8"));
    const normalized = normalizeConfig(config);
    const resolved = resolveArchitecture(normalized, { modelId: entry.model_id });
    const structure = materializeModelStructure(createStructureIr({
      network: buildNetwork(resolved, normalized), normalized, resolved,
    }));
    const nodes = structure.graph.nodes;
    // attn_sink 必须与每个 dsv4 注意力叶一一对应（无一遗漏；非 dsv4 不得出现）。
    const dsv4Attn = nodes.filter((n) => n.type === "operator" && DSV4_ATTN.has(n.attributes.operator_id));
    const sinkNodes = nodes.filter(
      (n) => n.type === "operator" && (n.attributes.weightMatrices || []).some((m) => m.param_dtype === "attn_sink"),
    );
    assert.equal(sinkNodes.length, dsv4Attn.length,
      `${entry.model_id}: attn_sink 应与 dsv4 注意力叶一一对应（应 ${dsv4Attn.length}，实得 ${sinkNodes.length}）`);
    // e_score_correction_bias ⇔ noaux_tc；bias_vl ⇔ routerBiasVl。
    assert.equal(hasParam(nodes, "router_correction_bias"), Boolean(normalized.routerCorrectionBias),
      `${entry.model_id}: e_score_correction_bias 存在性应与 noaux_tc(${Boolean(normalized.routerCorrectionBias)}) 一致`);
    assert.equal(hasParam(nodes, "router_bias_vl"), Boolean(normalized.routerBiasVl),
      `${entry.model_id}: bias_vl 存在性应与 routerBiasVl(${Boolean(normalized.routerBiasVl)}) 一致`);
    // compressor ape（checkpoint position_bias）：仅 DeepSeek-V4 嵌套 compressor 架构有，V4.1 flat 无。
    const hasCompressor = nodes.some((n) => n.type === "operator" && n.attributes.operator_id === "mla_kv_compress");
    const expectApe = resolved.architecture === "DeepseekV4ForCausalLM" && hasCompressor;
    assert.equal(hasParam(nodes, "compressor_ape"), expectApe,
      `${entry.model_id}: compressor_ape 存在性应与「DeepseekV4 嵌套 compressor」(${expectApe}) 一致`);
    // PLE inject（qwen4_exp）：key_proj 输出宽 = hidden·hc_count（含 hc 因子），且 3 个 grouped RMSNorm——
    // 锁死修正后的结构，防止回退到旧的 [2·ple_embed] 合并投影 + 单 norm（漏 hc 因子/value/2 norms）。
    const pleInject = nodes.find((n) => n.type === "operator" && /(^|\.)inject$/.test(n.canonical_id || "") && n.attributes.operator_id === "ple");
    if (pleInject) {
      const wm = pleInject.attributes.weightMatrices || [];
      const hcHidden = (normalized.hiddenSize || 0) * (normalized.hyperConnectionCount || 1);
      const norms = wm.filter((m) => Array.isArray(m.shape) && m.shape.length === 1 && m.shape[0] === hcHidden);
      assert.equal(norms.length, 3, `${entry.model_id}: PLE 应有 3 个 [hidden·hc=${hcHidden}] RMSNorm，实得 ${norms.length}`);
      assert.ok(wm.some((m) => Array.isArray(m.shape) && m.shape[0] === hcHidden && m.shape[1] === (normalized.pleEmbedDim || 0)),
        `${entry.model_id}: PLE key_proj 输出宽应=hidden·hc=${hcHidden}（含 hc 因子），输入=ple_embed`);
    }
  }
});
