import assert from "node:assert/strict";
import test from "node:test";
import { aggregateNodeCosts, computeNodeCosts, nodeMacs } from "../compute.js";
import { normalizeConfig } from "../../structure/config/normalize.js";
import { materializeStructureGraph } from "../../structure/graph/materializeStructureGraph.js";
import { aggregateCost } from "../aggregate.js";

// P7（步骤 7）：cost 链只遍历 Graph IR——夹具 tree root 统一经
// materializeStructureGraph 转图（节点 canonical_id / 乘子语义不变）。
const toGraph = (root) => materializeStructureGraph(root);

test("packed qweight is unknown without logical shape metadata", () => {
  assert.equal(nodeMacs({ weight_shapes: { qweight: [4, 1] } }, {}, { batch: 1, sequence: 1 }), null);
});

test("unknown linear MACs remain unknown in model totals", () => {
  const node = {
    type: "operator",
    attributes: { operator_id: "linear" },
    weight_shapes: { qweight: [4, 1] },
    children: [],
  };
  const result = aggregateCost({ graph: toGraph({ children: [node] }), config: {}, activationPeak: 0, runtimeConst: 0 });

  assert.equal(result.totalMacs, null);
  assert.equal(result.totalFlops, null);
  assert.equal(result.computeComplete, false);
  assert.deepEqual(result.unknownComputePaths, ["root.0"]);
});

test("checkpoint skeleton linear leaves contribute to model totals", () => {
  const node = {
    id: "model.layers.0.mlp.gate_proj",
    type: "module",
    weight_shapes: { weight: [4, 2] },
    children: [],
  };
  const result = aggregateCost({ graph: toGraph({ children: [node] }), config: {}, batch: 1, sequence: 3, activationPeak: 0, runtimeConst: 0 });

  assert.equal(result.totalMacs, 24);
  assert.equal(result.computeComplete, true);
  assert.equal(result.nodes[1].macs_source, "formula"); // W5-1：weight_shapes 由 counts 表直接消费，来源类目归并
});

test("template linear operators derive MACs from numeric tensor shapes", () => {
  const node = { type: "operator", attributes: { operator_id: "linear" }, input_shape: [-1, -1, 4], output_shape: [-1, -1, 8], children: [] };
  assert.equal(nodeMacs(node, {}, { batch: 1, sequence: 3, phase: "prefill" }), 96);
  const result = aggregateCost({ graph: toGraph({ children: [node, { type: "normalization", output_shape: [-1, -1, 8], children: [] }] }), config: { hiddenSize: 4, vocabSize: 0, tieWordEmbeddings: true }, sequence: 3, activationPeak: 0, runtimeConst: 0 });
  assert.equal(result.totalMacs, 96);
  assert.equal(result.macsPerToken, 32);
  assert.equal(result.totalFlops, 192);
  assert.equal(result.macsSources["formula"], 1);
  assert.equal(result.nodes.find((row) => row.node.type === "normalization").macs_source, "not-compute");
});

test("二维专家投影按逻辑输入输出宽度估算 MACs", () => {
  const node = { type: "operator", attributes: { operator_id: "linear" }, input_shape: [-1, -1, 8], output_shape: [-1, 4], children: [] };
  assert.equal(nodeMacs(node, {}, { batch: 2, sequence: 3, phase: "prefill" }), 192);
});

test("父节点 lens 可以汇总叶子成本，但模型总量不重复计费", () => {
  const root = { id: "root", children: [{ id: "decoder", children: [{ id: "decoder.linear", type: "operator", attributes: { operator_id: "linear" }, input_shape: [-1, -1, 4], output_shape: [-1, -1, 8], children: [] }] }] };
  const rows = computeNodeCosts(toGraph(root), {}, { batch: 1, sequence: 2, phase: "prefill" });
  const aggregate = aggregateNodeCosts(rows);
  assert.equal(aggregate.find((row) => row.path === "root.0").aggregate_macs, 64);
  assert.equal(aggregate.find((row) => row.path === "root").aggregate_macs, 64);
  assert.equal(rows.find((row) => row.path === "root").compute_macs, 0);
});

test("Graph IR 父节点只做汇总，不能把父级 attention 再计一次", () => {
  const graph = {
    root_id: "root",
    nodes: [
      { id: "root", canonical_id: "root", parent_id: null, order: 0, type: "model", name: "model" },
      { id: "root.0", canonical_id: "decoder.0.self_attn", parent_id: "root", order: 0, type: "attention", name: "GQA Attention", attributes: { attention_kind: "gqa" } },
      { id: "root.0.0", canonical_id: "decoder.0.self_attn.scores", parent_id: "root.0", order: 0, type: "operator", name: "attention scores", attributes: { operator_id: "matmul" }, input_shape: [-1, -1, 2, 4], output_shape: [-1, 2, -1, -1] },
      { id: "root.0.1", canonical_id: "decoder.0.self_attn.context", parent_id: "root.0", order: 1, type: "operator", name: "weighted value", attributes: { operator_id: "matmul" }, input_shape: [-1, 2, -1, -1], output_shape: [-1, -1, 2, 6] },
    ],
    edges: [],
  };
  const result = aggregateCost({
    graph,
    config: { attentionHeads: 2, headDim: 4, valueHeadDim: 6 },
    batch: 1,
    sequence: 3,
    activationPeak: 0,
    runtimeConst: 0,
  });
  // W3-①因果口径手算：T=S=3 → 每头可见对数 = 1+2+3 = 6，两头共 12 对。
  // scores = 12·headDim(4) = 48；context = 12·valueDim(6) = 72；合计 120。
  // （W3 前两相位通吃 T·S=9 对/头，算得 180。）
  assert.equal(result.totalMacs, 120);
  assert.equal(result.nodes.find((row) => row.path === "root.0").compute_macs, 0);
});

test("父有 counts 用父、子孙不进账（§2.4 计费主语）", () => {
  const graph = {
    root_id: "root",
    nodes: [
      { id: "root", canonical_id: "root", parent_id: null, order: 0, type: "model", name: "model" },
      {
        id: "root.0",
        canonical_id: "decoder.0.self_attn.sdpa",
        parent_id: "root",
        order: 0,
        type: "operator",
        name: "SDPA attention",
        attributes: { operator_id: "sdpa_attention", attention_kind: "gqa" },
      },
      {
        id: "root.0.0",
        canonical_id: "decoder.0.self_attn.sdpa.scores",
        parent_id: "root.0",
        order: 0,
        type: "operator",
        name: "attention scores",
        attributes: { operator_id: "matmul" },
        input_shape: [-1, -1, 2, 4],
        output_shape: [-1, 2, -1, -1],
      },
      {
        id: "root.0.1",
        canonical_id: "decoder.0.self_attn.sdpa.context",
        parent_id: "root.0",
        order: 1,
        type: "operator",
        name: "weighted value",
        attributes: { operator_id: "matmul" },
        input_shape: [-1, 2, -1, -1],
        output_shape: [-1, -1, 2, 6],
      },
    ],
    edges: [],
  };
  const result = aggregateCost({
    graph,
    config: { attentionHeads: 2, headDim: 4, valueHeadDim: 6, kvHeads: 2 },
    batch: 1,
    sequence: 3,
    activationPeak: 0,
    runtimeConst: 0,
  });
  const parent = result.nodes.find((row) => row.path === "root.0");
  const scores = result.nodes.find((row) => row.path === "root.0.0");
  const context = result.nodes.find((row) => row.path === "root.0.1");
  // 融合主语：12 对 × (4+6) = 120，与拆叶相加相同；子孙 compute 必须为 0。
  assert.equal(parent.compute_macs, 120);
  assert.equal(parent.macs_source, "formula");
  assert.equal(scores.compute_macs, 0);
  assert.equal(context.compute_macs, 0);
  assert.equal(result.totalMacs, 120);
});

test("参数无关叶节点不计入 MACs，KDA state 和短卷积保留维度公式", () => {
  const root = {
    children: [
      { type: "operator", name: "gated RMSNorm", attributes: { operator_id: "gated_rmsnorm" }, output_shape: [-1, -1, 8], children: [] },
      { type: "operator", name: "qkv causal short convolution", attributes: { operator_id: "causal_conv1d" }, output_shape: [-1, -1, 8], children: [] },
      { type: "operator", name: "KDA recurrent state", attributes: { operator_id: "gated_delta_attention" }, output_shape: [-1, -1, 8], children: [] },
    ],
  };
  const config = { linearAttentionMode: "qwen3_5", hiddenSize: 16, linearKeyHeads: 1, linearValueHeads: 1, linearKeyDim: 2, linearValueDim: 2, linearConvKernelSize: 3 };
  const result = aggregateCost({ graph: toGraph(root), config, batch: 1, sequence: 2, activationPeak: 0, runtimeConst: 0 });
  assert.equal(result.nodes[1].compute_macs, 0);
  assert.equal(result.nodes[2].compute_macs, 36);
  assert.equal(result.nodes[3].compute_macs, 24);
  assert.equal(result.totalMacs, 60);
  assert.equal(result.nodes.filter((row) => row.macs_source === "not-compute").length >= 1, true); // root 现标 aggregate（W5-1）
});

test("模板 MoE expert 叶节点按活跃专家和逻辑宽度估算 FFN MACs", () => {
  // N2-4 W-A：路由专家融合叶改独立 id fused_moe_mlp（对标 vLLM FusedMoE），
  // 计数语义与拆分前逐位相同（T·k·3·EH·EI）。
  const standard = {
    id: "decoder.0.mlp.expert_mlp",
    type: "operator",
    name: "expert MLP",
    attributes: { operator_id: "fused_moe_mlp" },
    children: [],
  };
  const latent = {
    id: "decoder.0.mlp.expert_mlp",
    type: "operator",
    name: "latent expert MLP",
    attributes: { operator_id: "fused_moe_mlp", latent_size: 2 },
    children: [],
  };
  const config = { hiddenSize: 4, intermediateSize: 6, experts: 8, expertsPerToken: 2 };
  // W5-1 语义修正：routed swiglu 按 k 全激活（T·k·3·EH·EI），旧链的 ·(k/E) 少乘 E
  // ——identity 收敛校准结论（39B≈官方 37B）。
  assert.equal(computeNodeCosts(toGraph(standard), config, { batch: 1, sequence: 3 })[0].compute_macs, 432);
  assert.equal(computeNodeCosts(toGraph(latent), config, { batch: 1, sequence: 3 })[0].compute_macs, 216);
});

test("F8 Linear MACs 区分 Prefill 的 B×T 与 Decode 的 B×1", () => {
  const node = { weight_shapes: { weight: [4, 2] } };
  assert.equal(nodeMacs(node, {}, { batch: 2, sequence: 3, phase: "prefill" }), 48);
  assert.equal(nodeMacs(node, {}, { batch: 2, sequence: 3, phase: "decode" }), 16);
});

test("F9 Attention core MACs 区分 Prefill 的 T² 与 Decode 的 T", () => {
  const config = { attentionHeads: 2, headDim: 4, valueHeadDim: 6 };
  const node = { type: "attention", attributes: { attention_kind: "gqa" }, id: "decoder.0.self_attn" };
  assert.equal(nodeMacs(node, config, { batch: 2, sequence: 3, phase: "prefill" }), 360);
  assert.equal(nodeMacs(node, config, { batch: 2, sequence: 3, phase: "decode" }), 120);
});

test("Qwen3.5 GDN MACs include qkvz/ba projections and value-head recurrent state", () => {
  const config = {
    linearAttentionMode: "qwen3_5",
    hiddenSize: 4,
    linearKeyHeads: 1,
    linearValueHeads: 2,
    linearKeyDim: 2,
    linearValueDim: 2,
    linearConvKernelSize: 3,
  };
  const node = { type: "attention", attributes: { attention_kind: "linear" }, id: "decoder.0.self_attn" };
  assert.equal(nodeMacs(node, config, { batch: 1, sequence: 5, phase: "prefill" }), 700);
  assert.equal(nodeMacs(node, config, { batch: 1, sequence: 5, phase: "decode" }), 140);
});

test("MiniMax M3 sparse attention MACs use selected blocks plus local/init blocks", () => {
  const node = { type: "attention", attributes: { attention_kind: "sparse" }, id: "text_decoder.3.self_attn" };
  const config = { modelType: "minimax_m3_vl", attentionHeads: 2, headDim: 3, sparseTopkBlocks: 2, sparseBlockSize: 4, sparseInitBlock: 1, sparseLocalBlock: 0 };
  assert.equal(computeNodeCosts(toGraph(node), config, { batch: 1, sequence: 5, phase: "prefill" })[0].compute_macs, 720);
});

test("F16 MoE expert fraction 逐层应用且不影响 dense 层", () => {
  const root = { children: [
    { id: "decoder.0.mlp.gate_proj", weight_shapes: { weight: [4, 2] }, children: [] },
    { id: "decoder.1.mlp.experts.0", weight_shapes: { weight: [4, 2] }, children: [] },
  ] };
  const rows = computeNodeCosts(toGraph(root), { experts: 8, expertsPerToken: 2, layerSchedule: ["dense", "moe"] }, { batch: 1, sequence: 1 });
  assert.equal(rows[1].compute_macs, 8);
  assert.equal(rows[2].compute_macs, 2);
});

test("F16 真实专家路径在缺少 layerSchedule 时仍使用活跃比例", () => {
  const root = { children: [{ id: "decoder.0.mlp.experts.0", weight_shapes: { weight: [4, 2] }, children: [] }] };
  const rows = computeNodeCosts(toGraph(root), { experts: 8, expertsPerToken: 2 }, { batch: 1, sequence: 1 });
  assert.equal(rows[1].compute_macs, 2);
});

test("layernorm 名称包含 attention 时不应误判为 attention 核心", () => {
  const node = { type: "normalization", name: "post attention layernorm", output_shape: [-1, -1, 8] };
  assert.equal(computeNodeCosts(toGraph(node), { attentionHeads: 2, headDim: 4 }, { batch: 1, sequence: 2 })[0].compute_macs, 0);
});

test("父节点和范围子节点同时有 repeat 时只计算一次范围倍数", () => {
  const root = { repeat: 4, children: [{ id: "decoder.0", repeat: 4, children: [{ weight_shapes: { weight: [2, 2] }, dtype: "BF16", children: [] }] }] };
  const rows = computeNodeCosts(toGraph(root), {}, { batch: 1, sequence: 1 });
  assert.equal(rows[2].compute_macs, 16);
  assert.equal(rows[2].multiplier, 4);
});

test("V3 用户可调 visionTokens：vision 域随 tokens 线性变化，文本域不变", () => {
  const config = normalizeConfig({
    text_config: { hidden_size: 4, num_attention_heads: 2 },
    vision_config: { hidden_size: 4, num_attention_heads: 2, num_position_embeddings: 4096, spatial_merge_size: 2 },
  });
  assert.equal(config.visionTokens, 1024); // normalize 推导兜底值
  const root = { children: [
    { id: "vision_tower.blocks.0.attn.proj", type: "operator", attributes: { operator_id: "linear", modality: "vision" }, weight_shapes: { weight: [4, 2] }, children: [] },
    { id: "model.layers.0.mlp.gate_proj", type: "operator", attributes: { operator_id: "linear" }, weight_shapes: { weight: [4, 2] }, children: [] },
  ] };
  const options = { batch: 1, sequence: 8, phase: "prefill" };
  const small = computeNodeCosts(toGraph(root), config, { ...options, visionTokens: 512 });
  const large = computeNodeCosts(toGraph(root), config, { ...options, visionTokens: 2048 });
  const byId = (rows, id) => rows.find((row) => row.node.id === id).compute_macs;
  // vision 域叶子：macs = 4*2*tokens，512→4096，2048→16384（线性 4×）
  assert.equal(byId(small, "vision_tower.blocks.0.attn.proj"), 4096);
  assert.equal(byId(large, "vision_tower.blocks.0.attn.proj"), 16384);
  // 文本域叶子只随 sequence，不受 visionTokens 影响
  assert.equal(byId(small, "model.layers.0.mlp.gate_proj"), 64);
  assert.equal(byId(large, "model.layers.0.mlp.gate_proj"), 64);
});
