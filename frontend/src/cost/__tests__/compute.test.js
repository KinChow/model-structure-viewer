import assert from "node:assert/strict";
import test from "node:test";
import { aggregateNodeCosts, attentionMacs, computeNodeCosts, linearAttentionMacs, linearMacs } from "../compute.js";
import { aggregateCost } from "../aggregate.js";

test("packed qweight is unknown without logical shape metadata", () => {
  assert.equal(linearMacs({ weight_shapes: { qweight: [4, 1] } }, { batch: 1, sequence: 1 }), null);
});

test("unknown linear MACs remain unknown in model totals", () => {
  const node = {
    type: "operator",
    attributes: { operator_id: "linear" },
    weight_shapes: { qweight: [4, 1] },
    children: [],
  };
  const result = aggregateCost({ root: { children: [node] }, config: {}, activationPeak: 0, runtimeConst: 0 });

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
  const result = aggregateCost({ root: { children: [node] }, config: {}, batch: 1, sequence: 3, activationPeak: 0, runtimeConst: 0 });

  assert.equal(result.totalMacs, 24);
  assert.equal(result.computeComplete, true);
  assert.equal(result.nodes[1].macs_source, "checkpoint-shape");
});

test("template linear operators derive MACs from numeric tensor shapes", () => {
  const node = { type: "operator", attributes: { operator_id: "linear" }, input_shape: [-1, -1, 4], output_shape: [-1, -1, 8], children: [] };
  assert.equal(linearMacs(node, { batch: 1, sequence: 3, phase: "prefill" }), 96);
  const result = aggregateCost({ root: { children: [node, { type: "normalization", output_shape: [-1, -1, 8], children: [] }] }, config: { hiddenSize: 4, vocabSize: 0, tieWordEmbeddings: true }, sequence: 3, activationPeak: 0, runtimeConst: 0 });
  assert.equal(result.totalMacs, 96);
  assert.equal(result.macsPerToken, 32);
  assert.equal(result.totalFlops, 192);
  assert.equal(result.macsSources["config-derived-shape"], 1);
  assert.equal(result.nodes.find((row) => row.node.type === "normalization").macs_source, "not-compute");
});

test("二维专家投影按逻辑输入输出宽度估算 MACs", () => {
  const node = { type: "operator", attributes: { operator_id: "linear" }, input_shape: [-1, -1, 8], output_shape: [-1, 4], children: [] };
  assert.equal(linearMacs(node, { batch: 2, sequence: 3, phase: "prefill" }), 192);
});

test("父节点 lens 可以汇总叶子成本，但模型总量不重复计费", () => {
  const root = { id: "root", children: [{ id: "decoder", children: [{ id: "decoder.linear", type: "operator", attributes: { operator_id: "linear" }, input_shape: [-1, -1, 4], output_shape: [-1, -1, 8], children: [] }] }] };
  const rows = computeNodeCosts(root, {}, { batch: 1, sequence: 2, phase: "prefill" });
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
  assert.equal(result.totalMacs, 180);
  assert.equal(result.nodes.find((row) => row.path === "root.0").compute_macs, 0);
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
  const result = aggregateCost({ root, config, batch: 1, sequence: 2, activationPeak: 0, runtimeConst: 0 });
  assert.equal(result.nodes[1].compute_macs, 0);
  assert.equal(result.nodes[2].compute_macs, 36);
  assert.equal(result.nodes[3].compute_macs, 24);
  assert.equal(result.totalMacs, 60);
  assert.equal(result.nodes.filter((row) => row.macs_source === "not-compute").length >= 2, true);
});

test("模板 MoE expert 叶节点按活跃专家和逻辑宽度估算 FFN MACs", () => {
  const standard = {
    id: "decoder.0.moe.expert_mlp",
    type: "operator",
    name: "expert MLP",
    attributes: { operator_id: "swiglu" },
    children: [],
  };
  const latent = {
    id: "decoder.0.moe.expert_mlp",
    type: "operator",
    name: "latent expert MLP",
    attributes: { operator_id: "swiglu", latent_size: 2 },
    children: [],
  };
  const config = { hiddenSize: 4, intermediateSize: 6, experts: 8, expertsPerToken: 2 };
  assert.equal(computeNodeCosts(standard, config, { batch: 1, sequence: 3 })[0].compute_macs, 54);
  assert.equal(computeNodeCosts(latent, config, { batch: 1, sequence: 3 })[0].compute_macs, 27);
});

test("F8 Linear MACs 区分 Prefill 的 B×T 与 Decode 的 B×1", () => {
  const node = { weight_shapes: { weight: [4, 2] } };
  assert.equal(linearMacs(node, { batch: 2, sequence: 3, phase: "prefill" }), 48);
  assert.equal(linearMacs(node, { batch: 2, sequence: 3, phase: "decode" }), 16);
});

test("F9 Attention core MACs 区分 Prefill 的 T² 与 Decode 的 T", () => {
  const config = { attentionHeads: 2, headDim: 4, valueHeadDim: 6 };
  assert.equal(attentionMacs(config, { batch: 2, sequence: 3, phase: "prefill" }), 360);
  assert.equal(attentionMacs(config, { batch: 2, sequence: 3, phase: "decode" }), 120);
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
  assert.equal(linearAttentionMacs(config, { batch: 1, sequence: 5, phase: "prefill" }), 700);
  assert.equal(linearAttentionMacs(config, { batch: 1, sequence: 5, phase: "decode" }), 140);
});

test("MiniMax M3 sparse attention MACs use selected blocks plus local/init blocks", () => {
  const node = { type: "attention", attributes: { attention_kind: "sparse" }, id: "text_decoder.3.self_attn" };
  const config = { modelType: "minimax_m3_vl", attentionHeads: 2, headDim: 3, sparseTopkBlocks: 2, sparseBlockSize: 4, sparseInitBlock: 1, sparseLocalBlock: 0 };
  assert.equal(computeNodeCosts(node, config, { batch: 1, sequence: 5, phase: "prefill" })[0].macs, 720);
});

test("F16 MoE expert fraction 逐层应用且不影响 dense 层", () => {
  const root = { children: [
    { id: "decoder.0.mlp.gate_proj", weight_shapes: { weight: [4, 2] }, children: [] },
    { id: "decoder.1.mlp.experts.0", weight_shapes: { weight: [4, 2] }, children: [] },
  ] };
  const rows = computeNodeCosts(root, { experts: 8, expertsPerToken: 2, layerSchedule: ["dense", "moe"] }, { batch: 1, sequence: 1 });
  assert.equal(rows[1].macs, 8);
  assert.equal(rows[2].macs, 2);
});

test("F16 真实专家路径在缺少 layerSchedule 时仍使用活跃比例", () => {
  const root = { children: [{ id: "decoder.0.mlp.experts.0", weight_shapes: { weight: [4, 2] }, children: [] }] };
  const rows = computeNodeCosts(root, { experts: 8, expertsPerToken: 2 }, { batch: 1, sequence: 1 });
  assert.equal(rows[1].macs, 2);
});

test("layernorm 名称包含 attention 时不应误判为 attention 核心", () => {
  const node = { type: "normalization", name: "post attention layernorm", output_shape: [-1, -1, 8] };
  assert.equal(computeNodeCosts(node, { attentionHeads: 2, headDim: 4 }, { batch: 1, sequence: 2 })[0].macs, 0);
});

test("父节点和范围子节点同时有 repeat 时只计算一次范围倍数", () => {
  const root = { repeat: 4, children: [{ id: "decoder.0", repeat: 4, children: [{ weight_shapes: { weight: [2, 2] }, dtype: "BF16", children: [] }] }] };
  const rows = computeNodeCosts(root, {}, { batch: 1, sequence: 1 });
  assert.equal(rows[2].macs, 16);
  assert.equal(rows[2].multiplier, 4);
});
