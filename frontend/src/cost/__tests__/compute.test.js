import assert from "node:assert/strict";
import test from "node:test";
import { attentionMacs, computeNodeCosts, linearAttentionMacs, linearMacs } from "../compute.js";

test("packed qweight is unknown without logical shape metadata", () => {
  assert.equal(linearMacs({ weight_shapes: { qweight: [4, 1] } }, { batch: 1, sequence: 1 }), null);
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
  assert.equal(computeNodeCosts(node, { attentionHeads: 2, headDim: 4 }, { batch: 1, sequence: 2 })[0].macs, 16);
});

test("父节点和范围子节点同时有 repeat 时只计算一次范围倍数", () => {
  const root = { repeat: 4, children: [{ id: "decoder.0", repeat: 4, children: [{ weight_shapes: { weight: [2, 2] }, dtype: "BF16", children: [] }] }] };
  const rows = computeNodeCosts(root, {}, { batch: 1, sequence: 1 });
  assert.equal(rows[2].macs, 16);
  assert.equal(rows[2].multiplier, 4);
});
