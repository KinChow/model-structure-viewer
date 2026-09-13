import assert from "node:assert/strict";
import test from "node:test";
import { activationTensorBytes, declaredElementsForHeader, graphWeightCapacity, kvBytesPerToken, linearStateBytesPerSequence, memoryBreakdown, tensorElements } from "../memory.js";
import { materializeStructureGraph } from "../../structure/graph/materializeStructureGraph.js";
import { aggregateCost } from "../aggregate.js";

// P7（步骤 7）：夹具 tree root 经 materializeStructureGraph 转 Graph IR。
const toGraph = (root) => materializeStructureGraph(root);

function cacheLeaf(id, { kv = 0, index = 0, state = 0, repeat } = {}) {
  return {
    id,
    attributes: {
      cache_kv_elements: kv,
      cache_index_elements: index,
      state_elements: state,
    },
    children: [],
    ...(repeat != null ? { repeat } : {}),
  };
}

test("F3 KV cache 使用 K/V 两份张量和 KV heads", () => {
  const graph = toGraph({
    id: "model",
    children: [
      { id: "layers", repeat: 2, children: [cacheLeaf("layers.0.sdpa", { kv: 2 * 4 * 8 })] },
    ],
  });
  assert.equal(kvBytesPerToken(graph, 2), 2 * 4 * 8 * 2 * 2);
});

test("F4 MLA KV 使用压缩 latent 与 rotary 分量", () => {
  const graph = toGraph({
    id: "model",
    children: [
      { id: "layers", repeat: 2, children: [cacheLeaf("layers.0.sdpa", { kv: 512 + 64 })] },
    ],
  });
  assert.equal(kvBytesPerToken(graph, 2), 2 * (512 + 64) * 2);
});

test("memory breakdown exposes token KV and request state separately", () => {
  const graph = toGraph({
    id: "model",
    children: [cacheLeaf("layers.0.sdpa", { kv: 2 })],
  });
  const result = memoryBreakdown({ weightBytes: 10, graph, batch: 1, tokens: 2,
    activationPeak: 3, runtimeConst: 4, commBuffer: 5, kvBytes: 1, bufferBytes: 0 });
  assert.deepEqual(result, { weightBytes: 10, bufferBytes: 0, kvBytes: 4, kvBytesPerToken: 2, stateBytes: 0, stateBytesPerSequence: 0, activationBytes: 3, runtimeBytes: 4, commBufferBytes: 5, totalBytes: 26 });
});

test("KDA recurrent and convolution state is request-scoped, not token KV", () => {
  const graph = toGraph({
    id: "model",
    children: [
      cacheLeaf("layers.0.state_update", { state: 80 }),
      cacheLeaf("layers.1.sdpa", { kv: 2 * 2 * 4 }),
    ],
  });
  assert.equal(linearStateBytesPerSequence(graph, 2), 160);
  assert.equal(kvBytesPerToken(graph, 2), 2 * 2 * 4 * 2);
  const result = memoryBreakdown({ weightBytes: 0, graph, batch: 2, tokens: 100, kvBytes: 2, activationPeak: 0, runtimeConst: 0, bufferBytes: 0 });
  assert.equal(result.stateBytes, 320);
  assert.equal(result.kvBytes, 6400);
});

test("Qwen3.5 GDN state uses separate key/value heads and dimensions", () => {
  const graph = toGraph({
    id: "model",
    children: [cacheLeaf("layers.0.state_update", { state: 24 })],
  });
  assert.equal(linearStateBytesPerSequence(graph, 2), 48);
});

test("MiniMax M3 sparse layers include index KV side cache", () => {
  const graph = toGraph({
    id: "model",
    children: [
      cacheLeaf("layers.0.sdpa", { kv: 2 * 2 * 4 }),
      cacheLeaf("layers.1.sparse_attention", { kv: 2 * 2 * 4, index: 1 * 2 }),
    ],
  });
  assert.equal(kvBytesPerToken(graph, 2), 68);
});

test("offline weight fallback multiplies folded layer repeats", () => {
  const root = { weight_shapes: {}, children: [{ repeat: 3, weight_shapes: {}, children: [
    { weight_shapes: { weight: [2, 2] }, dtype: "BF16", children: [] },
  ] }] };
  const result = aggregateCost({ graph: toGraph(root), config: { layers: 3, kvHeads: 1, headDim: 1 }, sequence: 1, activationPeak: 0, runtimeConst: 0 });
  assert.equal(result.memory.weightBytes, 3 * 2 * 2 * 2);
});

test("declaredElementsForHeader 按 checkpoint MTP key 计数，不近邻总量", () => {
  const graph = toGraph({
    id: "model",
    children: [
      {
        id: "layers.mlp",
        attributes: { weightMatrices: [{ out: 8, in: 4, count: 1, matrices: 1 }] },
        children: [],
      },
      {
        id: "mtp",
        type: "mtp",
        repeat: 0,
        attributes: {
          modules: 1,
          weightMatrices: [{ out: 8, in: 4, count: 1, matrices: 1 }],
        },
        children: [
          {
            id: "mtp.eh_proj",
            attributes: { weightMatrices: [{ out: 8, in: 4, count: 1, matrices: 1 }] },
            children: [],
          },
        ],
      },
    ],
  });
  const stem = graphWeightCapacity(graph, { includeMtp: false }).elements;
  const full = graphWeightCapacity(graph).elements;
  assert.equal(stem, 32);
  assert.ok(full > stem);
  assert.equal(declaredElementsForHeader(graph, { mtp_tensor_count: 0 }).declared, stem);
  assert.equal(declaredElementsForHeader(graph, { mtp_tensor_count: 0 }).includeMtp, false);
  assert.equal(declaredElementsForHeader(graph, { mtp_tensor_count: 12 }).declared, full);
  assert.equal(declaredElementsForHeader(graph, { mtp_tensor_count: 12 }).includeMtp, true);
  assert.equal(declaredElementsForHeader(graph, { parameterTotal: stem }).includeMtp, false);
  assert.equal(declaredElementsForHeader(graph, { parameterTotal: full }).includeMtp, true);
});

test("empty parameterCount falls back to node weights", () => {
  const root = { weight_shapes: { weight: [2, 2] }, dtype: "BF16", children: [] };
  const result = aggregateCost({ graph: toGraph(root), config: {}, parameterCount: {}, activationPeak: 0, runtimeConst: 0 });
  assert.equal(result.memory.weightBytes, 8);
});

test("动态数值 shape 分别解析普通张量和 attention 矩阵", () => {
  assert.equal(tensorElements([-1, -1, 4], { batch: 2, sequence: 3 }), 24);
  assert.equal(tensorElements([-1, -1, -1, -1], { batch: 2, sequence: 3, phase: "prefill", attentionHeads: 2 }), 36);
  assert.equal(tensorElements([-1, -1, -1, -1], { batch: 2, sequence: 3, phase: "decode", attentionHeads: 2 }), 12);
  assert.equal(activationTensorBytes([-1, -1, 4], { batch: 2, sequence: 3 }, 2), 48);
});

test("未知视觉输入尺寸不被当作文本 sequence", () => {
  assert.equal(tensorElements([-1, -1, -1, -1, -1], { batch: 1, sequence: 2048 }), 0);
});
