import assert from "node:assert/strict";
import test from "node:test";
import { activationTensorBytes, bytesPerDtype, cacheAccountingFromGraph, declaredElementsForHeader, draftKvBytesPerToken, draftWeightBytes, graphWeightCapacity, kvBytesPerToken, linearStateBytesPerSequence, memoryBreakdown, tensorElements } from "../memory.js";
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
      state_elements: state
    },
    children: [],
    ...(repeat != null ? { repeat } : {})
  };
}

test("F3 KV cache 使用 K/V 两份张量和 KV heads", () => {
  const graph = toGraph({
    id: "model",
    children: [
      { id: "layers", repeat: 2, children: [cacheLeaf("layers.0.sdpa", { kv: 2 * 4 * 8 })] },
    ]
  });
  assert.equal(kvBytesPerToken(graph, 2), 2 * 4 * 8 * 2 * 2);
});

test("F4 MLA KV 使用压缩 latent 与 rotary 分量", () => {
  const graph = toGraph({
    id: "model",
    children: [
      { id: "layers", repeat: 2, children: [cacheLeaf("layers.0.sdpa", { kv: 512 + 64 })] },
    ]
  });
  assert.equal(kvBytesPerToken(graph, 2), 2 * (512 + 64) * 2);
});

test("memory breakdown exposes token KV and request state separately", () => {
  const graph = toGraph({
    id: "model",
    children: [cacheLeaf("layers.0.sdpa", { kv: 2 })]
  });
  const result = memoryBreakdown({ weightBytes: 10, graph, batch: 1, tokens: 2, kvBytes: 1, bufferBytes: 0 });
  assert.equal(result.weightBytes, 10);
  assert.equal(result.kvBytes, 4);
  assert.equal(result.mainKvBytes, 4);
  assert.equal(result.draftKvBytes, 0);
  assert.equal(result.totalBytes, 14);
});

test("KDA recurrent and convolution state is request-scoped, not token KV", () => {
  const graph = toGraph({
    id: "model",
    children: [
      cacheLeaf("layers.0.state_update", { state: 80 }),
      cacheLeaf("layers.1.sdpa", { kv: 2 * 2 * 4 }),
    ]
  });
  assert.equal(linearStateBytesPerSequence(graph, 2), 160);
  assert.equal(kvBytesPerToken(graph, 2), 2 * 2 * 4 * 2);
  const result = memoryBreakdown({ weightBytes: 0, graph, batch: 2, tokens: 100, kvBytes: 2, bufferBytes: 0 });
  assert.equal(result.stateBytes, 320);
  assert.equal(result.kvBytes, 6400);
});

test("Qwen3.5 GDN state uses separate key/value heads and dimensions", () => {
  const graph = toGraph({
    id: "model",
    children: [cacheLeaf("layers.0.state_update", { state: 24 })]
  });
  assert.equal(linearStateBytesPerSequence(graph, 2), 48);
});

test("MiniMax M3 sparse layers include index KV side cache", () => {
  const graph = toGraph({
    id: "model",
    children: [
      cacheLeaf("layers.0.sdpa", { kv: 2 * 2 * 4 }),
      cacheLeaf("layers.1.sparse_attention", { kv: 2 * 2 * 4, index: 1 * 2 }),
    ]
  });
  assert.equal(kvBytesPerToken(graph, 2), 68);
});

test("offline weight fallback multiplies folded layer repeats", () => {
  const root = { weight_shapes: {}, children: [{ repeat: 3, weight_shapes: {}, children: [
    { weight_shapes: { weight: [2, 2] }, dtype: "BF16", children: [] },
  ] }] };
  const result = aggregateCost({ graph: toGraph(root), config: { layers: 3, kvHeads: 1, headDim: 1 }, sequence: 1});
  assert.equal(result.memory.weightBytes, 3 * 2 * 2 * 2);
});

test("declaredElementsForHeader 按 checkpoint MTP key 计数，不近邻总量", () => {
  const graph = toGraph({
    id: "model",
    children: [
      {
        id: "layers.mlp",
        attributes: { weightMatrices: [{ out: 8, in: 4, count: 1, matrices: 1 }] },
        children: []
      },
      {
        id: "mtp",
        type: "mtp",
        repeat: 0,
        attributes: {
          modules: 1,
          weightMatrices: [{ out: 8, in: 4, count: 1, matrices: 1 }]
        },
        children: [
          {
            id: "mtp.eh_proj",
            attributes: { weightMatrices: [{ out: 8, in: 4, count: 1, matrices: 1 }] },
            children: []
          },
        ]
      },
    ]
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
  const result = aggregateCost({ graph: toGraph(root), config: {}, parameterCount: {}});
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

// C6：MTP/投机草稿常驻 KV/token —— 直接读图里草稿层(mtp/dspark 子树)的 cache 叶，逐 dtype 累加。
// 用 root_id + parent_id 的真实图形状，让 walkStructure 能下降到草稿注意力叶。
const draftGraph = (attn) => ({
  root_id: "root",
  nodes: [
    { id: "root", parent_id: null, type: "model" },
    { id: "mtp", parent_id: "root", type: "mtp", repeat: 0, attributes: { modules: 1 } },
    { id: "mtp.layer.self_attn", parent_id: "mtp", type: "operator", attributes: attn },
  ],
});
test("draftKvBytesPerToken：无草稿 cache 叶返回 0（含 dsv4 未实装 draft attn）", () => {
  assert.equal(draftKvBytesPerToken({ root_id: "root", nodes: [{ id: "root", parent_id: null, type: "model" }] }, {}, 2), 0);
  // 只有 mtp 汇总节点、无展开的 draft 注意力 cache 叶（dsv4/V4.1 情形）→ 0
  assert.equal(draftKvBytesPerToken({ root_id: "root", nodes: [{ id: "root", parent_id: null }, { id: "mtp", parent_id: "root", type: "mtp" }] }, {}, 2), 0);
});
test("draftKvBytesPerToken：MLA 草稿层从图读取，draftTokens 叠加 verify 窗口", () => {
  const g = draftGraph({ cache_kv_elements: 576 }); // MLA latent 512+64
  assert.equal(draftKvBytesPerToken(g, {}, 2), 576 * 2); // 1152
  assert.equal(draftKvBytesPerToken(g, {}, 2, { draftTokens: 6 }), 576 * 2 * 7);
});
test("draftKvBytesPerToken：压缩(dsv4) draft 叶按逐 dtype 边际口径", () => {
  // 已实装的压缩 draft 叶（growth + 亚字节 dtype）→ 按 F4 字节算，不再失真
  const g = draftGraph({ cache_kv_dtype: "F4_E4M3S16", cache_kv_growth_elements: 32, cache_index_dtype: "F4_E8M0S32", cache_index_growth_elements: 32 });
  const expect = 32 * bytesPerDtype("F4_E4M3S16", 2) + 32 * bytesPerDtype("F4_E8M0S32", 2);
  assert.equal(draftKvBytesPerToken(g, {}, 2), expect);
});

const weightSplitGraph = () => ({
  root_id: "root",
  nodes: [
    { id: "root", parent_id: null, type: "model" },
    { id: "backbone.mlp", parent_id: "root", type: "operator", attributes: { weightMatrices: [{ out: 100, in: 100 }] } },
    { id: "mtp", parent_id: "root", type: "mtp", repeat: 0, attributes: { modules: 1 } },
    { id: "mtp.proj", parent_id: "mtp", type: "operator", attributes: { weightMatrices: [{ out: 10, in: 10 }] } },
  ],
});

test("draftWeightBytes：草稿子树权重按图占比拆分（含 total 摊分）", () => {
  const g = weightSplitGraph();
  // 图绝对：草稿 10×10×2=200；主干 100×100×2=20000；图总 20200
  assert.equal(draftWeightBytes(g), 200);
  // 传实际总量=图总量 → 等于图绝对
  assert.equal(draftWeightBytes(g, 20200), 200);
  // 按草稿占比 200/20200 摊到给定 checkpoint 总量
  assert.equal(draftWeightBytes(g, 40400), 400);
});

test("draftWeightBytes：无 mtp/dspark 子树返回 0（主干逐字节不变）", () => {
  const g = {
    root_id: "root",
    nodes: [
      { id: "root", parent_id: null, type: "model" },
      { id: "backbone.mlp", parent_id: "root", type: "operator", attributes: { weightMatrices: [{ out: 100, in: 100 }] } },
    ],
  };
  assert.equal(draftWeightBytes(g), 0);
  assert.equal(draftWeightBytes(g, 20000), 0);
});

test("framework accounting keeps private draft KV separate and shared pools unique", () => {
  const graph = {
    root_id: "root",
    nodes: [
      { id: "root", parent_id: null, type: "model" },
      { id: "target", parent_id: "root", type: "operator", attributes: {
        cache_kv_elements: 10, cache_pool_id: "target",
      } },
      { id: "mtp", parent_id: "root", type: "mtp", repeat: 0, attributes: { modules: 1 } },
      { id: "mtp.attn", parent_id: "mtp", type: "operator", attributes: {
        cache_kv_elements: 3, cache_pool_id: "draft",
      } },
    ],
  };
  const neutral = cacheAccountingFromGraph(graph, { kvBytes: 2, batch: 1, tokens: 4 });
  assert.equal(neutral.mainKvBytes, 80);
  assert.equal(neutral.draftKvBytes, 24);
  assert.equal(neutral.totalKvBytes, 104);

  const shared = {
    ...graph,
    nodes: graph.nodes.map((node) => node.id === "mtp.attn"
      ? { ...node, attributes: { ...node.attributes, cache_pool_id: "target", cache_pool_shared: true } }
      : node),
  };
  const accounting = cacheAccountingFromGraph(shared, { kvBytes: 2, batch: 1, tokens: 4 });
  assert.equal(accounting.sharedKvBytes, 80);
  assert.equal(accounting.draftKvBytes, 0);
  assert.equal(accounting.totalKvBytes, accounting.sharedKvBytes);
});
