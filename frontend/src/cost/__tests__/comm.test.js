import assert from "node:assert/strict";
import test from "node:test";
import { expertAllToAllBytes, nodeCommunicationBytes, pdKvTransferBytes, pipelineP2PBytes, planCommunicationBytes, ringAllReduceBytes } from "../comm.js";
import { materializeStructureGraph } from "../../structure/graph/materializeStructureGraph.js";

// P7（步骤 7）：planCommunicationBytes 只收 Graph IR——夹具 tree root 统一转图。
const toGraph = (root) => materializeStructureGraph(root);

test("F11 TP ring all-reduce 每层两次时包含 2×(TP-1)/TP 系数", () => {
  assert.equal(ringAllReduceBytes({ batch: 2, tokens: 3, hidden: 4, bytesPerElement: 2, tp: 4 }), 144);
  assert.equal(ringAllReduceBytes({ hidden: 4, tp: 1 }), 0);
});

test("F12 EP all-to-all 使用 expertsPerToken 而不是专家总数", () => {
  assert.equal(expertAllToAllBytes({ batch: 2, tokens: 3, hidden: 4, expertsPerToken: 2, bytesPerElement: 2 }), 192);
});

test("PP P2P 按相邻 stage 边界计算", () => {
  assert.equal(pipelineP2PBytes({ batch: 1, tokens: 2, hidden: 4, bytesPerElement: 2, pp: 3 }), 32);
});

test("节点路径可识别 TP 与 EP 通信模块", () => {
  assert.equal(nodeCommunicationBytes({ id: "decoder.0.self_attn.o_proj" }, { hiddenSize: 4 }, { tp: 2 }, { batch: 1, tokens: 1, bytesPerElement: 2 }), 8);
  assert.equal(nodeCommunicationBytes({ id: "decoder.0.mlp.dispatch" }, { hiddenSize: 4, expertsPerToken: 2 }, { tp: 2, ep: 2 }, { batch: 1, tokens: 1, bytesPerElement: 2 }), 16);
  assert.equal(nodeCommunicationBytes({ id: "decoder.0.mlp.combine" }, { hiddenSize: 4, expertsPerToken: 2 }, { tp: 2, ep: 2 }, { batch: 1, tokens: 1, bytesPerElement: 2 }), 16);
  assert.equal(nodeCommunicationBytes({ id: "decoder.0.mlp.expert_mlp" }, { hiddenSize: 4, expertsPerToken: 2 }, { tp: 2, ep: 2 }, { batch: 1, tokens: 1, bytesPerElement: 2 }), 0);
  assert.equal(nodeCommunicationBytes({ id: "decoder.0.mlp.experts.0.down_proj" }, { hiddenSize: 4, expertsPerToken: 2 }, { tp: 2, ep: 2 }, { batch: 1, tokens: 1, bytesPerElement: 2 }), 0);
});

test("显式通信语义不依赖算子路径命名", () => {
  const config = { hiddenSize: 4, expertsPerToken: 2 };
  const options = { batch: 1, tokens: 1, bytesPerElement: 2 };
  assert.equal(nodeCommunicationBytes(
    { id: "renamed.output", attributes: { communication_role: "tp_attention_output" } },
    config,
    { tp: 2 },
    options,
  ), 8);
  assert.equal(nodeCommunicationBytes(
    { id: "renamed.output", attributes: { communication_role: "tp_attention_output" } },
    config,
    { tp: 2, attnMode: "dp" },
    options,
  ), 0);
  assert.equal(nodeCommunicationBytes(
    { id: "renamed.route", attributes: { communication_role: "ep_dispatch" } },
    config,
    { ep: 2 },
    options,
  ), 16);
});

test("EP=1 不产生 all-to-all，DP-attention 不产生 attention TP all-reduce", () => {
  assert.equal(nodeCommunicationBytes({ id: "decoder.0.mlp.dispatch" }, { hiddenSize: 4, expertsPerToken: 2 }, { ep: 1 }), 0);
  assert.equal(nodeCommunicationBytes({ id: "decoder.0.self_attn.o_proj" }, { hiddenSize: 4 }, { tp: 4, attnMode: "dp" }), 0);
  assert.ok(nodeCommunicationBytes({ id: "decoder.0.mlp.down_proj" }, { hiddenSize: 4 }, { tp: 4, attnMode: "dp" }) > 0);
});

test("F15 PD KV 传输量按 decode 侧 GQA 布局计算", () => {
  const result = pdKvTransferBytes({ totalKvBytes: 160, config: { kvHeads: 4 }, pdPlan: {
    prefill_plan: { tp: 1 }, decode_plan: { tp: 8, dp: 1 },
  } });
  assert.equal(result.ok, true);
  assert.equal(result.shardFactor, 4);
  assert.equal(result.perDecodeRankBytes, 40);
});

test("PD KV 传输量按 decode 侧 MLA 布局复制", () => {
  const result = pdKvTransferBytes({ totalKvBytes: 160, config: { kvLoraRank: 512, qkRopeHeadDim: 64 }, pdPlan: {
    prefill_plan: { tp: 1 }, decode_plan: { tp: 8, dp: 1 },
  } });
  assert.equal(result.ok, true);
  assert.equal(result.shardFactor, 1);
  assert.equal(result.perDecodeRankBytes, 160);
  assert.equal(result.aggregateBytes, 1280);
});

test("PD transfer includes request-scoped KDA state", () => {
  const result = pdKvTransferBytes({ totalKvBytes: 160, totalStateBytes: 80, config: { kvHeads: 4 }, pdPlan: {
    prefill_plan: { tp: 1 }, decode_plan: { tp: 4, dp: 1 },
  } });
  assert.equal(result.perDecodeRankBytes, 40);
  assert.equal(result.perDecodeRankStateBytes, 20);
  assert.equal(result.aggregateBytes, 240);
});

test("PD 链路带宽取两侧可用链路的较小值", () => {
  const result = pdKvTransferBytes({ totalKvBytes: 10, config: { kvHeads: 1 }, pdPlan: { prefill_plan: {}, decode_plan: {} },
    prefillChip: { interconnect: { inter_node: { bandwidth: 20e9 } } },
    decodeChip: { interconnect: { inter_node: { bandwidth: 10e9 } } },
  });
  assert.equal(result.linkBandwidth, 10e9);
  assert.equal(result.linkSource, "两侧 inter_node");
});

test("PD 两侧布局不同只标记重排，不估算重排开销", () => {
  const result = pdKvTransferBytes({ totalKvBytes: 10, config: { kvHeads: 1 }, pdPlan: { prefill_plan: { tp: 1 }, decode_plan: { tp: 2 } } });
  assert.equal(result.layoutRepackRequired, true);
  assert.equal("repackBytes" in result, false);
});

test("通信汇总包含重复层节点通信和 PP 边界通信", () => {
  const result = planCommunicationBytes({ graph: toGraph({ repeat: 2, children: [{ id: "decoder.0.self_attn.o_proj" }] }), config: { hiddenSize: 4 }, plan: { tp: 2, pp: 2 }, tokens: 1, bytesPerElement: 2 });
  assert.equal(result.nodeBytes, 16);
  assert.equal(result.ppBytes, 8);
  assert.equal(result.totalBytes, 24);
});

test("通信汇总不重复计算父列表和范围子节点 repeat", () => {
  const root = { repeat: 4, children: [{ repeat: 4, children: [{ id: "decoder.0.self_attn.o_proj" }] }] };
  const result = planCommunicationBytes({ graph: toGraph(root), config: { hiddenSize: 4 }, plan: { tp: 2 }, tokens: 1, bytesPerElement: 2 });
  assert.equal(result.nodeBytes, 32);
});

test("通信汇总按 Graph IR 节点计算（root 入参已退役）", () => {
  const result = planCommunicationBytes({
    graph: {
      version: 2,
      schema_version: 2,
      root_id: "root",
      nodes: [
        { id: "root", module_id: "model", parent_id: null, order: 0, type: "model" },
        { id: "root.0", module_id: "decoder.layers.0.attention.o_proj", parent_id: "root", order: 0, type: "operator", attributes: { communication_role: "tp_attention_output" } },
      ],
      edges: [],
    },
    config: { hiddenSize: 4 },
    plan: { tp: 2 },
    tokens: 1,
    bytesPerElement: 2,
  });
  assert.equal(result.nodeBytes, 8);
});

// ---------------------------------------------------------------------------
// P10：AllToAll dp>1（Q4/Q6：无 EP 时 DP 切专家，DP 副本间仍需 all-to-all）与
// PD 传输时间（Q7②：bytes / min(两侧带宽)）。
// ---------------------------------------------------------------------------
test("P10：无 EP 但 dp>1 时 dispatch/combine 触发 all-to-all（DP-shards-experts）", () => {
  const node = { id: "decoder.0.mlp.dispatch" };
  const plan = { tp: 1, dp: 2, ep: 1 };
  const bytes = nodeCommunicationBytes(node, { hiddenSize: 512, expertsPerToken: 4 }, plan, { batch: 1, tokens: 8 });
  // dispatch 一次：1 × 8 tokens × topk 4 × 512 × 2B
  assert.equal(bytes, 8 * 4 * 512 * 2);
  // 单 DP 无 EP：无 all-to-all
  assert.equal(nodeCommunicationBytes(node, { hiddenSize: 512, expertsPerToken: 4 }, { tp: 1, dp: 1, ep: 1 }, { batch: 1, tokens: 8 }), 0);
});

test("P10：PD 传输时间 = per-rank bytes / 链路带宽（Q7②）", () => {
  const result = pdKvTransferBytes({
    totalKvBytes: 1000, totalStateBytes: 0, config: { kvHeads: 1, attentionHeads: 1 },
    pdPlan: { prefill_plan: { tp: 1 }, decode_plan: { tp: 1 } },
    prefillChip: { interconnect: { intra_node: { bandwidth: 500 } } },
    decodeChip: { interconnect: { intra_node: { bandwidth: 500 } } },
  });
  assert.equal(result.ok, true);
  assert.equal(result.transferSeconds, result.perDecodeRankBytes / 500);
});
