import assert from "node:assert/strict";
import test from "node:test";
import { buildCostAccounting } from "../memory.js";
import { recommendSingleNodePlan } from "../parallelDefaults.js";
import { materializeStructureGraph } from "../../structure/graph/materializeStructureGraph.js";

function weightedGraph() {
  return materializeStructureGraph({
    id: "model",
    children: [{
      id: "decoder.layers.0.mlp",
      attributes: {
        weightMatrices: [{ class: "tp", out: 10, in: 10, count: 1, matrices: 1 }],
      },
      children: [],
    }],
  });
}

function recommendation(weightBytes, memoryBytes = 100) {
  const graph = weightedGraph();
  const accounting = buildCostAccounting({ graph, weightBytes, tokens: 1 });
  return recommendSingleNodePlan({
    graph,
    accounting,
    config: { layers: 1 },
    chip: { memory_bytes: memoryBytes },
  });
}

test("small model defaults to the smallest fitting 1/2/4/8 card tier", () => {
  assert.equal(recommendation(100).cards, 1);
  assert.equal(recommendation(101).cards, 2);
  assert.equal(recommendation(200).cards, 2);
  assert.equal(recommendation(201).cards, 4);
  assert.equal(recommendation(400).cards, 4);
  assert.equal(recommendation(400).plan.tp, 4);
  assert.deepEqual(recommendation(400).nodes, { centralized: 1, prefill: 1, decode: 1 });
  assert.equal(recommendation(400).gpusPerNode, 8);
  assert.deepEqual(recommendation(400).plans.prefill, recommendation(400).plans.decode);
});

test("model that needs the full single node defaults to TP8", () => {
  const result = recommendation(600);
  assert.equal(result.cards, 8);
  assert.equal(result.fit, true);
  assert.deepEqual(result.candidates.map((candidate) => candidate.fit), [false, false, false, true]);
});

test("model that still cannot fit keeps the one-node TP8 default and reports no fit", () => {
  const result = recommendation(1000);
  assert.equal(result.cards, 8);
  assert.equal(result.plan.tp, 8);
  assert.equal(result.fit, false);
});

test("unknown capacity or weight accounting is not a proved one-card fit", () => {
  for (const bytes of [0, -1, NaN, Infinity]) {
    assert.equal(recommendation(100, bytes).fit, null);
    assert.equal(recommendation(100, bytes).cards, 8);
  }
  assert.equal(recommendSingleNodePlan().fit, null);
  assert.equal(recommendation(0).fit, null);
});

test("main, draft, shared KV, state and buffers all affect the default tier", () => {
  const graph = weightedGraph();
  graph.nodes.push(
    { id: "cache", parent_id: graph.root_id, attributes: { cache_pool_id: "shared", cache_pool_shared: true, cache_kv_elements: 20, state_elements: 10, buffer_elements: 5 } },
    { id: "mtp", parent_id: graph.root_id, type: "mtp", attributes: {} },
    { id: "mtp.alias", parent_id: "mtp", attributes: { cache_pool_id: "shared", cache_pool_shared: true, cache_kv_elements: 20, state_elements: 10 } },
    { id: "mtp.private", parent_id: "mtp", attributes: { cache_kv_elements: 20 } },
  );
  const accounting = buildCostAccounting({ graph, weightBytes: 100 });
  // TP1: 100 weight + 40 shared KV + 40 private KV + 20 state + 20 buffer = 220.
  // TP2: 50 + 20 + 20 + 10 + 20 = 120; TP4: 25 + 10 + 10 + 5 + 20 = 70.
  assert.equal(accounting.total.vramBytes, 220);
  const result = recommendSingleNodePlan({ graph, accounting, config: { kvHeads: 8 }, chip: { memory_bytes: 100 } });
  assert.equal(result.cards, 4);
});

test("MLA KV replication cannot be hidden by dividing total bytes by TP", () => {
  const graph = weightedGraph();
  graph.nodes.push({ id: "cache", parent_id: graph.root_id, attributes: { cache_kv_elements: 60 } });
  const accounting = buildCostAccounting({ graph, weightBytes: 100 });
  const result = recommendSingleNodePlan({ graph, accounting, config: { kvLoraRank: 8, qkRopeHeadDim: 4 }, chip: { memory_bytes: 100 } });
  assert.equal(result.cards, 8);
  assert.equal(result.fit, false, "120 bytes of replicated MLA KV cannot fit a 100-byte card at any TP");
});

test("vLLM and SGLang MoE defaults keep EP disabled and valid TP-only plans", () => {
  const graph = weightedGraph();
  graph.nodes.find((node) => node.attributes?.weightMatrices)?.attributes.weightMatrices
    .forEach((group) => { group.class = "ep"; });
  for (const frameworkProfile of ["neutral", "vllm", "sglang"]) {
    const accounting = buildCostAccounting({ graph, weightBytes: 300, frameworkProfile });
    const result = recommendSingleNodePlan({ graph, accounting, config: { layers: 1, experts: 16 }, chip: { memory_bytes: 100 } });
    assert.equal(result.cards, 4, frameworkProfile);
    assert.equal(result.plan.ep, 1);
    assert.equal(result.plan.moeTp, undefined);
    assert.equal(result.fit, true);
  }
});
