import assert from "node:assert/strict";
import test from "node:test";
import fs from "node:fs";
import { buildCostAccounting, cacheAccountingFromGraph } from "../memory.js";
import { aggregateCost } from "../aggregate.js";
import { maxContextForStages, planFitsCard, projectPlan, projectPdFit } from "../parallel.js";
import { getFrameworkRuntimeProfile } from "../../frameworkProfiles.js";
import { buildStructureFromConfig } from "../../structure/buildStructure.js";
import { normalizeConfig } from "../../structure/config/normalize.js";
import { computeNodeCosts } from "../compute.js";

const graphOf = (attrs, draftAttrs = {}, type = "mtp") => ({
  root_id: "root",
  nodes: [
    { id: "root", parent_id: null },
    { id: "layers.0.attn", parent_id: "root", attributes: attrs },
    { id: "draft", parent_id: "root", type, repeat: 0, attributes: { modules: 1 } },
    { id: "draft.attn", parent_id: "draft", attributes: draftAttrs },
  ],
});

test("profile selection and GDN/KDA dtype are implementation-specific; explicit config wins", () => {
  assert.equal(getFrameworkRuntimeProfile("missing").id, "neutral");
  const graph = graphOf({ state_conv_elements: 4, state_recurrent_elements: 8, state_recurrent_dtype: "F32" });
  const state = (frameworkProfile, config = {}) => cacheAccountingFromGraph(graph, { frameworkProfile, config }).totalStateBytes;
  assert.equal(state("neutral"), 40);
  assert.equal(state("sglang"), 40);
  assert.equal(state("vllm"), 24);
  assert.equal(state("vllm", { mambaSsmDtype: "float32" }), 40);
  graph.nodes[1].attributes.model_kind = "kimi_k3";
  assert.equal(state("vllm"), 40, "KDA auto must not become GDN BF16");
});

test("DSA explicit index and FP4 dtype are unaffected by fallback; plain cache changes", () => {
  const graph = graphOf({
    cache_kv_elements: 576, cache_kv_dtype: "BF16",
    cache_index_elements: 128, cache_index_dtype: "F8_E8M0S128",
  }, { cache_kv_elements: 16, cache_kv_dtype: "F4" });
  for (const kvBytes of [2, 1, 0.5]) {
    const a = cacheAccountingFromGraph(graph, { kvBytes });
    assert.equal(a.mainKvBytes, 1152 + 132);
    assert.equal(a.draftKvBytes, 8);
  }
  assert.equal(cacheAccountingFromGraph(graphOf({ cache_kv_elements: 10 }), { kvBytes: 0.5 }).totalKvBytes, 5);
  const indexOnly = graphOf({ cache_kv_elements: 10, cache_index_elements: 128, cache_index_dtype: "F8_E8M0S128" });
  assert.equal(cacheAccountingFromGraph(indexOnly, { kvBytes: 0.5 }).mainKvBytes, 5 + 132);
});

test("roofline state traffic uses the same profile dtype formula as residency", () => {
  const attrs = { operator_id: "gated_delta_attention", model_kind: "qwen3_5",
    state_elements: 14, state_conv_elements: 6, state_recurrent_elements: 8, state_recurrent_dtype: "F32" };
  const graph = graphOf(attrs);
  const config = { linearKeyHeads: 1, linearValueHeads: 2, linearKeyDim: 2, linearValueDim: 2, linearConvKernelSize: 2 };
  for (const frameworkProfile of ["vllm", "sglang"]) {
    const state = cacheAccountingFromGraph(graph, { config, frameworkProfile }).mainStateBytes;
    const row = computeNodeCosts(graph, config, { phase: "decode", sequence: 100, batch: 1, frameworkProfile })
      .find((entry) => entry.node.id === "layers.0.attn");
    assert.equal(row.actions.bytes.actIn, state);
    assert.equal(row.actions.bytes.actOut, state);
  }
});

test("one explicit shared DSpark pool is counted once regardless of visit order", () => {
  for (const frameworkProfile of ["neutral", "vllm", "sglang"]) {
    const attrs = { cache_pool_id: "same", cache_kv_elements: 10, state_elements: 3 };
    const graph = graphOf(attrs, { ...attrs, cache_pool_shared: true }, "dspark");
    for (const nodes of [graph.nodes, [...graph.nodes].reverse()]) {
      const a = buildCostAccounting({ graph: { ...graph, nodes }, frameworkProfile, weightBytes: 100, tokens: 8 });
      assert.equal(a.pools.length, 1);
      assert.equal(a.main.kvBytes, 0);
      assert.equal(a.draft.kvBytes, 0);
      assert.equal(a.shared.kvBytes, 160);
      assert.equal(a.shared.stateBytes, 6);
      assert.equal(a.total.vramBytes, 266);
    }
  }
});

test("private MTP KV, state and buffers drive Fit, Max Context and both PD sides", () => {
  const graph = graphOf({ cache_kv_elements: 10, buffer_elements: 5 }, { cache_kv_elements: 5, state_elements: 3 });
  const config = { layers: 1, kvHeads: 1 };
  const a = buildCostAccounting({ graph, weightBytes: 100, tokens: 10 });
  assert.equal(a.total.vramBytes, 426); // 100 weights + 20 buffers + 300 KV + 6 state
  const projection = projectPlan({ graph, accounting: a, config });
  assert.equal(projection.stages[0].totalBytes, 426);
  assert.equal(planFitsCard(projection, 400), false);
  assert.equal(maxContextForStages(projection.stages, { capacityBytes: 400, sequence: 10 }), 9);
  const p = buildCostAccounting({ graph, weightBytes: 100, tokens: 5 });
  const pd = projectPdFit({ graph, config, prefillAccounting: p, decodeAccounting: a,
    prefillChip: { memory_bytes: 400 }, decodeChip: { memory_bytes: 400 } });
  assert.equal(pd.prefill.fit, true);
  assert.equal(pd.decode.fit, false);
});

test("uneven PP places cache by layer range and private draft on last stage", () => {
  const graph = graphOf({ cache_kv_elements: 10 }, { cache_kv_elements: 5, state_elements: 3 });
  const a = buildCostAccounting({ graph, tokens: 10 });
  const p = projectPlan({ graph, accounting: a, config: { layers: 2, kvHeads: 1 }, plan: { pp: 2 } });
  assert.deepEqual(p.stages.map((s) => s.kvBytes), [200, 100]);
  assert.deepEqual(p.stages.map((s) => s.stateBytes), [0, 6]);
  assert.equal(p.stages.reduce((n, s) => n + s.totalBytes, 0), a.total.vramBytes);
});

test("runtime DSpark retains private bounded storage, not a fake shared zero or linear context multiplier", () => {
  const graph = graphOf({ cache_kv_elements: 10 }, {
    cache_kv_elements: 512, cache_kv_growth_elements: 0, cache_kv_dtype: "F4", sliding_window: 128, attention_kind: "dsv4_swa_mqa",
  }, "dspark");
  const runtime = (tokens) => buildCostAccounting({ graph, frameworkProfile: "sglang", tokens });
  assert.equal(runtime(1000).draft.kvBytes, 512 * 2 * 128);
  assert.equal(runtime(1000).draft.kvBytes, runtime(2000).draft.kvBytes);
  assert.equal(runtime(1000).draft.kvBytesPerToken, 0);
  assert.equal(runtime(1000).shared.kvBytes, 0);
  assert.ok(runtime(1000).evidence.unknownFields.includes("dsparkBackendPackingAndPageHeadroom"));
});

test("all catalog models x profiles reconcile resident roll-up and single-card projection", () => {
  const modelsRoot = new URL("../../../../models/", import.meta.url);
  const catalog = JSON.parse(fs.readFileSync(new URL("catalog.json", modelsRoot)));
  for (const entry of catalog.models) {
    const raw = JSON.parse(fs.readFileSync(new URL(entry.config_path, modelsRoot)));
    const config = normalizeConfig(raw);
    for (const frameworkProfile of ["neutral", "vllm", "sglang"]) {
      const { graph } = buildStructureFromConfig(raw, { frameworkProfile });
      const cost = aggregateCost({ graph, config, frameworkProfile, sequence: 2048 });
      const a = cost.memory.accounting;
      assert.ok(Number.isFinite(a.total.vramBytes), entry.model_id);
      assert.ok(Math.abs(a.main.vramBytes + a.draft.vramBytes + a.shared.vramBytes - a.total.vramBytes) < 0.01, entry.model_id);
      const p = projectPlan({ graph, accounting: a, config });
      assert.equal(p.ok, true, entry.model_id);
      assert.ok(Math.abs(p.stages[0].totalBytes - a.total.vramBytes) < Math.max(0.01, a.total.vramBytes * 1e-12), entry.model_id);
    }
  }
});
