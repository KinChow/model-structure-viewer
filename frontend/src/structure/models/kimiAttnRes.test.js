import assert from "node:assert/strict";
import fs from "node:fs";
import test from "node:test";
import { buildStructureFromConfig, buildStructureFromArtifacts } from "../buildStructure.js";
import { normalizeConfig } from "../config/normalize.js";
import { countsForNode } from "../operators/formulas/extractor.js";
import { computeNodeCosts } from "../../cost/compute.js";
import { MODULES } from "../operators/formulas/modules.js";
import { evaluateDecomposition } from "../operators/formulas/atoms.js";
import { buildSkeleton } from "../truth/skeleton.js";
import { layoutGraph } from "../../diagram/layout.js";

const raw = JSON.parse(fs.readFileSync(new URL("../../../../models/moonshotai/Kimi-K3/config.json", import.meta.url)));
test("K3 AttnRes has immutable depth states, dual pre-sublayer aggregation and final mix", () => {
  const { graph } = buildStructureFromConfig(raw);
  const get = id => graph.nodes.find(n => n.canonical_id === id);
  const has = (s, t) => graph.edges.some(e => e.source_canonical_id === s && e.target_canonical_id === t);
  // Independent published schedule: writes at 0,12,...84; 93 decoder layers.
  for (let layer = 0; layer < 93; layer++) {
    const p = `layers.${layer}`;
    const before = Math.ceil(layer / 12), after = Math.floor(layer / 12) + 1;
    const boundary = layer % 12 === 0;
    assert.equal(get(p)?.repeat, 1, "each immutable state endpoint names a real physical layer");
    assert.equal(get(`${p}.attn_res_pre`)?.attributes.candidate_states, layer === 0 ? 1 : before + 1);
    assert.equal(get(`${p}.attn_res_pre`).attributes.execution_skipped, layer === 0);
    assert.equal(get(`${p}.attn_res_mlp`)?.attributes.candidate_states, after + 1);
    assert.equal(get(`${p}.bank_in`).attributes.snapshot_count, before);
    assert.equal(get(`${p}.bank_out`).attributes.snapshot_count, after);
    assert.equal(get(`${p}.bank_out`).attributes.snapshot_write, boundary);
    assert.equal(get(`${p}.bank_out`).input_shape[2], before);
    assert.equal(get(`${p}.bank_out`).output_shape[2], after);
    assert.equal(get(`${p}.prefix_after_attn`).attributes.operator_id, boundary ? "identity" : "residual_add");
    assert.equal(has(`${p}.layer_in`, `${p}.prefix_after_attn`), !boundary);
    assert.ok(has(`${p}.self_attn`, `${p}.prefix_after_attn`));
    assert.ok(has(`${p}.attn_res_pre`, `${p}.input_layernorm`));
    assert.ok(has(`${p}.prefix_after_attn`, `${p}.attn_res_mlp`));
    assert.ok(has(`${p}.bank_out`, `${p}.attn_res_mlp`));
    assert.ok(has(`${p}.attn_res_mlp`, `${p}.post_attention_layernorm`));
    assert.equal(get(`${p}.attn_residual_add`), undefined);
    assert.equal(get(`${p}.ffn_residual_add`), undefined);
    if (layer) {
      assert.ok(has(`layers.${layer - 1}.bank_out`, `${p}.bank_in`));
      assert.ok(has(`layers.${layer - 1}.prefix_out`, `${p}.layer_in`));
    }
  }
  assert.equal(get("output_attn_residual").attributes.candidate_states, 9);
  assert.ok(has("layers.92.bank_out", "output_attn_residual"));
  assert.ok(has("layers.92.prefix_out", "output_attn_residual"));
  assert.ok(has("output_attn_residual", "norm"));
  assert.ok(get("output_attn_res_norm") && get("output_attn_res_proj"));
  assert.ok(get("layers.12.self_attention_res_norm") && get("layers.12.self_attention_res_proj"));
  // Declared graph is acyclic even though state is carried across depth.
  const indegree = new Map(graph.nodes.map(n => [n.id, 0]));
  const outgoing = new Map();
  for (const e of graph.edges) {
    indegree.set(e.target, indegree.get(e.target) + 1);
    outgoing.set(e.source, [...(outgoing.get(e.source) || []), e.target]);
  }
  const queue = [...indegree].filter(([, d]) => d === 0).map(([id]) => id);
  let seen = 0;
  while (queue.length) {
    const id = queue.pop(); seen++;
    for (const t of outgoing.get(id) || []) {
      indegree.set(t, indegree.get(t) - 1);
      if (!indegree.get(t)) queue.push(t);
    }
  }
  assert.equal(seen, graph.nodes.length);
});

test("AttnRes candidate-sensitive cost, skipped first aggregation and capacity ownership", () => {
  const small = { ...raw, text_config: { ...raw.text_config, hidden_size: 4, num_hidden_layers: 3, attn_res_block_size: 2 } };
  const config = normalizeConfig(small), { graph } = buildStructureFromConfig(small);
  const get = id => graph.nodes.find(n => n.canonical_id === id);
  const env = { config, options: { phase: "prefill", batch: 1, sequence: 2 }, bytesPerElement: 2 };
  const first = countsForNode(get("layers.0.attn_res_pre"), env);
  assert.deepEqual(first, { matrix: 0, vector: 0, sfu: 0, bytes: { weights: 0, actIn: 0, actOut: 0 } });
  // _apply_attn_res: normalize, combine norm/proj weight, score reduction,
  // softmax over C (not H), weighted matmul. T=2,H=4,C=3.
  const final = countsForNode(get("output_attn_residual"), env);
  assert.deepEqual(final, { matrix: 24, vector: 142, sfu: 18, bytes: { weights: 16, actIn: 48, actOut: 16 } });
  const rows = computeNodeCosts(graph, config, env.options);
  const childIds = new Set(graph.nodes.filter(n => n.parent_id === get("layers.0.attn_res_pre").id).map(n => n.canonical_id));
  const children = rows.filter(r => childIds.has(r.node.id));
  assert.equal(children.reduce((sum, r) => sum + r.weightBytes, 0), 16, "skipped execution still owns its real weights");
  assert.ok(children.every(r => r.compute_macs === 0 && r.actions === null), "parent bills computation only once");
  assert.ok(graph.nodes.filter(n => /bank_(in|out)$/.test(n.canonical_id)).every(n => !n.attributes.state_elements && !n.attributes.cache_kv_elements));
  const bankWrite = countsForNode(get("layers.0.bank_out"), env);
  assert.equal(bankWrite.bytes.actIn, null);
  assert.equal(bankWrite.bytes.actOut, null);
  assert.equal(bankWrite.matrix, 0);
});

test("AttnRes real norm/proj canonical paths bind both tensor and skeleton truth", () => {
  const f = JSON.parse(fs.readFileSync(new URL("./__fixtures__/kimi-k3-attnres-header.json", import.meta.url)));
  for (const checkpointTruth of [{ tensors: f.tensors },
    { skeleton: buildSkeleton(f.tensors), tensor_count: f.tensors.length }]) {
    const { graph } = buildStructureFromArtifacts({ config: raw, modelId: f.model_id, checkpointTruth });
    for (const t of f.tensors) {
      const bound = graph.nodes.filter(n => n.tensor_names?.includes(t.name));
      assert.equal(bound.length, 1);
      assert.equal(bound[0].canonical_id, t.name.replace(/^language_model\.model\./, "").replace(/\.weight$/, ""));
      assert.deepEqual(bound[0].weight_shapes.weight, t.shape);
    }
    assert.ok(graph.nodes.filter(n => n.attributes.operator_id === "attention_residual").every(n => !n.tensor_names?.length));
  }
});

test("AttnRes norm/score/softmax/mix decomposition scales with candidate count and batch", () => {
  const entry = MODULES.attention_residual;
  for (const tokens of [1, 6]) for (const candidates of [2, 3, 9]) {
    const p = { tokens, candidates, hidden: 4, b: 2 };
    const a = entry.fused(p), d = evaluateDecomposition(entry.decompose(p));
    for (const key of ["matrix", "vector", "sfu"]) assert.equal(a[key], d[key]);
    assert.equal(a.matrix, tokens * candidates * 4);
    assert.equal(a.sfu, tokens * candidates * 3);
    assert.equal(a.bytes.weights, 16);
    assert.equal(a.bytes.actIn, tokens * candidates * 8);
    assert.equal(a.bytes.actOut, tokens * 8);
  }
});

test("depth edges project to real layer containers and retain original endpoints", () => {
  const structure = buildStructureFromArtifacts({ config: raw, modelId: "moonshotai/Kimi-K3" });
  const get = id => structure.graph.nodes.find(n => n.canonical_id === id);
  const view = layoutGraph(structure, new Set([structure.graph.root_id, get("layers").id]));
  const edge = view.edges.find(e => e.source_canonical_id === "layers.11.bank_out" && e.target_canonical_id === "layers.12.bank_in");
  assert.ok(edge);
  assert.equal(edge.source, get("layers.11").id);
  assert.equal(edge.target, get("layers.12").id);
  assert.equal(edge.originalSource, get("layers.11.bank_out").id);
  assert.equal(edge.originalTarget, get("layers.12.bank_in").id);
  assert.equal(edge.relation, "depth-state");
  assert.equal(get("layers.12").attributes.range, "12..12");
});
