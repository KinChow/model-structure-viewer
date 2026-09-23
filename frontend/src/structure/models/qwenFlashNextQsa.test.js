import assert from "node:assert/strict";
import fs from "node:fs";
import test from "node:test";
import { buildStructureFromArtifacts, buildStructureFromConfig } from "../buildStructure.js";
import { normalizeConfig } from "../config/normalize.js";
import { countsForNode } from "../operators/formulas/extractor.js";
import { computeNodeCosts } from "../../cost/compute.js";
import { buildSkeleton } from "../truth/skeleton.js";
import { buildNodeLens } from "../../diagram/lens.js";

const modelsRoot = new URL("../../../../models/", import.meta.url);
const read = url => JSON.parse(fs.readFileSync(url, "utf8"));

function artifactsFor(modelId) {
  const dir = new URL(`${modelId}/`, modelsRoot);
  return {
    config: read(new URL("config.json", dir)),
    modelId,
    checkpointTruth: read(new URL("header-truth.json", dir)),
    sourceRef: read(new URL("source-ref.json", dir)),
  };
}

function qsaNodes(graph) {
  return graph.nodes.filter(node => node.canonical_id.startsWith("layers.") && node.attributes?.operator_id === "qsa_indexer");
}

function qsaAttentionNodes(graph) {
  return graph.nodes.filter(node => node.canonical_id.startsWith("layers.") && node.attributes?.operator_id === "qsa_sparse_attention");
}

for (const loading of ["config", "artifacts"]) {
  test(`Qwen3.8-Flash-Next QSA expands block selection and tail semantics (${loading})`, () => {
    for (const modelId of ["Qwen/Qwen3.8-Flash-Next", "Qwen/Qwen3.8-Flash-Next-FP8"]) {
      const structure = loading === "config"
        ? buildStructureFromConfig(read(new URL(`${modelId}/config.json`, modelsRoot)), { modelId })
        : buildStructureFromArtifacts(artifactsFor(modelId));
      const graph = structure.graph;
      const normalized = normalizeConfig(read(new URL(`${modelId}/config.json`, modelsRoot)));
      const indexers = qsaNodes(graph);
      const attentions = qsaAttentionNodes(graph);
      assert.equal(indexers.length, 12, `${modelId}: QSA layer count`);
      assert.equal(attentions.length, 12, `${modelId}: QSA sparse attention count`);

      const indexer = indexers[0];
      assert.equal(indexer.attributes.block_size, 4);
      assert.equal(indexer.attributes.block_budget, 512);
      assert.equal(indexer.attributes.budget, 2048);
      assert.equal(indexer.attributes.tail_max_tokens, 3);
      assert.deepEqual(indexer.attributes.index_pipeline, [
        "index_qk_projection", "q_norm", "key_block_mean_pool", "k_norm",
        "query_and_block_position", "relu_score_sum", "block_topk", "block_expand", "tail_append",
      ]);
      const projection = graph.nodes.find(node => node.canonical_id === "layers.3.self_attn.indexer.index_qk_proj");
      assert.deepEqual(projection.attributes.weightMatrices[0].shape, [640, 2560]);
      assert.equal(graph.nodes.filter(node => node.canonical_id.startsWith("layers.") && node.attributes?.semantic_role === "qsa_index_qk_projection").length, 12);
      assert.equal(graph.nodes.filter(node => node.canonical_id.startsWith("layers.") && node.attributes?.semantic_role === "qsa_key_block_mean_pool").length, 12);
      assert.equal(graph.nodes.filter(node => node.canonical_id.startsWith("layers.") && node.attributes?.semantic_role === "qsa_tail_append").length, 12);

      const sparse = attentions[0];
      assert.equal(sparse.attributes.complete_block_budget, 512);
      assert.equal(sparse.attributes.selected_tokens, 2048);
      assert.equal(sparse.attributes.selected_tokens_max, 2051);
      assert.equal(sparse.attributes.tail_max_tokens, 3);
      const incoming = graph.edges.filter(edge => edge.target === sparse.id).map(edge => edge.source_canonical_id);
      assert.ok(incoming.includes(indexer.canonical_id));
      assert.ok(incoming.some(source => source.endsWith(".rope")));
      const hasEdge = (from, to) => graph.edges.some(edge =>
        edge.source_canonical_id === `${indexer.canonical_id}.${from}` &&
        edge.target_canonical_id === `${indexer.canonical_id}.${to}`);
      for (const [from, to] of [
        ["raw_key_cache", "key_block_mean_pool"], ["key_block_mean_pool", "k_layernorm"],
        ["k_layernorm", "k_rope"], ["q_layernorm", "q_rope"],
        ["q_rope", "score"], ["k_rope", "score"], ["score", "block_select"],
        ["block_select", "block_expand"], ["block_expand", "tail_append"],
        ["visible_indices", "tail_append"],
      ]) assert.ok(hasEdge(from, to), `${modelId}: ${from} -> ${to}`);
      assert.equal(hasEdge("k_layernorm", "key_block_mean_pool"), false);

      const ctx = { config: normalized, options: { batch: 1, sequence: 1, phase: "decode" }, bytesPerElement: 2 };
      const expected = [
        [1, 1], [4, 4], [2048, 2048], [2049, 2049], [2051, 2051], [2052, 2048],
      ];
      for (const [sequence, selected] of expected) {
        const actions = countsForNode(sparse, {
          ...ctx,
          options: { ...ctx.options, sequence },
        });
        assert.equal(actions.bytes.kvRead / (2 * 2 * 256 * 2), selected, `${modelId}: sequence ${sequence}`);
      }
    }
  });
}

test("Qwen3.8-Flash-Next QSA does not instantiate per-GQA-group selectors", () => {
  const graph = buildStructureFromConfig(read(new URL("Qwen/Qwen3.8-Flash-Next/config.json", modelsRoot))).graph;
  assert.equal(graph.nodes.filter(node => node.canonical_id.startsWith("layers.") && node.attributes?.semantic_role === "qsa_index_qk_projection").length, 12);
  assert.equal(graph.nodes.filter(node => node.canonical_id.startsWith("layers.") && node.attributes?.semantic_role === "qsa_block_topk").length, 12);
  assert.equal(graph.nodes.filter(node => node.canonical_id.startsWith("layers.") && node.attributes?.semantic_role === "qsa_block_expand").length, 12);
  const score = graph.nodes.find(node => node.attributes?.semantic_role === "qsa_relu_score_sum");
  assert.equal(score.attributes.reduction, "sum_over_index_heads");
  assert.equal(score.attributes.no_gqa_group_selection, true);
});

test("QSA composite bills once, while checkpoint-shaped children own its weights", () => {
  const raw = read(new URL("Qwen/Qwen3.8-Flash-Next/config.json", modelsRoot));
  const structure = buildStructureFromConfig(raw);
  const rows = computeNodeCosts(structure.graph, normalizeConfig(raw), { batch: 2, sequence: 5, phase: "prefill" });
  const prefix = "layers.3.self_attn.indexer";
  const parent = rows.find(row => row.node.id === prefix);
  const children = rows.filter(row => row.node.id.startsWith(`${prefix}.`));
  assert.equal(parent.compute_macs, 10 * 2560 * 640 + 4 * 4 * 128);
  assert.equal(parent.weightBytes, 0);
  assert.ok(children.every(row => row.compute_macs === 0 && row.actions === null));
  assert.equal(children.reduce((sum, row) => sum + row.weightBytes, 0), (2560 * 640 + 128 * 2) * 2);
  assert.equal(parent.actions.bytes.indexRead, 2 * 4 * 128 * 2);
  const lens = buildNodeLens(structure, { memory_bandwidth: 100, peak_flops: { bf16: 1000 } });
  const pool = structure.graph.nodes.find(node => node.canonical_id === `${prefix}.key_block_mean_pool`);
  assert.equal(lens.nodes[pool.id].metrics.vramBytes, null, "do not fake a full-sequence pooled allocation");
  assert.equal(lens.nodes[pool.id].metrics.memoryBytes, null);
});

test("QSA core keeps the checkpoint Q/gate projection and both output-gate inputs", () => {
  const raw = read(new URL("Qwen/Qwen3.8-Flash-Next/config.json", modelsRoot));
  const { graph } = buildStructureFromConfig(raw);
  const get = name => graph.nodes.find(node => node.canonical_id === `layers.3.self_attn.${name}`);
  // Independently read from official safetensors header, revision de4b8e4d.
  for (const [name, shape] of [
    ["q_proj", [12288, 2560]], ["k_proj", [512, 2560]],
    ["v_proj", [512, 2560]], ["o_proj", [2560, 6144]],
  ]) assert.deepEqual(get(name)?.attributes.weightMatrices[0].shape, shape);
  const gate = get("output_gate");
  assert.deepEqual(new Set(graph.edges.filter(edge => edge.target === gate.id).map(edge => edge.source_canonical_id)),
    new Set(["layers.3.self_attn.q_gate_split", "layers.3.self_attn.sparse_attention"]));
  assert.equal(get("q_norm").attributes.operator_id, "gemma_rmsnorm");
  assert.equal(get("k_norm").attributes.operator_id, "gemma_rmsnorm");
  const counts = countsForNode(gate, { config: normalizeConfig(raw),
    options: { batch: 2, sequence: 5, phase: "prefill" }, bytesPerElement: 2 });
  assert.equal(counts.matrix, 0);
  assert.equal(counts.bytes.weights, 0);
  assert.equal(counts.vector, 10 * 6144);
  assert.equal(counts.sfu, 2 * 10 * 6144);
  assert.equal(get("qkv_proj"), undefined, "do not duplicate fused runtime weights and checkpoint projections");
});

test("both release headers bind each QSA tensor exactly once, including indexer children", () => {
  const fixture = read(new URL("./__fixtures__/qwen-flash-next-qsa-header.json", import.meta.url));
  for (const model of fixture.models) {
    const config = read(new URL(`${model.model_id}/config.json`, modelsRoot));
    for (const checkpointTruth of [{ tensors: model.tensors }, { skeleton: buildSkeleton(model.tensors), tensor_count: model.tensors.length }]) {
      const structure = buildStructureFromArtifacts({ config, modelId: model.model_id,
        revision: model.revision, checkpointTruth });
      for (const tensor of model.tensors) {
        const matches = structure.graph.nodes.filter(node => node.tensor_names?.includes(tensor.name));
        assert.equal(matches.length, 1, `${model.model_id}: ${tensor.name}`);
        assert.equal(matches[0].canonical_id, tensor.name.replace(/^model.language_model\./, "").replace(/\.weight$/, ""));
        assert.deepEqual(matches[0].weight_shapes.weight, tensor.shape);
        assert.equal(matches[0].value_source, "checkpoint");
      }
      assert.ok(structure.graph.nodes.filter(node =>
        node.attributes?.semantic_role?.startsWith("qsa_") && !node.attributes?.weightMatrices?.length
      ).every(node => !node.tensor_names?.length), "semantic stages must not impersonate checkpoint modules");
    }
  }
});
