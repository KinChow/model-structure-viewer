import assert from "node:assert/strict";
import fs from "node:fs";
import test from "node:test";
import { buildStructureFromArtifacts, buildStructureFromConfig } from "../buildStructure.js";
import { normalizeConfig } from "../config/normalize.js";
import { countsForNode } from "../operators/formulas/extractor.js";
import { computeNodeCosts } from "../../cost/compute.js";

const root = new URL("../../../../models/MiniMaxAI/", import.meta.url);
const read = file => JSON.parse(fs.readFileSync(new URL(file, root)));

for (const variant of ["MiniMax-M3", "MiniMax-M3-MXFP8"]) for (const loading of ["config-only", "production-artifacts"]) {
  test(`MiniMax M3 MSA preserves one independent selector per GQA group: ${loading}`, () => {
    const config = read(`${variant}/config.json`);
    const normalized = normalizeConfig(config);
    const structure = loading === "config-only" ? buildStructureFromConfig(config)
      : buildStructureFromArtifacts({
        config, modelId: `MiniMaxAI/${variant}`,
        checkpointTruth: read(`${variant}/header-truth.json`),
        sourceRef: read(`${variant}/source-ref.json`),
      });
    const indexer = structure.graph.nodes.find(n => n.attributes?.operator_id === "minimax_sparse_indexer");
    const sparse = structure.graph.nodes.find(n => n.attributes?.operator_id === "minimax_sparse_attention");
    assert.ok(indexer && sparse);
    assert.equal(indexer.attributes.index_selection_scope, "gqa_group");
    assert.equal(indexer.attributes.index_group_count, normalized.kvHeads);
    assert.equal(indexer.attributes.query_heads_per_group,
      normalized.attentionHeads / normalized.kvHeads);
    assert.equal(indexer.attributes.selection_shared_by_query_heads, true);
    assert.equal(indexer.attributes.index_key_heads, 1);
    assert.equal(indexer.attributes.block_score_reduction, "max");
    assert.equal(indexer.attributes.local_block_always_included, true);
    assert.equal(indexer.attributes.index_value_path, "disabled");
    assert.equal(sparse.attributes.selection_scope, "gqa_group");
    assert.equal(sparse.attributes.query_heads_per_group,
      normalized.attentionHeads / normalized.kvHeads);
    assert.equal(sparse.attributes.index_value_path, "disabled");
    assert.equal(sparse.attributes.selected_blocks_per_group, normalized.sparseTopkBlocks);
    assert.equal(sparse.attributes.selection_budget_includes_local, true);
    assert.equal(sparse.attributes.block_token_budget,
      normalized.sparseTopkBlocks * normalized.sparseBlockSize);
    const base = indexer.canonical_id;
    const children = ["group_scores", "block_max", "local_boost", "group_topk", "valid_block_ids"];
    const find = suffix => structure.graph.nodes.find(n => n.canonical_id === `${base}.${suffix}`);
    for (const suffix of children) {
      assert.ok(find(suffix), `missing MSA explanatory stage ${suffix}`);
      assert.equal(find(suffix).attributes.checkpoint_module, false);
      assert.equal(find(suffix).attributes.activation_materialization, "unknown");
    }
    assert.equal(find("group_topk").output_shape.at(-2), normalized.kvHeads);
    assert.equal(find("group_topk").output_shape.at(-1), normalized.sparseTopkBlocks);
    for (const [from, to] of children.slice(0, -1).map((n, i) => [n, children[i + 1]])) {
      assert.ok(structure.graph.edges.some(e => e.source === find(from).id && e.target === find(to).id),
        `${from} -> ${to}`);
    }
    assert.ok(structure.graph.edges.some(e => e.source === find("valid_block_ids").id &&
      e.target === sparse.id));
  });
}

test("MiniMax M3 indexer cost scales by four group selectors, not 64 query heads", () => {
  const config = read("MiniMax-M3/config.json");
  const normalized = normalizeConfig(config);
  const structure = buildStructureFromConfig(config);
  const indexer = structure.graph.nodes.find(n => n.attributes?.operator_id === "minimax_sparse_indexer");
  const actions = countsForNode(indexer, {
    config: normalized,
    options: { batch: 1, sequence: 128, phase: "decode" },
    bytesPerElement: 2,
  });
  // Q_idx has one head per KV/GQA group (4×128), while K_idx is one shared head.
  // The fused QKV/index projection is a separate linear leaf. This composite
  // counts only the index score scan: T_decode=1, S=128, four groups.
  assert.equal(actions.matrix, 4 * 128 * 128);
  assert.equal(actions.bytes.indexRead, 128 * 128 * 2);
});

test("MSA cost does not include QSA/DSA ReLU or cross-head sum, and local uses an existing TopK slot", () => {
  const config = read("MiniMax-M3/config.json");
  const normalized = normalizeConfig(config);
  const graph = buildStructureFromConfig(config).graph;
  const indexer = graph.nodes.find(n => n.attributes?.operator_id === "minimax_sparse_indexer");
  const sparse = graph.nodes.find(n => n.attributes?.operator_id === "minimax_sparse_attention");
  const ctx = { config: normalized, options: { batch: 1, sequence: 256, phase: "decode" }, bytesPerElement: 2 };
  const c = countsForNode(indexer, ctx);
  // Four independent group scores over 256 keys; max of two 128-token
  // blocks per group; one TopK scan over 2 block candidates per group.
  assert.equal(c.matrix, 4 * 256 * 128);
  assert.equal(c.vector, 4 * 256 + 4 * (256 - 2) + 4 * 2);
  assert.equal(c.bytes.indexRead, 256 * 128 * 2, "one shared index K cache");
  const d = countsForNode(sparse, ctx);
  assert.equal(d.bytes.kvRead, 4 * 256 * (128 + 128) * 2);
  const rows = computeNodeCosts(graph, normalized, ctx.options);
  const parent = rows.find(row => row.node.id === indexer.canonical_id);
  assert.ok(parent.compute_macs > 0);
  const children = rows.filter(row => row.node.id.startsWith(`${indexer.canonical_id}.`));
  assert.equal(children.length, 5);
  assert.ok(children.every(row => row.compute_macs === 0 && row.actions === null));
});
