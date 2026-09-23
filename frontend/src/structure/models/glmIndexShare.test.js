import assert from "node:assert/strict";
import fs from "node:fs";
import test from "node:test";
import { buildStructureFromArtifacts, buildStructureFromConfig } from "../buildStructure.js";
import { normalizeConfig } from "../config/normalize.js";
import { countsForNode } from "../operators/formulas/extractor.js";

const modelsRoot = new URL("../../../../models/", import.meta.url);
const read = url => JSON.parse(fs.readFileSync(url, "utf8"));

function artifactsFor(modelId) {
  const configUrl = new URL(`${modelId}/config.json`, modelsRoot);
  const dir = new URL(`${modelId}/`, modelsRoot);
  return {
    config: read(configUrl),
    modelId,
    checkpointTruth: read(new URL("header-truth.json", dir)),
    sourceRef: read(new URL("source-ref.json", dir)),
  };
}

function graphFor(modelId, loading) {
  return loading === "config"
    ? buildStructureFromConfig(read(new URL(`${modelId}/config.json`, modelsRoot)), { modelId }).graph
    : buildStructureFromArtifacts(artifactsFor(modelId)).graph;
}

function childrenOf(graph, parentId) {
  return graph.nodes.filter(node => node.parent_id === parentId);
}

function indexShareNodes(graph) {
  return {
    indexers: graph.nodes.filter(node => node.canonical_id.startsWith("layers.") && node.attributes?.operator_id === "dsa_indexer"),
    reuses: graph.nodes.filter(node => node.attributes?.semantic_role === "index_reuse"),
  };
}

for (const loading of ["config", "artifacts"]) {
  test(`GLM-5.2/5.3 IndexShare uses source indexers and shared top-k references (${loading})`, () => {
    for (const modelId of ["zai-org/GLM-5.2", "zai-org/GLM-5.2-FP8", "zai-org/GLM-5.3", "zai-org/GLM-5.3-BF16"]) {
      const graph = graphFor(modelId, loading);
      const layers = graph.nodes.find(node => node.canonical_id === "layers");
      assert.ok(layers, `${modelId}: decoder stack missing`);
      const groups = childrenOf(graph, layers.id);
      assert.deepEqual(groups.slice(0, 4).map(node => node.attributes?.range), ["0..1", "2..2", "3..5", "6..6"]);

      const { indexers, reuses } = indexShareNodes(graph);
      assert.equal(indexers.length, 20, `${modelId}: compute representatives, excluding the draft head`);
      assert.equal(reuses.length, 19, `${modelId}: one folded reference per shared run`);
      const byId = new Map(graph.nodes.map(node => [node.id, node]));
      const logicalLayers = nodes => nodes.reduce((total, node) => {
        const attention = byId.get(node.parent_id);
        const layerGroup = byId.get(attention?.parent_id);
        return total + (layerGroup?.repeat || 1);
      }, 0);
      assert.equal(logicalLayers(indexers), 21, `${modelId}: compute layers`);
      assert.equal(logicalLayers(reuses), 57, `${modelId}: folded references cover all shared layers`);
      assert.deepEqual(new Set(reuses.map(node => node.attributes?.index_source_layer)), new Set(
        [2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 66, 70, 74],
      ));
      assert.ok(reuses.every(node => node.attributes?.index_source_layer_id
        === `layers.${node.attributes.index_source_layer}.self_attn.indexer`));
      assert.ok(indexers.every(node => !node.attributes?.reuse_previous_indices));

      const relations = graph.edges.filter(edge => edge.relation === "index-reuse");
      assert.equal(relations.length, 19, `${modelId}: one cross-layer relation per shared run`);
      assert.ok(relations.every(edge =>
        edge.source_canonical_id.endsWith(".self_attn.indexer")
        && edge.target_canonical_id.endsWith(".self_attn.index_reuse")));
      assert.ok(relations.some(edge =>
        edge.source_canonical_id === "layers.2.self_attn.indexer"
        && edge.target_canonical_id === "layers.3.self_attn.index_reuse"));
      for (const node of reuses) {
        const attention = byId.get(node.parent_id);
        const children = childrenOf(graph, attention.id);
        assert.ok(!children.some(child => child.canonical_id.includes(".indexer")), "shared layer does not instantiate an indexer");
        const sparse = children.find(child => child.attributes?.operator_id === "dsa_sparse_mla");
        assert.equal(sparse.attributes.cache_index_elements, 0);
        assert.equal(sparse.attributes.cache_index_growth_elements, 0);
        assert.equal(sparse.attributes.cache_kv_elements, 576, "independent MLA KV stays resident");
        for (const suffix of ["q_a_proj", "q_b_proj", "kv_a_proj", "kv_b_proj", "o_proj"]) {
          assert.ok(children.some(child => child.canonical_id.endsWith(`.${suffix}`)), suffix);
        }
        assert.equal(node.attributes.index_storage, "alias");
        assert.ok(!node.attributes.weightMatrices?.length);
        assert.equal(graph.edges.filter(edge => edge.target === sparse.id && edge.source === node.id).length, 1);
      }
    }
  });
}

test("GLM-5.3-Flash k-pool remains independent from IndexShare", () => {
  const graph = graphFor("zai-org/GLM-5.3-Flash", "artifacts");
  const { indexers, reuses } = indexShareNodes(graph);
  assert.equal(indexers.length, 0);
  assert.equal(reuses.length, 0);
  assert.equal(graph.nodes.filter(node => node.attributes?.operator_id === "dsa_kpool_indexer").length, 12);
  assert.equal(graph.edges.filter(edge => edge.relation === "index-reuse").length, 0);
});

test("IndexShare rejects a shared layer without a preceding source", () => {
  const config = read(new URL("zai-org/GLM-5.2/config.json", modelsRoot));
  config.indexer_types[0] = "shared";
  assert.throws(() => buildStructureFromConfig(config, { modelId: "zai-org/GLM-5.2" }),
    /shared layer 0 has no preceding full indexer/);
});

test("IndexShare reference has zero compute and weights, but shared attention still executes", () => {
  const config = read(new URL("zai-org/GLM-5.2/config.json", modelsRoot));
  const normalized = normalizeConfig(config);
  const graph = buildStructureFromConfig(config).graph;
  const reference = graph.nodes.find(node => node.canonical_id === "layers.3.self_attn.index_reuse");
  const attention = graph.nodes.find(node => node.canonical_id === "layers.3.self_attn.sparse_attention");
  for (const phase of ["prefill", "decode"]) {
    const ctx = { config: normalized, options: { phase, batch: 2, sequence: 8 }, bytesPerElement: 2 };
    const alias = countsForNode(reference, ctx);
    assert.equal(alias.matrix, 0);
    assert.equal(alias.vector, 0);
    assert.equal(alias.sfu, 0);
    assert.equal(alias.bytes.weights, 0);
    assert.equal(alias.bytes.actIn + alias.bytes.actOut, 0, "alias is not a copied index tensor");
    assert.ok(countsForNode(attention, ctx).matrix > 0);
  }
});
