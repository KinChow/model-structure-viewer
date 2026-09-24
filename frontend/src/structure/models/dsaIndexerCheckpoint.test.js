import assert from "node:assert/strict";
import fs from "node:fs";
import test from "node:test";
import { buildStructureFromArtifacts, buildStructureFromConfig } from "../buildStructure.js";
import { normalizeConfig } from "../config/normalize.js";
import { countsForNode } from "../operators/formulas/extractor.js";
import { buildSkeleton } from "../truth/skeleton.js";

const root = new URL("../../../../models/", import.meta.url);
const read = url => JSON.parse(fs.readFileSync(url, "utf8"));
const variants = ["deepseek-ai/DeepSeek-V3.2", "zai-org/GLM-5", "zai-org/GLM-5.1",
  "zai-org/GLM-5.2", "zai-org/GLM-5.2-FP8", "zai-org/GLM-5.3", "zai-org/GLM-5.3-BF16",
  "zai-org/GLM-5.3-Flash", "zai-org/GLM-5.3-Flash-BF16"];

for (const modelId of variants) {
  for (const loading of ["config-only", "production-artifacts"]) {
    test(`DSA indexer uses checkpoint-owned projections: ${modelId} ${loading}`, () => {
      const dir = new URL(`${modelId}/`, root);
      const config = read(new URL("config.json", dir));
      const graph = (loading === "config-only"
        ? buildStructureFromConfig(config, { modelId })
        : buildStructureFromArtifacts({ config, modelId,
          checkpointTruth: read(new URL("header-truth.json", dir)),
          sourceRef: read(new URL("source-ref.json", dir)) })).graph;
      const indexer = graph.nodes.find(node =>
        ["dsa_indexer", "dsa_kpool_indexer"].includes(node.attributes?.operator_id) &&
        node.canonical_id.includes(".self_attn."));
      assert.ok(indexer, modelId);
      const prefix = indexer.canonical_id;
      const get = suffix => graph.nodes.find(n => n.canonical_id === `${prefix}.${suffix}`);
      const norm = normalizeConfig(config);
      assert.deepEqual(get("wq_b")?.attributes.weightMatrices[0].shape,
        [norm.dsaIndexHeads * norm.dsaIndexHeadDim, norm.qLoraRank]);
      assert.deepEqual(get("wk")?.attributes.weightMatrices[0].shape,
        [norm.dsaIndexHeadDim, norm.hiddenSize]);
      assert.deepEqual(get("weights_proj")?.attributes.weightMatrices[0].shape,
        [norm.dsaIndexHeads, norm.hiddenSize]);
      assert.equal(get("q_proj"), undefined);
      assert.equal(get("wk_weights_proj"), undefined);
      const edge = (from, to) => graph.edges.some(e =>
        e.source_canonical_id === `${prefix}.${from}` && e.target_canonical_id === to);
      assert.ok(edge("wk", `${prefix}.k_norm`));
      assert.ok(edge("wq_b", prefix));
      assert.ok(edge("weights_proj", prefix));
      assert.ok(graph.edges.some(e =>
        e.source_canonical_id === `${prefix.slice(0, -".indexer".length)}.q_a_norm`
        && e.target_canonical_id === `${prefix}.wq_b`));
      const count = suffix => countsForNode(get(suffix), {
        config: norm, options: { batch: 2, sequence: 5, phase: "prefill" }, bytesPerElement: 2,
      });
      assert.equal(count("wq_b").matrix, 10 * norm.qLoraRank * norm.dsaIndexHeads * norm.dsaIndexHeadDim);
      assert.equal(count("wk").matrix + count("weights_proj").matrix,
        10 * norm.hiddenSize * (norm.dsaIndexHeadDim + norm.dsaIndexHeads));
    });
  }
}

test("GLM-5.2 IndexCache release contains 21 physical indexers, never 57 shared-layer copies", () => {
  const config = read(new URL("zai-org/GLM-5.2/config.json", root));
  // Full-layer IDs and exact child names independently checked against the
  // published model.safetensors.index.json and HF forward.
  const full = [0, 1, 2, ...Array.from({ length: 18 }, (_, i) => 6 + i * 4)];
  const graph = buildStructureFromConfig(config, { modelId: "zai-org/GLM-5.2" }).graph;
  // The display folds identical layers 0..1 into one representative with
  // repeat=2. Physical release IDs must be checked against the weight map,
  // not inferred by counting visible Graph IR representatives.
  const stack = graph.nodes.find(n => n.canonical_id === "layers");
  const groups = graph.nodes.filter(n => n.parent_id === stack.id);
  assert.equal(groups.find(n => n.canonical_id === "layers.0")?.repeat, 2);
  for (const layer of full.filter(layer => layer !== 1)) {
    for (const suffix of ["wq_b", "wk", "weights_proj", "k_norm"]) {
      assert.ok(graph.nodes.find(n => n.canonical_id === `layers.${layer}.self_attn.indexer.${suffix}`));
    }
  }
  for (const layer of [3, 4, 5, 7, 75, 76, 77]) {
    assert.equal(graph.nodes.some(n => n.canonical_id.startsWith(`layers.${layer}.self_attn.indexer.`)), false);
  }
});

test("published GLM indexer tensor paths bind once through both tensor and skeleton truth", () => {
  const modelId = "zai-org/GLM-5.2";
  const config = read(new URL(`${modelId}/config.json`, root));
  const tensors = [
    ["wq_b.weight", [4096, 2048]], ["wk.weight", [128, 6144]],
    ["weights_proj.weight", [32, 6144]], ["k_norm.weight", [128]],
    ["k_norm.bias", [128]],
  ].map(([suffix, shape]) => ({
    name: `model.layers.2.self_attn.indexer.${suffix}`, shape, dtype: "BF16",
  }));
  for (const checkpointTruth of [{ tensors }, { skeleton: buildSkeleton(tensors), tensor_count: tensors.length }]) {
    const { graph } = buildStructureFromArtifacts({ config, modelId, checkpointTruth });
    for (const tensor of tensors) {
      const owners = graph.nodes.filter(n => n.tensor_names?.includes(tensor.name));
      assert.equal(owners.length, 1, tensor.name);
      assert.equal(owners[0].canonical_id,
        tensor.name.replace(/^model\./, "").replace(/\.(weight|bias)$/, ""));
      assert.equal(owners[0].value_source, "checkpoint");
    }
    assert.ok(!graph.nodes.some(n => n.canonical_id.includes(".wk_weights_proj")));
    assert.ok(!graph.nodes.find(n => n.canonical_id === "layers.3.self_attn.index_reuse").tensor_names?.length);
  }
});
