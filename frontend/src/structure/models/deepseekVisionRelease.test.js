import assert from "node:assert/strict";
import fs from "node:fs";
import test from "node:test";
import { buildStructureFromArtifacts, buildStructureFromConfig } from "../buildStructure.js";
import { buildSkeleton } from "../truth/skeleton.js";
import { normalizeConfig } from "../config/normalize.js";
import { isVisionPath } from "../operators/formulas/extractor.js";
import { walkStructure } from "../../cost/traverse.js";

const root = new URL("../../../../models/", import.meta.url);
const fixture = JSON.parse(fs.readFileSync(new URL("./__fixtures__/deepseek-v4-vision-release-header.json", import.meta.url), "utf8"));
const read = url => JSON.parse(fs.readFileSync(url, "utf8"));

for (const released of fixture.models) {
  test(`${released.model_id}: published ViT forward and tensor layout`, () => {
    const config = read(new URL(`${released.model_id}/config.json`, root));
    const graph = buildStructureFromConfig(config, { modelId: released.model_id }).graph;
    const node = id => graph.nodes.find(n => n.canonical_id === id);
    const edge = (source, target) => graph.edges.some(e =>
      e.source_canonical_id === source && e.target_canonical_id === target);
    assert.equal(node("vision")?.type, "vision-encoder");
    assert.equal(node("visual"), undefined);
    assert.ok(isVisionPath("vision.blocks.0.attn.wqkv"));
    for (const id of ["vision.patch_embed.proj", "vision.blocks.0.norm1",
      "vision.blocks.0.attn.wqkv", "vision.blocks.0.attn.wo",
      "vision.blocks.0.norm2", "vision.blocks.0.mlp.w1", "vision.blocks.0.mlp.w2",
      "vision.norm"]) assert.ok(node(id), `${released.model_id}: ${id}`);
    for (const id of ["vision.patch_embed", "vision.blocks.0.input_norm",
      "vision.blocks.0.qkv_proj", "vision.position", "vision.blocks.0.post_norm"])
      assert.equal(node(id), undefined, `${id}: no fictitious checkpoint module`);
    assert.equal(node("vision.blocks.0")?.repeat, 32);
    assert.ok(edge("vision.patch_embed.proj", "vision.blocks.0"));
    assert.ok(edge("vision.blocks.0", "vision.norm"));
    assert.ok(edge("vision.blocks.0.attn.wqkv", "vision.blocks.0.attn.qkv_split"));
    assert.ok(edge("vision.blocks.0.attn.qkv_split", "vision.blocks.0.attn.rope"));
    assert.ok(edge("vision.blocks.0.block_input", "vision.blocks.0.residual_attn"));
    assert.ok(edge("vision.blocks.0.attn.context_merge", "vision.blocks.0.attn.wo"));
    assert.ok(edge("vision.blocks.0.attn.wo", "vision.blocks.0.residual_attn"));
    assert.ok(edge("vision.blocks.0.mlp.w2", "vision.blocks.0.residual_mlp"));
    assert.equal(node("vision.blocks.0.mlp.w1").attributes.bias, undefined);
    assert.deepEqual(node("vision.blocks.0.mlp.w1").attributes.weightMatrices[0].shape, [5632, 1024]);
    assert.deepEqual(node("vision.blocks.0.mlp.w2").attributes.weightMatrices[0].shape, [1024, 2816]);
    assert.equal(node("vision.blocks.0.attn.rope").attributes.position_encoding, "rope_2d");
    assert.equal(normalizeConfig(config).visionTokens > 0, true);
    // One released block: 2 RMS scales + QKV/WO (with biases) + SwiGLU
    // W1(2*2816,1024), W2(1024,2816). 32 independent copies plus patch
    // Linear(1024,588) with bias and final RMSNorm.
    const visionWeights = [];
    walkStructure(graph, ({ node: n, multiplier }) => {
      if (!n.id.startsWith("vision.")) return;
      for (const matrix of n.attributes?.weightMatrices || []) {
        visionWeights.push(matrix.shape.reduce((product, dim) => product * dim, 1)
          * (matrix.count || 1) * (matrix.matrices || 1) * multiplier);
      }
    });
    assert.equal(visionWeights.reduce((sum, count) => sum + count, 0), 411842560,
      "no fictitious position weights or duplicate repeated blocks");

    const actual = buildStructureFromArtifacts({
      config, modelId: released.model_id,
      checkpointTruth: { skeleton: buildSkeleton(released.tensors), tensor_count: released.tensors.length },
    }).graph;
    for (const tensor of released.tensors) {
      const owner = actual.nodes.filter(n => n.tensor_names?.includes(tensor.name));
      assert.equal(owner.length, 1, `${tensor.name}: one owner`);
      assert.equal(owner[0].canonical_id, tensor.name.replace(/\.(weight|bias)$/, ""));
      assert.equal(owner[0].value_source, "checkpoint");
      assert.equal(owner[0].params,
        released.tensors.filter(t => t.name.replace(/\.(weight|bias)$/, "") === owner[0].canonical_id)
          .reduce((sum, t) => sum + t.shape.reduce((p, d) => p * d, 1), 0));
    }
  });
}
