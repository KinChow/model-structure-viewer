import assert from "node:assert/strict";
import fs from "node:fs";
import test from "node:test";
import { buildStructureFromArtifacts, buildStructureFromConfig } from "../buildStructure.js";
import { normalizeConfig } from "../config/normalize.js";
import { countsForNode } from "../operators/formulas/extractor.js";

const root = new URL("../../../../models/", import.meta.url);
const read = url => JSON.parse(fs.readFileSync(url, "utf8"));

test("MiniMax-M3 production wrapper paths bind to the readable vision graph", () => {
  for (const variant of ["MiniMax-M3", "MiniMax-M3-MXFP8"]) {
    const modelId = `MiniMaxAI/${variant}`;
    const config = read(new URL(`${modelId}/config.json`, root));
    const graph = buildStructureFromConfig(config, { modelId }).graph;
    const ids = new Set(graph.nodes.map(node => node.canonical_id));
    for (const id of [
      "vision_tower.embeddings.proj",
      "vision_tower.layers.0.self_attn.q_proj",
      "vision_tower.layers.0.self_attn.k_proj",
      "vision_tower.layers.0.self_attn.v_proj",
      "vision_tower.pre_layrnorm",
      "vision_tower.rotary_emb",
    ]) assert.ok(ids.has(id), `${modelId}: missing ${id}`);
    assert.equal(ids.has("vision_tower.vision_model"), false);

    const tensors = [
      ["vision_tower.vision_model.embeddings.patch_embedding.weight", [1280, 3, 2, 14, 14]],
      ["vision_tower.vision_model.encoder.layers.0.self_attn.q_proj.weight", [1280, 1280]],
      ["vision_tower.vision_model.encoder.layers.1.self_attn.q_proj.weight", [1280, 1280]],
      ["vision_tower.vision_model.pre_layrnorm.weight", [1280]],
    ].map(([name, shape]) => ({ name, shape, dtype: "BF16" }));
    const production = buildStructureFromArtifacts({
      modelId, config, checkpointTruth: { tensors },
    });
    assert.equal(production.graph.nodes.some(node => node.canonical_id === "checkpoint_gaps"), false);
    assert.equal(production.source.diagnostics.bound_tensors, 3);
  }
});

test("GLM-5.3-Flash production graph includes post norm, downsample, and released merger paths", () => {
  for (const variant of ["GLM-5.3-Flash", "GLM-5.3-Flash-BF16"]) {
    const modelId = `zai-org/${variant}`;
    const config = read(new URL(`${modelId}/config.json`, root));
    const graph = buildStructureFromConfig(config, { modelId }).graph;
    const ids = new Set(graph.nodes.map(node => node.canonical_id));
    for (const id of [
      "visual.blocks.0.attn.qkv",
      "visual.blocks.0.mlp.gate_proj",
      "visual.blocks.0.attn.q_norm",
      "visual.post_layernorm",
      "visual.downsample",
      "visual.merger.proj",
      "visual.merger.post_projection_norm",
      "visual.merger.gate_proj",
      "visual.merger.up_proj",
      "visual.merger.down_proj",
    ]) assert.ok(ids.has(id), `${modelId}: missing ${id}`);
    assert.equal(ids.has("visual.0.qkv_proj"), false);
    const tensors = [
      ["model.visual.blocks.0.attn.qkv.weight", [3072, 1024]],
      ["model.visual.blocks.1.attn.qkv.weight", [3072, 1024]],
      ["model.visual.post_layernorm.weight", [1024]],
      ["model.visual.downsample.weight", [4096, 1024, 2, 2]],
      ["model.visual.downsample.bias", [4096]],
      ["model.visual.merger.proj.weight", [4096, 4096]],
      ["model.visual.merger.post_projection_norm.weight", [4096]],
      ["model.visual.merger.post_projection_norm.bias", [4096]],
      ["model.visual.merger.gate_proj.weight", [10240, 4096]],
      ["model.visual.merger.up_proj.weight", [10240, 4096]],
      ["model.visual.merger.down_proj.weight", [4096, 10240]],
    ].map(([name, shape]) => ({ name, shape, dtype: "BF16" }));
    const production = buildStructureFromArtifacts({
      modelId, config, checkpointTruth: { tensors },
    });
    assert.equal(production.graph.nodes.some(node => node.canonical_id === "checkpoint_gaps"), false);
    assert.ok(production.graph.edges.some(edge =>
      edge.source_canonical_id === "visual.post_layernorm"
      && edge.target_canonical_id === "visual.downsample"));
    assert.ok(production.graph.edges.some(edge =>
      edge.source_canonical_id === "visual.downsample"
      && edge.target_canonical_id === "visual.merger"));
    const downsample = production.graph.nodes.find(node => node.canonical_id === "visual.downsample");
    assert.equal(downsample.attributes.operator_id, "vision_downsample");
    const normalized = normalizeConfig(config);
    const counts = countsForNode(downsample, {
      config: normalized,
      options: {
        batch: 1,
        sequence: normalized.visionTokens,
        phase: "prefill",
        vision: true,
        visionTokens: normalized.visionTokens,
      },
      path: downsample.id,
      bytesPerElement: 2,
    });
    // The 2x2 Conv2d emits one token per spatial merge, not one GEMM row
    // for every pre-merge patch token.
    assert.equal(counts.matrix, 256 * 4096 * 4096);
  }
});
