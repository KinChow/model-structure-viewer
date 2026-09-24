import assert from "node:assert/strict";
import fs from "node:fs";
import test from "node:test";
import { buildStructureFromArtifacts, buildStructureFromConfig } from "../buildStructure.js";
import { normalizeConfig } from "../config/normalize.js";

const root = new URL("../../../../models/", import.meta.url);
const read = url => JSON.parse(fs.readFileSync(url, "utf8"));
const variants = ["Kimi-K2.5", "Kimi-K2.6", "Kimi-K2.7-Code"];

test("Kimi K2.5-family vision graph follows the published encoder.blocks checkpoint layout", () => {
  for (const variant of variants) {
    const modelId = `moonshotai/${variant}`;
    const config = read(new URL(`${modelId}/config.json`, root));
    const graph = buildStructureFromConfig(config, { modelId }).graph;
    const ids = new Set(graph.nodes.map(node => node.canonical_id));
    for (const id of [
      "vision_tower.patch_embed.proj",
      "vision_tower.encoder.blocks.0.norm0",
      "vision_tower.encoder.blocks.0.wqkv",
      "vision_tower.encoder.blocks.0.wo",
      "vision_tower.encoder.blocks.0.norm1",
      "vision_tower.encoder.blocks.0.mlp.fc0",
      "vision_tower.encoder.blocks.0.mlp.fc1",
      "vision_tower.encoder.final_layernorm",
    ]) assert.ok(ids.has(id), `${modelId}: missing ${id}`);
    assert.equal(ids.has("vision_tower.0.qkv_proj"), false);
    assert.equal(ids.has("vision_tower.0.wqkv"), false);
    const group = graph.nodes.find(node => node.canonical_id === "vision_tower.encoder.blocks.0");
    assert.equal(group.attributes.range, `0..${normalizeConfig(config).visionLayers - 1}`);
    assert.equal(group.repeat, normalizeConfig(config).visionLayers);
  }
});

test("Kimi K2.5-family production tensor paths bind to the folded encoder block", () => {
  const modelId = "moonshotai/Kimi-K2.5";
  const config = read(new URL(`${modelId}/config.json`, root));
  const tensors = [
    ["vision_tower.patch_embed.proj.weight", [1152, 3, 2, 14, 14]],
    ["vision_tower.encoder.blocks.0.wqkv.weight", [3456, 1152]],
    ["vision_tower.encoder.blocks.1.wqkv.weight", [3456, 1152]],
    ["vision_tower.encoder.blocks.0.wo.weight", [1152, 1152]],
    ["vision_tower.encoder.blocks.1.wo.weight", [1152, 1152]],
    ["vision_tower.encoder.final_layernorm.weight", [1152]],
  ].map(([name, shape]) => ({ name, shape, dtype: "BF16" }));
  const graph = buildStructureFromArtifacts({
    modelId,
    config,
    checkpointTruth: { tensors },
  }).graph;
  assert.equal(graph.nodes.some(node => node.canonical_id === "checkpoint_gaps"), false);
  for (const tensor of tensors.filter(tensor => tensor.name.includes(".0.")
    || !tensor.name.includes(".blocks."))) {
    const owners = graph.nodes.filter(node => node.tensor_names?.includes(tensor.name));
    assert.equal(owners.length, 1, `${tensor.name}: representative tensor owner`);
  }
  const group = graph.nodes.find(node => node.canonical_id === "vision_tower.encoder.blocks.0");
  assert.equal(group.repeat, normalizeConfig(config).visionLayers);
  assert.equal(group.attributes.range, `0..${normalizeConfig(config).visionLayers - 1}`);
  assert.ok(graph.nodes.some(node =>
    node.canonical_id === "vision_tower.encoder.blocks.0.wqkv"
    && node.tensor_names?.includes("vision_tower.encoder.blocks.0.wqkv.weight")));
});
