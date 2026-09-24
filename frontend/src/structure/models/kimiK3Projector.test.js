import assert from "node:assert/strict";
import fs from "node:fs";
import test from "node:test";
import { buildStructureFromArtifacts, buildStructureFromConfig } from "../buildStructure.js";
import { buildSkeleton } from "../truth/skeleton.js";

const read = url => JSON.parse(fs.readFileSync(url, "utf8"));
const raw = read(new URL("../../../../models/moonshotai/Kimi-K3/config.json", import.meta.url));
const fixture = read(new URL("./__fixtures__/kimi-k3-projector-header.json", import.meta.url));

test("K3 PatchMergerMLPV2 is post-norm and bias-free on config and production paths", () => {
  for (const structure of [
    buildStructureFromConfig(raw, { modelId: fixture.model_id }),
    buildStructureFromArtifacts({ config: raw, modelId: fixture.model_id, revision: fixture.revision,
      checkpointTruth: { skeleton: buildSkeleton(fixture.tensors), tensor_count: fixture.tensors.length } }),
  ]) {
    const graph = structure.graph;
    const byId = new Map(graph.nodes.map(node => [node.canonical_id, node]));
    for (const path of ["mm_projector.proj.0", "mm_projector.proj.1", "mm_projector.proj.2", "mm_projector.post_norm"]) {
      assert.ok(byId.has(path), `${path}: published V2 forward`);
    }
    assert.equal(byId.has("mm_projector.pre_norm"), false);
    const pairs = graph.edges.map(edge => [edge.source_canonical_id, edge.target_canonical_id]);
    for (const [a, b] of [["0", "1"], ["1", "2"], ["2", "post_norm"]]) {
      assert.ok(pairs.some(([from, to]) =>
        from === `mm_projector.proj.${a}` && to === `mm_projector.${b === "post_norm" ? b : `proj.${b}`}`));
    }
    for (const tensor of fixture.tensors) {
      const bound = graph.nodes.filter(node => node.tensor_names?.includes(tensor.name));
      if (structure.diagnostics?.strategy?.includes("truth")) {
        assert.equal(bound.length, 1, `${tensor.name}: exactly one checkpoint owner`);
        assert.deepEqual(bound[0].weight_shapes.weight, tensor.shape);
      }
      const node = byId.get(tensor.name.replace(/\.weight$/, ""));
      assert.deepEqual(node.attributes.weightMatrices.flatMap(matrix =>
        Array.from({ length: matrix.matrices || 1 }, () => matrix.shape)), [tensor.shape]);
    }
    assert.equal(byId.get("mm_projector").attributes.class, "PatchMergerMLPV2");
  }
});
