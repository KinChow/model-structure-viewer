import assert from "node:assert/strict";
import fs from "node:fs";
import test from "node:test";
import { buildStructureFromArtifacts, buildStructureFromConfig } from "../buildStructure.js";
import { normalizeConfig } from "../config/normalize.js";
import { aggregateCost } from "../../cost/aggregate.js";

const root = new URL("../../../../models/", import.meta.url);
const catalog = JSON.parse(fs.readFileSync(new URL("catalog.json", root), "utf8"));
const read = url => fs.existsSync(url) ? JSON.parse(fs.readFileSync(url, "utf8")) : null;

for (const loading of ["config", "artifacts"]) {
  test(`all 39 vision models charge image encoding only on image prefill (${loading})`, () => {
    let checked = 0;
    for (const entry of catalog.models) {
      const config = read(new URL(entry.config_path, root));
      const normalized = normalizeConfig(config);
      if (!normalized.hasVision) continue;
      checked++;
      const dir = new URL("./", new URL(entry.config_path, root));
      const graph = loading === "config"
        ? buildStructureFromConfig(config, { modelId: entry.model_id }).graph
        : buildStructureFromArtifacts({ config, modelId: entry.model_id,
          checkpointTruth: read(new URL("skeleton-truth.json", dir)) || read(new URL("header-truth.json", dir)),
          sourceRef: read(new URL("source-ref.json", dir)) }).graph;
      const ctx = { graph, config: normalized, batch: 1, sequence: 64 };
      const prefill = aggregateCost({ ...ctx, phase: "prefill" });
      const decode = aggregateCost({ ...ctx, phase: "decode" });
      const visual = cost => cost.nodes.filter(row => row.node.attributes?.modality === "vision" && row.actions);
      const prefillVisual = visual(prefill);
      const decodeVisual = visual(decode);
      assert.ok(prefillVisual.some(row => row.actions.matrix > 0), `${entry.model_id}: image encoding still runs at prefill`);
      assert.ok(prefillVisual.some(row => row.actions.bytes.weights > 0), `${entry.model_id}: vision weights execute at prefill`);
      assert.ok(decodeVisual.length > 0, `${entry.model_id}: vision path remains inspectable`);
      for (const row of decodeVisual) {
        assert.deepEqual([row.actions.matrix, row.actions.vector, row.actions.sfu,
          row.actions.bytes.weights, row.actions.bytes.actIn, row.actions.bytes.actOut,
          row.actions.bytes.kvRead ?? 0, row.actions.bytes.indexRead ?? 0],
        [0, 0, 0, 0, 0, 0, 0, 0], `${entry.model_id}: ${row.node.id} must not rerun for text decode`);
      }
      assert.equal(decode.memory.weightBytes, prefill.memory.weightBytes,
        `${entry.model_id}: skipping execution cannot remove resident vision weights`);
      assert.ok(decode.totalMacs > 0, `${entry.model_id}: language decoder still runs`);
    }
    assert.equal(checked, 39);
  });
}
