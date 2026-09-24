import assert from "node:assert/strict";
import fs from "node:fs";
import test from "node:test";
import { buildStructureFromArtifacts, buildStructureFromConfig } from "../buildStructure.js";
import { normalizeConfig } from "../config/normalize.js";
import { countsForNode } from "../operators/formulas/extractor.js";
import { aggregateCost } from "../../cost/aggregate.js";
import { actionsByFormulaGroup } from "../../cost/ui.js";
import { buildNodeLens } from "../../diagram/lens.js";

const root = new URL("../../../../models/", import.meta.url);
const read = url => fs.existsSync(url) ? JSON.parse(fs.readFileSync(url, "utf8")) : null;
const catalog = read(new URL("catalog.json", root));
// Frozen expectations from primary forward implementations, not the builder.
const semantics = {
  qwen3_5: "placeholder_scatter", qwen3_5_moe: "placeholder_scatter",
  qwen4_exp: "placeholder_scatter", glm5_next: "placeholder_scatter",
  minimax_m3_vl: "placeholder_scatter", kimi_k25: "placeholder_expand",
  kimi_k3: "placeholder_expand", deepseek_v41: "image_span_overwrite",
  deepseek_v4: "image_span_overwrite",
};

for (const loading of ["config", "artifacts"]) {
  test(`39 multimodal entries have independent text/vision branches (${loading})`, () => {
    let checked = 0;
    for (const entry of catalog.models) {
      const config = read(new URL(entry.config_path, root));
      const n = normalizeConfig(config);
      const dir = new URL("./", new URL(entry.config_path, root));
      const s = loading === "config" ? buildStructureFromConfig(config, { modelId: entry.model_id })
        : buildStructureFromArtifacts({ config, modelId: entry.model_id,
          checkpointTruth: read(new URL("skeleton-truth.json", dir)) || read(new URL("header-truth.json", dir)),
          sourceRef: read(new URL("source-ref.json", dir)) });
      const g = s.graph;
      const fusion = g.nodes.find(node => node.canonical_id === "multimodal_fusion");
      if (!n.hasVision) {
        assert.equal(fusion, undefined, entry.model_id);
        continue;
      }
      checked++;
      assert.ok(fusion, entry.model_id);
      assert.equal(fusion.attributes.fusion_semantics, semantics[n.modelType], entry.model_id);
      assert.equal(fusion.attributes.sequence_policy, "workload_is_post_fusion");
      const embed = g.nodes.find(node => node.canonical_id === "embed_tokens");
      const vision = g.nodes.find(node => node.type === "vision-encoder");
      const projector = g.nodes.find(node => node.canonical_id === "projector");
      const imageInput = g.nodes.find(node => node.canonical_id === "image_input");
      const textInput = g.nodes.find(node => node.canonical_id === "text_input");
      const incoming = target => g.edges.filter(e => e.target === target.id).map(e => e.source);
      assert.deepEqual(incoming(embed), [textInput.id], "visual features are never embedding lookup indices");
      assert.deepEqual(incoming(vision), [imageInput.id]);
      assert.deepEqual(new Set(incoming(fusion)), new Set([embed.id, (projector || vision).id, textInput.id]));
      const entryStack = g.nodes.find(node => node.type === (n.modelType === "deepseek_v41" ? "encoder" : "decoder"));
      assert.ok(g.edges.some(e => e.source === fusion.id && e.target === entryStack.id));
      assert.equal(g.version, 2);
      assert.equal(g.schema_version, 2);
      assert.ok(!fusion.attributes.weightMatrices?.length);
      assert.ok(!fusion.tensor_names?.length);
      const decode = countsForNode(fusion, { config: n, options: { phase: "decode", batch: 2, sequence: 8 }, bytesPerElement: 2 });
      assert.equal(decode.matrix, 0);
      assert.equal(decode.bytes.weights, 0);
      assert.equal(decode.bytes.actIn, 0, "cached vision is not scattered again every decode token");
      const prefill = countsForNode(fusion, { config: n, options: { phase: "prefill", batch: 2, sequence: 8 }, bytesPerElement: 2 });
      assert.equal(prefill.matrix, 0);
      assert.equal(prefill.bytes.weights, 0);
      assert.equal(prefill.bytes.actIn, null, "placeholder occupancy and materialization are not supplied by sequence alone");
      if (n.modelType === "deepseek_v41") {
        assert.ok(g.edges.some(e => e.relation === "kv-projection"));
        const draft = g.nodes.find(node => node.type === "dspark");
        assert.ok(g.edges.some(e => e.target === draft.id && g.nodes.find(n => n.id === e.source)?.type === "decoder"));
      }
    }
    assert.equal(checked, 39);
  });
}

test("fusion unknown traffic survives totals, per-group UI and node lens; compute stays known", () => {
  const config = read(new URL("Qwen/Qwen3.5-0.8B/config.json", root));
  const s = buildStructureFromConfig(config);
  const n = normalizeConfig(config);
  const cost = aggregateCost({ graph: s.graph, config: n, batch: 1, sequence: 16 });
  assert.equal(cost.computeComplete, true);
  assert.ok(cost.totalMacs > 0);
  assert.equal(cost.actions.actIn, null);
  assert.equal(cost.actions.actOut, null);
  const memory = actionsByFormulaGroup(cost).find(group => group.group === "memory");
  assert.equal(memory.actions.bytes.actIn, null);
  const fusion = s.graph.nodes.find(node => node.canonical_id === "multimodal_fusion");
  const lens = buildNodeLens(s, { peak_flops: { bf16: 1000 }, memory_bandwidth: 100 });
  assert.equal(lens.nodes[fusion.id].times.memory, null);
  assert.equal(lens.nodes[fusion.id].metrics.vramBytes, null);
});

test("semantic fusion cannot bind a checkpoint module with a coincidentally matching path", () => {
  const config = read(new URL("Qwen/Qwen3.5-0.8B/config.json", root));
  const truth = { tensors: [{ name: "model.multimodal_fusion.weight", shape: [2, 2], dtype: "BF16" }] };
  const s = buildStructureFromArtifacts({ config, checkpointTruth: truth });
  const fusion = s.graph.nodes.find(node => node.canonical_id === "multimodal_fusion");
  assert.ok(!fusion.tensor_names?.length);
  assert.equal(s.graph.nodes.filter(node => node.tensor_names?.includes(truth.tensors[0].name)).length, 1,
    "unmatched checkpoint fact must survive as a gap, not be discarded");
});
