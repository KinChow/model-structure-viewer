import assert from "node:assert/strict";
import fs from "node:fs";
import test from "node:test";
import { buildStructureFromArtifacts, buildStructureFromConfig } from "../buildStructure.js";
import { normalizeConfig } from "../config/normalize.js";
import { countsForNode } from "../operators/formulas/extractor.js";
import { gatedResidualCounts, gatedResidualDecomposition } from "../operators/formulas/gatedResidual.js";
import { evaluateDecomposition } from "../operators/formulas/atoms.js";
import { computeNodeCosts } from "../../cost/compute.js";
import { buildSkeleton } from "../truth/skeleton.js";

// Independent contract: pinned Qwen3.8-Flash-Next report §2.2 Eq(30–34),
// Fig.1 and TextModel.forward. See qwen_flash_next_gr_repair.md.
// Expectations must not be generated from the structure builder.
const root = new URL("../../../../models/", import.meta.url);
const read = url => JSON.parse(fs.readFileSync(url, "utf8"));
const grFixture = read(new URL("./__fixtures__/qwen-flash-next-gr-header.json", import.meta.url));

for (const variant of ["Qwen3.8-Flash-Next", "Qwen3.8-Flash-Next-FP8"]) {
  for (const loading of ["config-only", "production-artifacts"]) {
    test(`GR retains four wide streams and contracts only on read: ${variant} ${loading}`, () => {
      const dir = new URL(`Qwen/${variant}/`, root);
      const config = read(new URL("config.json", dir));
      const structure = loading === "config-only" ? buildStructureFromConfig(config)
        : buildStructureFromArtifacts({
          config, modelId: `Qwen/${variant}`,
          checkpointTruth: read(new URL("header-truth.json", dir)),
          sourceRef: read(new URL("source-ref.json", dir)),
        });
      const get = id => {
        const node = structure.graph.nodes.find(n => n.canonical_id === id);
        assert.ok(node, `missing ${id}`);
        return node;
      };
      assert.equal(structure.graph.version, 2);
      const truth = grFixture.find(item => item.model_id === `Qwen/${variant}`);
      assert.ok(truth);
      for (const [tensor, metadata] of Object.entries(truth.tensors)) {
        const id = tensor.replace(/^model\.language_model\./, "").replace(/\.weight$/, "");
        assert.deepEqual(get(id).attributes.weightMatrices[0].shape, metadata.shape, tensor);
        assert.equal(get(id).attributes.weightMatrices[0].quantizable, false);
        assert.equal(metadata.dtype, "BF16");
      }
      // Published H=2560, nr=4. The residual state is NOT a single H vector.
      assert.deepEqual(get("layers.0").input_shape, [-1, -1, 10240]);
      assert.deepEqual(get("layers.0").output_shape, [-1, -1, 10240]);
      // TextDecoderLayer.forward adds the PLE delta to the original wide
      // stream BEFORE predicting either GR gate. It is not a replacement.
      for (const id of ["layers.1.ple", "layers.1.ple_residual_add"]) {
        assert.deepEqual(get(id).input_shape, [-1, -1, 10240]);
        assert.deepEqual(get(id).output_shape, [-1, -1, 10240]);
      }
      const inputs = structure.graph.edges.filter(e =>
        e.target_canonical_id === "layers.1.ple_residual_add").map(e => e.source_canonical_id);
      assert.deepEqual(new Set(inputs), new Set(["layers.1.layer_in", "layers.1.ple"]));
      assert.ok(structure.graph.edges.some(e => e.source_canonical_id === "layers.1.ple_residual_add"
        && e.target_canonical_id === "layers.1.attn_hyper_connection.streams"));
      for (const name of ["attn_hyper_connection", "mlp_hyper_connection"]) {
        const module = get(`layers.0.${name}`);
        assert.deepEqual(module.input_shape, [-1, -1, 10240]);
        assert.deepEqual(module.output_shape, [-1, -1, 2560]);
        const descendants = structure.graph.nodes.filter(n => n.canonical_id.startsWith(`${module.canonical_id}.`));
        // Separate data-dependent elementwise read and per-stream scalar write.
        const readGate = descendants.find(n => n.attributes?.semantic_role === "gr_read_gate");
        const writeGate = descendants.find(n => n.attributes?.semantic_role === "gr_write_gate");
        assert.ok(readGate && writeGate, "both gates must be independently visible");
        assert.equal(readGate.output_shape.at(-1), 10240);
        assert.equal(writeGate.output_shape.at(-1), 4);
        assert.ok(structure.graph.edges.some(e => e.source === writeGate.id &&
          e.target_canonical_id?.startsWith("layers.0.") &&
          !e.target_canonical_id.startsWith(`${module.canonical_id}.`)),
        "write gate must reach the corresponding widened residual update");
      }
      const final = get("hyper_connection_mixer");
      assert.deepEqual(final.input_shape, [-1, -1, 10240]);
      assert.deepEqual(final.output_shape, [-1, -1, 2560]);
      assert.ok(!structure.graph.nodes.some(n => n.canonical_id.startsWith(`${final.canonical_id}.`) &&
        n.attributes?.semantic_role === "gr_write_gate"), "final read has no write gate");
      assert.equal(structure.graph.nodes.some(n => n.canonical_id === "norm" || n.canonical_id === "norm.rmsnorm"),
        false, "released text model has no extra final normalization");
      assert.ok(structure.graph.edges.some(e => e.source_canonical_id === "hyper_connection_mixer"
        && e.target_canonical_id === "lm_head"));
    });
  }
}

test("GR final read normalizes each branch, not the flattened stream, and owns no write projection", () => {
  const config = normalizeConfig(read(new URL("Qwen/Qwen3.8-Flash-Next/config.json", root)));
  const tiny = { ...config, hiddenSize: 3, hyperConnectionCount: 4, hyperConnectionLowrank: 2 };
  const node = { attributes: { operator_id: "hyper_connection", hc_use_combine: false } };
  const actions = countsForNode(node, {
    config: tiny, options: { batch: 1, sequence: 2, phase: "prefill" }, bytesPerElement: 2,
  });
  // T=2: two low-rank linear maps; one rsqrt per branch; sigmoid=2 SFU.
  // No injection projection or write sigmoid in the final read.
  assert.equal(actions.matrix, 2 * (12 * 2 + 2 * 12));
  assert.equal(actions.bytes.weights, (12 + 12 * 2 + 2 * 12) * 2);
  assert.equal(actions.sfu, 2 * (4 + 2 * 2 + 2 * 12));
});

test("GR independent tiny read/write counts and atom decomposition agree", () => {
  // Literal hand calculation, T=2,H=3,nr=4,rank=2,b=2.
  for (const [stage, useCombine, matrix, vector, sfu, weights, actIn, actOut] of [
    ["read", true, 192, 188, 80, 216, 48, 28],
    ["read", false, 96, 172, 64, 120, 48, 12],
    ["write", true, 0, 48, 0, 0, 76, 48],
  ]) {
    const p = { tokens: 2, hidden: 3, streams: 4, lowrank: 2, b: 2, stage, useCombine };
    const counts = gatedResidualCounts(p);
    assert.deepEqual(counts, { matrix, vector, sfu, bytes: { weights, actIn, actOut } });
    const atoms = evaluateDecomposition(gatedResidualDecomposition(p));
    for (const key of ["matrix", "vector", "sfu"]) assert.equal(atoms[key], counts[key]);
  }
});

test("GR composite read bills once while real child projections own capacity", () => {
  const raw = read(new URL("Qwen/Qwen3.8-Flash-Next/config.json", root));
  const { graph } = buildStructureFromConfig(raw);
  const config = normalizeConfig(raw);
  const rows = computeNodeCosts(graph, config, { batch: 1, sequence: 2, phase: "prefill" });
  const id = "layers.0.attn_hyper_connection";
  const parent = rows.find(r => r.node.id === id);
  assert.ok(parent.compute_macs > 0);
  const children = rows.filter(r => r.node.id.startsWith(`${id}.`));
  assert.equal(children.reduce((n, row) => n + (row.compute_macs || 0), 0), 0);
  const owner = graph.nodes.filter(n => n.canonical_id.startsWith(`${id}.`))
    .flatMap(n => n.attributes?.weightMatrices || []);
  assert.equal(owner.reduce((n, w) => n + w.shape.reduce((a, b) => a * b, 1), 0),
    10240 + 2 * 10240 * 320 + 4 * 10240);
  assert.deepEqual(graph.nodes.find(n => n.canonical_id === "residual_expand").output_shape, [-1, -1, 10240]);
});

test("GR real header tensors bind once through raw and skeleton truth paths", () => {
  for (const fixture of grFixture) {
    const tensors = Object.entries(fixture.tensors).map(([name, meta]) => ({ name, ...meta }));
    const config = read(new URL(`${fixture.model_id}/config.json`, root));
    for (const checkpointTruth of [{ tensors }, { skeleton: buildSkeleton(tensors), tensor_count: tensors.length }]) {
      const { graph } = buildStructureFromArtifacts({
        config, modelId: fixture.model_id, revision: fixture.revision, checkpointTruth,
      });
      for (const tensor of tensors) {
        const matches = graph.nodes.filter(n => n.tensor_names?.includes(tensor.name));
        assert.equal(matches.length, 1);
        assert.equal(matches[0].canonical_id, tensor.name.replace(/^model\.language_model\./, "").replace(/\.weight$/, ""));
        assert.deepEqual(matches[0].weight_shapes.weight, tensor.shape);
      }
      assert.ok(graph.nodes.filter(n => n.attributes?.checkpoint_module === false &&
        n.attributes?.semantic_role?.startsWith("gr_")).every(n => !n.tensor_names?.length));
    }
  }
});

test("MTP shares one hidden projection over four branches and explicitly adds the broadcast embedding", () => {
  const raw = read(new URL("Qwen/Qwen3.8-Flash-Next/config.json", root));
  const { graph } = buildStructureFromConfig(raw);
  const get = id => graph.nodes.find(n => n.canonical_id === id);
  const proj = get("mtp.fc_hidden");
  assert.deepEqual(proj.attributes.weightMatrices[0].shape, [2560, 2560]);
  assert.deepEqual(proj.input_shape, [-1, -1, 10240]);
  assert.deepEqual(proj.output_shape, [-1, -1, 10240]);
  const c = countsForNode(proj, { config: normalizeConfig(raw),
    options: { batch: 1, sequence: 2, phase: "prefill" }, bytesPerElement: 2 });
  assert.equal(c.matrix, 2 * 4 * 2560 * 2560);
  assert.equal(c.bytes.weights, 2560 * 2560 * 2, "shared matrix capacity/read not multiplied by branches");
  const edges = graph.edges;
  for (const [from, to] of [["layers", "mtp.pre_fc_norm_hidden"],
    ["embed_tokens", "mtp.pre_fc_norm_embedding"], ["mtp.fc_hidden", "mtp.input_add"],
    ["mtp.embedding_expand", "mtp.input_add"], ["mtp.input_add", "mtp.layer"]]) {
    assert.ok(edges.some(e => e.source_canonical_id === from && e.target_canonical_id === to), `${from} -> ${to}`);
  }
});
