import assert from "node:assert/strict";
import fs from "node:fs";
import test from "node:test";
import { buildStructureFromArtifacts, buildStructureFromConfig } from "../buildStructure.js";
import { normalizeConfig } from "../config/normalize.js";
import { attentionModule } from "../layers/attention.js";
import { countsForNode } from "../operators/formulas/extractor.js";
import { buildSkeleton } from "../truth/skeleton.js";

const dir = new URL("../../../../models/moonshotai/Kimi-K3/", import.meta.url);
const read = name => JSON.parse(fs.readFileSync(new URL(name, dir), "utf8"));
const raw = read("config.json");
const config = normalizeConfig(raw);
// K3 technical report §2.1.2, Eq(7); released forward KimiMLAAttention.
// Query width 96*(128+64) differs from output/gate width 96*128.
for (const loading of ["config", "artifacts"]) {
  test(`K3 MLA keeps NoPE channels and independent full-rank output gate (${loading})`, () => {
    const structure = loading === "config" ? buildStructureFromConfig(raw)
      : buildStructureFromArtifacts({ config: raw, modelId: "moonshotai/Kimi-K3" });
    const { graph } = structure;
    const attentions = graph.nodes.filter(n => n.canonical_id.startsWith("layers.") && n.type === "attention" && n.attributes.attention_kind === "mla");
    assert.equal(attentions.reduce((sum, n) => sum + (graph.nodes.find(p => p.id === n.parent_id).repeat ?? 1), 0), 24);
    const get = suffix => graph.nodes.find(n => n.canonical_id === `layers.3.self_attn.${suffix}`);
    assert.equal(get("rope"), undefined);
    assert.equal(get("mla_gate"), undefined);
    const visualRope = graph.nodes.find(n => n.canonical_id === "vision_tower.0.rope");
    assert.equal(visualRope?.attributes.position_encoding, "rope_2d");
    assert.deepEqual(get("q_b_proj").output_shape, [-1, -1, 96, 192]);
    assert.deepEqual(get("kv_a_proj_with_mqa").attributes.weightMatrices[0].shape, [576, 7168]);
    assert.deepEqual(get("g_proj").attributes.weightMatrices[0].shape, [12288, 7168]);
    assert.equal(get("g_proj").attributes.operator_id, "linear");
    assert.ok(get("q_a_layernorm") && get("kv_a_layernorm"));
    const incoming = node => graph.edges.filter(e => e.target === node.id).map(e => e.source_canonical_id);
    assert.deepEqual(new Set(incoming(get("output_gate"))),
      new Set(["layers.3.self_attn.g_proj", "layers.3.self_attn.sdpa"]));
    assert.deepEqual(new Set(incoming(get("sdpa"))),
      new Set(["layers.3.self_attn.q_b_proj", "layers.3.self_attn.kv_b_proj", "layers.3.self_attn.kv_split"]));
    assert.deepEqual(incoming(get("o_proj")), ["layers.3.self_attn.output_gate"]);
    const ctx = { config, options: { batch: 2, sequence: 5, phase: "prefill" }, bytesPerElement: 2 };
    const gate = countsForNode(get("output_gate"), ctx);
    assert.equal(gate.matrix, 0);
    assert.equal(gate.bytes.weights, 0);
    assert.equal(gate.vector, 10 * 12288);
    assert.equal(gate.sfu, 2 * 10 * 12288);
    assert.equal(gate.bytes.actIn, 2 * 10 * 12288 * 2, "read gate logits and attention output");
    const projection = countsForNode(get("g_proj"), ctx);
    assert.equal(projection.matrix, 10 * 12288 * 7168);
    assert.equal(projection.bytes.weights, 2 * 12288 * 7168);
    const vision = countsForNode(visualRope, ctx);
    assert.equal(vision.matrix, 0);
    assert.equal(vision.bytes.weights, 0);
    // 2 batch * 1024 visual tokens * (Q+K) * 12 heads * 128 dims * 3 FLOPs.
    assert.equal(vision.vector, 2 * 1024 * 2 * 12 * 128 * 3);
  });
}

test("MLA optional query compression, positional encoding and gate emit only valid dependencies", () => {
  for (const mlaUseNope of [false, true]) for (const mlaUseOutputGate of [false, true]) {
    for (const qLoraRank of [null, 16]) {
      const module = attentionModule("attn", { ...config, mlaUseNope, mlaUseOutputGate, qLoraRank }, "mla");
      const ids = new Set(module.children.map(c => c.id.replace(/^attn\./, "")));
      for (const edge of module.attributes.dataflow_edges) {
        assert.ok(edge.every(id => ids.has(id)), edge.join(" -> "));
      }
      assert.equal(ids.has("rope"), !mlaUseNope);
      assert.equal(ids.has("g_proj"), mlaUseOutputGate);
      assert.equal(ids.has("output_gate"), mlaUseOutputGate);
      assert.equal(ids.has("q_proj"), qLoraRank === null);
    }
  }
});

test("released MLA header binds each real projection once, not to a semantic gate", () => {
  const fixture = JSON.parse(fs.readFileSync(new URL("./__fixtures__/kimi-k3-mla-header.json", import.meta.url)));
  for (const checkpointTruth of [{ tensors: fixture.tensors },
    { skeleton: buildSkeleton(fixture.tensors), tensor_count: fixture.tensors.length }]) {
    const structure = buildStructureFromArtifacts({ config: raw, modelId: fixture.model_id,
      revision: fixture.revision, checkpointTruth });
    for (const tensor of fixture.tensors) {
      const matches = structure.graph.nodes.filter(n => n.tensor_names?.includes(tensor.name));
      assert.equal(matches.length, 1, tensor.name);
      assert.equal(matches[0].canonical_id, tensor.name.replace(/^language_model\.model\./, "").replace(/\.weight$/, ""));
      assert.deepEqual(matches[0].weight_shapes.weight, tensor.shape);
    }
    assert.ok(structure.graph.nodes.filter(n => n.attributes.operator_id === "mla_output_gate").every(n => !n.tensor_names?.length));
  }
});

test("small MLA projects to value width, bills its gate once and respects decode tokens", () => {
  const small = { ...config, hiddenSize: 8, attentionHeads: 2, kvHeads: 2, headDim: 6,
    valueHeadDim: 4, qkNopeHeadDim: 4, qkRopeHeadDim: 2, qLoraRank: 3, kvLoraRank: 5 };
  const module = attentionModule("attn", small, "mla");
  const leaf = name => {
    const s = module.children.find(n => n.id === `attn.${name}`);
    return { ...s, canonical_id: s.id, attributes: { ...s.attributes, operator_id: s.operatorId } };
  };
  const projection = leaf("g_proj"), gate = leaf("output_gate");
  assert.deepEqual(projection.attributes.weightMatrices[0].shape, [8, 8]);
  for (const [phase, tokens] of [["prefill", 6], ["decode", 2]]) {
    const ctx = { config: small, options: { phase, batch: 2, sequence: 3 }, bytesPerElement: 2 };
    const p = countsForNode(projection, ctx), g = countsForNode(gate, ctx);
    assert.deepEqual([p.matrix, p.bytes.weights], [tokens * 64, 128]);
    assert.deepEqual([g.matrix, g.vector, g.sfu, g.bytes.weights], [0, tokens * 8, tokens * 16, 0]);
    assert.deepEqual([g.bytes.actIn, g.bytes.actOut], [tokens * 32, tokens * 16]);
  }
});
