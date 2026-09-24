import assert from "node:assert/strict";
import fs from "node:fs";
import test from "node:test";
import { buildStructureFromConfig, buildStructureFromArtifacts } from "../buildStructure.js";
import { normalizeConfig } from "../config/normalize.js";
import { countsForNode } from "../operators/formulas/extractor.js";
import { FORMULAS } from "../operators/formulas/index.js";
import { MODULES } from "../operators/formulas/modules.js";
import { evaluateDecomposition } from "../operators/formulas/atoms.js";
import { mlpModule } from "../layers/mlp.js";
import { buildSkeleton } from "../truth/skeleton.js";

const raw = JSON.parse(fs.readFileSync(new URL("../../../../models/moonshotai/Kimi-K3/config.json", import.meta.url)));
// Frozen release evidence: technical report §2.3.2 Eq(12), SituAndMul.forward.
// beta=4, linear_beta=25 are scalars, not nn.Parameter objects.
for (const loading of ["config", "artifacts"]) {
  test(`SiTU covers dense/shared/routed K3 FFNs (${loading})`, () => {
    const s = loading === "config" ? buildStructureFromConfig(raw)
      : buildStructureFromArtifacts({ config: raw, modelId: "moonshotai/Kimi-K3" });
    const get = id => s.graph.nodes.find(n => n.canonical_id === id);
    const dense = get("layers.0.mlp.situ_glu");
    const shared = get("layers.1.block_sparse_moe.shared_experts.situ_glu");
    const routed = get("layers.1.block_sparse_moe.expert_mlp");
    for (const n of [dense, shared, routed]) {
      assert.ok(n);
      assert.equal(n.attributes.activation, "situ");
      assert.equal(n.attributes.situ_beta, 4);
      assert.equal(n.attributes.situ_linear_beta, 25);
      assert.equal(n.attributes.activation_compute_dtype, "float32");
      assert.match(n.attributes.formula, /tanh/);
      assert.equal(n.attributes.operator_id, n === routed ? "fused_moe_mlp" : "situ_glu");
    }
    assert.ok(!dense.attributes.weightMatrices && !shared.attributes.weightMatrices);
    assert.ok(!get("layers.0.mlp.swiglu") && !get("layers.1.block_sparse_moe.shared_experts.swiglu"));
    for (const n of [dense, shared]) {
      assert.equal(n.attributes.checkpoint_module, false);
      assert.deepEqual(new Set(s.graph.edges.filter(e => e.target === n.id).map(e => e.source_canonical_id)),
        new Set(["gate_proj", "up_proj"].map(p => n.canonical_id.replace(/situ_glu$/, p))));
    }
    // LatentMoE routed compression and independent full-width shared path survive.
    assert.deepEqual(get("layers.1.block_sparse_moe.routed_expert_down_proj").output_shape, [-1, 3584]);
    assert.deepEqual(routed.attributes.weightMatrices[0].shape, [3072, 3584]);
    assert.deepEqual(get("layers.1.block_sparse_moe.shared_experts.gate_proj").attributes.weightMatrices[0].shape, [6144, 7168]);
    const ctx = { config: normalizeConfig(raw), options: { batch: 2, sequence: 3, phase: "prefill" }, bytesPerElement: 2 };
    const counts = countsForNode(routed, ctx);
    assert.equal(counts.matrix, 6 * 16 * 3 * 3584 * 3072);
    assert.equal(counts.vector, 6 * 16 * 3072 * 6);
    assert.equal(counts.sfu, 6 * 16 * 3072 * 4);
    assert.equal(counts.bytes.weights, 96 * 3 * 3584 * 3072 * 2);
  });
}

test("SiTU exact tiny actions and optional up softcap agree with independent decomposition", () => {
  for (const [linearBeta, v, sf] of [[25, 36, 24], [undefined, 24, 18]]) {
    const config = normalizeConfig({ ...raw, text_config: { ...raw.text_config, activation_situ_linear_beta: linearBeta } });
    const spec = mlpModule("mlp", { ...config, hiddenSize: 4, intermediateSize: 3 }).children.find(n => n.operatorId === "situ_glu");
    assert.ok(spec);
    const node = { ...spec, attributes: { ...spec.attributes, operator_id: spec.operatorId } };
    const c = countsForNode(node, { config, options: { batch: 1, sequence: 2, phase: "prefill" }, bytesPerElement: 2 });
    assert.deepEqual(c, { matrix: 0, vector: v, sfu: sf, bytes: { weights: 0, actIn: 24, actOut: 12 } });
    const p = { tokens: 2, intermediate: 3, linearBeta, b: 2 };
    const dec = evaluateDecomposition(MODULES.situ_glu.decompose(p));
    for (const key of ["matrix", "vector", "sfu"]) assert.equal(c[key], dec[key]);
    assert.equal(MODULES.situ_glu.compulsoryBytes(p), 36);
    const decode = countsForNode(node, { config, options: { batch: 1, sequence: 999, phase: "decode" }, bytesPerElement: 2 });
    assert.equal(decode.vector, v / 2);
    assert.equal(decode.sfu, sf / 2);
  }
  assert.match(FORMULAS.situ_glu.formula, /linear_beta/);
});

test("SiTU activation identity follows explicit hidden_act, not model/display names", () => {
  const config = normalizeConfig({ model_type: "llama", hidden_size: 4, intermediate_size: 3,
    hidden_act: "situ", activation_situ_beta: 2, activation_situ_linear_beta: 10 });
  const nodes = mlpModule("mlp", config).children;
  assert.equal(nodes.find(n => n.operatorId === "situ_glu")?.attributes.situ_beta, 2);
  const fallback = normalizeConfig({ ...raw, text_config: { ...raw.text_config, hidden_act: "silu" } });
  assert.ok(mlpModule("mlp", fallback).children.some(n => n.operatorId === "swiglu"));
  const defaults = normalizeConfig({ ...raw, text_config: { ...raw.text_config,
    activation_situ_beta: undefined, activation_situ_linear_beta: undefined } });
  const activation = mlpModule("mlp", defaults).children.find(n => n.operatorId === "situ_glu");
  assert.equal(activation.attributes.situ_beta, 1);
  assert.equal(activation.attributes.situ_linear_beta, undefined);
});

test("shared expert header paths stay intact and semantic SiTU cannot acquire checkpoint weights", () => {
  const fixture = JSON.parse(fs.readFileSync(new URL("./__fixtures__/kimi-k3-shared-header.json", import.meta.url)));
  for (const checkpointTruth of [{ tensors: fixture.tensors },
    { skeleton: buildSkeleton(fixture.tensors), tensor_count: fixture.tensors.length }]) {
    const { graph } = buildStructureFromArtifacts({ config: raw, modelId: fixture.model_id, checkpointTruth });
    for (const tensor of fixture.tensors) {
      const matches = graph.nodes.filter(n => n.tensor_names?.includes(tensor.name));
      assert.equal(matches.length, 1);
      assert.equal(matches[0].canonical_id, tensor.name.replace(/^language_model\.model\./, "").replace(/\.weight$/, ""));
      assert.deepEqual(matches[0].weight_shapes.weight, tensor.shape);
    }
  }
  const name = "language_model.model.layers.3.block_sparse_moe.shared_experts.situ_glu.weight";
  const { graph } = buildStructureFromArtifacts({ config: raw,
    checkpointTruth: { tensors: [{ name, shape: [2], dtype: "BF16" }] } });
  const activation = graph.nodes.find(n => n.canonical_id === "layers.3.block_sparse_moe.shared_experts.situ_glu");
  assert.ok(!activation.tensor_names?.length);
  assert.equal(graph.nodes.filter(n => n.tensor_names?.includes(name)).length, 1, "conflicting fact remains a gap, not activation weight");
});
