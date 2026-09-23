import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import { buildStructureFromConfig, buildStructureFromArtifacts } from "../buildStructure.js";
import { normalizeConfig } from "../config/normalize.js";
import { countsForNode } from "../operators/formulas/extractor.js";

const modelsRoot = new URL("../../../../models/", import.meta.url);
const read = url => fs.existsSync(url) ? JSON.parse(fs.readFileSync(url, "utf8")) : null;
const catalog = read(new URL("catalog.json", modelsRoot));
// 独立判据：Qwen 官方模型卡 Model Overview + Gated Attention 论文 §2.2 Eq(5)。
// full gate 需要 O/G；GDN 输出 norm 需要 recurrent output/z，z 不参与卷积。
function checkInputs(graph) {
  const byCanonical = new Map(graph.nodes.map(n => [n.canonical_id, n]));
  let gates = 0, norms = 0;
  for (const node of graph.nodes) {
    const prefix = node.canonical_id.slice(0, node.canonical_id.lastIndexOf("."));
    const incoming = graph.edges.filter(e => e.target === node.id);
    if (node.attributes?.operator_id === "attention_output_gate") {
      gates++;
      const qsa = node.attributes.semantic_role === "qsa_output_gate";
      const split = qsa ? "q_gate_split" : "qkv_gate_split";
      const attention = qsa ? "sparse_attention" : "sdpa";
      assert.deepEqual(new Set(incoming.map(e => e.source_canonical_id)),
        new Set([`${prefix}.${attention}`, `${prefix}.${split}`]));
      assert.equal(incoming.find(e => e.source_canonical_id.endsWith(`.${split}`)).label, "gate");
      assert.ok(!node.attributes.weightMatrices?.length, "gate application must not own projection weights");
    }
    if (node.attributes?.operator_id === "gated_rmsnorm" && byCanonical.has(`${prefix}.qkvz_split`)) {
      norms++;
      assert.deepEqual(new Set(incoming.map(e => e.source_canonical_id)),
        new Set([`${prefix}.state_update`, `${prefix}.qkvz_split`]));
      assert.equal(incoming.find(e => e.source_canonical_id.endsWith(".qkvz_split")).label, "z");
      const conv = byCanonical.get(`${prefix}.short_conv`);
      assert.deepEqual(conv.attributes.branches, ["q", "k", "v"]);
      assert.ok(!Object.hasOwn(conv.attributes.channel_layout, "z"), "z is not a convolution channel");
      assert.equal(graph.edges.find(e => e.target === conv.id && e.source_canonical_id.endsWith(".qkvz_split")).label, "q, k, v");
    }
  }
  return { gates, norms };
}

for (const loading of ["config", "artifacts"]) {
  test(`Qwen required gate inputs across all 31 variants (${loading})`, () => {
    let variants = 0, fullVariants = 0;
    for (const entry of catalog.models.filter(e => e.model_id.startsWith("Qwen/"))) {
      const config = read(new URL(entry.config_path, modelsRoot));
      const dir = new URL("./", new URL(entry.config_path, modelsRoot));
      const structure = loading === "config" ? buildStructureFromConfig(config, { modelId: entry.model_id })
        : buildStructureFromArtifacts({ config, modelId: entry.model_id,
          checkpointTruth: read(new URL("skeleton-truth.json", dir)) || read(new URL("header-truth.json", dir)),
          sourceRef: read(new URL("source-ref.json", dir)) });
      const result = checkInputs(structure.graph);
      assert.ok(result.norms > 0, entry.model_id);
      variants++;
      if (result.gates) fullVariants++;
    }
    assert.equal(variants, 31);
    assert.equal(fullVariants, 31);
  });
}

test("small full-attention gate counts only sigmoid/multiply, not a second GEMM", () => {
  const config = { architectures: ["Qwen3_5ForConditionalGeneration"], model_type: "qwen3_5",
    text_config: { num_hidden_layers: 1, hidden_size: 16, num_attention_heads: 2, num_key_value_heads: 1,
      head_dim: 4, intermediate_size: 32, vocab_size: 64, attn_output_gate: true, layer_types: ["full_attention"] } };
  const graph = buildStructureFromConfig(config).graph;
  checkInputs(graph);
  const gate = graph.nodes.find(n => n.attributes?.operator_id === "attention_output_gate");
  const projection = graph.nodes.find(n => n.canonical_id.endsWith(".qkv_gate_proj"));
  assert.deepEqual(projection.attributes.weightMatrices[0].shape, [24, 16]);
  const counts = countsForNode(gate, { config: normalizeConfig(config),
    options: { phase: "prefill", batch: 1, sequence: 3 }, bytesPerElement: 2 });
  assert.equal(counts.matrix, 0);
  assert.equal(counts.bytes.weights, 0);
  assert.equal(counts.vector, 24);
  assert.equal(counts.sfu, 48);
});
