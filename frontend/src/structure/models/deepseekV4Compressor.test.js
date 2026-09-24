import assert from "node:assert/strict";
import fs from "node:fs";
import test from "node:test";
import { buildStructureFromArtifacts, buildStructureFromConfig } from "../buildStructure.js";
import { normalizeConfig } from "../config/normalize.js";
import { countsForNode } from "../operators/formulas/extractor.js";

const root = new URL("../../../../models/deepseek-ai/", import.meta.url);
const read = path => JSON.parse(fs.readFileSync(new URL(path, root)));

for (const modelId of ["DeepSeek-V4-Flash", "DeepSeek-V4-Flash-0731",
  "DeepSeek-V4-Flash-Vision-Exp", "DeepSeek-V4-Pro", "DeepSeek-V4-Pro-0813"]) {
  for (const loading of ["config-only", "production-artifacts"]) {
    test(`DeepSeek V4 optional compressor/indexer paths are connected: ${modelId} ${loading}`, () => {
      const config = read(`${modelId}/config.json`);
      const structure = loading === "config-only" ? buildStructureFromConfig(config)
        : buildStructureFromArtifacts({
          config, modelId: `deepseek-ai/${modelId}`,
          checkpointTruth: read(`${modelId}/header-truth.json`),
          sourceRef: read(`${modelId}/source-ref.json`),
        });
      const ratio4 = structure.graph.nodes.find(n =>
        n.canonical_id?.match(/layers\.\d+\.self_attn\.indexer$/) &&
        n.attributes?.operator_id === "dsv4_indexer");
      assert.ok(ratio4, "at least one C4 indexer");
      const prefix = ratio4.canonical_id.replace(/\.indexer$/, "");
      const get = suffix => structure.graph.nodes.find(n => n.canonical_id === `${prefix}.${suffix}`);
      for (const suffix of ["indexer.q_proj", "indexer.weights_proj",
        "indexer.compressor.wkv_gate", "indexer.compressor.norm", "indexer"]) {
        assert.ok(get(suffix), `missing ${prefix}.${suffix}`);
      }
      const edge = (from, to) => structure.graph.edges.some(e =>
        e.source_canonical_id === `${prefix}.${from}` && e.target_canonical_id === `${prefix}.${to}`);
      assert.ok(edge("indexer.compressor.wkv_gate", "indexer.compressor.norm"));
      assert.ok(edge("indexer.compressor.norm", "indexer"));
      assert.ok(edge("indexer.q_proj", "indexer"));
      assert.ok(edge("indexer.weights_proj", "indexer"));

      const compressor = get("compressor");
      assert.ok(compressor);
      const compressorPrefix = compressor.canonical_id;
      const compressorNorm = structure.graph.nodes.find(n =>
        n.canonical_id === `${compressorPrefix}.norm`);
      assert.ok(compressorNorm);
      const ratio = compressor.attributes.compress_ratio;
      assert.equal(compressor.attributes.operator_id, "mla_kv_compress");
      assert.equal(compressor.attributes.weightMatrices[0].shape[0],
        2 * (ratio === 4 ? 2 : 1) * compressorNorm.output_shape.at(-1));
      assert.ok(structure.graph.edges.some(e =>
        e.source_canonical_id === compressorNorm.canonical_id && e.target_canonical_id === compressorPrefix));
      assert.ok(structure.graph.edges.some(e =>
        e.source_canonical_id === compressorPrefix &&
        e.target_canonical_id === `${prefix}.attention`));
      assert.ok(!structure.graph.edges.some(e =>
        e.source_canonical_id === compressorPrefix && e.target_canonical_id === compressorNorm.canonical_id),
      "the composite compressor already applies its own norm; do not normalize its output twice");
      const packed = get("indexer.compressor.wkv_gate");
      const indexNorm = get("indexer.compressor.norm");
      assert.equal(packed.output_shape.at(-1), 4 * indexNorm.input_shape.at(-1),
        "packed K/V plus overlap is compressed before head-dimension normalization");
      const ratio128 = structure.graph.nodes.find(n =>
        n.canonical_id?.match(/layers\.\d+\.self_attn\.compressor$/) &&
        n.attributes?.compress_ratio === 128);
      assert.ok(ratio128, "HCA ratio-128 compressor must also be represented");
      const hcaPrefix = ratio128.canonical_id.replace(/\.compressor$/, "");
      assert.ok(structure.graph.edges.some(e =>
        e.source_canonical_id === `${hcaPrefix}.compressor.norm`
        && e.target_canonical_id === ratio128.canonical_id));
      assert.ok(structure.graph.edges.some(e =>
        e.source_canonical_id === ratio128.canonical_id
        && e.target_canonical_id === `${hcaPrefix}.attention`));
      assert.ok(!structure.graph.nodes.some(n => n.canonical_id === `${hcaPrefix}.indexer`),
        "HCA is not a C4 sparse-indexer path");
      // A ratio-0 sliding-window layer has no compressor or C4 indexer.
      const zero = structure.graph.nodes.find(n =>
        n.attributes?.attention_variant === "deepseek_v4" &&
        n.attributes?.compress_ratio === 0 &&
        n.canonical_id?.match(/layers\.\d+\.self_attn$/));
      if (zero) {
        assert.ok(!structure.graph.nodes.some(n =>
          n.canonical_id === `${zero.canonical_id}.compressor` ||
          n.canonical_id === `${zero.canonical_id}.indexer`));
      }
    });
  }
}

test("DeepSeek V4.1 CSA2 does not inherit V4's nested compressor norm", () => {
  const config = read("DeepSeek-V4.1-Flash/config.json");
  const structure = buildStructureFromConfig(config);
  const compressor = structure.graph.nodes.find(n =>
    n.canonical_id?.match(/layers\.(?:encoder|decoder)\.\d+\.self_attn\.compressor$/) &&
    n.attributes?.operator_id === "mla_kv_compress");
  assert.ok(compressor);
  assert.ok(!structure.graph.nodes.some(n => n.canonical_id === `${compressor.canonical_id}.norm`));
  assert.ok(structure.graph.edges.some(e =>
    e.source_canonical_id === compressor.canonical_id &&
    e.target_canonical_id === compressor.canonical_id.replace(/\.compressor$/, ".attention")));
});

test("V4 compressor RMSNorm executes on emitted compressed blocks, not every raw token", () => {
  // SGLang pinned compressor.py: forward_compress passes compress_forward's
  // output to fused_norm_rope; c4/c128.cuh decode only compress at ratio boundaries.
  // The indexer's nested compressor has the same C4 output schedule.
  for (const modelId of ["DeepSeek-V4-Flash", "DeepSeek-V4-Flash-0731",
    "DeepSeek-V4-Flash-Vision-Exp", "DeepSeek-V4-Pro", "DeepSeek-V4-Pro-0813"]) {
    const config = read(`${modelId}/config.json`);
    const normalized = normalizeConfig(config);
    const structure = buildStructureFromArtifacts({
      config, modelId: `deepseek-ai/${modelId}`,
      checkpointTruth: read(`${modelId}/header-truth.json`),
      sourceRef: read(`${modelId}/source-ref.json`),
    });
    const nodes = structure.graph.nodes;
    for (const ratio of [4, 128]) {
      const compressor = nodes.find(n => n.attributes?.operator_id === "mla_kv_compress"
        && n.attributes?.compress_ratio === ratio);
      assert.ok(compressor, `${modelId}: missing C${ratio}`);
      const norm = nodes.find(n => n.canonical_id === `${compressor.canonical_id}.norm`);
      assert.ok(norm);
      const related = ratio === 4
        ? [norm, nodes.find(n => n.canonical_id === compressor.canonical_id.replace(/\.compressor$/, ".indexer.compressor.norm"))]
        : [norm];
      for (const leaf of related) {
        assert.ok(leaf, `${modelId}: missing nested C4 compressor norm`);
        const count = (sequence, phase = "prefill") => countsForNode(leaf, {
          config: normalized, options: { batch: 2, sequence, phase }, bytesPerElement: 2,
        });
        const width = leaf.output_shape.at(-1);
        const blocks = Math.floor((ratio + 1) / ratio) * 2;
        assert.equal(count(ratio + 1).sfu, blocks);
        assert.equal(count(ratio + 1).vector, blocks * (4 * width - 1));
        assert.equal(count(ratio + 1).bytes.actOut, blocks * width * 2);
        assert.equal(count(ratio - 1).sfu, 0);
        assert.equal(count(ratio - 1).bytes.weights, 0, "no norm kernel reads its scale before a block boundary");
        assert.equal(count(ratio - 1, "decode").sfu, 0);
        assert.equal(count(ratio, "decode").sfu, 2);
      }
    }
  }
});
