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
        n.canonical_id?.match(/layers\.\d+\.self_attn\.compressor\.indexer$/) &&
        n.attributes?.operator_id === "dsv4_indexer");
      assert.ok(ratio4, "at least one C4 indexer");
      const prefix = ratio4.canonical_id.replace(/\.compressor\.indexer$/, "");
      const get = suffix => structure.graph.nodes.find(n => n.canonical_id === `${prefix}.${suffix}`);
      for (const suffix of ["compressor.indexer.q_b_proj", "compressor.indexer.scorer.weights_proj",
        "compressor.indexer.kv_proj", "compressor.indexer.gate_proj",
        "compressor.indexer.kv_norm", "compressor.indexer", "compressor"]) {
        assert.ok(get(suffix), `missing ${prefix}.${suffix}`);
      }
      const edge = (from, to) => structure.graph.edges.some(e =>
        e.source_canonical_id === `${prefix}.${from}` && e.target_canonical_id === `${prefix}.${to}`);
      for (const branch of ["compressor", "compressor.indexer"]) {
        assert.ok(get(`${branch}.window_reduce`), `${branch} must expose gated compression`);
        for (const input of ["kv_proj", "gate_proj", "position_bias"]) {
          assert.ok(edge(`${branch}.${input}`, `${branch}.window_reduce`), `${input} enters window reduction`);
        }
        assert.ok(edge(`${branch}.window_reduce`, `${branch}.kv_norm`));
        assert.ok(!edge(`${branch}.position_bias`, `${branch}.gate_proj`),
          "position bias adds to projected logits, not to the projection's input");
      }
      assert.ok(!edge("compressor.indexer", "compressor.rotary_emb"),
        "index selection is a sibling result; it must not feed the compressor's RoPE");
      assert.ok(edge("compressor.indexer.kv_norm", "compressor.indexer.rotary_emb"));
      assert.ok(edge("compressor.indexer.rotary_emb", "compressor.indexer.scorer"));
      assert.ok(edge("compressor.indexer.scorer", "compressor.indexer"));
      assert.ok(edge("compressor.indexer.scorer.weights_proj", "compressor.indexer.scorer"));
      assert.ok(edge("compressor.rotary_emb", "compressor"),
        "compressed KV must contribute to the outer compressor output");
      assert.ok(edge("compressor.indexer", "compressor"),
        "C4 index selection must contribute to the outer compressor's block mask");
      assert.ok(structure.graph.edges.some(e =>
        e.source_canonical_id === `${prefix}.compressor.indexer`
        && e.target_canonical_id === `${prefix}.compressor`
        && e.relation === "index-control"), "selected blocks must be labelled as an index-control dependency");

      const compressor = get("compressor");
      assert.ok(compressor);
      const compressorPrefix = compressor.canonical_id;
      const compressorNorm = structure.graph.nodes.find(n =>
        n.canonical_id === `${compressorPrefix}.kv_norm`);
      assert.ok(compressorNorm);
      const ratio = compressor.attributes.compress_ratio;
      assert.equal(compressor.attributes.operator_id, "mla_kv_compress");
      assert.equal(compressor.attributes.weightMatrices, undefined,
        "published compressor weights belong to checkpoint-shaped children");
      assert.equal(get("compressor.kv_proj").attributes.weightMatrices[0].shape[0],
        (ratio === 4 ? 2 : 1) * compressorNorm.output_shape.at(-1));
      assert.ok(structure.graph.edges.some(e =>
        e.source_canonical_id === compressorNorm.canonical_id &&
        e.target_canonical_id === `${compressorPrefix}.rotary_emb`));
      assert.ok(structure.graph.edges.some(e =>
        e.source_canonical_id === compressorPrefix &&
        e.target_canonical_id === `${prefix}.attention`));
      const indexNorm = get("compressor.indexer.kv_norm");
      assert.equal(get("compressor.indexer.kv_proj").output_shape.at(-1), 2 * indexNorm.input_shape.at(-1),
        "indexer K/V projections emit the published 2*index_head_dim packed width");
      const ratio128 = structure.graph.nodes.find(n =>
        n.canonical_id?.match(/layers\.\d+\.self_attn\.compressor$/) &&
        n.attributes?.compress_ratio === 128);
      assert.ok(ratio128, "HCA ratio-128 compressor must also be represented");
      const hcaPrefix = ratio128.canonical_id.replace(/\.compressor$/, "");
      assert.ok(structure.graph.edges.some(e =>
        e.source_canonical_id === `${hcaPrefix}.compressor`
        && e.target_canonical_id === `${hcaPrefix}.attention`));
      assert.ok(structure.graph.nodes.some(n =>
        n.canonical_id === `${hcaPrefix}.compressor.kv_norm`));
      assert.ok(structure.graph.edges.some(e =>
        e.source_canonical_id === `${hcaPrefix}.compressor.kv_norm`
        && e.target_canonical_id === `${hcaPrefix}.compressor.rotary_emb`));
      assert.ok(structure.graph.edges.some(e =>
        e.source_canonical_id === `${hcaPrefix}.compressor.rotary_emb`
        && e.target_canonical_id === `${hcaPrefix}.compressor`));
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

test("DeepSeek V4 production source-ref binds the published compressor module path", () => {
  const config = read("DeepSeek-V4-Flash/config.json");
  const sourceRef = read("DeepSeek-V4-Flash/source-ref.json");
  const graph = buildStructureFromArtifacts({
    config,
    modelId: "deepseek-ai/DeepSeek-V4-Flash",
    checkpointTruth: read("DeepSeek-V4-Flash/header-truth.json"),
    sourceRef,
  });
  const compressor = graph.graph.nodes.find(n =>
    n.canonical_id === "layers.2.self_attn.compressor");
  assert.equal(compressor.source_ref.className, "DeepseekV4CSACompressor");
  assert.equal(compressor.source_ref.line, 580);
  assert.ok(graph.source.diagnostics.source_ref.bound > 0);
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
      const norm = nodes.find(n => n.canonical_id === `${compressor.canonical_id}.kv_norm`);
      assert.ok(norm);
      const related = ratio === 4
        ? [norm, nodes.find(n => n.canonical_id === `${compressor.canonical_id}.indexer.kv_norm`)]
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

test("V4 C4 indexer scores compressed keys, not raw sequence positions", () => {
  const config = read("DeepSeek-V4-Flash/config.json");
  const normalized = normalizeConfig(config);
  const structure = buildStructureFromConfig(config);
  const indexer = structure.graph.nodes.find(n =>
    n.attributes?.operator_id === "dsv4_indexer" &&
    n.attributes?.compress_ratio === 4);
  assert.ok(indexer);
  assert.equal(indexer.attributes.index_key_domain, "compressed_window");
  assert.equal(indexer.attributes.score_mask_stage, "after_dense_scores");
  const count = sequence => countsForNode(indexer, {
    config: normalized,
    options: { batch: 1, sequence, phase: "decode" },
    bytesPerElement: 2,
  });
  const short = count(4);
  const long = count(4096);
  // The published indexer compresses each complete C4 window before scoring;
  // indexRead is therefore sequence/4 * index_dim * bytes, not sequence * ...
  assert.equal(short.bytes.indexRead, 1 * normalized.dsaIndexHeadDim * 2);
  assert.equal(long.bytes.indexRead, 1024 * normalized.dsaIndexHeadDim * 2);
  assert.equal(long.bytes.indexRead / short.bytes.indexRead, 1024);
  assert.equal(short.matrix, normalized.dsaIndexHeads * normalized.dsaIndexHeadDim);
  assert.equal(long.matrix, normalized.dsaIndexHeads * 1024 * normalized.dsaIndexHeadDim);
  assert.equal(short.bytes.actOut - count(5).bytes.actOut,
    normalized.dsaIndexHeadDim * 2,
    "a non-boundary decode step must not write another compressed index key");
  const prefill = countsForNode(indexer, {
    config: normalized,
    options: { batch: 1, sequence: 8, phase: "prefill" },
    bytesPerElement: 2,
  });
  assert.equal(prefill.matrix, normalized.dsaIndexHeads * 8 * 2 * normalized.dsaIndexHeadDim,
    "reference scorer does the dense Q×compressed-K matmul before masking invalid blocks");
  const production = buildStructureFromArtifacts({
    config, modelId: "deepseek-ai/DeepSeek-V4-Flash",
    checkpointTruth: read("DeepSeek-V4-Flash/header-truth.json"),
    sourceRef: read("DeepSeek-V4-Flash/source-ref.json"),
  }).graph.nodes.find(n => n.canonical_id === indexer.canonical_id);
  assert.ok(production);
  assert.equal(countsForNode(production, {
    config: normalized,
    options: { batch: 1, sequence: 4096, phase: "decode" },
    bytesPerElement: 2,
  }).bytes.indexRead, long.bytes.indexRead, "production artifacts use the same compressed-key geometry");
});
