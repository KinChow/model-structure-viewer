import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";
import { normalizeConfig } from "./config/normalize.js";
import { resolveArchitecture } from "./registry/resolveArchitecture.js";
import { buildNetwork } from "./model_executor/models/index.js";
import { createStructureIr } from "./ir/createStructureIr.js";
import { materializeModelStructure } from "./materializers/toStructureNode.js";
import { formulaForOperator } from "./formulas/index.js";
import { TEMPLATE_FAMILIES } from "./truth/mergeSemantics.js";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../../..");

test("normalizes common config fields before architecture resolution", () => {
  const normalized = normalizeConfig({
    model_type: "deepseek_v3",
    architectures: ["DeepseekV3ForCausalLM"],
    num_hidden_layers: 4,
    hidden_size: 7168,
    num_attention_heads: 128,
    intermediate_size: 18432,
    vocab_size: 129280,
    n_routed_experts: 256,
    num_experts_per_tok: 8,
  });

  assert.equal(normalized.layers, 4);
  assert.equal(normalized.hiddenSize, 7168);
  assert.equal(normalized.attentionHeads, 128);
  assert.equal(normalized.headDim, 56);
  assert.equal(normalized.intermediateSize, 18432);
  assert.equal(normalized.vocabSize, 129280);
  assert.equal(normalized.experts, 256);
  assert.equal(normalized.expertsPerToken, 8);
});

test("resolves architecture by config architecture before field inference", () => {
  const normalized = normalizeConfig({
    model_type: "deepseek_v3",
    architectures: ["DeepseekV3ForCausalLM"],
    num_hidden_layers: 4,
  });

  const resolved = resolveArchitecture(normalized, { modelId: "deepseek-ai/DeepSeek-V3.1" });

  assert.equal(resolved.canonicalArchitecture, "mla-moe-decoder");
  assert.equal(resolved.resolution, "architecture-alias");
});

test("resolves known vendor aliases without string inference", () => {
  const cases = [
    ["DeepseekV32ForCausalLM", "deepseek_v32", "mla-moe-decoder"],
    ["Glm4MoeForCausalLM", "glm4_moe", "gqa-moe-decoder"],
    ["Qwen3_5MoeForConditionalGeneration", "qwen3_5_moe", "gqa-moe-decoder"],
    ["KimiK25ForConditionalGeneration", "kimi_k25", "mla-moe-decoder"],
    ["MiniMaxM2ForCausalLM", "minimax_m2", "gqa-moe-decoder"],
    ["KimiK3ForConditionalGeneration", "kimi_k3", "hybrid-multimodal-moe-decoder"],
    ["Glm5NextForConditionalGeneration", "glm5_next", "hybrid-multimodal-moe-decoder"],
  ];

  for (const [architecture, modelType, canonical] of cases) {
    const resolved = resolveArchitecture(
      normalizeConfig({
        model_type: modelType,
        architectures: [architecture],
        num_hidden_layers: 2,
      }),
    );
    assert.equal(resolved.canonicalArchitecture, canonical);
    assert.equal(resolved.resolution, "architecture-alias");
  }
});

test("selects dedicated model builders by canonical architecture", () => {
  const normalized = normalizeConfig({
    model_type: "minimax_m3",
    architectures: ["MiniMaxM3SparseForConditionalGeneration"],
    text_config: {
      num_hidden_layers: 3,
      hidden_size: 4096,
      num_attention_heads: 32,
      num_local_experts: 64,
    },
    vision_config: {
      num_hidden_layers: 2,
      hidden_size: 1152,
    },
  });
  const resolved = resolveArchitecture(normalized, { modelId: "MiniMaxAI/MiniMax-M3" });
  const network = buildNetwork(resolved, normalized);

  assert.equal(network.children[0].id, "vision_tower");
  assert.equal(network.children[1].id, "projector");
  assert.equal(network.children[2].id, "text_decoder");
  assert.equal(network.children[2].attributes.class, "DecoderStack");
});

test("builds Qwen multimodal models with vision tower and projector", () => {
  const normalized = normalizeConfig({
    architectures: ["Qwen4ExpForConditionalGeneration"],
    model_type: "qwen4_exp",
    text_config: {
      model_type: "qwen4_exp_text",
      num_hidden_layers: 2,
      hidden_size: 2560,
      num_attention_heads: 24,
      num_key_value_heads: 2,
      head_dim: 256,
      num_experts: 512,
      num_experts_per_tok: 10,
      moe_intermediate_size: 640,
      vocab_size: 248320,
    },
    vision_config: {
      model_type: "qwen4_exp",
      depth: 27,
      hidden_size: 1152,
      out_hidden_size: 2560,
    },
  });
  const resolved = resolveArchitecture(normalized, { modelId: "Qwen/Qwen3.8-Flash-Next" });
  const network = buildNetwork(resolved, normalized);
  const structure = materializeModelStructure(createStructureIr({ network, normalized, resolved }));

  assert.equal(resolved.canonicalArchitecture, "multimodal-gqa-moe-decoder");
  assert.deepEqual(network.children.map((child) => child.id), ["vision_tower", "projector", "embed_tokens", "decoder", "norm", "lm_head"]);
  assert.equal(structure.root.children[0].attributes.output_shape, "[batch, visual_tokens, vision hidden size=2560]");
  assert.equal(structure.root.children[1].attributes.input_shape, "[batch, visual_tokens, vision hidden size=2560]");
  assert.equal(structure.summary.vision_layers, 27);
  assert.equal(structure.summary.vision_output_size, 2560);
  assert.equal(TEMPLATE_FAMILIES.has(resolved.canonicalArchitecture), true);
});

test("maps real Qwen, Kimi, and DeepSeek vision configs to multimodal networks", () => {
  const cases = [
    ["Qwen/Qwen3.6-27B", "multimodal-gqa-decoder", true],
    ["moonshotai/Kimi-K2.5", "multimodal-mla-moe-decoder", true],
    ["deepseek-ai/DeepSeek-V4-Flash-Vision-Exp", "multimodal-mla-moe-decoder", false],
  ];
  for (const [modelId, canonicalArchitecture, hasProjector] of cases) {
    const config = JSON.parse(fs.readFileSync(path.join(repoRoot, `models/${modelId}/config.json`), "utf8"));
    const normalized = normalizeConfig(config);
    const resolved = resolveArchitecture(normalized, { modelId });
    const network = buildNetwork(resolved, normalized);

    assert.equal(normalized.hasVision, true, modelId);
    assert.equal(resolved.canonicalArchitecture, canonicalArchitecture, modelId);
    assert.equal(network.children.some((node) => node.id === "vision_tower"), true, modelId);
    assert.equal(network.children.some((node) => node.id === "projector"), hasProjector, modelId);
    assert.equal(network.children.find((node) => node.id === "vision_tower").children.length > 0, true, modelId);
  }
});

test("keeps inferred architecture diagnostics in the IR", () => {
  const normalized = normalizeConfig({
    model_type: "qwen3",
    num_hidden_layers: 2,
    hidden_size: 2048,
    num_attention_heads: 16,
  });
  const resolved = resolveArchitecture(normalized, { modelId: "Qwen/Qwen3.5-0.8B" });
  const network = buildNetwork(resolved, normalized);
  const ir = createStructureIr({ network, normalized, resolved });

  assert.equal(ir.resolved.canonicalArchitecture, "gqa-decoder");
  assert.equal(ir.diagnostics.resolution, "model-type");
  assert.equal(ir.diagnostics.warnings[0].code, "architecture-inferred");
});

test("builds network modules and materializes operator formulas", () => {
  const normalized = normalizeConfig({
    model_type: "deepseek_v3",
    architectures: ["DeepseekV3ForCausalLM"],
    num_hidden_layers: 4,
    hidden_size: 7168,
    num_attention_heads: 128,
    num_key_value_heads: 128,
    first_k_dense_replace: 1,
    n_routed_experts: 256,
    num_experts_per_tok: 8,
  });
  const resolved = resolveArchitecture(normalized, { modelId: "deepseek-ai/DeepSeek-V3.1" });
  const network = buildNetwork(resolved, normalized);
  const ir = createStructureIr({
    network,
    normalized,
    resolved,
    options: {
      modelId: "deepseek-ai/DeepSeek-V3.1",
      source: "pasted",
    },
  });
  const structure = materializeModelStructure(ir);

  assert.equal(network.kind, "network");
  assert.equal(ir.version, 3);
  assert.equal(structure.graph.version, 2);
  assert.equal(structure.graph.schema_version, 2);
  assert.ok(structure.graph.nodes.length > 0);
  assert.ok(structure.graph.edges.some((edge) => edge.evidence === "declared"));
  assert.equal(ir.diagnostics.operator_count > 0, true);
  assert.equal(network.children[1].id, "decoder");
  assert.equal(structure.summary.canonical_architecture, "mla-moe-decoder");
  assert.equal(structure.source.diagnostics.operator_count, ir.diagnostics.operator_count);
  assert.equal(structure.root.children[1].children[0].attributes.range, "0..0");
  assert.equal(structure.root.children[1].children[1].attributes.range, "1..3");

  const attention = structure.root.children[1].children[1].children.find((node) => node.type === "attention");
  const softmax = attention.children.find((node) => node.attributes.operator_id === "softmax");
  assert.equal(softmax.attributes.formula, formulaForOperator("softmax").formula);
});

test("adds readable tensor shapes to modules and operators", () => {
  const normalized = normalizeConfig({
    model_type: "qwen3_5",
    architectures: ["Qwen3_5ForCausalLM"],
    num_hidden_layers: 2,
    hidden_size: 1024,
    num_attention_heads: 8,
    num_key_value_heads: 2,
    head_dim: 256,
    intermediate_size: 3584,
    vocab_size: 248320,
  });
  const resolved = resolveArchitecture(normalized, { modelId: "Qwen/Qwen3.5-0.8B" });
  const network = buildNetwork(resolved, normalized);
  const structure = materializeModelStructure(createStructureIr({ network, normalized, resolved }));

  const embedding = structure.root.children.find((node) => node.type === "embedding");
  assert.equal(embedding.attributes.input_shape, "[batch, sequence]");
  assert.equal(embedding.attributes.output_shape, "[batch, sequence, hidden size=1024]");

  const decoder = structure.root.children.find((node) => node.name === "Decoder Layers");
  const layer = decoder.children[0];
  const attention = layer.children.find((node) => node.type === "attention");
  const qProjection = attention.children.find((node) => node.name === "q projection");

  assert.equal(layer.attributes.input_shape, "[batch, sequence, hidden size=1024]");
  assert.equal(attention.attributes.query_shape, "[batch, sequence, attention heads=8, head dimension=256]");
  assert.equal(attention.attributes.key_shape, "[batch, sequence, key value heads=2, head dimension=256]");
  assert.equal(qProjection.attributes.output_shape, "[batch, sequence, attention heads=8, head dimension=256]");
  const rope = attention.children.find((node) => node.name === "rotary position embedding");
  assert.equal(rope.attributes.input_shape, "[batch, sequence, attention heads=8, head dimension=256], [batch, sequence, key value heads=2, head dimension=256]");
  assert.equal(rope.attributes.position_shape, "[batch, sequence]");
  const scores = attention.children.find((node) => node.name === "attention scores");
  assert.equal(scores.attributes.input_shape, "[batch, sequence, attention heads=8, head dimension=256], [batch, sequence, key value heads=2, head dimension=256]");
  assert.equal(scores.attributes.output_shape, "[batch, attention heads, query sequence, key sequence]");
  assert.equal(scores.attributes.formula, "S = Q K^T / sqrt(d)");
  assert.deepEqual(scores.attributes.inputs, ["Q", "K"]);
  const weighted = attention.children.find((node) => node.name === "weighted value");
  assert.equal(weighted.attributes.input_shape, "[batch, attention heads, query sequence, key sequence], [batch, sequence, key value heads=2, value head dimension=256]");
  assert.equal(weighted.attributes.output_shape, "[batch, sequence, attention heads=8, value head dimension=256]");
  assert.equal(weighted.attributes.formula, "O = P V");
  assert.deepEqual(weighted.attributes.inputs, ["probabilities", "V"]);
});

test("multi-input MLP and MoE operators expose complete shape flows", () => {
  const denseNormalized = normalizeConfig({
    model_type: "qwen3",
    architectures: ["Qwen3ForCausalLM"],
    num_hidden_layers: 1,
    hidden_size: 1024,
    num_attention_heads: 8,
    num_key_value_heads: 2,
    head_dim: 128,
    intermediate_size: 4096,
    vocab_size: 32000,
  });
  const denseResolved = resolveArchitecture(denseNormalized, { modelId: "Qwen/Qwen3-1B" });
  const denseNetwork = buildNetwork(denseResolved, denseNormalized);
  const denseStructure = materializeModelStructure(createStructureIr({ network: denseNetwork, normalized: denseNormalized, resolved: denseResolved }));
  const denseLayer = denseStructure.root.children.find((node) => node.name === "Decoder Layers").children[0];
  const mlp = denseLayer.children.find((node) => node.type === "mlp");
  const swiglu = mlp.children.find((node) => node.name === "SwiGLU activation");
  assert.match(swiglu.attributes.input_shape, /^\[.*\], \[.*\]$/);
  assert.ok(swiglu.attributes.output_shape);

  const moeNormalized = normalizeConfig({
    model_type: "qwen3_moe",
    architectures: ["Qwen3MoeForCausalLM"],
    num_hidden_layers: 1,
    hidden_size: 1024,
    num_attention_heads: 8,
    num_key_value_heads: 2,
    head_dim: 128,
    intermediate_size: 4096,
    moe_intermediate_size: 2048,
    n_routed_experts: 16,
    num_experts_per_tok: 2,
    vocab_size: 32000,
  });
  const moeResolved = resolveArchitecture(moeNormalized, { modelId: "Qwen/Qwen3-MoE" });
  const moeNetwork = buildNetwork(moeResolved, moeNormalized);
  const moeStructure = materializeModelStructure(createStructureIr({ network: moeNetwork, normalized: moeNormalized, resolved: moeResolved }));
  const moeLayer = moeStructure.root.children.find((node) => node.name === "Decoder Layers").children[0];
  const moe = moeLayer.children.find((node) => node.type === "moe");
  const dispatch = moe.children.find((node) => node.name === "expert dispatch");
  const combine = moe.children.find((node) => node.name === "expert combine");
  assert.match(dispatch.attributes.input_shape, /^\[.*\], \[.*\]$/);
  assert.match(combine.attributes.input_shape, /^\[.*\], \[.*\]$/);
  assert.ok(dispatch.attributes.output_shape);
  assert.ok(combine.attributes.output_shape);
});

test("maps GLM-5.3-Flash KDA, QSA, and mHC to the published layer layout", () => {
  const config = JSON.parse(fs.readFileSync(path.join(repoRoot, "models/zai-org/GLM-5.3-Flash/config.json"), "utf8"));
  const normalized = normalizeConfig(config);
  assert.equal(normalized.headDim, 256);
  assert.equal(normalized.attentionSchedule.filter((kind) => kind === "linear").length, 34);
  assert.equal(normalized.attentionSchedule.filter((kind) => kind === "qsa").length, 11);
  assert.equal(normalized.mhcNumResidualStreams, 4);
  assert.equal(normalized.mhcSinkhornIterations, 20);
  assert.equal(normalized.linearLowerBound, -5);

  const resolved = resolveArchitecture(normalized, { modelId: "zai-org/GLM-5.3-Flash" });
  const structure = materializeModelStructure(createStructureIr({
    network: buildNetwork(resolved, normalized),
    normalized,
    resolved,
  }));
  const decoder = structure.root.children.find((node) => node.id === "decoder");
  const firstLayer = decoder.children[0];
  const firstAttention = firstLayer.children.find((node) => node.type === "attention");
  assert.equal(firstLayer.children[0].name, "mHC attention pre");
  assert.equal(firstLayer.children[2].name, "mHC fused post + FFN pre");
  assert.ok(firstLayer.children.every((node) => !["input layernorm", "post attention layernorm"].includes(node.name)));
  assert.deepEqual(firstAttention.children.map((node) => node.name), [
    "QKV projection",
    "beta projection",
    "forget/decay gate projection",
    "qkv causal short convolution",
    "KDA recurrent state",
    "gated RMSNorm",
    "output projection",
  ]);
  const stateUpdate = firstAttention.children.find((node) => node.name === "KDA recurrent state");
  assert.equal(stateUpdate.attributes.formula_id, "gated_delta_attention");
  assert.equal(stateUpdate.attributes.safe_gate, true);
  assert.equal(stateUpdate.attributes.gate_lower_bound, -5);
  const glmProjection = firstAttention.children.find((node) => node.name === "QKV projection");
  assert.equal(glmProjection.output_shape[2], 24896);
  assert.equal(glmProjection.attributes.fused_projection_width, 24896);

  const qsaLayer = decoder.children.find((node) => node.attributes.range === "3..3");
  assert.equal(qsaLayer.children.find((node) => node.type === "attention").attributes.attention_kind, "qsa");
  const lastLayer = decoder.children.at(-1);
  assert.equal(lastLayer.children.at(-2).name, "mHC final post");
  assert.equal(lastLayer.children.at(-1).name, "mHC contract");
});

test("keeps Kimi-K3 KDA semantics canonical while retaining its model-specific implementations", () => {
  const config = JSON.parse(fs.readFileSync(path.join(repoRoot, "models/moonshotai/Kimi-K3/config.json"), "utf8"));
  const normalized = normalizeConfig(config);
  assert.equal(normalized.linearAttentionMode, "kimi_k3");
  assert.equal(normalized.sharedExpertIntermediateSize, 6144);
  assert.equal(normalized.routedExpertHiddenSize, 3584);
  const resolved = resolveArchitecture(normalized, { modelId: "moonshotai/Kimi-K3" });
  const structure = materializeModelStructure(createStructureIr({
    network: buildNetwork(resolved, normalized),
    normalized,
    resolved,
  }));
  const decoder = structure.root.children.find((node) => node.id === "decoder");
  const kdaLayer = decoder.children[0];
  const attention = kdaLayer.children.find((node) => node.type === "attention");
  assert.deepEqual(attention.children.map((node) => node.name), [
    "QKV projection",
    "beta projection",
    "forget/decay gate projection",
    "qkv causal short convolution",
    "KDA recurrent state",
    "gated RMSNorm",
    "output projection",
  ]);
  const kimiProjection = attention.children.find((node) => node.name === "QKV projection");
  assert.equal(kimiProjection.output_shape[2], 49376);
  assert.equal(kimiProjection.attributes.fused_projection_width, 49376);
  assert.deepEqual(attention.children[0].attributes.implementation, {
    input_projection: "in_proj_qkvgfab",
    beta_projection: "b_proj",
    decay_projection: ["f_a_proj", "f_b_proj"],
    short_convolution: "conv1d",
    output_gate: "in_proj_qkvgfab.g",
  });
  const moeLayer = decoder.children.find((node) => node.attributes.range === "1..2");
  const moe = moeLayer.children.find((node) => node.type === "moe");
  assert.ok(moe.children.some((node) => node.name === "routed expert latent down projection"));
  assert.ok(moe.children.some((node) => node.name === "routed expert latent up projection"));
  const shared = moe.children.find((node) => node.id.endsWith(".shared_experts"));
  assert.equal(shared.attributes.intermediate_size, 6144);
  const layerZeroResidual = kdaLayer.children.find((node) => node.type === "residual");
  assert.equal(layerZeroResidual.attributes.block_write, true);
  assert.equal(layerZeroResidual.attributes.previous_blocks, 0);
  assert.equal(layerZeroResidual.children.find((node) => node.name === "attention residual norm").attributes.snapshot_write, true);
  const layerOne = decoder.children.find((node) => node.attributes.range === "1..2");
  const layerOneResidual = layerOne.children.find((node) => node.type === "residual");
  assert.equal(layerOneResidual.attributes.block_write, false);
  assert.equal(layerOneResidual.attributes.previous_blocks, 0);
  assert.equal(layerOneResidual.children.find((node) => node.name === "attention residual norm").attributes.snapshot_write, false);
  assert.equal(structure.root.children.find((node) => node.id === "output_attn_residual").attributes.snapshot_blocks, 8);
});

test("maps DeepSeek V4 compression variants and hash MoE without duplicating framework kernels", () => {
  const config = JSON.parse(fs.readFileSync(path.join(repoRoot, "models/deepseek-ai/DeepSeek-V4-Flash/config.json"), "utf8"));
  const normalized = normalizeConfig(config);
  assert.deepEqual(normalized.attentionSchedule.every((kind) => kind === "dsv4"), true);
  assert.deepEqual(normalized.compressRatios.slice(0, 4), [0, 0, 4, 128]);
  assert.equal(normalized.numHashLayers, 3);
  assert.equal(normalized.indexerHeadDim, 128);
  assert.equal(normalized.multiHyperConnection, true);
  assert.equal(normalized.mhcNumResidualStreams, 4);

  const resolved = resolveArchitecture(normalized, { modelId: "deepseek-ai/DeepSeek-V4-Flash" });
  const structure = materializeModelStructure(createStructureIr({
    network: buildNetwork(resolved, normalized),
    normalized,
    resolved,
  }));
  const decoder = structure.root.children.find((node) => node.id === "decoder");
  const hashLayer = decoder.children[0];
  const hashAttention = hashLayer.children.find((node) => node.type === "attention");
  const hashMoe = hashLayer.children.find((node) => node.type === "moe");
  assert.equal(hashAttention.attributes.attention_kind, "dsv4");
  assert.equal(hashAttention.attributes.compress_ratio, 0);
  assert.equal(hashAttention.children.find((node) => node.name === "sliding-window MQA").attributes.formula_id, "dsv4_swa_attention");
  assert.ok(hashMoe.children.some((node) => node.attributes.formula_id === "dsv4_hash_route"));
  assert.equal(hashMoe.children.some((node) => node.name === "top-k expert routing"), false);
  assert.ok(hashMoe.children.some((node) => node.name === "shared expert branch add"));

  const sparseLayer = decoder.children.find((node) => node.attributes.range === "2..2");
  const sparseAttention = sparseLayer.children.find((node) => node.type === "attention");
  assert.equal(sparseAttention.attributes.compress_ratio, 4);
  assert.equal(sparseAttention.children.find((node) => node.name === "C4 sparse indexer").attributes.implementation[0], "vLLM.SparseAttnIndexer");
  assert.equal(sparseAttention.children.find((node) => node.name === "C4 sparse MLA attention").attributes.formula_id, "qsa_attention");

  const compressedLayer = decoder.children.find((node) => node.attributes.range === "3..3");
  const compressedAttention = compressedLayer.children.find((node) => node.type === "attention");
  assert.equal(compressedAttention.attributes.compress_ratio, 128);
  assert.equal(compressedAttention.children.find((node) => node.name === "compressed MLA attention").attributes.formula_id, "dsv4_compressed_attention");
  assert.equal(compressedLayer.children.find((node) => node.type === "moe").children.some((node) => node.name === "top-k expert routing"), true);
});

test("maps Qwen4Exp GDN, QSA, PLE, and delayed HyperConnection boundaries", () => {
  const config = JSON.parse(fs.readFileSync(path.join(repoRoot, "models/Qwen/Qwen3.8-Flash-Next/config.json"), "utf8"));
  const normalized = normalizeConfig(config);
  assert.equal(normalized.linearAttentionMode, "qwen4_exp");
  assert.deepEqual(normalized.attentionSchedule.reduce((counts, kind) => {
    counts[kind] = (counts[kind] || 0) + 1;
    return counts;
  }, {}), { linear: 36, qsa: 12 });
  assert.equal(normalized.hyperConnectionCount, 4);
  assert.equal(normalized.hyperConnectionLowrank, 320);
  assert.deepEqual(normalized.pleLayerIds, [2]);

  const resolved = resolveArchitecture(normalized, { modelId: "Qwen/Qwen3.8-Flash-Next" });
  const structure = materializeModelStructure(createStructureIr({
    network: buildNetwork(resolved, normalized),
    normalized,
    resolved,
  }));
  const decoder = structure.root.children.find((node) => node.id === "decoder");
  const firstLayer = decoder.children[0];
  assert.deepEqual(firstLayer.children.map((node) => node.name), [
    "HyperConnection attention mix",
    "LINEAR Attention",
    "HyperConnection MLP combine + mix",
    "Routed MoE",
  ]);
  const linear = firstLayer.children.find((node) => node.type === "attention");
  assert.equal(linear.children.length, 8);
  assert.equal(linear.children.find((node) => node.name === "qkvz split").attributes.formula_id, "qwen_qkvz_split");
  assert.equal(linear.children.find((node) => node.name === "output projection").attributes.communication_role, "tp_attention_output");
  assert.equal(linear.children.find((node) => node.name === "KDA recurrent state").attributes.decay_activation, "softplus");
  const pleLayer = decoder.children.find((node) => node.attributes.range === "1..1");
  assert.equal(pleLayer.children[0].name, "PLE");
  const qsaLayer = decoder.children.find((node) => node.attributes.range === "3..3");
  assert.equal(qsaLayer.children.find((node) => node.type === "attention").attributes.attention_kind, "qsa");
  assert.equal(structure.root.children.find((node) => node.id === "hyper_connection_mixer").name, "HyperConnection final mixer");
  const ple = pleLayer.children.find((node) => node.type === "ple");
  assert.equal(ple.attributes.ngram_size, 3);
  assert.equal(ple.attributes.heads_per_ngram, 8);
  assert.equal(ple.attributes.conv_dilation, 3);
  assert.equal(ple.children[0].attributes.formula_id, "ple");
});

test("maps Qwen3.5/3.6/3.8 GDN, full attention gate, and shared expert semantics", () => {
  const denseConfig = JSON.parse(fs.readFileSync(path.join(repoRoot, "models/Qwen/Qwen3.5-27B/config.json"), "utf8"));
  const denseNormalized = normalizeConfig(denseConfig);
  assert.equal(denseNormalized.linearAttentionMode, "qwen3_5");
  assert.equal(denseNormalized.normMode, "gemma_rmsnorm");
  assert.equal(denseNormalized.partialRotaryFactor, 0.25);
  assert.deepEqual(denseNormalized.attentionSchedule.reduce((counts, kind) => {
    counts[kind] = (counts[kind] || 0) + 1;
    return counts;
  }, {}), { linear: 48, qwen35_full: 16 });

  const denseResolved = resolveArchitecture(denseNormalized, { modelId: "Qwen/Qwen3.5-27B" });
  const denseStructure = materializeModelStructure(createStructureIr({
    network: buildNetwork(denseResolved, denseNormalized),
    normalized: denseNormalized,
    resolved: denseResolved,
  }));
  const denseDecoder = denseStructure.root.children.find((node) => node.id === "decoder");
  const linear = denseDecoder.children[0].children.find((node) => node.type === "attention");
  assert.deepEqual(linear.children.map((node) => node.name), [
    "QKV projection",
    "qkvz split",
    "beta projection",
    "forget/decay gate projection",
    "qkv causal short convolution",
    "KDA recurrent state",
    "gated RMSNorm",
    "output projection",
  ]);
  assert.equal(linear.children.find((node) => node.name === "qkvz split").attributes.formula_id, "qwen_qkvz_split");
  const full = denseDecoder.children.find((node) => node.attributes.range === "3..3").children.find((node) => node.type === "attention");
  assert.equal(full.name, "Qwen3.5 Full Attention");
  assert.equal(full.attributes.attention_kind, "qwen35_full");
  assert.equal(full.children.find((node) => node.name === "rotary position embedding").attributes.partial_rotary_factor, 0.25);
  assert.equal(full.children.find((node) => node.name === "Q attention Gemma RMSNorm").attributes.formula_id, "gemma_rmsnorm");
  assert.equal(full.children.find((node) => node.name === "attention output gate").attributes.formula_id, "attention_output_gate");

  const moeConfig = JSON.parse(fs.readFileSync(path.join(repoRoot, "models/Qwen/Qwen3.5-35B-A3B/config.json"), "utf8"));
  const moeNormalized = normalizeConfig(moeConfig);
  assert.equal(moeNormalized.sharedExperts, 1);
  assert.equal(moeNormalized.sharedExpertGate, true);
  const moeResolved = resolveArchitecture(moeNormalized, { modelId: "Qwen/Qwen3.5-35B-A3B" });
  const moeStructure = materializeModelStructure(createStructureIr({
    network: buildNetwork(moeResolved, moeNormalized),
    normalized: moeNormalized,
    resolved: moeResolved,
  }));
  const moe = moeStructure.root.children.find((node) => node.id === "decoder").children[0].children.find((node) => node.type === "moe");
  assert.ok(moe.children.some((node) => node.id.endsWith(".shared_experts")));
  assert.ok(moe.children.some((node) => node.name === "Shared Expert Gate"));
  assert.ok(moe.children.some((node) => node.name === "shared expert branch add"));
});

test("maps DeepSeek V3.2 and GLM DSA latent/indexer paths with top-k reuse schedule", () => {
  const deepseekConfig = JSON.parse(fs.readFileSync(path.join(repoRoot, "models/deepseek-ai/DeepSeek-V3.2/config.json"), "utf8"));
  const deepseekNormalized = normalizeConfig(deepseekConfig);
  assert.equal(deepseekNormalized.attentionSchedule.length, 61);
  assert.equal(deepseekNormalized.attentionSchedule.every((kind) => kind === "qsa"), true);
  assert.equal(deepseekNormalized.indexerSchedule.every((kind) => kind === "compute"), true);
  const deepseekResolved = resolveArchitecture(deepseekNormalized, { modelId: "deepseek-ai/DeepSeek-V3.2" });
  const deepseekStructure = materializeModelStructure(createStructureIr({
    network: buildNetwork(deepseekResolved, deepseekNormalized),
    normalized: deepseekNormalized,
    resolved: deepseekResolved,
  }));
  const deepseekAttention = deepseekStructure.root.children.find((node) => node.id === "decoder").children[0].children.find((node) => node.type === "attention");
  assert.deepEqual(deepseekAttention.children.map((node) => node.name), [
    "query down projection",
    "query latent RMSNorm",
    "query up projection",
    "KV compression projection",
    "KV latent and rope split",
    "KV latent RMSNorm",
    "KV expansion projection",
    "rotary position embedding",
    "indexer query projection",
    "indexer key and weight projection",
    "indexer key RMSNorm",
    "DSA indexer",
    "DSA sparse MLA attention",
    "output projection",
  ]);
  assert.equal(deepseekAttention.children.find((node) => node.name === "DSA indexer").attributes.indexer_mode, "compute");
  assert.equal(deepseekAttention.children.find((node) => node.name === "KV latent and rope split").attributes.formula_id, "mla_kv_split");

  const glmConfig = JSON.parse(fs.readFileSync(path.join(repoRoot, "models/zai-org/GLM-5.2/config.json"), "utf8"));
  const glmNormalized = normalizeConfig(glmConfig);
  assert.deepEqual(glmNormalized.indexerSchedule.slice(0, 8), ["compute", "compute", "compute", "reuse", "reuse", "reuse", "compute", "reuse"]);
  const glmResolved = resolveArchitecture(glmNormalized, { modelId: "zai-org/GLM-5.2" });
  const glmStructure = materializeModelStructure(createStructureIr({
    network: buildNetwork(glmResolved, glmNormalized),
    normalized: glmNormalized,
    resolved: glmResolved,
  }));
  const glmDecoder = glmStructure.root.children.find((node) => node.id === "decoder");
  assert.equal(glmDecoder.children[0].attributes.range, "0..2");
  assert.equal(glmDecoder.children[1].attributes.range, "3..5");
  assert.equal(glmDecoder.children[1].children.find((node) => node.type === "attention").children.find((node) => node.name === "DSA indexer").attributes.indexer_mode, "reuse");
});

test("maps MiniMax M3 dense/sparse attention and sigmoid-routed shared MoE", () => {
  const config = JSON.parse(fs.readFileSync(path.join(repoRoot, "models/MiniMaxAI/MiniMax-M3/config.json"), "utf8"));
  const normalized = normalizeConfig(config);
  assert.deepEqual(normalized.attentionSchedule.reduce((counts, kind) => {
    counts[kind] = (counts[kind] || 0) + 1;
    return counts;
  }, {}), { gqa: 3, sparse: 57 });
  assert.deepEqual(normalized.layerSchedule.reduce((counts, kind) => {
    counts[kind] = (counts[kind] || 0) + 1;
    return counts;
  }, {}), { dense: 3, moe: 57 });
  assert.equal(normalized.normMode, "gemma_rmsnorm");
  assert.equal(normalized.sparseIndexHeads, 4);
  assert.equal(normalized.sparseIndexDim, 128);
  assert.equal(normalized.sparseTopkBlocks, 16);
  assert.equal(normalized.sparseBlockSize, 128);
  assert.equal(normalized.sharedExperts, 1);
  assert.equal(normalized.sharedExpertIntermediateSize, 3072);

  const resolved = resolveArchitecture(normalized, { modelId: "MiniMaxAI/MiniMax-M3" });
  const structure = materializeModelStructure(createStructureIr({
    network: buildNetwork(resolved, normalized),
    normalized,
    resolved,
  }));
  const decoder = structure.root.children.find((node) => node.id === "text_decoder");
  assert.equal(decoder.children[0].attributes.range, "0..2");
  assert.equal(decoder.children[1].attributes.range, "3..59");
  const sparseLayer = decoder.children[1];
  const attention = sparseLayer.children.find((node) => node.type === "attention");
  assert.deepEqual(attention.children.map((node) => node.name), [
    "fused QKV + index projection",
    "main/index QKV split",
    "Q Gemma RMSNorm",
    "K Gemma RMSNorm",
    "partial rotary position embedding",
    "index Q Gemma RMSNorm",
    "index K Gemma RMSNorm",
    "index partial rotary position embedding",
    "MiniMax M3 block indexer",
    "MiniMax M3 block-sparse GQA",
    "output projection",
  ]);
  assert.equal(attention.children.find((node) => node.name === "MiniMax M3 block indexer").attributes.disable_index_value, true);
  const moe = sparseLayer.children.find((node) => node.type === "moe");
  assert.equal(moe.children.find((node) => node.name === "router logits").attributes.scoring_func, "sigmoid");
  assert.equal(moe.children.find((node) => node.name === "expert MLP").attributes.activation, "swigluoai_uninterleave");
  assert.ok(moe.children.some((node) => node.name === "shared expert branch add"));
});

test("maps MiniMax M2 fused QKV, QK norms, and partial RoPE", () => {
  const config = JSON.parse(fs.readFileSync(path.join(repoRoot, "models/MiniMaxAI/MiniMax-M2.7/config.json"), "utf8"));
  const normalized = normalizeConfig(config);
  assert.equal(normalized.modelType, "minimax_m2");
  assert.equal(normalized.rotaryDim, 64);
  assert.equal(normalized.qkNormType, "per_layer");
  const resolved = resolveArchitecture(normalized, { modelId: "MiniMaxAI/MiniMax-M2.7" });
  const structure = materializeModelStructure(createStructureIr({
    network: buildNetwork(resolved, normalized),
    normalized,
    resolved,
  }));
  const decoder = structure.root.children.find((node) => node.id === "decoder");
  const attention = decoder.children[0].children.find((node) => node.type === "attention");
  assert.deepEqual(attention.children.map((node) => node.name), [
    "fused QKV projection",
    "QKV split",
    "Q RMSNorm",
    "K RMSNorm",
    "partial rotary position embedding",
    "attention scores",
    "attention probabilities",
    "weighted value",
    "output projection",
  ]);
  assert.equal(attention.children.find((node) => node.name === "QKV split").attributes.formula_id, "attention_qkv_split");
  const moe = decoder.children[0].children.find((node) => node.type === "moe");
  assert.equal(moe.children.find((node) => node.name === "router logits").attributes.scoring_func, "sigmoid");
});

test("maps Kimi K2 family MLA latent norms and shared expert branch", () => {
  const config = JSON.parse(fs.readFileSync(path.join(repoRoot, "models/moonshotai/Kimi-K2-Base/config.json"), "utf8"));
  const normalized = normalizeConfig(config);
  assert.equal(normalized.qLoraRank, 1536);
  assert.equal(normalized.kvLoraRank, 512);
  assert.equal(normalized.sharedExperts, 1);
  assert.equal(normalized.sharedExpertIntermediateSize, 2048);
  const resolved = resolveArchitecture(normalized, { modelId: "moonshotai/Kimi-K2-Base" });
  const structure = materializeModelStructure(createStructureIr({
    network: buildNetwork(resolved, normalized),
    normalized,
    resolved,
  }));
  const decoder = structure.root.children.find((node) => node.id === "decoder");
  const mlaLayer = decoder.children.find((node) => node.children?.some((child) => child.type === "attention" && child.attributes.attention_kind === "mla"));
  const attention = mlaLayer.children.find((node) => node.type === "attention");
  assert.deepEqual(attention.children.slice(0, 8).map((node) => node.name), [
    "query down projection",
    "query latent RMSNorm",
    "query up projection",
    "KV compression projection",
    "KV latent and rope split",
    "KV latent RMSNorm",
    "KV expansion projection",
    "rotary position embedding",
  ]);
  const moeLayer = decoder.children.find((node) => node.children?.some((child) => child.type === "moe"));
  const moe = moeLayer.children.find((node) => node.type === "moe");
  assert.ok(moe.children.some((node) => node.name === "shared expert branch add"));
});

test("maps GLM4.7 fused QKV, QK norm, partial RoPE, and shared MoE", () => {
  const config = JSON.parse(fs.readFileSync(path.join(repoRoot, "models/zai-org/GLM-4.7/config.json"), "utf8"));
  const normalized = normalizeConfig(config);
  assert.equal(normalized.modelType, "glm4_moe");
  assert.equal(normalized.attentionBias, true);
  assert.equal(normalized.useQkNorm, true);
  assert.equal(normalized.partialRotaryFactor, 0.5);
  assert.equal(normalized.sharedExperts, 1);
  assert.equal(normalized.sharedExpertIntermediateSize, 1536);
  const resolved = resolveArchitecture(normalized, { modelId: "zai-org/GLM-4.7" });
  const structure = materializeModelStructure(createStructureIr({
    network: buildNetwork(resolved, normalized),
    normalized,
    resolved,
  }));
  const decoder = structure.root.children.find((node) => node.id === "decoder");
  const attention = decoder.children[0].children.find((node) => node.type === "attention");
  assert.equal(attention.children[0].name, "fused QKV projection");
  assert.equal(attention.children.find((node) => node.name === "Q RMSNorm").attributes.formula_id, "rmsnorm");
  assert.equal(attention.children.find((node) => node.name === "partial rotary position embedding").attributes.partial_rotary_factor, 0.5);
  const moeLayer = decoder.children.find((node) => node.children?.some((child) => child.type === "moe"));
  const moe = moeLayer.children.find((node) => node.type === "moe");
  assert.equal(moe.children.find((node) => node.name === "router logits").attributes.scoring_func, "sigmoid");
  assert.ok(moe.children.some((node) => node.name === "shared expert branch add"));
});

test("MiniMax M2/M3 attention edges are builder-declared", () => {
  for (const modelPath of ["models/MiniMaxAI/MiniMax-M2.7/config.json", "models/MiniMaxAI/MiniMax-M3/config.json", "models/zai-org/GLM-4.7/config.json"]) {
    const config = JSON.parse(fs.readFileSync(path.join(repoRoot, modelPath), "utf8"));
    const normalized = normalizeConfig(config);
    const resolved = resolveArchitecture(normalized, { modelId: modelPath });
    const structure = materializeModelStructure(createStructureIr({
      network: buildNetwork(resolved, normalized),
      normalized,
      resolved,
    }));
    const attentionIds = new Set(structure.graph.nodes.filter((node) => node.type === "attention").map((node) => node.id));
    const semanticAttentionEdges = structure.graph.edges.filter((edge) => edge.evidence === "semantic-flow" && [...attentionIds].some((id) => edge.source.startsWith(`${id}.`)));
    assert.equal(semanticAttentionEdges.length, 0, modelPath);
    assert.ok(structure.graph.edges.some((edge) => edge.evidence === "declared"), modelPath);
  }
});

test("Qwen3.5/3.6 linear and full attention edges are builder-declared", () => {
  for (const modelPath of ["models/Qwen/Qwen3.5-0.8B/config.json", "models/Qwen/Qwen3.6-27B/config.json"]) {
    const config = JSON.parse(fs.readFileSync(path.join(repoRoot, modelPath), "utf8"));
    const normalized = normalizeConfig(config);
    const resolved = resolveArchitecture(normalized, { modelId: modelPath });
    const structure = materializeModelStructure(createStructureIr({
      network: buildNetwork(resolved, normalized),
      normalized,
      resolved,
    }));
    const attentionIds = new Set(structure.graph.nodes.filter((node) => node.type === "attention").map((node) => node.id));
    assert.equal(structure.graph.edges.filter((edge) => edge.evidence === "semantic-flow" && [...attentionIds].some((id) => edge.source.startsWith(`${id}.`))).length, 0, modelPath);
  }
});

test("QSA variants use builder-declared graph edges", () => {
  for (const modelPath of [
    "models/deepseek-ai/DeepSeek-V3.2/config.json",
    "models/zai-org/GLM-5.3/config.json",
    "models/zai-org/GLM-5.3-Flash/config.json",
    "models/Qwen/Qwen3.8-Flash-Next/config.json",
  ]) {
    const config = JSON.parse(fs.readFileSync(path.join(repoRoot, modelPath), "utf8"));
    const normalized = normalizeConfig(config);
    const resolved = resolveArchitecture(normalized, { modelId: modelPath });
    const structure = materializeModelStructure(createStructureIr({
      network: buildNetwork(resolved, normalized),
      normalized,
      resolved,
    }));
    const qsaIds = new Set(structure.graph.nodes.filter((node) => node.type === "attention" && node.attributes.attention_kind === "qsa").map((node) => node.id));
    assert.equal(structure.graph.edges.filter((edge) => edge.evidence === "semantic-flow" && [...qsaIds].some((id) => edge.source.startsWith(`${id}.`))).length, 0, modelPath);
  }
});

test("DeepSeek V4 DSV4 attention edges are builder-declared", () => {
  for (const modelPath of ["models/deepseek-ai/DeepSeek-V4-Flash/config.json", "models/deepseek-ai/DeepSeek-V4-Pro/config.json"]) {
    const config = JSON.parse(fs.readFileSync(path.join(repoRoot, modelPath), "utf8"));
    const normalized = normalizeConfig(config);
    const resolved = resolveArchitecture(normalized, { modelId: modelPath });
    const structure = materializeModelStructure(createStructureIr({
      network: buildNetwork(resolved, normalized),
      normalized,
      resolved,
    }));
    const dsv4Ids = new Set(structure.graph.nodes.filter((node) => node.type === "attention" && node.attributes.attention_kind === "dsv4").map((node) => node.id));
    assert.equal(structure.graph.edges.filter((edge) => edge.evidence === "semantic-flow" && [...dsv4Ids].some((id) => edge.source.startsWith(`${id}.`))).length, 0, modelPath);
  }
});

test("DeepSeek V4 Flash Vision DSV4 graph connects every attention node", () => {
  const config = JSON.parse(fs.readFileSync(path.join(repoRoot, "models/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp/config.json"), "utf8"));
  const normalized = normalizeConfig(config);
  const resolved = resolveArchitecture(normalized, { modelId: "deepseek-ai/DeepSeek-V4-Flash-Vision-Exp" });
  const structure = materializeModelStructure(createStructureIr({ network: buildNetwork(resolved, normalized), normalized, resolved }));
  const attention = structure.graph.nodes.find((node) => node.type === "attention" && node.attributes.attention_kind === "dsv4");
  const children = structure.graph.nodes.filter((node) => node.parent_id === attention.id);
  const edges = structure.graph.edges.filter((edge) => edge.source.startsWith(`${attention.id}.`) && edge.target.startsWith(`${attention.id}.`));
  assert.equal(children.length, 10);
  assert.equal(edges.length, 10);
  assert.ok(children.every((node) => edges.some((edge) => edge.source === node.id || edge.target === node.id)));
});

test("Kimi K3, GLM5 Flash, and Qwen4Exp linear edges are builder-declared", () => {
  for (const modelPath of [
    "models/moonshotai/Kimi-K3/config.json",
    "models/zai-org/GLM-5.3-Flash/config.json",
    "models/Qwen/Qwen3.8-Flash-Next/config.json",
  ]) {
    const config = JSON.parse(fs.readFileSync(path.join(repoRoot, modelPath), "utf8"));
    const normalized = normalizeConfig(config);
    const resolved = resolveArchitecture(normalized, { modelId: modelPath });
    const structure = materializeModelStructure(createStructureIr({
      network: buildNetwork(resolved, normalized),
      normalized,
      resolved,
    }));
    const linearIds = new Set(structure.graph.nodes.filter((node) => node.type === "attention" && node.attributes.attention_kind === "linear").map((node) => node.id));
    assert.equal(structure.graph.edges.filter((edge) => edge.evidence === "semantic-flow" && [...linearIds].some((id) => edge.source.startsWith(`${id}.`))).length, 0, modelPath);
  }
});
