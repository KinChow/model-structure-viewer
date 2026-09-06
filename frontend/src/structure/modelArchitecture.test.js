import assert from "node:assert/strict";
import test from "node:test";
import { normalizeConfig } from "./config/normalize.js";
import { resolveArchitecture } from "./registry/resolveArchitecture.js";
import { buildNetwork } from "./model_executor/models/index.js";
import { createStructureIr } from "./ir/createStructureIr.js";
import { materializeModelStructure } from "./materializers/toStructureNode.js";
import { formulaForOperator } from "./formulas/index.js";
import { TEMPLATE_FAMILIES } from "../cost/mergeSemantics.js";

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
  assert.equal(ir.version, 2);
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
