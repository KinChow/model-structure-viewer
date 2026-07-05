import assert from "node:assert/strict";
import test from "node:test";
import { normalizeConfig } from "./config/normalize.js";
import { resolveArchitecture } from "./registry/resolveArchitecture.js";
import { buildNetwork } from "./model_executor/models/index.js";
import { createStructureIr } from "./ir/createStructureIr.js";
import { materializeModelStructure } from "./materializers/toStructureNode.js";
import { formulaForOperator } from "./formulas/index.js";

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
  assert.equal(ir.version, 1);
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
});
