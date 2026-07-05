import assert from "node:assert/strict";
import test from "node:test";
import { buildStructureForPayload } from "./useStructure.js";

test("buildStructureForPayload handles pasted config in the frontend without API", async () => {
  let apiCalled = false;
  const payload = {
    source: "config",
    config_json: {
      model_type: "deepseek_v3",
      architectures: ["DeepseekV3ForCausalLM"],
      num_hidden_layers: 2,
      hidden_size: 7168,
      num_attention_heads: 128,
      n_routed_experts: 256,
    },
  };

  const structure = await buildStructureForPayload(payload, async () => {
    apiCalled = true;
    throw new Error("API should not be called");
  });

  assert.equal(apiCalled, false);
  assert.equal(structure.summary.strategy, "frontend-architecture-template");
  assert.equal(structure.summary.canonical_architecture, "mla-moe-decoder");
});

test("buildStructureForPayload falls back to backend when auto local config is missing", async () => {
  const expected = { summary: { strategy: "backend" }, source: {}, root: { id: "root" } };
  const structure = await buildStructureForPayload(
    { source: "auto", model_id: "Qwen/Qwen3.5-0.8B" },
    async (payload) => {
      assert.equal(payload.source, "auto");
      return expected;
    },
    async () => {
      throw new Error("missing local config");
    },
  );

  assert.equal(structure, expected);
});

test("buildStructureForPayload reads local config then builds in frontend", async () => {
  let apiCalled = false;
  const structure = await buildStructureForPayload(
    {
      source: "local",
      model_id: "Qwen/Qwen3.5-0.8B",
      revision: "main",
    },
    async () => {
      apiCalled = true;
      throw new Error("structure API should not be called");
    },
    async ({ modelId }) => ({
      model_id: modelId,
      source: { kind: "local cache" },
      config: {
        model_type: "qwen3",
        architectures: ["Qwen3ForCausalLM"],
        num_hidden_layers: 2,
        hidden_size: 1024,
        num_attention_heads: 16,
      },
    }),
  );

  assert.equal(apiCalled, false);
  assert.equal(structure.source.kind, "local cache");
  assert.equal(structure.summary.canonical_architecture, "gqa-decoder");
});

test("buildStructureForPayload reads built-in config without backend API", async () => {
  let apiCalled = false;
  let localCalled = false;
  const structure = await buildStructureForPayload(
    {
      source: "builtin",
      model_id: "Qwen/Qwen3.5-0.8B",
      revision: "main",
    },
    async () => {
      apiCalled = true;
      throw new Error("structure API should not be called");
    },
    async () => {
      localCalled = true;
      throw new Error("local API should not be called");
    },
    async () => {
      throw new Error("HF API should not be called");
    },
    async ({ modelId }) => ({
      model_id: modelId,
      source: { kind: "built-in config" },
      config: {
        model_type: "qwen3_5",
        architectures: ["Qwen3_5ForConditionalGeneration"],
        num_hidden_layers: 2,
        hidden_size: 1024,
        num_attention_heads: 16,
      },
    }),
  );

  assert.equal(apiCalled, false);
  assert.equal(localCalled, false);
  assert.equal(structure.source.kind, "built-in config");
  assert.equal(structure.summary.canonical_architecture, "gqa-decoder");
});

test("buildStructureForPayload reads HF config then builds in frontend", async () => {
  let apiCalled = false;
  const structure = await buildStructureForPayload(
    {
      source: "hf",
      model_id: "moonshotai/Kimi-K2.7-Code",
      revision: "main",
    },
    async () => {
      apiCalled = true;
      throw new Error("structure API should not be called");
    },
    async () => {
      throw new Error("local config should not be called");
    },
    async () => ({
      model_type: "kimi_k2",
      architectures: ["DeepseekV3ForCausalLM"],
      num_hidden_layers: 4,
      hidden_size: 4096,
      num_attention_heads: 32,
      n_routed_experts: 64,
    }),
  );

  assert.equal(apiCalled, false);
  assert.equal(structure.source.kind, "hf config");
  assert.equal(structure.summary.canonical_architecture, "mla-moe-decoder");
});
