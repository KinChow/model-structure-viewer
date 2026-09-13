import assert from "node:assert/strict";
import test from "node:test";
import { buildStructureForPayload } from "./useStructure.js";
import { graphRoot } from "../structure/graph/selectors.js";

// P7（步骤 7）：legacy structure.root 断言退役——按 Graph IR 检查
// （层级断言走 graphRoot 图视图，节点查找按 canonical_id）。
function findNode(structure, canonicalId) {
  return structure.graph.nodes.find((node) => (node.canonical_id || node.id) === canonicalId) || null;
}

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
  assert.equal(structure.summary.architecture, "DeepseekV3ForCausalLM");
});

test("buildStructureForPayload keeps local safetensors truth for a picked directory", async () => {
  const structure = await buildStructureForPayload({
    source: "config",
    source_label: "local directory",
    checkpoint_truth: { parameterTotal: 12345, parameterCount: { BF16: 12345 }, tensors: [] },
    config_json: {
      model_type: "qwen3",
      architectures: ["Qwen3ForCausalLM"],
      num_hidden_layers: 2,
      hidden_size: 1024,
      num_attention_heads: 16,
    },
  }, async () => { throw new Error("API should not be called"); });
  assert.equal(structure.source.kind, "local directory");
  assert.equal(structure.summary.parameters_total, 12345);
  assert.deepEqual(structure.summary.parameters_by_dtype, { BF16: 12345 });
});

test("buildStructureForPayload falls back to backend when auto local config is missing", async () => {
  // 后端契约：graph 是唯一结构载荷（P7 步骤 7）
  const expected = { summary: { strategy: "backend" }, source: {}, graph: { version: 2, schema_version: 2, root_id: "root", nodes: [], edges: [] } };
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
  assert.equal(structure.summary.architecture, "Qwen3ForCausalLM");
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
    async () => null,
  );

  assert.equal(apiCalled, false);
  assert.equal(localCalled, false);
  assert.equal(structure.source.kind, "built-in config");
  assert.equal(structure.summary.architecture, "Qwen3_5ForConditionalGeneration");
});

test("built-in model enriches its config with remote safetensors truth", async () => {
  let truthRequest = null;
  const structure = await buildStructureForPayload(
    {
      source: "builtin",
      model_id: "Qwen/Qwen3.5-0.8B",
      endpoint: "huggingface",
      revision: "main",
    },
    async () => { throw new Error("structure API should not be called"); },
    async () => { throw new Error("local config should not be called"); },
    async () => { throw new Error("HF config should not be called"); },
    async ({ modelId }) => ({
      model_id: modelId,
      source: { kind: "built-in config" },
      config: {
        model_type: "qwen3",
        architectures: ["Qwen3ForCausalLM"],
        num_hidden_layers: 1,
        hidden_size: 1024,
        num_attention_heads: 16,
        num_key_value_heads: 8,
      },
    }),
    async (request) => {
      truthRequest = request;
      return {
        tensors: [{ name: "model.embed_tokens.weight", dtype: "BF16", shape: [32000, 1024] }],
        parameterCount: { BF16: 32000 * 1024 },
        parameterTotal: 32000 * 1024,
      };
    },
  );

  assert.deepEqual(truthRequest, {
    modelId: "Qwen/Qwen3.5-0.8B",
    revision: "main",
    hubUrl: "https://huggingface.co",
    resolvePrefix: "",
  });
  assert.equal(structure.source.kind, "built-in config");
  assert.equal(structure.source.checkpoint_truth, "available");
  assert.equal(structure.summary.strategy, "template+truth");
  assert.equal(structure.summary.parameters_total, 32000 * 1024);
});

test("built-in header-truth sidecar supplies parameterTotal without remote tensors", async () => {
  let remoteTruthCalls = 0;
  const structure = await buildStructureForPayload(
    {
      source: "builtin",
      model_id: "Qwen/Qwen3.5-0.8B",
      endpoint: "huggingface",
      revision: "main",
    },
    async () => { throw new Error("structure API should not be called"); },
    async () => { throw new Error("local config should not be called"); },
    async () => { throw new Error("HF config should not be called"); },
    async ({ modelId }) => ({
      model_id: modelId,
      source: { kind: "built-in config" },
      config: {
        model_type: "qwen3",
        architectures: ["Qwen3ForCausalLM"],
        num_hidden_layers: 1,
        hidden_size: 1024,
        num_attention_heads: 16,
        num_key_value_heads: 8,
      },
    }),
    async () => {
      remoteTruthCalls += 1;
      throw new Error("remote header should not be fetched when sidecar exists");
    },
    undefined,
    undefined,
    async () => ({
      generated: "safetensors header (fetch-header-truth)",
      method: "hub",
      tensor_count: 9,
      parameterTotal: 123456789,
      parameterCount: { BF16: 123456789 },
    }),
  );
  assert.equal(remoteTruthCalls, 0);
  assert.equal(structure.source.checkpoint_truth, "available");
  assert.equal(structure.summary.strategy, "template+header-truth");
  assert.equal(structure.summary.parameters_total, 123456789);
  assert.deepEqual(structure.summary.parameters_by_dtype, { BF16: 123456789 });
});

test("built-in config returns before deferred safetensors truth and updates in background", async () => {
  let resolveTruth;
  let truthRequests = 0;
  let backgroundStructure = null;
  let resolveBackgroundUpdate;
  const backgroundUpdate = new Promise((resolve) => { resolveBackgroundUpdate = resolve; });
  const truthPromise = new Promise((resolve) => { resolveTruth = resolve; });
  const structure = await buildStructureForPayload(
    {
      source: "builtin",
      model_id: "Qwen/Qwen3.5-0.8B",
      endpoint: "huggingface",
      revision: "main",
    },
    async () => { throw new Error("structure API should not be called"); },
    async () => { throw new Error("local config should not be called"); },
    async () => { throw new Error("HF config should not be called"); },
    async ({ modelId }) => ({
      model_id: modelId,
      source: { kind: "built-in config" },
      config: {
        model_type: "qwen3",
        architectures: ["Qwen3ForCausalLM"],
        num_hidden_layers: 1,
        hidden_size: 1024,
        num_attention_heads: 16,
      },
    }),
    async () => {
      truthRequests += 1;
      return truthPromise;
    },
    (updated) => {
      backgroundStructure = updated;
      resolveBackgroundUpdate();
    },
  );

  assert.equal(truthRequests, 1);
  assert.equal(structure.source.checkpoint_truth, "not-requested");
  assert.equal(backgroundStructure, null);

  resolveTruth({
    tensors: [{ name: "model.embed_tokens.weight", dtype: "BF16", shape: [32000, 1024] }],
    parameterCount: { BF16: 32000 * 1024 },
    parameterTotal: 32000 * 1024,
  });
  await backgroundUpdate;
  assert.equal(backgroundStructure?.source.checkpoint_truth, "available");
  assert.equal(backgroundStructure?.summary.parameters_total, 32000 * 1024);
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
    async () => null, // 离线：无真值 → 降级模板路径
    async () => null,
  );

  assert.equal(apiCalled, false);
  assert.equal(structure.source.kind, "hf config (huggingface)");
  assert.equal(structure.summary.architecture, "DeepseekV3ForCausalLM");
});

test("HF config failure falls back to ModelScope before backend", async () => {
  const configRequests = [];
  const structure = await buildStructureForPayload(
    {
      source: "hf",
      model_id: "org/example",
      revision: "main",
      endpoint: "huggingface",
    },
    async () => { throw new Error("structure API should not be called"); },
    async () => { throw new Error("local config should not be called"); },
    async ({ endpoint }) => {
      configRequests.push(endpoint);
      if (endpoint === "huggingface") throw new Error("HF unavailable");
      return {
        model_type: "qwen3",
        architectures: ["Qwen3ForCausalLM"],
        num_hidden_layers: 1,
        hidden_size: 1024,
        num_attention_heads: 16,
        num_key_value_heads: 8,
      };
    },
    async () => { throw new Error("built-in config should not be called"); },
    async ({ hubUrl, revision }) => ({
      tensors: [{ name: "model.embed_tokens.weight", dtype: "BF16", shape: [32000, 1024] }],
      parameterCount: { BF16: 32000 * 1024 },
      parameterTotal: 32000 * 1024,
      method: `${hubUrl}:${revision}`,
    }),
  );

  assert.deepEqual(configRequests, ["huggingface", "modelscope"]);
  assert.equal(structure.source.kind, "hf config (modelscope)");
  assert.equal(structure.source.config_endpoint, "modelscope");
  assert.equal(structure.source.checkpoint_truth_endpoint, "modelscope");
  assert.equal(structure.summary.strategy, "template+truth");
});

test("buildStructureForPayload enriches HF tree with checkpoint truth when available", async () => {
  let apiCalled = false;
  const structure = await buildStructureForPayload(
    {
      source: "hf",
      model_id: "Qwen/Qwen3-0.6B",
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
      model_type: "qwen3",
      architectures: ["Qwen3ForCausalLM"],
      num_hidden_layers: 2,
      hidden_size: 1024,
      num_attention_heads: 16,
      num_key_value_heads: 8,
      vocab_size: 151936,
    }),
    async () => null,
    async () => ({
      tensors: [
        { name: "model.embed_tokens.weight", dtype: "BF16", shape: [151936, 1024] },
        { name: "model.layers.0.self_attn.q_proj.weight", dtype: "BF16", shape: [1024, 1024] },
        { name: "model.layers.0.mlp.gate_proj.weight", dtype: "BF16", shape: [4096, 1024] },
        { name: "model.layers.1.self_attn.q_proj.weight", dtype: "BF16", shape: [1024, 1024] },
        { name: "model.layers.1.mlp.gate_proj.weight", dtype: "BF16", shape: [4096, 1024] },
        { name: "model.norm.weight", dtype: "BF16", shape: [1024] },
      ],
      parameterCount: { BF16: 151936 * 1024 + 1024 * 1024 * 2 + 4096 * 1024 * 2 + 1024 },
      parameterTotal: 151936 * 1024 + 1024 * 1024 * 2 + 4096 * 1024 * 2 + 1024,
    }),
  );

  assert.equal(apiCalled, false);
  assert.equal(structure.summary.parameters_total, 151936 * 1024 + 1024 * 1024 * 2 + 4096 * 1024 * 2 + 1024);
  assert.equal(structure.summary.strategy, "template+truth");
  assert.equal(structure.source.strategy, "template+truth");

  // 模板算子 q_proj 绑定到 trie 层 0 的真值
  const qProj = findNode(structure, "layers.0.self_attn.q_proj");
  assert.equal(qProj.params, 1024 * 1024);
  assert.equal(qProj.value_source, "checkpoint");
  assert.deepEqual(qProj.weight_shapes.weight, [1024, 1024]);
  assert.equal(qProj.dtype, "BF16");
  // 无模板对应物（rope）不绑定（图 schema：未知扩展字段为 null，不是 undefined）
  const rope = findNode(structure, "layers.0.self_attn.rope");
  assert.equal(rope.params, null);
});

test("无模板 HF 架构把 checkpoint trie 报告为骨架真值", async () => {
  const structure = await buildStructureForPayload(
    { source: "hf", model_id: "HuggingFaceTB/SmolLM2-135M", revision: "main" },
    async () => { throw new Error("structure API should not be called"); },
    async () => { throw new Error("local config should not be called"); },
    async () => ({
      model_type: "llama",
      architectures: ["LlamaForCausalLM"],
      num_hidden_layers: 1,
      hidden_size: 576,
      num_attention_heads: 9,
    }),
    async () => null,
    async () => ({
      tensors: [{ name: "model.layers.0.self_attn.q_proj.weight", dtype: "BF16", shape: [576, 576] }],
      parameterCount: { BF16: 576 * 576 },
      parameterTotal: 576 * 576,
    }),
  );

  assert.equal(structure.summary.strategy, "skeleton-truth");
  assert.equal(structure.source.strategy, "skeleton-truth");
  assert.equal(graphRoot(structure.graph).children[0].value_source, "checkpoint");
});
