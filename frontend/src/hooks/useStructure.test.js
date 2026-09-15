import assert from "node:assert/strict";
import test from "node:test";
import { buildStructureForPayload } from "./useStructure.js";
import { graphRoot } from "../structure/graph/selectors.js";
function findNode(structure, canonicalId) {
  return structure.graph.nodes.find(node => (node.canonical_id || node.id) === canonicalId) || null;
}
test("buildStructureForPayload handles pasted config in the frontend without API", async () => {
  const payload = {
    source: "config",
    config_json: {
      model_type: "deepseek_v3",
      architectures: ["DeepseekV3ForCausalLM"],
      num_hidden_layers: 2,
      hidden_size: 7168,
      num_attention_heads: 128,
      n_routed_experts: 256
    }
  };
  const structure = await buildStructureForPayload(payload);
  assert.equal(structure.summary.strategy, "frontend-architecture-template");
  assert.equal(structure.summary.architecture, "DeepseekV3ForCausalLM");
});
test("buildStructureForPayload keeps local safetensors truth for a picked directory", async () => {
  const structure = await buildStructureForPayload({
    source: "config",
    source_label: "local directory",
    checkpoint_truth: {
      parameterTotal: 12345,
      parameterCount: {
        BF16: 12345
      },
      tensors: []
    },
    config_json: {
      model_type: "qwen3",
      architectures: ["Qwen3ForCausalLM"],
      num_hidden_layers: 2,
      hidden_size: 1024,
      num_attention_heads: 16
    }
  });
  assert.equal(structure.source.kind, "local directory");
  assert.equal(structure.summary.parameters_total, 12345);
  assert.deepEqual(structure.summary.parameters_by_dtype, {
    BF16: 12345
  });
});
test("buildStructureForPayload reads built-in config without backend API", async () => {
  const structure = await buildStructureForPayload({
    source: "builtin",
    model_id: "Qwen/Qwen3.5-0.8B",
    revision: "main"
  }, {
    fetchHfConfig: async () => {
      throw new Error("HF API should not be called");
    },
    fetchBuiltinConfig: async ({
      modelId
    }) => ({
      model_id: modelId,
      source: {
        kind: "built-in config"
      },
      config: {
        model_type: "qwen3_5",
        architectures: ["Qwen3_5ForConditionalGeneration"],
        num_hidden_layers: 2,
        hidden_size: 1024,
        num_attention_heads: 16
      }
    }),
    fetchTruth: async () => null
  });
  assert.equal(structure.source.kind, "built-in config");
  assert.equal(structure.summary.architecture, "Qwen3_5ForConditionalGeneration");
});
test("built-in model enriches its config with remote safetensors truth", async () => {
  let truthRequest = null;
  const structure = await buildStructureForPayload({
    source: "builtin",
    model_id: "Qwen/Qwen3.5-0.8B",
    endpoint: "huggingface",
    revision: "main"
  }, {
    fetchHfConfig: async () => {
      throw new Error("HF config should not be called");
    },
    fetchBuiltinConfig: async ({
      modelId
    }) => ({
      model_id: modelId,
      source: {
        kind: "built-in config"
      },
      config: {
        model_type: "qwen3",
        architectures: ["Qwen3ForCausalLM"],
        num_hidden_layers: 1,
        hidden_size: 1024,
        num_attention_heads: 16,
        num_key_value_heads: 8
      }
    }),
    fetchTruth: async request => {
      truthRequest = request;
      return {
        tensors: [{
          name: "model.embed_tokens.weight",
          dtype: "BF16",
          shape: [32000, 1024]
        }],
        parameterCount: {
          BF16: 32000 * 1024
        },
        parameterTotal: 32000 * 1024
      };
    }
  });
  assert.deepEqual(truthRequest, {
    modelId: "Qwen/Qwen3.5-0.8B",
    revision: "main",
    hubUrl: "https://huggingface.co",
    resolvePrefix: ""
  });
  assert.equal(structure.source.kind, "built-in config");
  assert.equal(structure.source.checkpoint_truth, "available");
  assert.equal(structure.summary.strategy, "template+truth");
  assert.equal(structure.summary.parameters_total, 32000 * 1024);
});
test("built-in header-truth sidecar supplies parameterTotal without remote tensors", async () => {
  let remoteTruthCalls = 0;
  const structure = await buildStructureForPayload({
    source: "builtin",
    model_id: "Qwen/Qwen3.5-0.8B",
    endpoint: "huggingface",
    revision: "main"
  }, {
    fetchHfConfig: async () => {
      throw new Error("HF config should not be called");
    },
    fetchBuiltinConfig: async ({
      modelId
    }) => ({
      model_id: modelId,
      source: {
        kind: "built-in config"
      },
      config: {
        model_type: "qwen3",
        architectures: ["Qwen3ForCausalLM"],
        num_hidden_layers: 1,
        hidden_size: 1024,
        num_attention_heads: 16,
        num_key_value_heads: 8
      }
    }),
    fetchTruth: async () => {
      remoteTruthCalls += 1;
      throw new Error("remote header should not be fetched when sidecar exists");
    },
    fetchBuiltinSkeletonTruth: async () => ({
      generated: "safetensors header (fetch-header-truth)",
      method: "hub",
      tensor_count: 9,
      parameterTotal: 123456789,
      parameterCount: {
        BF16: 123456789
      }
    })
  });
  assert.equal(remoteTruthCalls, 0);
  // source 是对象（`{kind, ...}`），展示层只取 kind 字符串——不得出现 "[object Object] + header-truth"
  assert.equal(structure.source.kind, 'built-in config + header-truth');
  assert.equal(structure.source.checkpoint_truth, "available");
  assert.equal(structure.summary.strategy, "template+header-truth");
  assert.equal(structure.summary.parameters_total, 123456789);
  assert.deepEqual(structure.summary.parameters_by_dtype, {
    BF16: 123456789
  });
});
test("built-in config returns before deferred safetensors truth and updates in background", async () => {
  let resolveTruth;
  let truthRequests = 0;
  let backgroundStructure = null;
  let resolveBackgroundUpdate;
  const backgroundUpdate = new Promise(resolve => {
    resolveBackgroundUpdate = resolve;
  });
  const truthPromise = new Promise(resolve => {
    resolveTruth = resolve;
  });
  const structure = await buildStructureForPayload({
    source: "builtin",
    model_id: "Qwen/Qwen3.5-0.8B",
    endpoint: "huggingface",
    revision: "main"
  }, {
    fetchHfConfig: async () => {
      throw new Error("HF config should not be called");
    },
    fetchBuiltinConfig: async ({
      modelId
    }) => ({
      model_id: modelId,
      source: {
        kind: "built-in config"
      },
      config: {
        model_type: "qwen3",
        architectures: ["Qwen3ForCausalLM"],
        num_hidden_layers: 1,
        hidden_size: 1024,
        num_attention_heads: 16
      }
    }),
    fetchTruth: async () => {
      truthRequests += 1;
      return truthPromise;
    },
    onBackgroundUpdate: updated => {
      backgroundStructure = updated;
      resolveBackgroundUpdate();
    }
  });
  assert.equal(truthRequests, 1);
  assert.equal(structure.source.kind, 'built-in config');
  assert.equal(structure.source.checkpoint_truth, "not-requested");
  assert.equal(backgroundStructure, null);
  resolveTruth({
    tensors: [{
      name: "model.embed_tokens.weight",
      dtype: "BF16",
      shape: [32000, 1024]
    }],
    parameterCount: {
      BF16: 32000 * 1024
    },
    parameterTotal: 32000 * 1024
  });
  await backgroundUpdate;
  assert.equal(backgroundStructure?.source.checkpoint_truth, "available");
  assert.equal(backgroundStructure?.summary.parameters_total, 32000 * 1024);
});
test("buildStructureForPayload reads HF config then builds in frontend", async () => {
  const structure = await buildStructureForPayload({
    source: "hf",
    model_id: "moonshotai/Kimi-K2.7-Code",
    revision: "main"
  }, {
    fetchHfConfig: async () => ({
      model_type: "kimi_k2",
      architectures: ["DeepseekV3ForCausalLM"],
      num_hidden_layers: 4,
      hidden_size: 4096,
      num_attention_heads: 32,
      n_routed_experts: 64
    }),
    fetchBuiltinConfig: async () => null,
    fetchTruth: async () => null
  });
  assert.equal(structure.source.kind, "hf config (huggingface)");
  assert.equal(structure.summary.architecture, "DeepseekV3ForCausalLM");
});
test("HF config failure falls back to ModelScope directly", async () => {
  const configRequests = [];
  const structure = await buildStructureForPayload({
    source: "hf",
    model_id: "org/example",
    revision: "main",
    endpoint: "huggingface"
  }, {
    fetchHfConfig: async ({
      endpoint
    }) => {
      configRequests.push(endpoint);
      if (endpoint === "huggingface") throw new Error("HF unavailable");
      return {
        model_type: "qwen3",
        architectures: ["Qwen3ForCausalLM"],
        num_hidden_layers: 1,
        hidden_size: 1024,
        num_attention_heads: 16,
        num_key_value_heads: 8
      };
    },
    fetchBuiltinConfig: async () => {
      throw new Error("built-in config should not be called");
    },
    fetchTruth: async ({
      hubUrl,
      revision
    }) => ({
      tensors: [{
        name: "model.embed_tokens.weight",
        dtype: "BF16",
        shape: [32000, 1024]
      }],
      parameterCount: {
        BF16: 32000 * 1024
      },
      parameterTotal: 32000 * 1024,
      method: `${hubUrl}:${revision}`
    })
  });
  assert.deepEqual(configRequests, ["huggingface", "modelscope"]);
  assert.equal(structure.source.kind, "hf config (modelscope)");
  assert.equal(structure.source.config_endpoint, "modelscope");
  assert.equal(structure.source.checkpoint_truth_endpoint, "modelscope");
  assert.equal(structure.summary.strategy, "template+truth");
});
test("buildStructureForPayload enriches HF tree with checkpoint truth when available", async () => {
  const structure = await buildStructureForPayload({
    source: "hf",
    model_id: "Qwen/Qwen3-0.6B",
    revision: "main"
  }, {
    fetchHfConfig: async () => ({
      model_type: "qwen3",
      architectures: ["Qwen3ForCausalLM"],
      num_hidden_layers: 2,
      hidden_size: 1024,
      num_attention_heads: 16,
      num_key_value_heads: 8,
      vocab_size: 151936
    }),
    fetchBuiltinConfig: async () => null,
    fetchTruth: async () => ({
      tensors: [{
        name: "model.embed_tokens.weight",
        dtype: "BF16",
        shape: [151936, 1024]
      }, {
        name: "model.layers.0.self_attn.q_proj.weight",
        dtype: "BF16",
        shape: [1024, 1024]
      }, {
        name: "model.layers.0.mlp.gate_proj.weight",
        dtype: "BF16",
        shape: [4096, 1024]
      }, {
        name: "model.layers.1.self_attn.q_proj.weight",
        dtype: "BF16",
        shape: [1024, 1024]
      }, {
        name: "model.layers.1.mlp.gate_proj.weight",
        dtype: "BF16",
        shape: [4096, 1024]
      }, {
        name: "model.norm.weight",
        dtype: "BF16",
        shape: [1024]
      }],
      parameterCount: {
        BF16: 151936 * 1024 + 1024 * 1024 * 2 + 4096 * 1024 * 2 + 1024
      },
      parameterTotal: 151936 * 1024 + 1024 * 1024 * 2 + 4096 * 1024 * 2 + 1024
    })
  });
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
  const structure = await buildStructureForPayload({
    source: "hf",
    model_id: "HuggingFaceTB/SmolLM2-135M",
    revision: "main"
  }, {
    fetchHfConfig: async () => ({
      model_type: "llama",
      architectures: ["LlamaForCausalLM"],
      num_hidden_layers: 1,
      hidden_size: 576,
      num_attention_heads: 9
    }),
    fetchBuiltinConfig: async () => null,
    fetchTruth: async () => ({
      tensors: [{
        name: "model.layers.0.self_attn.q_proj.weight",
        dtype: "BF16",
        shape: [576, 576]
      }],
      parameterCount: {
        BF16: 576 * 576
      },
      parameterTotal: 576 * 576
    })
  });
  assert.equal(structure.summary.strategy, "skeleton-truth");
  assert.equal(structure.source.strategy, "skeleton-truth");
  assert.equal(graphRoot(structure.graph).children[0].value_source, "checkpoint");
});
const remoteConfig = {
  model_type: "qwen3",
  architectures: ["Qwen3ForCausalLM"],
  num_hidden_layers: 1,
  hidden_size: 64,
  num_attention_heads: 4,
  num_key_value_heads: 2,
  intermediate_size: 128,
  vocab_size: 256
};
test("legacy auto resolves builtin then remote without local or structure APIs", async () => {
  const calls = [];
  const structure = await buildStructureForPayload({
    source: "auto",
    model_id: "org/remote"
  }, {
    fetchBuiltinConfig: async () => {
      calls.push("builtin");
      throw new Error("not in catalog");
    },
    fetchHfConfig: async () => {
      calls.push("hf");
      return remoteConfig;
    },
    fetchTruth: async () => null
  });
  assert.deepEqual(calls, ["builtin", "hf"]);
  assert.equal(structure.summary.architecture, "Qwen3ForCausalLM");
});
test("legacy local and invalid sources fail before any network access", async t => {
  const calls = [];
  t.mock.method(globalThis, "fetch", async (...args) => {
    calls.push(args);
    throw new Error("unexpected fetch");
  });
  for (const payload of [{
    source: "local",
    model_id: "org/model"
  }, {
    source: "local",
    config_path: "/saved/path"
  }]) {
    await assert.rejects(buildStructureForPayload(payload), error => error.issue.code === "model.localDirectoryRequired");
  }
  await assert.rejects(buildStructureForPayload({
    source: "other"
  }), error => error.issue.code === "model.invalidSource");
  for (const value of [null, [], 3, "invalid"]) {
    await assert.rejects(buildStructureForPayload({
      source: "config",
      config_json: value
    }), error => error.issue.code === "model.invalidConfig");
  }
  assert.deepEqual(calls, []);
});
test("all remote failures report model and endpoints without server fallback", async () => {
  const calls = [];
  await assert.rejects(buildStructureForPayload({
    source: "hf",
    model_id: "org/missing"
  }, {
    fetchHfConfig: async ({
      endpoint
    }) => {
      calls.push(endpoint);
      throw new Error("HTTP 404");
    }
  }), error => {
    assert.equal(error.issue.code, "model.remoteConfigFailed");
    assert.equal(error.issue.params.modelId, "org/missing");
    assert.match(error.issue.params.detail, /huggingface: HTTP 404; modelscope: HTTP 404/);
    return true;
  });
  assert.deepEqual(calls, ["huggingface", "modelscope"]);
});
