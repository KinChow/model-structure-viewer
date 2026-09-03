import assert from "node:assert/strict";
import test from "node:test";

import { enrichNetworkWithTruth } from "../mergeSemantics.js";

const T = (name, dtype = "BF16", shape) => ({ name, dtype, shape });

function templateNetwork() {
  return {
    kind: "network",
    id: "model",
    name: "Qwen3ForCausalLM",
    canonicalArchitecture: "gqa-decoder",
    children: [
      {
        kind: "module",
        id: "embed_tokens",
        name: "embed tokens",
        type: "embedding",
        attributes: {},
        children: [],
      },
      {
        kind: "module",
        id: "decoder.0.mlp",
        name: "MLP",
        type: "mlp",
        attributes: {},
        children: [
          {
            kind: "operator",
            id: "decoder.0.mlp.gate_proj",
            name: "gate projection",
            operatorId: "linear",
            attributes: {},
            children: [],
          },
          {
            kind: "operator",
            id: "decoder.0.mlp.swiglu",
            name: "SwiGLU activation",
            operatorId: "swiglu",
            attributes: {},
            children: [],
          },
        ],
      },
    ],
  };
}

test("无模板时直接用 trie 树兜底（generic-* 架构）", () => {
  const truth = {
    tensors: [
      T("model.embed_tokens.weight", "BF16", [32000, 4096]),
      T("model.layers.0.self_attn.q_proj.weight", "BF16", [4096, 4096]),
      T("model.layers.0.self_attn.k_proj.weight", "BF16", [1024, 4096]),
      T("model.layers.1.self_attn.q_proj.weight", "BF16", [4096, 4096]),
      T("model.layers.1.self_attn.k_proj.weight", "BF16", [1024, 4096]),
    ],
    parameterTotal: 32000 * 4096 + 4096 * 4096 * 2 + 1024 * 4096 * 2,
  };
  const { network, diagnostics } = enrichNetworkWithTruth(templateNetwork(), truth, {
    hasTemplate: false,
    modelName: "LlamaForCausalLM",
    canonicalArchitecture: "generic-decoder",
  });
  assert.equal(diagnostics.strategy, "skeleton-truth");
  assert.equal(network.id, "skeleton");
  // 单顶层段展开：model 容器的直接子节点即为顶层模块
  const embed = network.children.find((c) => c.name === "embed_tokens");
  const layers = network.children.find((c) => c.name === "layers");
  assert.ok(embed, "embed_tokens 应为顶层模块");
  assert.ok(layers, "layers 应为顶层模块");
  // 层被折叠为 repeat
  assert.equal(layers.repeat, 2);
  assert.equal(layers.value_source, "checkpoint");
  // 叶节点携带真值
  assert.equal(embed.params, 32000 * 4096);
  assert.deepEqual(embed.weight_shapes.weight, [32000, 4096]);
});

test("有模板时绑定真值并报告未声明的含参模块（gap）", () => {
  const truth = {
    tensors: [
      T("model.embed_tokens.weight", "BF16", [32000, 4096]),
      T("model.layers.0.mlp.gate_proj.weight", "BF16", [12288, 4096]),
      T("model.layers.0.mlp.up_proj.weight", "BF16", [12288, 4096]),
      // 模板未声明的含参模块
      T("model.layers.0.self_attn.q_a_proj.weight", "BF16", [2048, 4096]),
    ],
  };
  const network = templateNetwork();
  const { network: enriched, diagnostics } = enrichNetworkWithTruth(network, truth, {
    hasTemplate: true,
    modelName: "DeepseekV3ForCausalLM",
    canonicalArchitecture: "mla-moe-decoder",
  });
  assert.equal(diagnostics.strategy, "template+truth");
  assert.equal(enriched, network);
  // 模板算子 gate_proj 绑定真值
  const gateProj = network.children[1].children[0];
  assert.equal(gateProj.params, 12288 * 4096);
  assert.equal(gateProj.value_source, "checkpoint");
  // 模板无对应物（swiglu）不绑定
  assert.equal(network.children[1].children[1].params, undefined);
  // 模板未声明的含参模块进入 gap 信号
  assert.ok(diagnostics.template_gaps.includes("model.layers.0.self_attn.q_a_proj"));
});

test("无真值时原样返回", () => {
  const network = templateNetwork();
  const { network: out, diagnostics } = enrichNetworkWithTruth(network, null, {
    hasTemplate: true,
  });
  assert.equal(out, network);
  assert.equal(diagnostics.strategy, "no-truth");
});
