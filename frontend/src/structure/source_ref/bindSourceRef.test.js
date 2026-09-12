import assert from "node:assert/strict";
import test from "node:test";
import { bindSourceRefToGraph, canonicalSourceRefPath, displaySourceRef } from "./bindSourceRef.js";

const CATALOG = {
  model_id: "Org/Demo",
  transformers_version: "4.40.0",
  modules: [
    {
      module_path: "root.model.layers.0.self_attn",
      class_name: "Qwen2Attention",
      has_params: false,
      source_ref: {
        framework: "transformers",
        module_path: "transformers.models.qwen2.modeling_qwen2",
        class_name: "Qwen2Attention",
        file: "src/transformers/models/qwen2/modeling_qwen2.py",
        line: 1456,
        version: "4.40.0",
        url: "https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/models/qwen2/modeling_qwen2.py#L1456",
      },
    },
    {
      module_path: "root.model.layers.0.group0",
      class_name: "Qwen2DecoderLayer",
      has_params: false,
      source_ref: {
        framework: "transformers",
        class_name: "Qwen2DecoderLayer",
        file: "src/transformers/models/qwen2/modeling_qwen2.py",
        line: 900,
        version: "4.40.0",
        url: "https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/models/qwen2/modeling_qwen2.py#L900",
      },
    },
  ],
};

test("canonicalSourceRefPath 与后端对账路径同折叠", () => {
  assert.equal(canonicalSourceRefPath("root.model.layers.0.self_attn"), "layers.self_attn");
  assert.equal(canonicalSourceRefPath("layers.0.self_attn"), "layers.self_attn");
  assert.equal(canonicalSourceRefPath("root.language_model.layers.0.linear_attn"), "layers.linear_attn");
  assert.equal(canonicalSourceRefPath("root"), "");
});

test("displaySourceRef 版本不一致时去掉 #L 锚点，不编造 url", () => {
  const matched = displaySourceRef(CATALOG.modules[0].source_ref, "4.40.0");
  assert.equal(matched.versionMismatch, false);
  assert.match(matched.url, /#L1456$/);
  const stale = displaySourceRef(CATALOG.modules[0].source_ref, "5.16.1");
  assert.equal(stale.versionMismatch, true);
  assert.equal(stale.url.endsWith("#L1456"), false);
  assert.equal(stale.line, 1456);
  assert.equal(displaySourceRef(null), null);
  assert.equal(displaySourceRef({ file: "/tmp/custom.py", line: 3, url: null }).url, null);
});

test("bindSourceRefToGraph 按路径绑定，聚合节点保持 null", () => {
  const graph = {
    nodes: [
      { id: "root.2.0", canonical_id: "layers.0.self_attn", type: "attention", attributes: { class: "Qwen2Attention" } },
      { id: "root.2", canonical_id: "layers.0.group0", type: "layer-group", attributes: { class: "Qwen2DecoderLayer" } },
      { id: "root.9", canonical_id: "scores", type: "operator", attributes: { class: "softmax" } },
    ],
  };
  const { graph: bound, diagnostics } = bindSourceRefToGraph(graph, CATALOG);
  assert.equal(bound.nodes[0].source_ref.file, "src/transformers/models/qwen2/modeling_qwen2.py");
  assert.equal(bound.nodes[0].source_ref.line, 1456);
  assert.equal(bound.nodes[1].source_ref, null);
  assert.equal(bound.nodes[2].source_ref, null);
  assert.equal(diagnostics.bound, 1);
  assert.equal(diagnostics.unmatched, 1);
  assert.equal(diagnostics.transformers_version, "4.40.0");
});

test("bindSourceRefToGraph 按 HF 属性名绑定 linear_attn 与 self_attn，同 class 多实例共享定义", () => {
  const catalog = {
    transformers_version: "5.16.1",
    modules: [
      {
        module_path: "root.language_model.layers.0.linear_attn",
        class_name: "Qwen3_5GatedDeltaNet",
        source_ref: { file: "modeling_qwen3_5.py", line: 383, class_name: "Qwen3_5GatedDeltaNet" },
      },
      {
        module_path: "root.language_model.layers.3.self_attn",
        class_name: "Qwen3_5Attention",
        source_ref: { file: "modeling_qwen3_5.py", line: 632, class_name: "Qwen3_5Attention" },
      },
      {
        module_path: "root.language_model.layers.0.mlp",
        class_name: "Qwen3_5MLP",
        source_ref: { file: "modeling_qwen3_5.py", line: 707, class_name: "Qwen3_5MLP" },
      },
      {
        module_path: "root.language_model.layers.3.mlp",
        class_name: "Qwen3_5MLP",
        source_ref: { file: "modeling_qwen3_5.py", line: 707, class_name: "Qwen3_5MLP" },
      },
      {
        module_path: "root.visual",
        class_name: "Qwen3_5VisionModel",
        source_ref: { file: "modeling_qwen3_5.py", line: 1009, class_name: "Qwen3_5VisionModel" },
      },
    ],
  };
  const graph = {
    nodes: [
      { id: "a", canonical_id: "layers.0.linear_attn", type: "attention", attributes: { class: "Qwen3_5GatedDeltaNet" } },
      { id: "b", canonical_id: "layers.3.self_attn", type: "attention", attributes: { class: "Qwen3_5Attention" } },
      { id: "c", canonical_id: "layers.0.mlp", type: "mlp", attributes: { class: "Qwen3_5MLP" } },
      { id: "d", canonical_id: "visual.0", type: "vision", attributes: { class: "Qwen3_5VisionBlock" } },
    ],
  };
  const { graph: bound, diagnostics } = bindSourceRefToGraph(graph, catalog);
  assert.equal(bound.nodes[0].source_ref.line, 383);
  assert.equal(bound.nodes[1].source_ref.line, 632);
  assert.equal(bound.nodes[2].source_ref.line, 707);
  assert.equal(bound.nodes[3].source_ref, null);
  assert.equal(diagnostics.bound, 3);
  assert.equal(diagnostics.unmatched, 1);
});
