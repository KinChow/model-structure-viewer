import assert from "node:assert/strict";
import test from "node:test";
import { hfAttentionAttr, hfFfnAttr, hfNamedClass } from "./index.js";

test("hfNamedClass：配方写全名则用全名，没写用词干，不剥 architectures[0] 再拼", () => {
  assert.equal(hfNamedClass({ architecture: "Qwen3ForCausalLM" }, "attentionClass", "Attention"), "Attention");
  assert.equal(hfNamedClass({ architecture: "DeepseekV3ForCausalLM" }, "moeClass", "MoE"), "MoE");
  assert.equal(hfNamedClass({ architecture: "MiniMaxM2ForCausalLM" }, "moeClass", "MoE"), "MiniMaxM2SparseMoeBlock");
  assert.equal(
    hfNamedClass({ architecture: "KimiK3ForConditionalGeneration" }, "attentionClass", "Attention", "Attention", { kind: "mla" }),
    "KimiMLAAttention",
  );
  assert.equal(
    hfNamedClass({ architecture: "KimiK3ForConditionalGeneration" }, "attentionClass", "Attention", "Attention", { kind: "linear" }),
    "KimiDeltaAttention",
  );
  assert.equal(
    hfNamedClass({ architecture: "Qwen4ExpForConditionalGeneration" }, "attentionClass", "Attention", "Attention", { kind: "linear" }),
    "Qwen4ExpTextGatedDeltaNet",
  );
  assert.equal(
    hfNamedClass({ architecture: "Qwen4ExpForConditionalGeneration" }, "attentionClass", "Attention", "Attention", { kind: "qsa" }),
    "Qwen4ExpTextAttention",
  );
  assert.equal(
    hfNamedClass({ architecture: "Qwen3_5ForConditionalGeneration" }, "attentionClass", "Attention", "Attention", { kind: "linear" }),
    "Qwen3_5GatedDeltaNet",
  );
  assert.equal(
    hfNamedClass({ architecture: "Qwen3_5ForConditionalGeneration" }, "attentionClass", "Attention", "Attention", { kind: "qwen35_full" }),
    "Qwen3_5Attention",
  );
  assert.equal(
    hfNamedClass({ architecture: "Qwen3_5ForConditionalGeneration" }, "modelClass", "Model"),
    "Qwen3_5TextModel",
  );
  assert.equal(hfNamedClass({}, "attentionClass", "Attention", "Attention"), "Attention");
});

test("hfAttentionAttr 跟随 transformers 属性名", () => {
  assert.equal(hfAttentionAttr({ architecture: "Qwen3_5ForConditionalGeneration" }, "linear"), "linear_attn");
  assert.equal(hfAttentionAttr({ architecture: "Qwen3_5ForConditionalGeneration" }, "qwen35_full"), "self_attn");
  assert.equal(hfAttentionAttr({ architecture: "Qwen4ExpForConditionalGeneration" }, "linear"), "linear_attn");
  assert.equal(hfAttentionAttr({ architecture: "Qwen4ExpForConditionalGeneration" }, "qsa"), "self_attn");
  assert.equal(hfAttentionAttr({ architecture: "KimiK3ForConditionalGeneration" }, "linear"), "self_attn");
  assert.equal(hfAttentionAttr({ architecture: "Glm5NextForConditionalGeneration" }, "linear"), "self_attn");
  assert.equal(hfAttentionAttr({ architecture: "DeepseekV3ForCausalLM" }, "mla"), "self_attn");
});

test("hfFfnAttr 跟随 HF _modules key，不按 layerKind 二分", () => {
  assert.equal(hfFfnAttr({ architecture: "DeepseekV3ForCausalLM" }, "moe"), "mlp");
  assert.equal(hfFfnAttr({ architecture: "DeepseekV3ForCausalLM" }, "dense"), "mlp");
  assert.equal(hfFfnAttr({ architecture: "Qwen3MoeForCausalLM" }, "moe"), "mlp");
  assert.equal(hfFfnAttr({ architecture: "MiniMaxM3SparseForConditionalGeneration" }, "moe"), "mlp");
  assert.equal(hfFfnAttr({ architecture: "MiniMaxM2ForCausalLM" }, "moe"), "block_sparse_moe");
  assert.equal(hfFfnAttr({ architecture: "MiniMaxM2ForCausalLM" }, "dense"), "block_sparse_moe");
  assert.equal(hfFfnAttr({ architecture: "KimiK3ForConditionalGeneration" }, "moe"), "block_sparse_moe");
  assert.equal(hfFfnAttr({ architecture: "KimiK3ForConditionalGeneration" }, "dense"), "mlp");
  assert.equal(hfFfnAttr({}, "moe"), "mlp");
});
