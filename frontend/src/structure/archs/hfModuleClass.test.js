import assert from "node:assert/strict";
import test from "node:test";
import { hfModulePrefix, hfNamedClass } from "./index.js";

test("hfModulePrefix 剥 transformers 任务后缀，必要时用配方覆盖", () => {
  assert.equal(hfModulePrefix("Qwen3ForCausalLM"), "Qwen3");
  assert.equal(hfModulePrefix("Qwen3_5ForConditionalGeneration"), "Qwen3_5");
  assert.equal(hfModulePrefix("MiniMaxM2ForCausalLM"), "MiniMaxM2");
  assert.equal(hfModulePrefix("MiniMaxM3SparseForConditionalGeneration"), "MiniMaxM3VL");
  assert.equal(hfModulePrefix("KimiK25ForConditionalGeneration"), "DeepseekV3");
  assert.equal(hfModulePrefix("KimiK3ForConditionalGeneration"), "Kimi");
});

test("hfNamedClass 按架构前缀拼模块类，混合注意力按 kind 分 stem", () => {
  assert.equal(hfNamedClass({ architecture: "Qwen3ForCausalLM" }, "attentionStem", "Attention"), "Qwen3Attention");
  assert.equal(hfNamedClass({ architecture: "MiniMaxM2ForCausalLM" }, "moeStem", "MoE"), "MiniMaxM2SparseMoeBlock");
  assert.equal(hfNamedClass({ architecture: "DeepseekV3ForCausalLM" }, "moeStem", "MoE"), "DeepseekV3MoE");
  assert.equal(
    hfNamedClass({ architecture: "KimiK3ForConditionalGeneration" }, "attentionStem", "Attention", "Attention", { kind: "mla" }),
    "KimiMLAAttention",
  );
  assert.equal(
    hfNamedClass({ architecture: "KimiK3ForConditionalGeneration" }, "attentionStem", "Attention", "Attention", { kind: "linear" }),
    "KimiDeltaAttention",
  );
  assert.equal(
    hfNamedClass({ architecture: "Qwen4ExpForConditionalGeneration" }, "attentionStem", "Attention"),
    "Qwen4ExpTextAttention",
  );
  assert.equal(hfNamedClass({}, "attentionStem", "Attention", "Attention"), "Attention");
});
