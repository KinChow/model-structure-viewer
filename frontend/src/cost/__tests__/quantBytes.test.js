// quantBytes 的手算单测：三种量化方案的权重/scale 字节 + 路径排除语义。
import assert from "node:assert/strict";
import test from "node:test";
import { quantLinearWeightBytes, isQuantizedPath, quantizationConfigOf } from "../quantBytes.js";

test("fp8 块量化：权重 1B/元素 + scale（ue8m0 1B/块，默认 fp32 4B/块）", () => {
  // [8192, 7168]，块 [128,128]：scale = 64×56 块
  const shape = { out: 8192, inn: 7168 };
  const ue8m0 = quantLinearWeightBytes({ ...shape, quant: { quant_method: "fp8", weight_block_size: [128, 128], scale_fmt: "ue8m0" } });
  assert.equal(ue8m0, 8192 * 7168 + 64 * 56 * 1);
  const fp32Scale = quantLinearWeightBytes({ ...shape, quant: { quant_method: "fp8", weight_block_size: [128, 128] } });
  assert.equal(fp32Scale, 8192 * 7168 + 64 * 56 * 4);
});

test("mxfp8：[1,32] 块，e8m0 scale 1B/块", () => {
  const b = quantLinearWeightBytes({ out: 4096, inn: 4096, quant: { quant_method: "mxfp8", weight_block_size: [1, 32] } });
  assert.equal(b, 4096 * 4096 + 4096 * 128);
});

test("gptq int4：0.5B/元素 + 每组 fp16 scale + int4 qzeros", () => {
  const b = quantLinearWeightBytes({ out: 4096, inn: 4096, quant: { quant_method: "gptq", bits: 4, group_size: 128 } });
  assert.equal(b, 4096 * 4096 * 0.5 + 4096 * 32 * (2 + 0.5));
});

test("未知方案 / 非正形状：返回 null（调用方退回标量）", () => {
  assert.equal(quantLinearWeightBytes({ out: 4, inn: 4, quant: { quant_method: "awq" } }), null);
  assert.equal(quantLinearWeightBytes({ out: 0, inn: 4, quant: { quant_method: "fp8", weight_block_size: [128, 128] } }), null);
  assert.equal(quantLinearWeightBytes({ out: 4, inn: 4, quant: null }), null);
});

test("dynamic 排除/包含：'-:' 前缀为排除正则，显式列出为包含，候选路径桥接命名差异", () => {
  // 照抄 Qwen3.5-27B-GPTQ-Int4 的真实 dynamic 表
  const quant = { quant_method: "gptq", dynamic: {
    lm_head: {},
    "model.language_model.embed_tokens": {},
    "-:.*attn.*": {},
    "-:.*shared_expert.*": {},
    "-:.*mtp.*": {},
    "-:.*visual.*": {},
  } };
  assert.equal(isQuantizedPath("lm_head", quant), true);
  // 字面路径 pattern（checkpoint 命名）经候选路径补全命中树 id
  assert.equal(isQuantizedPath("embed_tokens", quant), true);
  assert.equal(isQuantizedPath("decoder.3.self_attn.qkv_proj", quant), false);
  assert.equal(isQuantizedPath("decoder.3.mlp.down_proj", quant), true);
  assert.equal(isQuantizedPath("decoder.3.moe.shared_experts.down_proj", quant), false);
  assert.equal(isQuantizedPath("mtp.layer.mlp.down_proj", quant), false);
  // vision_tower → visual（Qwen3.5 checkpoint 对视觉塔的命名）
  assert.equal(isQuantizedPath("vision_tower.0.qkv_proj", quant), false);
  // 无 dynamic：全部量化
  assert.equal(isQuantizedPath("decoder.0.self_attn.qkv_proj", { quant_method: "fp8" }), true);
});

test("quantizationConfigOf：顶层 / raw 嵌套 / text_config 嵌套", () => {
  const q = { quant_method: "fp8" };
  assert.equal(quantizationConfigOf({ quantization_config: q }), q);
  assert.equal(quantizationConfigOf({ raw: { quantization_config: q } }), q);
  assert.equal(quantizationConfigOf({ text_config: { quantization_config: q } }), q);
  assert.equal(quantizationConfigOf({}), null);
});
