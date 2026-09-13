// quantBytes 的手算单测：三种量化方案的权重/scale 字节 + 路径排除语义。
import assert from "node:assert/strict";
import test from "node:test";
import { quantLinearWeightBytes, isQuantizedPath, quantizationConfigOf, logicalElementsFromHeader } from "../quantBytes.js";

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

test("compressed-tensors w4a16：0.5B/元素 + 每组 fp16 scale（symmetric 无零点）", () => {
  // 照抄 Kimi-K2-Thinking：int4 / group 32 / symmetric / pack-quantized
  const quant = { quant_method: "compressed-tensors", format: "pack-quantized", config_groups: { group_0: { targets: ["Linear"], weights: { num_bits: 4, type: "int", strategy: "group", group_size: 32, symmetric: true } } } };
  const b = quantLinearWeightBytes({ out: 2048, inn: 7168, quant });
  assert.equal(b, 2048 * 7168 * 0.5 + 2048 * Math.ceil(7168 / 32) * 2);
  // 非对称零点未取证 → null 诚实缺项
  const asym = { ...quant, config_groups: { group_0: { weights: { num_bits: 4, type: "int", strategy: "group", group_size: 32, symmetric: false } } } };
  assert.equal(quantLinearWeightBytes({ out: 8, inn: 8, quant: asym }), null);
});

test("compressed-tensors mxfp4（K3）：0.5B/元素 + e8m0 scale 1B/32 组", () => {
  const quant = { quant_method: "compressed-tensors", format: "mxfp4-pack-quantized", config_groups: { group_0: { targets: ["Linear"], weights: { num_bits: 4, type: "float", strategy: "group", group_size: 32, symmetric: true } } } };
  const b = quantLinearWeightBytes({ out: 3072, inn: 3584, quant });
  assert.equal(b, 3072 * 3584 * 0.5 + 3072 * Math.ceil(3584 / 32) * 1);
});

test("compressed-tensors 的 ignore 数组 = modules_to_not_convert 同义（'re:' 前缀正则）", () => {
  // 照抄 Kimi-K2-Thinking 的 ignore 表：路由专家量化，attention/shared/dense-MLP/lm_head 排除
  const quant = { quant_method: "compressed-tensors", ignore: ["lm_head", "re:.*self_attn.*", "re:.*shared_experts.*", "re:.*mlp\\.(gate|up|gate_up|down)_proj.*"] };
  assert.equal(isQuantizedPath("decoder.3.mlp.expert_mlp", quant), true);
  assert.equal(isQuantizedPath("decoder.3.self_attn.q_proj", quant), false);
  assert.equal(isQuantizedPath("decoder.3.mlp.shared_experts.down_proj", quant), false);
  assert.equal(isQuantizedPath("decoder.3.mlp.gate_proj", quant), false);
  assert.equal(isQuantizedPath("lm_head", quant), false);
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
  assert.equal(isQuantizedPath("layers.3.self_attn.qkv_proj", quant), false);
  assert.equal(isQuantizedPath("layers.3.mlp.down_proj", quant), true);
  assert.equal(isQuantizedPath("layers.3.mlp.shared_experts.down_proj", quant), false);
  assert.equal(isQuantizedPath("mtp.layer.mlp.down_proj", quant), false);
  assert.equal(isQuantizedPath("visual.0.qkv_proj", quant), false);
  // 无 dynamic：全部量化
  assert.equal(isQuantizedPath("layers.0.self_attn.qkv_proj", { quant_method: "fp8" }), true);
});

test("modules_to_not_convert 数组：命中即不量化，优先于 dynamic", () => {
  // HF/vLLM 数组约定（GPTQ/FP8 checkpoint 常见）；模式支持字面路径与正则
  const quant = { quant_method: "fp8", weight_block_size: [128, 128], modules_to_not_convert: ["lm_head", ".*embed_tokens.*", ".*visual.*"] };
  assert.equal(isQuantizedPath("lm_head", quant), false);
  assert.equal(isQuantizedPath("embed_tokens", quant), false);
  assert.equal(isQuantizedPath("visual.0.qkv_proj", quant), false);
  assert.equal(isQuantizedPath("layers.3.self_attn.qkv_proj", quant), true);
  // 空数组不排除
  assert.equal(isQuantizedPath("lm_head", { quant_method: "fp8", modules_to_not_convert: [] }), true);
});

test("logicalElementsFromHeader：GPTQ I32×8 扣 qzeros，scale 的 F16 不计", () => {
  // Qwen3.5-27B-GPTQ-Int4 sidecar。BF16 兄模型 parameterTotal = 27,781,427,952。
  const header = {
    parameterCount: { BF16: 10668659184, F32: 8448, I32: 2157576192, F16: 133693440 },
  };
  const logical = logicalElementsFromHeader(header, { quant_method: "gptq", bits: 4, group_size: 128 });
  assert.equal(logical, 2157576192 * 8 - 133693440 + 10668659184 + 8448);
  assert.ok(Math.abs(logical / 27781427952 - 1) < 0.01);
});

test("logicalElementsFromHeader：MXFP8 跳过 U8 scale，权重 F8 1:1", () => {
  // MiniMax-M3-MXFP8 sidecar。BF16 兄模型 parameterTotal = 427,040,140,160 精确相等。
  const header = {
    parameterCount: { BF16: 3323221760, F8_E4M3: 423670579200, U8: 13239705600, F32: 46339200 },
  };
  const logical = logicalElementsFromHeader(header, { quant_method: "mxfp8", weight_block_size: [1, 32] });
  assert.equal(logical, 427040140160);
});

test("logicalElementsFromHeader：FP8 跳过 F8_E8M0 scale，F8_E4M3 1:1", () => {
  const header = {
    parameterCount: { BF16: 2103729152, F8_E4M3: 751226191872, F32: 45872560 },
  };
  const logical = logicalElementsFromHeader(header, { quant_method: "fp8", weight_block_size: [128, 128] });
  assert.equal(logical, 2103729152 + 751226191872 + 45872560);
});

test("logicalElementsFromHeader：NVFP4 I8×2，跳过 E8M0 scale", () => {
  // DeepSeek-V4-Flash sidecar。I8 是 fp4 打包（1B 存 2 个逻辑元素）。
  const header = {
    parameterCount: {
      BF16: 1415259264, F32: 36168018, F8_E8M0: 8858737664, F8_E4M3: 6023020544, I8: 141733920768, I64: 2327040,
    },
  };
  const logical = logicalElementsFromHeader(header, { quant_method: "fp8", weight_block_size: [128, 128] });
  assert.equal(logical, 141733920768 * 2 + 6023020544 + 1415259264 + 36168018 + 2327040);
});

test("logicalElementsFromHeader：compressed-tensors I32×8，BF16 里扣 scale", () => {
  // Kimi-K2-Thinking sidecar。group_size=32，scale 进 BF16 桶。
  const header = {
    parameterCount: { BF16: 43431131776, I32: 126835891200, F32: 23040 },
  };
  const quant = {
    quant_method: "compressed-tensors",
    config_groups: { group_0: { weights: { num_bits: 4, type: "int", strategy: "group", group_size: 32, symmetric: true } } },
  };
  const packed = 126835891200 * 8;
  const logical = logicalElementsFromHeader(header, quant);
  assert.equal(logical, packed + 43431131776 + 23040 - packed / 32);
});

test("logicalElementsFromHeader：缺 parameterCount 或空桶 → null", () => {
  assert.equal(logicalElementsFromHeader({}, { quant_method: "fp8" }), null);
  assert.equal(logicalElementsFromHeader({ parameterCount: {} }, { quant_method: "gptq" }), null);
});

test("quantizationConfigOf：顶层 / raw 嵌套 / text_config 嵌套（normalized 的 raw.text_config 同样命中）", () => {
  const q = { quant_method: "fp8" };
  assert.equal(quantizationConfigOf({ quantization_config: q }), q);
  assert.equal(quantizationConfigOf({ raw: { quantization_config: q } }), q);
  assert.equal(quantizationConfigOf({ text_config: { quantization_config: q } }), q);
  // W-C 核实补位：Kimi K2.5 系（VLM）把 quantization_config 嵌在 raw.text_config
  assert.equal(quantizationConfigOf({ raw: { text_config: { quantization_config: q } } }), q);
  assert.equal(quantizationConfigOf({}), null);
});
