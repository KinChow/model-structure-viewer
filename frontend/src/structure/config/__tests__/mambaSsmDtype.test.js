// 守护：normalizeConfig 读取 HF 配置里的 recurrent(ssm) state dtype 覆盖键
// （mamba_ssm_dtype / ssm_dtype / mamba2_state_dtype），并让 memory lens 采用之。
// 修复前 normalize 从不读取该键 → 文档声称的 config 覆盖对 recurrent state 静默失效，
// 任何声明 bf16/fp16 的模型都会被按 fp32 高估 ~2×。
import assert from "node:assert/strict";
import test from "node:test";
import { normalizeConfig } from "../normalize.js";
import { linearStateResidentDecl } from "../../operators/ops/index.js";

const baseLinearConfig = (extra = {}) => ({
  model_type: "qwen3_next",
  hidden_size: 1024,
  num_hidden_layers: 4,
  num_attention_heads: 16,
  head_dim: 128,
  linear_num_key_heads: 16,
  linear_num_value_heads: 32,
  linear_key_head_dim: 128,
  linear_value_head_dim: 128,
  linear_conv_kernel_dim: 4,
  ...extra,
});

test("recurrent state dtype 缺省落到 fp32（无声明）", () => {
  const normalized = normalizeConfig(baseLinearConfig());
  assert.equal(normalized.mambaSsmDtype, undefined);
  assert.equal(linearStateResidentDecl(normalized).state_recurrent_dtype, "F32");
});

test("mamba_ssm_dtype=bfloat16 被读取并令 recurrent state 用 BF16", () => {
  const normalized = normalizeConfig(baseLinearConfig({ mamba_ssm_dtype: "bfloat16" }));
  assert.equal(normalized.mambaSsmDtype, "bfloat16");
  assert.equal(linearStateResidentDecl(normalized).state_recurrent_dtype, "BF16");
});

test("torch.float16 / 别名键 ssm_dtype 也被识别", () => {
  const a = normalizeConfig(baseLinearConfig({ mamba_ssm_dtype: "torch.float16" }));
  assert.equal(linearStateResidentDecl(a).state_recurrent_dtype, "F16");
  const b = normalizeConfig(baseLinearConfig({ ssm_dtype: "bfloat16" }));
  assert.equal(linearStateResidentDecl(b).state_recurrent_dtype, "BF16");
});
