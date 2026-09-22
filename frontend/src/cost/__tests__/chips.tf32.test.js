// TF32 费率行：公开芯片的 dense 值 + 费率解析 + 手工表单透传。
// 背景：Hopper/Blackwell + DeepGEMM 时 mHC 的 pre-GEMM 在 TF32 tensor core
// 上运行（vLLM deepseek_v4 tilelang，2026-09-09 kernel 取证）——芯片表此前
// 没有 TF32 速率行，roofline 对该类 GEMM 无法按正确算力判定。
import assert from "node:assert/strict";
import test from "node:test";
import { PUBLIC_CHIPS } from "../chips/public.js";
import { chipRates } from "../chips/rates.js";
import { createManualChip } from "../chips/manual.js";

test("NVIDIA 公开芯片带 TF32 dense 行，且 fp32 <= tf32 < bf16（防单位错）", () => {
  const nvidia = PUBLIC_CHIPS.filter((chip) => chip.vendor === "NVIDIA" && chip.peak_flops?.tf32);
  assert.ok(nvidia.length >= 3, "A100/H100/L40S 都应有 tf32");
  for (const chip of nvidia) {
    const { fp32, tf32, bf16 } = chip.peak_flops;
    // L40 的官方 datasheet 将 dense TF32 标为 90.5 TFLOPS，与 FP32 相同；
    // 因此这里不能假设所有 Ada 卡都严格高于 FP32。
    assert.ok(tf32 >= fp32, `${chip.id}: tf32 >= fp32`);
    assert.ok(tf32 < bf16, `${chip.id}: tf32 < bf16`);
    assert.ok(chip.field_sources?.peak_flops, `${chip.id}: 逐字段来源`);
  }
  // A100 dense TF32 = 156 TFLOPS（官方页 156/312 两列取未启用稀疏性列）
  assert.equal(PUBLIC_CHIPS.find((c) => c.id.includes("a100")).peak_flops.tf32, 156e12);
  assert.equal(PUBLIC_CHIPS.find((c) => c.id.includes("h100")).peak_flops.tf32, 494.5e12);
});

test("chipRates 以 dtype=tf32 解析矩阵费率（peak_flops·η/2）", () => {
  const chip = { peak_flops: { tf32: 494.5e12 }, memory_bandwidth: 3.35e12 };
  const rates = chipRates(chip, { dtype: "tf32", efficiency: { flops: 1, hbm: 1 } });
  assert.equal(rates.matrixPerSecond, 494.5e12 / 2);
  assert.ok(rates.missing.every((field) => !field.startsWith("peak_flops")));
});

test("手工芯片表单透传 tf32Tflops", () => {
  const chip = createManualChip({ name: "test", fp32Tflops: 10, tf32Tflops: 100, bf16Tflops: 200 });
  assert.equal(chip.peak_flops.tf32, 100e12);
  assert.equal(chip.peak_flops.bf16, 200e12);
});
