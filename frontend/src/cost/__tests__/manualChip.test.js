import assert from "node:assert/strict";
import test from "node:test";
import { createManualChip, validateManualChipInput } from "../chips/manual.js";

test("手动芯片输入转换为内部 SI 单位", () => {
  const chip = createManualChip({ name: "My Chip", memoryGb: 64, memoryBandwidthTb: 2, bf16Tflops: 300, interconnectGb: 100 });
  assert.equal(chip.id, "my-chip");
  assert.equal(chip.memory_bytes, 64e9);
  assert.equal(chip.memory_bandwidth, 2e12);
  assert.equal(chip.peak_flops.bf16, 300e12);
  assert.equal(chip.interconnect.intra_node.bandwidth, 100e9);
  assert.equal(chip.confidence, "local");
});

test("手动芯片拒绝空名称和非正数规格", () => {
  assert.equal(validateManualChipInput({}).length, 5);
});

test("手工条目带 field_sources：已填字段逐项标注 user-input（旁路 C）", () => {
  const chip = createManualChip({ name: "My Chip", memoryGb: 64, memoryBandwidthTb: 2, bf16Tflops: 300, interconnectGb: 100 });
  assert.deepEqual(chip.field_sources, {
    memory_bytes: "user-input",
    memory_bandwidth: "user-input",
    peak_flops: "user-input",
    interconnect: "user-input",
  });
  assert.equal(chip.source, "manual-session");
  // 与 public.js 公开卡同结构：扁平字段名 → 来源字符串
  for (const value of Object.values(chip.field_sources)) assert.equal(typeof value, "string");
  // 未填的 dtype 不进 peak_flops，也不单列来源
  assert.equal(chip.peak_flops.fp8, undefined);
});

test("手工条目来源可由表单 URL 覆盖 user-input，未填字段不登记来源", () => {
  const chip = createManualChip({ name: "Src", memoryGb: 64, memoryBandwidthTb: 2, bf16Tflops: 300, intraNodeGb: 100, interNodeGb: 25, sourceUrl: " https://example.com/spec " });
  for (const value of Object.values(chip.field_sources)) assert.equal(value, "https://example.com/spec");
  assert.equal(chip.field_sources.interconnect, "https://example.com/spec");
  assert.equal(chip.interconnect.inter_node.bandwidth, 25e9);
});
