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
