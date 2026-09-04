import assert from "node:assert/strict";
import test from "node:test";
import { PUBLIC_CHIPS, validatePublicChipCatalog } from "../chips/public.js";

test("公开芯片目录每条都有来源且 id 唯一", () => {
  assert.deepEqual(validatePublicChipCatalog(), []);
  assert.equal(new Set(PUBLIC_CHIPS.map((chip) => chip.id)).size, PUBLIC_CHIPS.length);
  for (const chip of PUBLIC_CHIPS) {
    assert.match(chip.source, /^https:\/\//);
    assert.equal(chip.confidence, "official");
  }
});

test("公开芯片目录不使用二进制 GiB 冒充厂商十进制 GB", () => {
  const a100 = PUBLIC_CHIPS.find((chip) => chip.id === "nvidia-a100-80gb-sxm");
  assert.equal(a100.memory_bytes, 80e9);
  assert.notEqual(a100.memory_bytes, 80 * 1024 ** 3);
});

test("H100 稠密算力保留由官方稀疏峰值换算的说明", () => {
  const h100 = PUBLIC_CHIPS.find((chip) => chip.id === "nvidia-h100-80gb-sxm");
  assert.ok(h100.notes.some((note) => note.includes("稀疏峰值除以 2")));
  assert.equal(h100.peak_flops.bf16, 989.5e12);
});

test("L40S 使用官方未启用稀疏性的 BF16 与双向 PCIe 数据", () => {
  const l40s = PUBLIC_CHIPS.find((chip) => chip.id === "nvidia-l40s-48gb");
  assert.equal(l40s.memory_bytes, 48e9);
  assert.equal(l40s.memory_bandwidth, 864e9);
  assert.equal(l40s.peak_flops.bf16, 362.05e12);
  assert.equal(l40s.interconnect.intra_node.bandwidth, 64e9);
  assert.ok(l40s.notes.some((note) => note.includes("双向带宽")));
});
