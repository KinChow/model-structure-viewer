import assert from "node:assert/strict";
import test from "node:test";
import { PUBLIC_CHIPS, validatePublicChipCatalog } from "../chips/public.js";
import { CONFIDENCE_VALUES } from "../chips/coverage.js";
import { chipRates } from "../chips/rates.js";

test("公开芯片目录每条都有来源且 id 唯一", () => {
  assert.deepEqual(validatePublicChipCatalog(), []);
  assert.equal(new Set(PUBLIC_CHIPS.map((chip) => chip.id)).size, PUBLIC_CHIPS.length);
  for (const chip of PUBLIC_CHIPS) {
    assert.match(chip.source, /^https:\/\//);
    assert.ok(CONFIDENCE_VALUES.has(chip.confidence));
    assert.ok(chip.peak_flops.fp32 > 0);
    // §7：每个规格子树都有字段级来源
    for (const key of ["memory_bytes", "memory_bandwidth", "peak_flops", "interconnect"]) {
      assert.match(chip.field_sources?.[key] || "", /^https:\/\//);
    }
  }
});

test("公开芯片目录保留独立 FP32 峰值", () => {
  assert.equal(PUBLIC_CHIPS.find((chip) => chip.id === "nvidia-a100-80gb-sxm").peak_flops.fp32, 19.5e12);
  assert.equal(PUBLIC_CHIPS.find((chip) => chip.id === "nvidia-h100-80gb-sxm").peak_flops.fp32, 67e12);
  assert.equal(PUBLIC_CHIPS.find((chip) => chip.id === "nvidia-l40s-48gb").peak_flops.fp32, 91.6e12);
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

test("昇腾 910B4 按保守口径入库，无独立 SFU 时声明 sfu→vector 语义映射", () => {
  const chip = PUBLIC_CHIPS.find((entry) => entry.id === "huawei-ascend-910b4");
  assert.ok(chip, "910B4 条目存在");
  assert.equal(chip.vendor, "Huawei");
  assert.equal(chip.memory_bytes, 32e9);
  assert.equal(chip.memory_bandwidth, 800e9);
  assert.equal(chip.peak_flops.bf16, 280e12);
  assert.equal(chip.peak_flops.fp16, 280e12);
  assert.equal(chip.peak_flops.int8, 560e12);
  assert.equal(chip.vector_flops, 9.2e12);
  assert.equal(chip.interconnect.intra_node.bandwidth, 392e9);
  assert.equal(chip.sfu_ops, undefined);
  assert.equal(chip.sfu_rate_source, "vector");
  assert.equal(chip.confidence, "community");
  assert.ok(chip.notes.some((note) => note.includes("官方口径有调整史")));
  // 完整 field_sources：每个规格子树都有可追溯来源
  for (const key of ["memory_bytes", "memory_bandwidth", "peak_flops", "interconnect", "vector_flops"]) {
    assert.match(chip.field_sources[key], /^https:\/\//);
  }
  // sfu 速率经向量单元语义映射可得（rates.js 预留插槽生效）
  const rates = chipRates(chip, { dtype: "bf16" });
  assert.ok(rates.sfuPerSecond > 0);
  assert.equal(rates.sfuPerSecond, rates.vectorPerSecond);
  assert.ok(!rates.missing.includes("sfu_ops"));
});

test("芯片 notes 双语：每条中文 notes 都有等长英文 notes_en（无汉字）", () => {
  const han = /\p{Script=Han}/u;
  for (const chip of PUBLIC_CHIPS) {
    if (!Array.isArray(chip.notes) || chip.notes.length === 0) continue;
    assert.ok(Array.isArray(chip.notes_en), `${chip.id}: 缺 notes_en`);
    assert.equal(chip.notes_en.length, chip.notes.length, `${chip.id}: notes_en 与 notes 条数不一致`);
    for (const note of chip.notes_en) {
      assert.ok(note && note.trim(), `${chip.id}: notes_en 有空条`);
      assert.doesNotMatch(note, han, `${chip.id}: notes_en 含汉字 -> ${note}`);
    }
  }
});
