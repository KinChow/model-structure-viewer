import assert from "node:assert/strict";
import test from "node:test";
import { getChipCoverage, validateChipEntry } from "../chips/coverage.js";

const COMPLETE = {
  id: "test-chip",
  vendor: "test",
  name: "Test chip",
  memory_bytes: 80 * 1024 ** 3,
  memory_bandwidth: 1e12,
  peak_flops: { bf16: 1e15 },
  interconnect: {
    intra_node: { kind: "PCIe", bandwidth: 1e11 },
    inter_node: { kind: "RoCE", bandwidth: 5e10 },
  },
  source: "https://example.com/spec",
  confidence: "official",
};

test("完整芯片规格开启对应能力", () => {
  const result = getChipCoverage(COMPLETE, "bf16");
  assert.deepEqual(result.missing, []);
  assert.equal(result.capabilities.fit, true);
  assert.equal(result.capabilities.compute_bound, true);
  assert.equal(result.capabilities.memory_bound, true);
  assert.equal(result.capabilities.comm_bound, true);
});

test("缺显存带宽时只禁用 memory-bound，不填估算值", () => {
  const result = getChipCoverage({ ...COMPLETE, memory_bandwidth: undefined }, "bf16");
  assert.deepEqual(result.missing, ["memory_bandwidth"]);
  assert.equal(result.capabilities.fit, true);
  assert.equal(result.capabilities.max_context, true);
  assert.equal(result.capabilities.compute_bound, false);
  assert.equal(result.capabilities.memory_bound, false);
});

test("缺跨节点带宽时保留能力并给出偏乐观警告", () => {
  const chip = { ...COMPLETE, interconnect: { intra_node: COMPLETE.interconnect.intra_node } };
  const result = getChipCoverage(chip, "bf16");
  assert.equal(result.capabilities.comm_bound, true);
  assert.match(result.warnings[0], /跨节点按节点内带宽估算/);
});

test("缺显存容量时禁用 fit 和 max_context", () => {
  const result = getChipCoverage({ ...COMPLETE, memory_bytes: null }, "bf16");
  assert.equal(result.capabilities.fit, false);
  assert.equal(result.capabilities.max_context, false);
  assert.equal(result.capabilities.compute_bound, true);
  assert.equal(result.capabilities.weight_share, true);
});

test("芯片条目必须有可追溯来源", () => {
  assert.deepEqual(validateChipEntry({ id: "x", vendor: "v", name: "n" }), ["缺少 source"]);
});
