import assert from "node:assert/strict";
import test from "node:test";
import { getChipCoverage, validateChipEntry } from "../chips/coverage.js";
import { createManualChip } from "../chips/manual.js";
import { formatIssue } from "../../i18n/format.js";

const COMPLETE = {
  id: "test-chip",
  vendor: "test",
  name: "Test chip",
  memory_bytes: 80 * 1024 ** 3,
  memory_bandwidth: 1e12,
  peak_flops: { bf16: 1e15 },
  vector_flops: 1e14,
  sfu_ops: 1e13,
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
  assert.equal(result.warnings[0].code, "chip.missingInterNode");
  assert.match(formatIssue("zh", result.warnings[0]), /跨节点按节点内带宽估算/);
  assert.match(formatIssue("en", result.warnings[0]), /intra-node bandwidth/);
});

test("缺显存容量时禁用 fit 和 max_context", () => {
  const result = getChipCoverage({ ...COMPLETE, memory_bytes: null }, "bf16");
  assert.equal(result.capabilities.fit, false);
  assert.equal(result.capabilities.max_context, false);
  assert.equal(result.capabilities.compute_bound, true);
  assert.equal(result.capabilities.weight_share, true);
});

test("芯片条目必须有可追溯来源", () => {
  assert.deepEqual(validateChipEntry({ id: "x", vendor: "v", name: "n" }), [{ code: "chip.missingField", params: { field: "source" } }]);
});

test("缺 vector_flops/sfu_ops 时对应单元能力关闭并提示补充路径", () => {
  const noVector = getChipCoverage({ ...COMPLETE, vector_flops: undefined }, "bf16");
  assert.deepEqual(noVector.missing, ["vector_flops"]);
  assert.equal(noVector.capabilities.vector_bound, false);
  assert.equal(noVector.capabilities.sfu_bound, true);
  assert.match(noVector.warnings.map((warning) => warning.code).join(" "), /missingVectorFlops/);

  const noSfu = getChipCoverage({ ...COMPLETE, sfu_ops: undefined }, "bf16");
  assert.deepEqual(noSfu.missing, ["sfu_ops"]);
  assert.equal(noSfu.capabilities.sfu_bound, false);
  assert.match(noSfu.warnings.map((warning) => warning.code).join(" "), /missingSfuOps/);
});

test("声明 sfu→vector 语义映射且向量吞吐在场时不再把 sfu_ops 记为缺项", () => {
  const mapped = getChipCoverage({ ...COMPLETE, sfu_ops: undefined, sfu_rate_source: "vector" }, "bf16");
  assert.deepEqual(mapped.missing, []);
  assert.equal(mapped.capabilities.sfu_bound, true);
  assert.ok(!mapped.warnings.some((warning) => warning.code === "chip.missingSfuOps"));

  // 映射依赖 vector_flops；向量吞吐缺失时 sfu 仍不可判
  const unmapped = getChipCoverage({ ...COMPLETE, sfu_ops: undefined, sfu_rate_source: "vector", vector_flops: undefined }, "bf16");
  assert.ok(unmapped.missing.includes("sfu_ops"));
  assert.equal(unmapped.capabilities.sfu_bound, false);
});

test("手工条目量级偏离所有公开卡超过 10 倍时给出双语单位警告（旁路 C）", () => {
  const normal = createManualChip({ name: "Normal chip", memoryGb: 80, memoryBandwidthTb: 2, bf16Tflops: 300, intraNodeGb: 400 });
  assert.equal(getChipCoverage(normal, "bf16").warnings.some((warning) => warning.code === "chip.unitAnomalyBandwidth"), false);

  // 带宽 5 GB/s：比最慢公开卡（800 GB/s）还低 160 倍 → 提示单位可能错误
  const slow = createManualChip({ name: "Slow chip", memoryGb: 80, memoryBandwidthTb: 0.005, bf16Tflops: 300, intraNodeGb: 400 });
  const bandwidthWarning = getChipCoverage(slow, "bf16").warnings.find((warning) => warning.code === "chip.unitAnomalyBandwidth");
  assert.equal(bandwidthWarning.code, "chip.unitAnomalyBandwidth");
  assert.match(formatIssue("en", bandwidthWarning), /memory_bandwidth/);
  assert.match(formatIssue("en", bandwidthWarning), /Unit may be wrong/);

  // BF16 300 GFLOPS：比最慢公开卡（280 TFLOPS）低约千倍 → 提示单位可能错误
  const tinyFlops = createManualChip({ name: "Tiny chip", memoryGb: 80, memoryBandwidthTb: 2, bf16Tflops: 0.3, intraNodeGb: 400 });
  const flopsWarning = getChipCoverage(tinyFlops, "bf16").warnings.find((warning) => warning.code === "chip.unitAnomalyFlops");
  assert.equal(flopsWarning.params.dtype, "bf16");
  assert.match(formatIssue("en", flopsWarning), /Unit may be wrong/);

  // 对照只针对手工条目；公开/本地形态条目不做量级警告
  assert.equal(getChipCoverage({ ...COMPLETE, memory_bandwidth: 5e9 }, "bf16").warnings.some((warning) => warning.code === "chip.unitAnomalyBandwidth"), false);
});
