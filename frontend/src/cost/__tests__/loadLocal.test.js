import assert from "node:assert/strict";
import test from "node:test";
import { loadLocalChipOverrides, mergeChipCatalog } from "../chips/loadLocal.js";

test("本地覆盖按字段深合并，不丢失公开规格", () => {
  const merged = mergeChipCatalog([
    { id: "a", vendor: "v", name: "A", peak_flops: { bf16: 1, fp16: 2 }, interconnect: { intra_node: { bandwidth: 3 } }, source: "s", confidence: "official" },
  ], [{ id: "a", peak_flops: { bf16: 9 }, interconnect: { intra_node: { kind: "NVLink" } }, confidence: "local" }]);
  assert.deepEqual(merged[0].peak_flops, { bf16: 9, fp16: 2 });
  assert.deepEqual(merged[0].interconnect.intra_node, { bandwidth: 3, kind: "NVLink" });
  assert.equal(merged[0].confidence, "local");
});

test("本地新增芯片自动标记为 local", () => {
  const merged = mergeChipCatalog([], [{ id: "local", vendor: "v", name: "L" }]);
  assert.equal(merged[0].confidence, "local");
});

test("本地配置 404 按可选空配置处理", async () => {
  const result = await loadLocalChipOverrides({ fetchImpl: async () => ({ ok: false, status: 404 }) });
  assert.deepEqual(result, []);
});

test("本地配置支持 {chips: []} 并校验基本字段", async () => {
  const result = await loadLocalChipOverrides({ fetchImpl: async () => ({ ok: true, status: 200, json: async () => ({ chips: [{ id: "x", vendor: "v", name: "n", source: "local" }] }) }) });
  assert.equal(result[0].id, "x");
});

test("同 id 本地覆盖后保留 local 标记", () => {
  const merged = mergeChipCatalog([{ id: "a", vendor: "v", name: "A", source: "s", confidence: "official" }], [{ id: "a", confidence: "local" }]);
  assert.equal(merged[0].confidence, "local");
});
