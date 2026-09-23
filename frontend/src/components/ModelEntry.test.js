import assert from "node:assert/strict";
import test from "node:test";
import { formatReleaseTime, sortModelsByName, sortModelsByReleaseTime } from "../structure/catalog/modelOrdering.js";

test("sorts published models newest first and keeps undated entries stable", () => {
  const models = [
    { modelId: "org/undated" },
    { modelId: "org/old", releaseTime: "2025-01-01T00:00:00Z" },
    { modelId: "org/new", releaseTime: "2026-01-01T00:00:00Z" },
    { modelId: "org/also-undated" },
  ];

  assert.deepEqual(sortModelsByReleaseTime(models).map((entry) => entry.modelId), [
    "org/new",
    "org/old",
    "org/undated",
    "org/also-undated",
  ]);
});

test("formats valid release times and preserves invalid values", () => {
  assert.equal(formatReleaseTime("2026-01-02T08:00:00Z", "en"), "01/02/2026");
  assert.equal(formatReleaseTime("not-a-date", "zh"), "not-a-date");
});

test("sorts models by display name when selected", () => {
  assert.deepEqual(sortModelsByName([
    { modelId: "org/z", displayName: "Zeta" },
    { modelId: "org/a", displayName: "Alpha" },
  ]).map((entry) => entry.modelId), ["org/a", "org/z"]);
});

// 入口统计块派生值：数量 = 条目数；最近更新 = 最新 releaseTime 经 formatReleaseTime。
// 组件渲染逻辑（builtinModels.length / sortModelsByReleaseTime[0].releaseTime）在此以纯函数取证，
// node --test 无 DOM，故断言组件将呈现的值而非 DOM 结构。
test("entry stats derive count and latest release date", () => {
  const builtinModels = [
    { modelId: "org/a", releaseTime: "2026-01-01T00:00:00Z" },
    { modelId: "org/b", releaseTime: "2026-09-10T00:00:00Z" },
    { modelId: "org/undated" },
  ];
  const latestReleaseTime = sortModelsByReleaseTime(builtinModels)[0]?.releaseTime || null;
  assert.equal(builtinModels.length, 3);
  assert.equal(latestReleaseTime, "2026-09-10T00:00:00Z");
  assert.equal(formatReleaseTime(latestReleaseTime, "en"), "09/10/2026");
  assert.equal(formatReleaseTime(latestReleaseTime, "zh"), "2026/09/10");
});

test("entry stats fall back to zero and em dash when catalog is empty", () => {
  const builtinModels = [];
  const latestReleaseTime = sortModelsByReleaseTime(builtinModels)[0]?.releaseTime || null;
  assert.equal(builtinModels.length, 0);
  assert.equal(latestReleaseTime, null);
  assert.equal(latestReleaseTime ? formatReleaseTime(latestReleaseTime, "en") : "—", "—");
});
