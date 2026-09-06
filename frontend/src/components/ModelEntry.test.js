import assert from "node:assert/strict";
import test from "node:test";
import { formatReleaseTime, sortModelsByReleaseTime } from "../structure/catalog/modelOrdering.js";

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
