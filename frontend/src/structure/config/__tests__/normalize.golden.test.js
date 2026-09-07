// W3-C oracle：① normalizeConfig 输出哈希与基线比对（重生成后 diff 必须只体现
// 方案字段删除）；② deriveBuildPlan 与方案字段 parity fixture 逐值相等（逐字搬迁
// 的保真证明，覆盖 59 模型 × 8 字段）。
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";
import { buildNormalizeMap, buildPlanFromDerive, hashJson, PLAN_FIELDS } from "./normalizeGoldenLib.js";
import { deriveBuildPlan } from "../../model_executor/plan.js";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../../../../..");
const goldenPath = path.join(repoRoot, "frontend/src/structure/config/__tests__/normalize.golden.json");
const fixturePath = path.join(repoRoot, "frontend/src/structure/config/__tests__/normalize.plan-fixture.json");

test("W3-C：normalizeConfig 输出哈希与基线一致", () => {
  const golden = JSON.parse(fs.readFileSync(goldenPath, "utf8"));
  const current = buildNormalizeMap();
  assert.deepEqual(Object.keys(current).sort(), Object.keys(golden).sort());
  const bad = [];
  for (const [model, json] of Object.entries(current)) {
    if (hashJson(json) !== golden[model]) {
      bad.push(model);
      const dump = path.join(os.tmpdir(), "msv-normalize-golden", `${model.replaceAll("/", "__")}.json`);
      fs.mkdirSync(path.dirname(dump), { recursive: true });
      fs.writeFileSync(dump, json);
    }
  }
  if (bad.length > 0) {
    console.error(`normalize 输出变更（已 dump 到 ${path.join(os.tmpdir(), "msv-normalize-golden")}）:\n  ${bad.join("\n  ")}`);
  }
  assert.deepEqual(bad, [], "normalize 输出变更：字段删除须重生成基线并审阅 diff（node scripts/gen-normalize-golden.mjs）");
});

test("W3-C：deriveBuildPlan 与搬迁前方案字段逐值相等（parity fixture）", () => {
  const fixture = JSON.parse(fs.readFileSync(fixturePath, "utf8"));
  const derived = buildPlanFromDerive();
  const fields = [...PLAN_FIELDS, "indexerSchedule"];
  assert.deepEqual(Object.keys(derived).sort(), Object.keys(fixture).sort());
  for (const model of Object.keys(fixture)) {
    assert.deepEqual(
      Object.fromEntries(fields.map((f) => [f, derived[model][f] ?? null])),
      Object.fromEntries(fields.map((f) => [f, fixture[model][f] ?? null])),
      `${model} 的 plan 字段与搬迁前不一致`,
    );
  }
});
