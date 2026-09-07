// W2 差分：全部内置模型的 spec 树必须与重构前基线逐字节等价（哈希比对）。
// 失败时把该模型的完整规范化 JSON dump 到 /tmp，供逐字节 diff 定位。
// 重构导致的有意变更 → 先跑 node ../scripts/gen-ops-spec-golden.mjs 重新生成并人工审阅 diff。
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";
import { buildSpecTreeMap, hashSpecTree } from "./opsSpecTreeGoldenLib.js";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../../../../..");
const goldenPath = path.join(repoRoot, "frontend/src/structure/model_executor/__tests__/ops-spec-tree.golden.json");

test("W2 差分：spec 树与重构前基线深度相等", () => {
  const golden = JSON.parse(fs.readFileSync(goldenPath, "utf8"));
  const current = buildSpecTreeMap();
  assert.deepEqual(Object.keys(current).sort(), Object.keys(golden).sort(), "catalog 模型集合与基线不一致，先重新生成基线");
  const bad = [];
  for (const [model, json] of Object.entries(current)) {
    if (hashSpecTree(json) !== golden[model]) {
      bad.push(model);
      const dump = path.join(os.tmpdir(), "msv-spec-golden", `${model.replaceAll("/", "__")}.json`);
      fs.mkdirSync(path.dirname(dump), { recursive: true });
      fs.writeFileSync(dump, json);
    }
  }
  if (bad.length > 0) {
    console.error(`spec 树变更（完整 JSON 已 dump 到 ${path.join(os.tmpdir(), "msv-spec-golden")}，与 git show HEAD:<golden> 的重算结果 diff）:\n  ${bad.join("\n  ")}`);
  }
  assert.deepEqual(bad, [], "spec 树与基线不一致：无意变更即重构引入回归；有意变更须先审阅 diff 再重新生成基线");
});
