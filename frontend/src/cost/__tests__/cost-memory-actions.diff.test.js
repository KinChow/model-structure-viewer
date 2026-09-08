// M11-P0-5 前置差分：内存侧数值与基线深度相等。
// 恒等式/结构基线不覆盖访存侧——counts.bytes 接入（P0-5）、actions 断链修复
// （P0-4）等改动会合法地改变本基线。失败时逐模型 dump 到 /tmp 供 diff：
// 无意变化 = 回归；有意变化须审阅 diff 确认只含该改动声称的字段，
// 再跑 node ../scripts/gen-cost-memory-golden.mjs 重新生成。
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";
import { buildMemoryActionsMap } from "./costMemoryGoldenLib.js";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../../../..");
const goldenPath = path.join(repoRoot, "frontend/src/cost/__tests__/cost-memory-actions.golden.json");

test("内存侧基线：actions/activation/roofline 与基线深度相等", () => {
  const golden = JSON.parse(fs.readFileSync(goldenPath, "utf8"));
  const current = buildMemoryActionsMap();
  assert.deepEqual(Object.keys(current).sort(), Object.keys(golden).sort(), "catalog 模型集合与基线不一致，先重新生成基线");
  const bad = [];
  for (const [model, snapshot] of Object.entries(current)) {
    if (JSON.stringify(snapshot) !== JSON.stringify(golden[model])) {
      bad.push(model);
      const dump = path.join(os.tmpdir(), "msv-memory-golden", `${model.replaceAll("/", "__")}.json`);
      fs.mkdirSync(path.dirname(dump), { recursive: true });
      fs.writeFileSync(dump, JSON.stringify(snapshot, null, 1) + "\n");
    }
  }
  if (bad.length > 0) {
    console.error(`内存侧数值变更（dump 到 ${path.join(os.tmpdir(), "msv-memory-golden")}，与 git show HEAD:<golden> diff）:\n  ${bad.join("\n  ")}`);
  }
  assert.deepEqual(bad, [], "内存侧数值与基线不一致：无意变化即回归；有意变化须审阅 diff 后重新生成基线");
});
