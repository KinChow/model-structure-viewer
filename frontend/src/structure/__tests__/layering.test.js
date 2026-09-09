// layering.test.js —— 结构目录级分层护栏（M11.5 子项 2 的棘轮）。
// 背景：formulas/extractor.js 曾 import ../model_executor/{dims,layers/vision}.js ——
// 文件级 madge 不报环，但目录分层被破坏（formulas 是 model_executor 的下层）。
// 修复后编码"今天成立"的两条规则；未来再出现反向依赖，在这里失败而不是悄悄扩散：
//   1. formulas/**（排除 __tests__）：import 行不得指向 model_executor；
//   2. config/**：不得 import formulas/ 或 model_executor/（config 与 archs 同为结构栈最底层）。
// 依赖仅 node 内建模块 —— 本测试自身不得引入被测层。

import assert from "node:assert/strict";
import test from "node:test";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

// 本文件在 frontend/src/structure/__tests__/ → structure/ 上溯一级。
const structureDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

/** 递归收集 .js 文件；跳过 __tests__（测试允许上探运行时各层，不构成分层违规）。 */
function listJsFiles(dir) {
  const files = [];
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    if (entry.name === "__tests__") continue;
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) files.push(...listJsFiles(full));
    else if (entry.name.endsWith(".js")) files.push(full);
  }
  return files;
}

/** 依赖边行 = static / side-effect / dynamic import，或 `export ... from` 再导出。
 *  先剥纯注释行（与 scripts/check_principles.sh 同口径：注释里的提及不构成依赖）。 */
function dependencyLines(file) {
  return fs
    .readFileSync(file, "utf8")
    .split("\n")
    .filter((line) => !/^\s*(\/\/|\*|\/\*)/.test(line))
    .filter((line) => /\bimport\b/.test(line) || (/^\s*export\b/.test(line) && /\bfrom\s*["']/.test(line)));
}

function offenders(dir, tokens) {
  return listJsFiles(dir).flatMap((file) =>
    dependencyLines(file)
      .filter((line) => tokens.some((token) => line.includes(token)))
      .map((line) => `${path.relative(structureDir, file)}: ${line.trim()}`),
  );
}

test("结构分层：formulas/ 不 import model_executor/；config/ 不 import formulas|model_executor（M11.5 棘轮）", () => {
  const lowerLayer = offenders(path.join(structureDir, "formulas"), ["model_executor"]);
  assert.deepEqual(
    lowerLayer,
    [],
    "formulas（下层）出现指向 model_executor（上层）的 import —— 分层倒置（docs/refactor_plan.md M11.5 子项 2）",
  );
  const bottom = offenders(path.join(structureDir, "config"), ["formulas", "model_executor"]);
  assert.deepEqual(
    bottom,
    [],
    "config（与 archs 同为结构栈最底层）不得 import formulas/ 或 model_executor/",
  );
});
