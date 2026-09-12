// layering.test.js —— 结构目录级分层护栏。
// 目录：config/archs 最底；operators（formulas + ops）不碰 models/layers；
// models 组网、layers 共享模块。测试允许上探，不构成分层违规。
// 依赖仅 node 内建模块 —— 本测试自身不得引入被测层。

import assert from "node:assert/strict";
import test from "node:test";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const structureDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

/** 递归收集 .js 文件；跳过 __tests__（测试允许上探运行时各层，不构成分层违规）。 */
function listJsFiles(dir) {
  if (!fs.existsSync(dir)) return [];
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

test("结构分层：operators/formulas 不 import models|layers；config 不 import operators|models|layers", () => {
  const formulasDir = path.join(structureDir, "operators", "formulas");
  const lowerLayer = offenders(formulasDir, ["/models/", "/layers/"]);
  assert.deepEqual(
    lowerLayer,
    [],
    "operators/formulas 出现指向 models/ 或 layers/ 的 import —— 分层倒置",
  );
  const bottom = offenders(path.join(structureDir, "config"), ["/operators/", "/models/", "/layers/"]);
  assert.deepEqual(
    bottom,
    [],
    "config（与 archs 同为结构栈最底层）不得 import operators/、models/ 或 layers/",
  );
});
