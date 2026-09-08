// M11-P0-5 前置：生成内存侧基线（actions 汇总 + activation + roofline 时间）。
// 访存侧变更（P0-4/P0-5）落地前先跑一次留底；有意变更后重跑并人工审阅 diff。
// 用法（cwd = frontend）：node ../scripts/gen-cost-memory-golden.mjs
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { buildMemoryActionsMap } from "../frontend/src/cost/__tests__/costMemoryGoldenLib.js";

const frontendRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "frontend");
const outPath = path.join(frontendRoot, "src/cost/__tests__/cost-memory-actions.golden.json");
const map = buildMemoryActionsMap();
fs.writeFileSync(outPath, JSON.stringify(map, null, 1) + "\n");
console.log(`written ${Object.keys(map).length} model memory/actions snapshots -> ${path.relative(process.cwd(), outPath)}`);
