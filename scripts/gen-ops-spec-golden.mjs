// W2-0：W2 重构动手前生成 spec 树哈希基线（重构保真 oracle）。
// 用法（cwd = frontend）：node ../scripts/gen-ops-spec-golden.mjs
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { buildSpecTreeMap, buildEdgeMap, hashSpecTree } from "../frontend/src/structure/model_executor/__tests__/opsSpecTreeGoldenLib.js";

const frontendRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "frontend");
const outPath = path.join(frontendRoot, "src/structure/model_executor/__tests__/ops-spec-tree.golden.json");
const map = buildSpecTreeMap();
const hashes = Object.fromEntries(Object.entries(map).map(([k, v]) => [k, hashSpecTree(v)]));
fs.writeFileSync(outPath, JSON.stringify(hashes, null, 1) + "\n");
const edgeMap = buildEdgeMap();
const edgeHashes = Object.fromEntries(Object.entries(edgeMap).map(([k, v]) => [k, hashSpecTree(v)]));
fs.writeFileSync(outPath.replace("ops-spec-tree.golden.json", "ops-edge.golden.json"), JSON.stringify(edgeHashes, null, 1) + "\n");
console.log(`written ${Object.keys(hashes).length} model hashes + edge hashes -> ${path.relative(process.cwd(), outPath)}`);
