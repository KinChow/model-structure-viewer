// W3-C：生成 normalizeConfig 输出哈希基线 + 方案字段 parity fixture。
// 用法（任意 cwd）：node scripts/gen-normalize-golden.mjs
// C 完成后重生成哈希基线，diff 必须只体现方案类字段删除。
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { buildNormalizeMap, hashJson } from "../frontend/src/structure/config/__tests__/normalizeGoldenLib.js";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const outDir = path.join(repoRoot, "frontend/src/structure/config/__tests__");
const hashes = Object.fromEntries(Object.entries(buildNormalizeMap()).map(([k, v]) => [k, hashJson(v)]));
fs.writeFileSync(path.join(outDir, "normalize.golden.json"), JSON.stringify(hashes, null, 1) + "\n");
console.log(`written ${Object.keys(hashes).length} model hashes（plan fixture 为冻结件，不随本脚本重写）`);
