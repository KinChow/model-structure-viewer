#!/usr/bin/env node
// gen-sglang-arch-registry.mjs —— 扫本地 SGLang 源码的 EntryClass 声明，落一份 committed 快照
// （arch 类名集合 + 来源文件），供 sglangArchAlignment 守卫测试用（避免 CI 依赖 SGLang 在场）。
//
// EntryClass 的类名 == SGLang 注册的 architectures[0]（模型 loader 按 config.architectures[0] 匹配类名）。
// 用法：node scripts/gen-sglang-arch-registry.mjs [--sglang=<models-dir>]
//   源目录优先级：--sglang=<dir> > 环境变量 SGLANG_MODELS_DIR > 约定默认。
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const sglangArg = process.argv.find((a) => a.startsWith("--sglang="))?.slice("--sglang=".length);
const sglangModels = sglangArg
  || process.env.SGLANG_MODELS_DIR
  || "/sgl-workspace/sglang/python/sglang/srt/models";
const outPath = path.join(repoRoot, "frontend/src/structure/__tests__/sglang_arch_registry.json");

if (!fs.existsSync(sglangModels)) {
  console.error(`SGLang models dir not found: ${sglangModels}`);
  console.error("  传 --sglang=<dir> 或设 SGLANG_MODELS_DIR 指向本地 sglang/python/sglang/srt/models；fixture 未更新（保留已提交快照）。");
  process.exit(2);
}

const arch = {}; // className -> source file (first seen)
for (const file of fs.readdirSync(sglangModels)) {
  if (!file.endsWith(".py")) continue;
  const text = fs.readFileSync(path.join(sglangModels, file), "utf8");
  // 匹配 `EntryClass = X` 或 `EntryClass = [X, Y, ...]`（跨行）
  const m = text.match(/EntryClass\s*=\s*(\[[\s\S]*?\]|[A-Za-z_][A-Za-z0-9_]*)/);
  if (!m) continue;
  const names = m[1].replace(/[[\]\s]/g, "").split(",").filter(Boolean);
  for (const n of names) if (!arch[n]) arch[n] = file;
}

const payload = {
  generated: "sglang EntryClass scan (gen-sglang-arch-registry.mjs)",
  // 记录扫描根的「sglang/…」相对片段，避免把机器本地绝对路径写进已提交快照。
  source: `sglang/${sglangModels.replace(/^.*?sglang[/\\]/, "")}`.replace(/\\/g, "/"),
  count: Object.keys(arch).length,
  architectures: Object.fromEntries(Object.entries(arch).sort(([a], [b]) => a.localeCompare(b))),
};
fs.writeFileSync(outPath, `${JSON.stringify(payload, null, 2)}\n`);
console.log(`wrote ${payload.count} SGLang EntryClass archs -> ${path.relative(repoRoot, outPath)}`);
