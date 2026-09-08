// fetch-evidence.mjs —— 模型证据库取证（M8-V2 基础设施）。
// 用法：node scripts/fetch-evidence.mjs <org>/<id> [家族别名]
// 下载 L1 config / L2 modeling 源码 / L3 index.json 到 evidence/<org>/<id>/，
// index 原件 gitignore（可重下），并生成 index 摘要（逐层张量模式）。
// 来源：HF 直连优先，失败回退 hf-mirror.com。
// 注意：modeling 文件名各家族不同，首次接入用 --probe 列出仓库文件后
// 在清单里登记（MAINTENANCE.md 纪律：新家族三源齐备）。
import fs from "node:fs";
import path from "node:path";
import { createHash } from "node:crypto";

const [orgId, alias] = process.argv.slice(2);
if (!orgId) {
  console.error("用法: node scripts/fetch-evidence.mjs <org>/<id> [--probe [file-list...]]");
  process.exit(1);
}
const HF = (file) => `https://huggingface.co/${orgId}/resolve/main/${file}`;
const MIRROR = (file) => `https://hf-mirror.com/${orgId}/resolve/main/${file}`;
const outDir = path.resolve("models", orgId); // HF hub 惯例：模型相关文件同仓（M8-V2 修正）
fs.mkdirSync(outDir, { recursive: true });

async function download(file, dest) {
  for (const url of [HF(file), MIRROR(file)]) {
    try {
      const res = await fetch(url, { redirect: "follow" });
      if (!res.ok) continue;
      const buf = Buffer.from(await res.arrayBuffer());
      if (buf.length < 10 || buf.slice(0, 20).toString().includes("git-lfs")) {
        // LFS 指针或空响应 → 换镜像重试
        continue;
      }
      fs.writeFileSync(dest, buf);
      return { file, bytes: buf.length, sha256: createHash("sha256").update(buf).digest("hex").slice(0, 16), url };
    } catch { /* 下一个源 */ }
  }
  return null;
}

const probe = process.argv.includes("--probe");
const probeFiles = process.argv.slice(process.argv.indexOf("--probe") + 1).filter((f) => f && !f.startsWith("-"));

if (probe) {
  for (const file of probeFiles) {
    const result = await download(file, path.join(outDir, path.basename(file)));
    console.log(result ? `✓ ${file} → ${path.basename(result.file)} (${(result.bytes / 1024).toFixed(0)}KB)` : `✗ ${file}`);
  }
  process.exit(0);
}

// 常规取证：config + 用户在命令行列出的 modeling/index 文件
const wanted = process.argv.slice(2).filter((f) => f.includes("."));
const results = [];
for (const file of ["config.json", ...wanted]) {
  const result = await download(file, path.join(outDir, path.basename(file)));
  results.push(result ? { file, ...result } : { file, failed: true });
  console.log(result ? `✓ ${file} (${(result.bytes / 1024).toFixed(0)}KB)` : `✗ ${file}`);
}
// index.json 摘要（逐层张量模式）
const indexFile = results.find((r) => r.file.includes("index.json"));
if (indexFile && !indexFile.failed) {
  const idx = JSON.parse(fs.readFileSync(path.join(outDir, "index.json"), "utf8"));
  const summary = {};
  for (const name of Object.keys(idx.weight_map)) {
    const m = name.match(/layers\.(\d+)\.(.+)/);
    if (!m) continue;
    (summary[m[1]] ||= new Set()).add(m[2].replace(/\.\d+\./g, ".#."));
  }
  fs.writeFileSync(
    path.join(outDir, "index-summary.json"),
    JSON.stringify(Object.fromEntries(Object.entries(summary).map(([l, t]) => [l, [...t].sort()])).map(([l, t]) => [Number(l), t]).sort((a, b) => a[0] - b[0]).map(([l, t]) => [String(l), t]).map(([l, t]) => [l, t]).reduce((o, [l, t]) => ((o[l] = t), o), {})),
  );
  console.log("index-summary.json written");
}
