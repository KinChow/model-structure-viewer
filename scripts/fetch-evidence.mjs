// fetch-evidence.mjs —— 模型证据库取证（M8-V2 基础设施）。
// 用法：node scripts/fetch-evidence.mjs <org>/<id> [家族别名] [--headers]
//   --headers：按 model.safetensors.index.json 的 weight_map 逐分片 Range 取
//   safetensors 头部（8B 长度 + JSON），构建**折叠 skeleton** 写
//   models/<org>/<id>/skeleton-truth.json（N2-2 离线 checkpoint 真值）。
//   原始逐张量表不入库（K3 量级 59.7MB，轻量元数据纪律），需要时重下。
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

// ---- --headers：离线 checkpoint 真值（N2-2）----
if (process.argv.includes("--headers")) {
  const indexPath = path.join(outDir, "model.safetensors.index.json");
  if (!fs.existsSync(indexPath)) {
    console.error(`✗ 需要 ${indexPath}（先常规取证一次）`);
    process.exit(1);
  }
  const weightMap = JSON.parse(fs.readFileSync(indexPath, "utf8")).weight_map || {};
  const shards = [...new Set(Object.values(weightMap))];
  console.log(`分片 ${shards.length} 个，逐个取头部（Range 206）…`);
  const tensors = [];
  for (const [tensorName, shard] of Object.entries(weightMap)) {
    const url = HF(shard);
    try {
      // 与 frontend/src/cost/safetensorsReader.js 同一格式协议（8B u64le + JSON）
      const lenRes = await fetch(url, { headers: { Range: "bytes=0-7" } });
      if (!lenRes.ok || lenRes.status !== 206) throw new Error(`HTTP ${lenRes.status}`);
      const lenBuf = new Uint8Array(await lenRes.arrayBuffer());
      let headerLen = 0n;
      for (let i = 7; i >= 0; i--) headerLen = (headerLen << 8n) | BigInt(lenBuf[i]);
      const jsonLen = Number(headerLen);
      if (!Number.isSafeInteger(jsonLen) || jsonLen <= 0 || jsonLen > 100 * 1024 * 1024) throw new Error(`头长非法 ${jsonLen}`);
      const jsonRes = await fetch(url, { headers: { Range: `bytes=8-${8 + jsonLen - 1}` } });
      if (!jsonRes.ok || jsonRes.status !== 206) throw new Error(`HTTP ${jsonRes.status}`);
      const header = JSON.parse(new TextDecoder().decode(new Uint8Array(await jsonRes.arrayBuffer())));
      const info = header[tensorName];
      if (info) tensors.push({ name: tensorName, dtype: info.dtype, shape: info.shape });
    } catch (error) {
      console.log(`  ✗ ${tensorName} ← ${shard}: ${error.message}`);
    }
  }
  // 折叠：直接复用前端 skeleton 构建器（folded tree，体积小几个数量级）
  const { buildSkeleton } = await import("../frontend/src/structure/truth/skeleton.js");
  const skeleton = buildSkeleton(tensors);
  const parameterTotal = tensors.reduce((sum, t) => sum + t.shape.reduce((a, b) => a * b, 1), 0);
  fs.writeFileSync(path.join(outDir, "skeleton-truth.json"), JSON.stringify({
    generated: "safetensors headers (fetch-evidence --headers)",
    source: `https://huggingface.co/${orgId}/`,
    tensor_count: tensors.length,
    parameterTotal,
    skeleton,
  }, null, 1));
  console.log(`✓ skeleton-truth.json：${tensors.length} 张量 → 折叠节点（参数 ${parameterTotal.toLocaleString("en-US")}）`);
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
