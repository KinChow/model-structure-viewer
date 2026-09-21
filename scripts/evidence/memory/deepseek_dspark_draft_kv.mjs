// 只读探针（前端侧基线锁定）：dump dsv4 家族 DSpark/MTP 草稿子树的每 token 常驻 draft KV 字节。
//
// 目的：锁定 MSV 前端建模的 draft KV per-token 权威值，供 H20 真机 DSpark draft 池对账
//   （validation_status.md R2/R3 draft 侧）。逐叶打印 dtype/elements，明确 fp4 设计口径
//   与「按 fp8 KV 重算」的期望值（真机代理 build KV=fp8_e4m3）。
//
// 用法（本地、零下载、纯前端图，不触发任何 GPU/前向）：
//   node scripts/evidence/memory/deepseek_dspark_draft_kv.mjs
//
// 回填 doc：docs/details/evidence/memory/deepseek_v41_dspark_runtime_h20.md（draft 池对账段）。
import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { buildStructureFromConfig } from "../../../frontend/src/structure/buildStructure.js";
import { normalizeConfig } from "../../../frontend/src/structure/config/normalize.js";
import { draftKvBytesPerToken, bytesPerDtype } from "../../../frontend/src/cost/memory.js";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "..", "..");
const catalog = JSON.parse(await fs.readFile(path.join(repoRoot, "models", "catalog.json"), "utf8"));
const targets = catalog.models
  .map((m) => m.config_path)
  .filter((cp) => /DeepSeek-V4(\.1)?-(Flash|Pro)\/config\.json$/.test(cp) && !/Vision|0731|0813/.test(cp));

for (const cp of targets) {
  const config = JSON.parse(await fs.readFile(path.join(repoRoot, "models", cp), "utf8"));
  const normalized = normalizeConfig(config);
  const structure = buildStructureFromConfig(config, { modelId: cp, source: "draft-kv baseline" });
  const g = structure.graph;
  const draft = draftKvBytesPerToken(g, normalized, 2);
  const draft6 = draftKvBytesPerToken(g, normalized, 2, { draftTokens: 6 });
  const rawById = new Map(g.nodes.map((n) => [n.id, n]));
  const leaves = [];
  let fp8Recompute = 0;
  for (const n of g.nodes) {
    let cur = n, isDraft = false;
    while (cur) { if (cur.type === "mtp" || cur.type === "dspark") { isDraft = true; break; } cur = cur.parent_id != null ? rawById.get(cur.parent_id) : null; }
    if (!isDraft) continue;
    const a = n.attributes || {};
    if (a.cache_kv_dtype == null && a.cache_kv_elements == null && a.cache_index_elements == null) continue;
    const kvB = bytesPerDtype(a.cache_kv_dtype, 2), idxB = bytesPerDtype(a.cache_index_dtype || a.cache_kv_dtype, 2);
    const kvE = a.cache_kv_growth_elements > 0 ? a.cache_kv_growth_elements : (a.cache_kv_elements || 0);
    const idxE = a.cache_index_growth_elements > 0 ? a.cache_index_growth_elements : (a.cache_index_elements || 0);
    fp8Recompute += kvE * 1 + idxE * 1; // fp8_e4m3 = 1 B/elem（真机代理 build KV dtype）
    leaves.push(`    leaf ${n.id}: kv=${a.cache_kv_dtype}(${kvB}B)×${kvE} idx=${a.cache_index_dtype}(${idxB}B)×${idxE} => ${kvE * kvB + idxE * idxB} B`);
  }
  console.log(`${cp.split("/")[1]}: draftKv/token=${draft} B（设计口径）; +6 verify 窗口=${draft6} B; 按 fp8_e4m3 重算=${fp8Recompute} B`);
  for (const l of leaves) console.log(l);
}
