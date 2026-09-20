// structure/v4_proxy_frontend.mjs —— dump DeepSeek-V4-Flash 逐注意力算子分布（operator_id × 折叠 multiplier 求和）。
// 与后端 v4_proxy_backend.py 的 transformers layer_type 计数对账。回填 evidence/structure/deepseek_v4_proxy.md。
// 用法：node scripts/evidence/structure/v4_proxy_frontend.mjs（config 取仓库内 DeepSeek-V4-Flash，输出到 MSV_EVIDENCE_OUT）。
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { buildStructureFromConfig } from "../../../frontend/src/structure/buildStructure.js";
import { walkStructure } from "../../../frontend/src/cost/traverse.js";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../../..");
const CFG = process.env.MSV_V4_CONFIG || path.join(repoRoot, "models/deepseek-ai/DeepSeek-V4-Flash/config.json");
const raw = JSON.parse(fs.readFileSync(CFG, "utf8"));
const structure = buildStructureFromConfig(raw, { modelId: "deepseek-ai/DeepSeek-V4-Flash", source: "v4-proxy" });

const counts = {};
walkStructure(structure.graph, ({ node, multiplier }) => {
  const opId = node?.attributes?.operator_id;
  if (!opId) return;
  if (/dsv4|attention|sparse_mla|swa|compressed/i.test(opId)) {
    counts[opId] = (counts[opId] || 0) + (multiplier || 1);
  }
});
console.log("前端 V4-Flash 注意力算子分布 (operator_id → Σmultiplier):");
console.log(JSON.stringify(counts, null, 2));
const outDir = process.env.MSV_EVIDENCE_OUT || path.join(repoRoot, "_evidence_out/structure");
fs.mkdirSync(outDir, { recursive: true });
fs.writeFileSync(path.join(outDir, "v4_proxy_frontend.json"), JSON.stringify(counts, null, 2));
