// gen-cost-counts.mjs —— 生成 docs/details/cost_counts.md「逐条清单」机器段。
//
// 动机（用户裁决）：逐条清单原为手写，必然与 FORMULAS 漂移（曾停在 49 行 vs 52 条）。
// 改为从注册表派生 + 单元探针分类，杜绝「每次补全」。人工规格（全局假设 A1–A7、
// 相位口径、F1–F9 符号 bytes 公式、复合节点分解声明）仍手写，不在本机器段内。
//
// 用法（任意 cwd）：
//   node scripts/gen-cost-counts.mjs            # 写回机器段
//   node scripts/gen-cost-counts.mjs --check    # 只比对，drift 即退出 1（pre-commit / docs:check 用）
//
// 不漂移保证：`--check`（check_principles.sh / docs:check）+ 冻结用例
// frontend/src/structure/operators/formulas/__tests__/costCountsDoc.test.js（npm test，CI 门禁）。
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { FORMULAS, FORMULA_GROUPS, UNGROUPED_FORMULAS } from "../frontend/src/structure/operators/formulas/index.js";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const DOC = path.join(repoRoot, "docs/details/cost_counts.md");
const BEGIN = "<!-- BEGIN GENERATED: cost-counts-roster -->";
const END = "<!-- END GENERATED: cost-counts-roster -->";

// 与 counts.test.js:300 同一单元探针（形状全 1），保证分类可复现、与「注册表完整性」用例同源。
const UNIT_PROBE = {
  elements: 1, tokens: 1, hidden: 1, bytesPerElement: 1, width: 1, intermediate: 1,
  experts: 1, topk: 1, expertHidden: 1, expertIntermediate: 1, keyDim: 1, valueDim: 1,
  keyTokens: 1, headDim: 1, heads: 1, queryTokens: 1, ropeDims: 1, channels: 1, kernel: 1,
  tableRows: 1, logicalShape: [1, 1], inElements: 1, outElements: 1, gateProjection: false,
  gateProjectionInput: 0, weightOne: false, gated: false, delta: false, normTopkProb: false,
  copy: false, part: "scores", selected: 1, batch: 1, sequence: 1, keyHeads: 1, valueHeads: 1,
  convKernelSize: 1,
};
// 复合节点 ctx 是嵌套结构，单元探针不适用（与 counts.test.js 同一豁免集）；标「分解」，
// 三分量以「复合」示意，符号分解见「复合节点」表。
const COMPOSITES = new Set([
  "mhc_pre", "mhc_fused_post_pre", "mhc_post", "mhc_contract", "mla_query_compress",
  "mla_kv_compress", "attention_residual", "hyper_connection", "ple", "qsa_indexer",
  "dsa_indexer", "dsa_kpool_indexer", "dsv4_indexer", "minimax_sparse_indexer",
]);

const groupOf = (id) => (UNGROUPED_FORMULAS.has(id) ? "—" : FORMULAS[id].group || "—");
const tick = (value) => (value > 0 ? "✓" : "0");

/** 由单元探针的动作向量判分类，与手写表同一词汇。复合节点不探针。 */
function classify(id) {
  if (COMPOSITES.has(id)) return { klass: "分解", matrix: "复合", vector: "复合", sfu: "复合", bytes: "复合" };
  const c = FORMULAS[id].counts(UNIT_PROBE);
  const b = c.bytes || {};
  const bytesNonzero = (b.weights || 0) > 0 || (b.actIn || 0) > 0 || (b.actOut || 0) > 0;
  const klass = (c.matrix || 0) > 0
    ? "计算+访存"
    : (c.vector || 0) > 0 || (c.sfu || 0) > 0
      ? "仅访存"
      : "仅搬运";
  return { klass, matrix: tick(c.matrix || 0), vector: tick(c.vector || 0), sfu: tick(c.sfu || 0), bytes: bytesNonzero ? "✓" : "0" };
}

export function buildCostCountsRoster() {
  const ids = Object.keys(FORMULAS).sort();
  const lines = [];
  lines.push(BEGIN);
  lines.push("");
  lines.push(`> 生成物（\`scripts/gen-cost-counts.mjs\`，勿手改）：逐条 = FORMULAS 注册表；\`分类\`/三分量`);
  lines.push(`> 由单元探针（形状全 1，同 counts.test.js）判定。符号 bytes 公式见上方 F1–F9；复合节点`);
  lines.push(`> 分解见「复合节点」表。共 **${ids.length}** 条。`);
  lines.push("");
  lines.push("| 条目 | group | 分类 | matrix | vector | sfu | bytes |");
  lines.push("|---|---|---|---|---|---|---|");
  for (const id of ids) {
    const c = classify(id);
    lines.push(`| \`${id}\` | ${groupOf(id)} | ${c.klass} | ${c.matrix} | ${c.vector} | ${c.sfu} | ${c.bytes} |`);
  }
  lines.push("");
  lines.push(END);
  return lines.join("\n");
}

function spliceSegment(existing, generated) {
  const hasMarkers = existing.includes(BEGIN) && existing.includes(END);
  if (!hasMarkers) {
    throw new Error(`cost_counts.md 缺少 ${BEGIN} / ${END} 标记，无法写入机器段`);
  }
  return existing.slice(0, existing.indexOf(BEGIN)) + generated + existing.slice(existing.indexOf(END) + END.length);
}

/** 抽取当前文件里的机器段（含 BEGIN/END），供用例冻结比对。 */
export function currentCostCountsSegment(docText) {
  const text = docText ?? fs.readFileSync(DOC, "utf8");
  const start = text.indexOf(BEGIN);
  const end = text.indexOf(END);
  if (start < 0 || end < 0) return null;
  return text.slice(start, end + END.length);
}

function main() {
  const existing = fs.existsSync(DOC) ? fs.readFileSync(DOC, "utf8") : "";
  const generated = buildCostCountsRoster();
  const next = spliceSegment(existing, generated);
  if (process.argv.includes("--check")) {
    if (existing !== next) {
      console.error("✗ cost_counts.md 逐条清单机器段过期：运行 `node scripts/gen-cost-counts.mjs` 重生成。");
      process.exit(1);
    }
    console.log(`cost_counts.md 逐条清单机器段：与注册表一致（${Object.keys(FORMULAS).length} 条）`);
    return;
  }
  fs.writeFileSync(DOC, next);
  console.log(`written 逐条清单机器段 -> docs/details/cost_counts.md（${Object.keys(FORMULAS).length} 条）`);
}

if (import.meta.url === pathToFileURL(process.argv[1] || "").href) {
  main();
}
