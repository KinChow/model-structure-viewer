#!/usr/bin/env node
// gen-model-reference.mjs —— 生成 docs/models_reference.md 与
// docs/architectures_reference.md 的机器段。
//
// 背景（route-closeout Task 9 / P9 版本和文档治理）：模型清单此前只有
// docs/details/models.md 的人工段（按 canonical 分组的手写列表 + 结构类台账），
// 架构识别登记（别名表 / canonical 目录 / 配方表）也没有台账视图——与 catalog、
// aliases.js、ARCH_RECIPES 的漂移只能靠人眼。本脚本照
// scripts/gen-operators-reference.mjs 的 --check 手法（重新生成内容与文档机器段
// 逐字节比对），把两张台账收进「跑一把就能复现」的生成器。
//
// 事实源（本脚本不复制任何清单，只读取并渲染）：
//   - models/catalog.json（59 模型：model_id / model_type / architectures / release_time）
//   - frontend/src/structure/config/normalize.js + registry/resolveArchitecture.js
//     （与 frontend/src/structure/builtinModels.test.js 同一条运行时链路）
//   - frontend/src/cost/memory.js 的 graphWeightCapacity（walk 图声明，与 UI 同口径）
//   - frontend/src/structure/models/index.js（MODELS，key=architectures[0]）
//   - frontend/src/structure/archs/index.js（ARCH_RECIPES）
//
// 用法：
//   node scripts/gen-model-reference.mjs           # 写入两份文档
//   node scripts/gen-model-reference.mjs --check   # 只比对，任一机器段不一致即退出 1
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { normalizeConfig } from "../frontend/src/structure/config/normalize.js";
import { resolveArchitecture } from "../frontend/src/structure/registry/resolveArchitecture.js";
import { MODELS, buildNetwork } from "../frontend/src/structure/models/index.js";
import { ARCH_RECIPES } from "../frontend/src/structure/archs/index.js";
import { createStructureIr } from "../frontend/src/structure/ir/createStructureIr.js";
import { materializeModelStructure } from "../frontend/src/structure/materializers/modelStructure.js";
import { graphWeightCapacity } from "../frontend/src/cost/memory.js";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

const MODELS_DOC = path.join(repoRoot, "docs", "models_reference.md");
const MODELS_BEGIN = "<!-- BEGIN GENERATED: models -->";
const MODELS_END = "<!-- END GENERATED: models -->";
const ARCH_DOC = path.join(repoRoot, "docs", "architectures_reference.md");
const ARCH_BEGIN = "<!-- BEGIN GENERATED: architectures -->";
const ARCH_END = "<!-- END GENERATED: architectures -->";

// 千分位：不用 toLocaleString（locale 随运行环境变化，破坏逐字节比对），手写分组。
const group3 = (n) => String(n).replace(/\B(?=(\d{3})+(?!\d))/g, ",");
// 空值统一渲染为 —（缺 architectures / model_type 的 catalog 条目不应产出空单元格）。
const dash = (v) => (v === undefined || v === null || v === "" ? "—" : v);
/**
 * 参数量级单元格 = 精确值（缩写）。closed-form 会带浮点尾差（如 MLA
 * kv_lora_rank/2 一类除法，DeepSeek-V3 实测 682099236125.3771），台账取整到
 * 参数个位——亚参数级尾差无物理意义，也不该让台账跟着抖动；缩写沿用
 * frontend/src/formatters.js `formatCount` 的 B/M 档位惯例并补 T 档（≥1e12）。
 */
function paramCell(n) {
  const magnitude = n >= 1e12 ? `${(n / 1e12).toFixed(2)}T`
    : n >= 1e9 ? `${(n / 1e9).toFixed(1)}B`
    : n >= 1e6 ? `${(n / 1e6).toFixed(1)}M`
    : "";
  const exact = group3(Math.round(n));
  return magnitude ? `${exact}（${magnitude}）` : exact;
}

/**
 * 全目录扫描：catalog 逐模型跑 normalizeConfig → resolveArchitecture →
 * buildNetwork → graphWeightCapacity。口径与 UI SummaryChips 同一函数。
 */
function collectModels() {
  const catalog = JSON.parse(fs.readFileSync(path.join(repoRoot, "models", "catalog.json"), "utf8"));
  const rows = catalog.models.map((entry) => {
    const raw = JSON.parse(fs.readFileSync(path.join(repoRoot, "models", entry.config_path), "utf8"));
    const normalized = normalizeConfig(raw);
    const resolved = resolveArchitecture(normalized, { modelId: entry.model_id });
    const structure = materializeModelStructure(createStructureIr({
      network: buildNetwork(resolved, normalized),
      normalized,
      resolved,
    }));
    const params = graphWeightCapacity(structure.graph).elements;
    const dir = path.join(repoRoot, "models", entry.model_id);
    const exists = fs.existsSync(dir);
    const manifest = exists && fs.existsSync(path.join(dir, "evidence-manifest.json"));
    return {
      modelId: entry.model_id,
      modelType: dash(entry.model_type),
      arch0: dash(Array.isArray(entry.architectures) ? entry.architectures[0] : ""),
      architecture: resolved.architecture || "unsupported",
      params,
      evidence: exists ? (manifest ? "manifest" : "有") : "无",
      // release_time 只取日期段（catalog 里是 ISO 8601 UTC 串，完整时刻以 catalog 为准）。
      releaseDate: typeof entry.release_time === "string" ? entry.release_time.slice(0, 10) : "—",
    };
  });
  // 排序用码元比较（<），不用 localeCompare —— locale 随环境变，破坏逐字节比对。
  rows.sort((a, b) => (a.modelId < b.modelId ? -1 : a.modelId > b.modelId ? 1 : 0));
  const countBy = (key) => {
    const counts = new Map();
    for (const row of rows) counts.set(row[key], (counts.get(row[key]) || 0) + 1);
    return counts;
  };
  return {
    rows,
    total: rows.length,
    // architectures[0] 计数：别名表 / 配方表的「catalog 命中」列；canonical 计数：
    // canonical 目录的「catalog 模型数」列。命中 0 的条目 = 死登记，台账上一眼可见。
    arch0Counts: countBy("arch0"),
    architectureCounts: countBy("architecture"),
    manifestCount: rows.filter((row) => row.evidence === "manifest").length,
  };
}

/** docs/models_reference.md 的机器段（59 模型台账表 + 按 architectures[0] 汇总）。 */
function renderModels({ rows, total, architectureCounts, manifestCount }) {
  const out = [];
  out.push(MODELS_BEGIN);
  out.push("");
  out.push("> **本节由 `node scripts/gen-model-reference.mjs` 生成，请勿手改。**");
  out.push("> 数据源 = `models/catalog.json` + 前端运行时链路（`normalizeConfig` → `resolveArchitecture`，");
  out.push("> 与 `frontend/src/structure/builtinModels.test.js` 同口径）；`参数量级` =");
  out.push("> `graphWeightCapacity(graph).elements`（walk 图声明，与 UI SummaryChips 同一函数；");
  out.push("> 无 checkpoint 时不编造闭式。取整到参数个位，缩写沿用 formatters.js 的 B/M 档位并补 T 档）；");
  out.push("> `证据库` = `models/<org>/<id>/` 目录存在性，");
  out.push("> `manifest` = 目录内另有 `evidence-manifest.json`（见 details/models.md「模型目录内的证据文件」）；");
  out.push("> `release_time` 取 catalog ISO 串的日期段，完整时刻以 `models/catalog.json` 为准。");
  out.push("");
  out.push(`## 模型台账（${total} 个内置模型，按 model_id 码元序）`);
  out.push("");
  out.push("| 模型 ID | family（model_type） | architectures[0] | 参数量级（图声明） | 证据库 | release_time |");
  out.push("|---|---|---|---|---|---|");
  for (const row of rows) {
    out.push(`| \`${row.modelId}\` | \`${row.modelType}\` | \`${row.arch0}\` | ${paramCell(row.params)} | ${row.evidence} | ${row.releaseDate} |`);
  }
  out.push("");
  out.push("## 按 architectures[0] 汇总（生成物）");
  out.push("");
  for (const [arch, count] of [...architectureCounts.entries()].sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : 1))) {
    out.push(`- \`${arch}\`：${count} 个`);
  }
  out.push("");
  out.push(`证据库 manifest（evidence-manifest.json）覆盖：**${manifestCount} / ${total}**`);
  out.push("");
  out.push(MODELS_END);
  return out.join("\n");
}

/** docs/architectures_reference.md 的机器段（MODELS 注册表 + 配方表）。 */
function renderArchitectures({ arch0Counts }) {
  const out = [];
  out.push(ARCH_BEGIN);
  out.push("");
  out.push("> **本节由 `node scripts/gen-model-reference.mjs` 生成，请勿手改。**");
  out.push("> 事实源 = `frontend/src/structure/models/index.js` 的 `MODELS`");
  out.push(">（key=`architectures[0]` 原字符串，对标 vLLM `_TEXT_GENERATION_MODELS` / SGLang `_ModelRegistry.models`）+");
  out.push("> `structure/archs/index.js` 的 `ARCH_RECIPES`（类名 / 路径例外）。");
  out.push("> `catalog 命中` = 59 内置模型的精确计数。");
  out.push("");
  const registered = Object.keys(MODELS);
  out.push(`## MODELS 注册表（${registered.length} 条）`);
  out.push("");
  out.push("| architectures[0] | catalog 命中 |");
  out.push("|---|---|");
  for (const arch of registered) {
    out.push(`| \`${arch}\` | ${arch0Counts.get(arch) || 0} |`);
  }
  out.push("");
  const recipes = Object.entries(ARCH_RECIPES);
  // 配方位列动态取「跨条目并集」：将来 recipe 增删字段时表列自动跟随，
  // 不需要改本脚本（首选顺序仅影响既有四位的展示次序）。
  const preferred = ["normMode", "linearAttentionMode", "visionInternalMerger", "sharedExpertsAreFused"];
  const recipeKeys = [
    ...preferred.filter((key) => recipes.some(([, recipe]) => key in recipe)),
    ...new Set(recipes.flatMap(([, recipe]) => Object.keys(recipe)).filter((key) => !preferred.includes(key))),
  ];
  out.push(`## 配方表 ARCH_RECIPES（${recipes.length} 条）`);
  out.push("");
  out.push(`> 四个配方位（${preferred.join(" / ")}）在 config 里没有对应字段，属人工登记的`);
  out.push("> 家族知识（archs/index.js 头注：显式声明比藏在 `model_type.includes(...)` 里诚实）；");
  out.push("> 能用 config 字段表达的判据一律走 `config/plan.js`，不进配方表。");
  out.push("");
  out.push(`| architectures[0] | ${recipeKeys.join(" | ")} |`);
  out.push(`|---|${recipeKeys.map(() => "---").join("|")}|`);
  for (const [arch, recipe] of recipes) {
    const cells = recipeKeys.map((key) => {
      const value = recipe[key];
      if (value === undefined) return "—";
      if (typeof value === "boolean") return value ? "✓" : "—";
      if (value && typeof value === "object") {
        return `\`${Object.entries(value).map(([k, v]) => `${k}=${v}`).join(", ")}\``;
      }
      return `\`${String(value)}\``;
    });
    out.push(`| \`${arch}\` | ${cells.join(" | ")} |`);
  }
  out.push("");
  out.push(ARCH_END);
  return out.join("\n");
}

// ---- 驱动：扫描一次（两份文档共用同一份扫描结果），再渲染 / 比对 / 写盘 ----
const data = collectModels();
const targets = [
  {
    doc: MODELS_DOC, begin: MODELS_BEGIN, end: MODELS_END,
    render: () => renderModels(data), label: "models_reference.md",
    stat: `${data.total} 行模型台账`,
  },
  {
    doc: ARCH_DOC, begin: ARCH_BEGIN, end: ARCH_END,
    render: () => renderArchitectures(data), label: "architectures_reference.md",
    stat: `${Object.keys(MODELS).length} MODELS + ${Object.keys(ARCH_RECIPES).length} 配方`,
  },
];

const check = process.argv.includes("--check");
let failed = false;
for (const target of targets) {
  const generated = target.render();
  const existing = fs.existsSync(target.doc) ? fs.readFileSync(target.doc, "utf8") : "";
  const hasMarkers = existing.includes(target.begin) && existing.includes(target.end);
  // 拼接手法照抄 gen-operators-reference.mjs：机器段整体替换 BEGIN..END 区间，
  // 人工段（含锚注解视图）原样保留；无标记时追加到文件尾。
  const next = hasMarkers
    ? existing.slice(0, existing.indexOf(target.begin)) + generated + existing.slice(existing.indexOf(target.end) + target.end.length)
    : `${existing.trimEnd()}\n\n${generated}\n`;

  if (check) {
    if (next !== existing) {
      console.error(`✗ ${target.label} 机器段与 catalog/registry/recipes 不一致。跑 \`node scripts/gen-model-reference.mjs\` 重生成。`);
      failed = true;
    } else {
      console.log(`${target.label} 机器段：与 catalog/registry/recipes 一致（${target.stat}）`);
    }
  } else {
    fs.writeFileSync(target.doc, next);
    console.log(`written 机器段 -> ${path.relative(repoRoot, target.doc)}（${target.stat}）`);
  }
}
if (failed) process.exit(1);
