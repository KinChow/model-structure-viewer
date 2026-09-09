#!/usr/bin/env node
// gen-operators-reference.mjs —— 生成 docs/details/operators_reference.md 的机器段。
//
// 背景（W6）：这张表原先 1500+ 行全手写，59 模型 x 40+ 算子的触发面数字靠人工
// 维护，必然与代码漂移——用户看到的「漏洞百出」有一部分就是这么来的。
// 现在机器段由本脚本从注册表 + 探针生成，人工段（已知近似、对齐勾选）单独维护。
//
// 用法：
//   node scripts/gen-operators-reference.mjs            # 写入文档
//   node scripts/gen-operators-reference.mjs --check     # 只比对，diff 非空即退出 1
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { normalizeConfig } from "../frontend/src/structure/config/normalize.js";
import { resolveArchitecture } from "../frontend/src/structure/registry/resolveArchitecture.js";
import { buildNetwork } from "../frontend/src/structure/model_executor/models/index.js";
import { createStructureIr } from "../frontend/src/structure/ir/createStructureIr.js";
import { materializeModelStructure } from "../frontend/src/structure/materializers/toStructureNode.js";
import { FORMULAS } from "../frontend/src/structure/formulas/index.js";
import { MODULES, DECOMPOSE_PENDING } from "../frontend/src/structure/formulas/modules.js";
import { OPERATOR_TO_MODULE, moduleParamsFor } from "../frontend/src/structure/formulas/moduleProbeParams.js";
import { evaluateDecomposition } from "../frontend/src/structure/formulas/atoms.js";
import { classifyRoofline } from "../frontend/src/cost/roofline.js";
import { countsForNode } from "../frontend/src/structure/formulas/extractor.js";
import { childRepeatMultiplier } from "../frontend/src/cost/traverse.js";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const DOC = path.join(repoRoot, "docs/details/operators_reference.md");
const BEGIN = "<!-- BEGIN GENERATED: operators -->";
const END = "<!-- END GENERATED: operators -->";
const B = 2;
const PHASES = [
  { name: "prefill", sequence: 128 },
  { name: "decode", sequence: 4096 },
];

// bound 判定用的参考芯片（A100 80G bf16）。与 modelIdentities.test.js 的 CHIP 同值，
// efficiency 取 1 —— 表里要的是「结构上偏算力还是偏访存」，不是某次实测利用率。
const CHIP = {
  id: "A100-80G", memory_bytes: 80e9, memory_bandwidth: 2.039e12,
  peak_flops: { bf16: 312e12 }, vector_flops: 19.5e12, sfu_ops: 4.875e12,
};

// 16 结构类的代表模型（plan §五）。逐类分节的机器段按这张表走。
const REPRESENTATIVES = [
  ["S01", "Qwen/Qwen3.5-0.8B"], ["S02", "Qwen/Qwen3.5-35B-A3B"], ["S03", "zai-org/GLM-5"],
  ["S04", "moonshotai/Kimi-K2-Instruct"], ["S05", "deepseek-ai/DeepSeek-V4-Pro"], ["S06", "moonshotai/Kimi-K2.5"],
  ["S07", "deepseek-ai/DeepSeek-V3.1"], ["S08", "zai-org/GLM-5.3-Flash"], ["S09", "MiniMaxAI/MiniMax-M3"],
  ["S10", "Qwen/Qwen3.8-2.4T-A95B"], ["S11", "Qwen/Qwen3.8-Flash-Next"], ["S12", "deepseek-ai/DeepSeek-V3.2"],
  ["S13", "deepseek-ai/DeepSeek-V4-Flash-Vision-Exp"], ["S14", "zai-org/GLM-4.7"],
  ["S15", "moonshotai/Kimi-K3"], ["S16", "MiniMaxAI/MiniMax-M2.7"],
];

function walk(root, visit) {
  const stack = [{ node: root, multiplier: 1 }];
  while (stack.length > 0) {
    const { node, multiplier } = stack.pop();
    const children = node?.children || [];
    if (children.length > 0) {
      const childMultiplier = childRepeatMultiplier(node, multiplier);
      for (const child of children) stack.push({ node: child, multiplier: childMultiplier });
      continue;
    }
    visit(node, multiplier);
  }
}

/** 深度优先**保序**遍历（表 B 的数据流顺序靠它；walk 用栈会倒序）。 */
function walkOrdered(node, visit) {
  const children = node?.children || [];
  if (children.length === 0) { visit(node); return; }
  for (const child of children) walkOrdered(child, visit);
}

/**
 * 结构槽位 = 叶子 id 去掉层号与叶名后的路径，例如
 * `decoder.0.self_attn.q_proj` -> `decoder.self_attn`、`mtp.layer.mlp.up_proj` -> `mtp.layer.mlp`。
 * 这样同一槽位在 59 模型 / 所有层之间可以合并统计。
 */
function slotPathOf(node) {
  const segments = String(node?.id || "").split(".");
  segments.pop();
  const kept = segments.filter((s) => !/^\d+$/.test(s));
  return kept.length > 0 ? kept.join(".") : "root";
}

// 逐结构类明细段的工作点。**prefill 用 T=S=2048，不是 128** —— bound 是
// arithmetic intensity 与 ridge point 的比较结果，T 太小时连投影类都会落在访存侧
//（实测 T=128 下 59 个模型全 memory，表就失去查错价值）。与
// modelIdentities.test.js 的 BOUND_PHASES 同一工作点，表与断言口径一致。
const CLASS_PHASES = [
  { name: "prefill", tokens: 2048, sequence: 2048 },
  { name: "decode", tokens: 1, sequence: 4096 },
];

/**
 * 逐结构类分节的数据：16 个代表模型，每模型逐算子逐相位聚合三分量与 bytes 三分量，
 * 再算 arithmetic intensity 与 bound；模块层的恒等式结果与融合收益按同一工作点现算
 *（数学在 modules.js / atoms.js，本处只调用，不复制）。
 */
function collectByStructureClass() {
  const catalog = JSON.parse(fs.readFileSync(path.join(repoRoot, "models/catalog.json"), "utf8"));
  const byId = new Map(catalog.models.map((m) => [m.model_id, m]));
  const sections = [];

  for (const [cls, modelId] of REPRESENTATIVES) {
    const entry = byId.get(modelId);
    if (!entry) continue;
    const raw = JSON.parse(fs.readFileSync(path.join(repoRoot, "models", entry.config_path), "utf8"));
    const normalized = normalizeConfig(raw);
    const resolved = resolveArchitecture(normalized, { modelId });
    const structure = materializeModelStructure(createStructureIr({
      network: buildNetwork(resolved, normalized), normalized, resolved,
    }));
    // op -> { nodes, instances, prefill:{...}, decode:{...} }
    const ops = new Map();
    walk(structure.root, (node, multiplier) => {
      const op = String(node?.attributes?.operator_id || (node?.type === "embedding" ? "embedding" : ""));
      if (!op) return;
      if (!ops.has(op)) {
        ops.set(op, {
          nodes: 0, instances: 0,
          prefill: { matrix: 0, vector: 0, sfu: 0, weights: 0, actIn: 0, actOut: 0 },
          decode: { matrix: 0, vector: 0, sfu: 0, weights: 0, actIn: 0, actOut: 0 },
        });
      }
      const row = ops.get(op);
      row.nodes += 1;
      row.instances += multiplier;
      const inVision = String(node?.id || "").includes("vision");
      for (const ph of CLASS_PHASES) {
        const options = inVision
          ? { batch: 1, sequence: normalized.visionTokens || 1, phase: ph.name, vision: true, visionTokens: normalized.visionTokens || 1 }
          : { batch: 1, sequence: ph.sequence, phase: ph.name };
        const a = countsForNode(node, { config: normalized, options, path: node?.id || "", bytesPerElement: B });
        if (!a) return;
        const acc = row[ph.name];
        acc.matrix += (a.matrix || 0) * multiplier;
        acc.vector += (a.vector || 0) * multiplier;
        acc.sfu += (a.sfu || 0) * multiplier;
        acc.weights += (a.bytes?.weights || 0) * multiplier;
        acc.actIn += (a.bytes?.actIn || 0) * multiplier;
        acc.actOut += (a.bytes?.actOut || 0) * multiplier;
      }
    });

    // 模块层：恒等式结果 + 融合收益（同一工作点，逐相位）
    const moduleFacts = new Map(); // op -> { prefill: {...}, decode: {...} }
    for (const [op, moduleId] of Object.entries(OPERATOR_TO_MODULE)) {
      if (!ops.has(op)) continue;
      const mod = MODULES[moduleId];
      if (typeof mod?.decompose !== "function") continue;
      const perPhase = {};
      for (const ph of CLASS_PHASES) {
        const p = moduleParamsFor(moduleId, normalized, ph, B);
        if (!p) continue;
        const fused = mod.fused(p);
        const dec = evaluateDecomposition(mod.decompose(p));
        const gain = (mod.residentIntermediates?.(p) || [])
          .reduce((s, x) => s + 2 * x.elements * (x.bytesPerElement ?? B), 0);
        const fusedBytes = fused.bytes.weights + fused.bytes.actIn + fused.bytes.actOut;
        const decBytes = dec.bytes.weights + dec.bytes.actIn + dec.bytes.actOut;
        const floor = mod.compulsoryBytes ? mod.compulsoryBytes(p) : null;
        const computeOk = fused.matrix === dec.matrix && fused.vector === dec.vector && fused.sfu === dec.sfu;
        const bytesOk = (floor == null || fusedBytes >= floor - 1e-9) && fusedBytes <= decBytes + 1e-9;
        perPhase[ph.name] = { moduleId, computeOk, bytesOk, gain };
      }
      if (Object.keys(perPhase).length > 0) moduleFacts.set(op, perPhase);
    }
    sections.push({ cls, modelId, ops, moduleFacts });
  }
  return sections;
}

/** AI = matrix(MAC) / bytesMoved。bytesMoved = 0 时无定义。 */
function arithmeticIntensity(acc) {
  const bytes = acc.weights + acc.actIn + acc.actOut;
  return bytes > 0 ? acc.matrix / bytes : null;
}

function boundOf(acc) {
  const { bound } = classifyRoofline(
    { actions: { matrix: acc.matrix, vector: acc.vector, sfu: acc.sfu, bytes: { weights: acc.weights, actIn: acc.actIn, actOut: acc.actOut } } },
    CHIP,
    { efficiency: { flops: 1, hbm: 1 } },
  );
  return bound || "—";
}

function collect() {
  const catalog = JSON.parse(fs.readFileSync(path.join(repoRoot, "models/catalog.json"), "utf8"));
  // op -> { models:Set, nodes, instances, nonzero:{matrix,vector,sfu,bytes}, slots:Set }
  const stats = new Map();
  const touch = (op) => {
    if (!stats.has(op)) {
      stats.set(op, {
        models: new Set(), nodes: 0, instances: 0, slots: new Set(),
        nonzero: { matrix: false, vector: false, sfu: false, bytes: false },
      });
    }
    return stats.get(op);
  };
  let unknownLeaves = 0;
  // 双向表数据：opSlots: op -> Map(slotPath -> 节点数)；slotOps: slotPath -> 有序算子序列
  const opSlots = new Map();
  const slotOps = new Map();

  for (const entry of catalog.models) {
    const raw = JSON.parse(fs.readFileSync(path.join(repoRoot, "models", entry.config_path), "utf8"));
    const normalized = normalizeConfig(raw);
    const resolved = resolveArchitecture(normalized, { modelId: entry.model_id });
    const structure = materializeModelStructure(createStructureIr({
      network: buildNetwork(resolved, normalized), normalized, resolved,
    }));
    walk(structure.root, (node, multiplier) => {
      const op = String(node?.attributes?.operator_id || (node?.type === "embedding" ? "embedding" : "")) || "(unknown)";
      if (op === "(unknown)") { unknownLeaves += 1; return; }
      const row = touch(op);
      const slotPath = slotPathOf(node);
      if (!opSlots.has(op)) opSlots.set(op, new Map());
      opSlots.get(op).set(slotPath, (opSlots.get(op).get(slotPath) || 0) + 1);
      row.models.add(entry.model_id);
      row.nodes += 1;
      row.instances += multiplier;
      row.matrix = row.matrix || { prefill: 0, decode: 0 };
      row.bytes = row.bytes || { prefill: 0, decode: 0 };
      // 结构槽位：取叶子 id 里紧邻的父段（self_attn / mlp / moe / vision_tower ...）
      const segments = String(node?.id || "").split(".");
      const slot = segments.length >= 2 ? segments[segments.length - 2] : "root";
      row.slots.add(/^\d+$/.test(slot) ? (segments[segments.length - 3] || slot) : slot);
      for (const ph of PHASES) {
        const inVision = String(node?.id || "").includes("vision");
        const options = inVision
          ? { batch: 1, sequence: normalized.visionTokens || 1, phase: ph.name, vision: true, visionTokens: normalized.visionTokens || 1 }
          : { batch: 1, sequence: ph.sequence, phase: ph.name };
        const a = countsForNode(node, { config: normalized, options, path: node?.id || "", bytesPerElement: B });
        if (!a) return;
        if (a.matrix > 0) row.nonzero.matrix = true;
        if (a.vector > 0) row.nonzero.vector = true;
        if (a.sfu > 0) row.nonzero.sfu = true;
        const moved = (a.bytes?.weights || 0) + (a.bytes?.actIn || 0) + (a.bytes?.actOut || 0);
        if (moved > 0) row.nonzero.bytes = true;
        // 占比段：按实例数加权累计（乘 multiplier），跨 59 模型求和
        row.matrix[ph.name] += (a.matrix || 0) * multiplier;
        row.bytes[ph.name] += moved * multiplier;
      }
    });
    // 表 B：槽位内的算子序列按子节点声明顺序取，跨模型做「首次出现即追加」的并集
    walkOrdered(structure.root, (node) => {
      const op = String(node?.attributes?.operator_id || (node?.type === "embedding" ? "embedding" : ""));
      if (!op) return;
      const slotPath = slotPathOf(node);
      if (!slotOps.has(slotPath)) slotOps.set(slotPath, []);
      const seq = slotOps.get(slotPath);
      if (!seq.includes(op)) seq.push(op);
    });
  }
  return { stats, total: catalog.models.length, unknownLeaves, opSlots, slotOps };
}

function refOf(op) {
  const src = fs.readFileSync(path.join(repoRoot, "frontend/src/structure/formulas/index.js"), "utf8");
  const at = src.indexOf(`\n  ${op}: {`);
  if (at === -1) return "";
  const block = src.slice(at, src.indexOf("\n  },", at));
  const line = block.split("\n").find((l) => l.includes("ref:"));
  if (!line) return "";
  const kind = /一等/.test(block) ? "一" : /二等/.test(block) ? "二" : /三等/.test(block) ? "三" : "?";
  return kind;
}

function render({ stats, total, unknownLeaves, opSlots, slotOps }) {
  const rows = [...stats.entries()].sort((a, b) => b[1].models.size - a[1].models.size || a[0].localeCompare(b[0]));
  const mark = (on) => (on ? "✓" : "0");
  const out = [];
  out.push(BEGIN);
  out.push("");
  out.push("> **本节由 `node scripts/gen-operators-reference.mjs` 生成，请勿手改。**");
  out.push("> 触发面按 `models/catalog.json` 全量模型实跑（prefill T=128 / decode T=1,S=4096 两相位）；");
  out.push("> `matrix|vector|sfu|bytes` 列的 ✓/0 表示该分量在任一相位是否非零（0 = 精确零，principles §3.3）。");
  out.push("> `来源` = principles §3.5 的三级体系（一 aten 锚点 / 二 modeling 对照 / 三 分解声明）。");
  out.push("");
  out.push(`## 总览表（生成物：${rows.length} 个算子 / ${total} 个模型）`);
  out.push("");
  out.push("| 算子 | matrix | vector | sfu | bytes | 来源 | 触发模型 | 节点 | 实例 | 出现槽位 |");
  out.push("|---|---|---|---|---|---|---|---|---|---|");
  for (const [op, r] of rows) {
    const slots = [...r.slots].sort().slice(0, 4).join(" · ") + (r.slots.size > 4 ? ` 等 ${r.slots.size}` : "");
    out.push(`| \`${op}\` | ${mark(r.nonzero.matrix)} | ${mark(r.nonzero.vector)} | ${mark(r.nonzero.sfu)} | ${mark(r.nonzero.bytes)} | ${refOf(op) || "—"} | ${r.models.size}/${total} | ${r.nodes} | ${r.instances} | ${slots} |`);
  }
  out.push("");
  out.push(`未识别叶子（无 operator_id 且非 embedding）：**${unknownLeaves}**`);
  out.push("");
  const registered = Object.keys(FORMULAS);
  const triggered = new Set(rows.map(([op]) => op));
  const zeroTrigger = registered.filter((op) => !triggered.has(op));
  // ---- 占比段（按实例加权，跨全部模型求和）----
  for (const phase of ["prefill", "decode"]) {
    const totalMatrix = rows.reduce((sum, [, r]) => sum + (r.matrix?.[phase] || 0), 0);
    const totalBytes = rows.reduce((sum, [, r]) => sum + (r.bytes?.[phase] || 0), 0);
    const ranked = rows
      .map(([op, r]) => ({ op, m: r.matrix?.[phase] || 0, b: r.bytes?.[phase] || 0 }))
      .filter((x) => x.m > 0 || x.b > 0)
      .sort((a, b) => (b.m + b.b) - (a.m + a.b))
      .slice(0, 15);
    out.push(`## 算力/访存占比（${phase}，${phase === "prefill" ? "T=128" : "T=1 S=4096"}，59 模型实例加权求和）`);
    out.push("");
    out.push("| 算子 | matrix (MACs) | matrix 占比 | bytes | bytes 占比 |");
    out.push("|---|---|---|---|---|");
    for (const x of ranked) {
      const mp = totalMatrix > 0 ? ((x.m / totalMatrix) * 100).toFixed(2) : "0.00";
      const bp = totalBytes > 0 ? ((x.b / totalBytes) * 100).toFixed(2) : "0.00";
      out.push(`| \`${x.op}\` | ${x.m.toExponential(3)} | ${mp}% | ${x.b.toExponential(3)} | ${bp}% |`);
    }
    out.push("");
    out.push(`合计：matrix ${totalMatrix.toExponential(4)} MACs · bytes ${totalBytes.toExponential(4)}（前 15 名之外的算子占比均 < 前列末位）`);
    out.push("");
  }

  out.push("## 注册表 ↔ 触发面对账（生成物）");
  out.push("");
  out.push(`- 注册表条目：**${registered.length}**`);
  out.push(`- 实际被触发：**${triggered.size}**（含结构节点 \`embedding\`）`);
  out.push(`- 零触发条目：**${zeroTrigger.length}**${zeroTrigger.length ? " —— " + zeroTrigger.map((o) => `\`${o}\``).join(" · ") : ""}`);
  out.push("");

  // ---- 双向表（生成物）：与手写 §3 的表 A/表 B 同义，但覆盖面由探针保证 ----
  out.push("## 双向表 A（生成物）· 算子 → 结构槽位");
  out.push("");
  out.push("> 槽位 = 叶子 id 去掉层号与叶名的路径（`decoder.0.self_attn.q_proj` → `decoder.self_attn`）。括号内为节点数。");
  out.push("");
  out.push("| 算子 | 槽位（节点数） |");
  out.push("|---|---|");
  for (const [op] of rows) {
    const slots = [...(opSlots.get(op) || new Map()).entries()].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]));
    out.push(`| \`${op}\` | ${slots.map(([s, n]) => `\`${s}\`(${n})`).join(" · ") || "—"} |`);
  }
  out.push("");
  out.push("## 双向表 B（生成物）· 结构槽位 → 算子序列（数据流顺序）");
  out.push("");
  out.push("> 序列按子节点声明顺序取，跨 59 模型做「首次出现即追加」的并集 —— 同一槽位不同结构类的算子会依次排在后面。");
  out.push("");
  out.push("| 结构槽位 | 算子序列 |");
  out.push("|---|---|");
  for (const [slot, seq] of [...slotOps.entries()].sort((a, b) => b[1].length - a[1].length || a[0].localeCompare(b[0]))) {
    out.push(`| \`${slot}\` | ${seq.map((o) => `\`${o}\``).join(" → ")} |`);
  }
  out.push("");

  // ---- 模块层分解台账（生成物）：直接读 modules.js，不复算恒等式 ----
  const declared = Object.values(MODULES).filter((m) => typeof m.decompose === "function");
  out.push("## 模块层分解台账（生成物）");
  out.push("");
  out.push(`- 已声明分解的模块：**${declared.length}** —— ${declared.map((m) => `\`${m.id}\``).join(" · ")}`);
  out.push(`- 尚未声明分解（\`DECOMPOSE_PENDING\`）：**${Object.keys(DECOMPOSE_PENDING).length}**`);
  out.push("");
  out.push("| 模块 | 待办原因 |");
  out.push("|---|---|");
  for (const [id, why] of Object.entries(DECOMPOSE_PENDING)) out.push(`| \`${id}\` | ${why} |`);
  out.push("");
  out.push("> 恒等式（融合分解 / 权重字节 / KV 读量 / 激活流形状连续性）的判定结果不在此生成，");
  out.push("> 由 `npm test` 的 `identities.test.js` 与 `modelIdentities.test.js` 断言并打印报表 —— 避免同一套数学写两遍。");
  out.push("");
  out.push(renderStructureClassSections());
  out.push(END);
  return out.join("\n");
}

const exp3 = (n) => (n === 0 ? "0" : Number.isFinite(n) ? n.toExponential(3) : String(n));

/**
 * 按结构类分节（plan §七 W6）：每算子两行（prefill / decode），列含
 * 三分量 + bytes 三分量 + arithmetic intensity + bound + 恒等式结果 + 融合收益 + 容差。
 */
function renderStructureClassSections() {
  const sections = collectByStructureClass();
  const out = [];
  out.push("## 逐结构类算子明细（生成物）");
  out.push("");
  out.push("> 16 个结构类各取一个代表模型（plan §五），**每算子两行**：prefill T=S=2048 / decode T=1 S=4096。");
  out.push("> 数值 = 该模型内该算子**按实例加权求和**（层组 repeat 已乘）。视觉塔叶按 `visionTokens` 计。");
  out.push(`> \`AI\` = matrix(MAC) / bytesMoved；\`bound\` = 五路 max（参考芯片 ${CHIP.id}，efficiency=1，仅判结构倾向）。`);
  out.push("> `恒等式` 只对「算子即模块」的那些算（映射见 `formulas/moduleProbeParams.js` 的 `OPERATOR_TO_MODULE`）：");
  out.push("> `计算✓` = fused 三分量与原子分解逐位相等；`字节✓` = fused 落在 [compulsory 下界, Σ分解] 之间。");
  out.push("> `融合收益` = Σ 2·residentIntermediates·b（分解下会落 HBM、融合下留在寄存器/SRAM 的量）。");
  out.push("> `容差` 一列单列：结构性恒等式一律 0（整数相等 / 不等式夹逼），非 0 只可能是显式声明的近似执行形态。");
  out.push("> **bound 的粒度是叶级**。「attention prefill 算力瓶颈」说的是**模块级** —— MLA 的 `matmul` 叶");
  out.push("> 单看是访存侧（latent 读占主导，AI 未过 ridge），把 q/kv 压缩 + 投影 + scores/context 合起来才是算力侧。");
  out.push("> 模块级断言在 `modelIdentities.test.js`（error 模式），不在本表。");
  out.push("");
  for (const { cls, modelId, ops, moduleFacts } of sections) {
    out.push(`### ${cls} · ${modelId}`);
    out.push("");
    out.push("| 算子 | 相位 | 节点/实例 | matrix | vector | sfu | weights | actIn | actOut | AI | bound | 恒等式 | 融合收益 | 容差 |");
    out.push("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|");
    const ordered = [...ops.entries()].sort((a, b) => (b[1].prefill.matrix + b[1].prefill.actIn) - (a[1].prefill.matrix + a[1].prefill.actIn) || a[0].localeCompare(b[0]));
    for (const [op, row] of ordered) {
      for (const ph of CLASS_PHASES) {
        const acc = row[ph.name];
        const ai = arithmeticIntensity(acc);
        const fact = moduleFacts.get(op)?.[ph.name];
        const identity = fact
          ? `${fact.computeOk ? "计算✓" : "计算✗"} ${fact.bytesOk ? "字节✓" : "字节✗"}`
          : "—";
        const gain = fact ? exp3(fact.gain) : "—";
        const tol = fact ? "0" : "—";
        out.push(`| \`${op}\` | ${ph.name} | ${row.nodes}/${row.instances} | ${exp3(acc.matrix)} | ${exp3(acc.vector)} | ${exp3(acc.sfu)} | ${exp3(acc.weights)} | ${exp3(acc.actIn)} | ${exp3(acc.actOut)} | ${ai == null ? "—" : ai.toFixed(2)} | ${boundOf(acc)} | ${identity} | ${gain} | ${tol} |`);
      }
    }
    out.push("");
  }
  return out.join("\n");
}

// --json：把探针原始数据打到 stdout（不落文件、不进 golden）。
// W6「探针从 /tmp 收回仓库」的落点就是这个 flag —— 探针不再是一次性的 /tmp 脚本，
// 而是本生成器本身，任何人可复现同一组数字。
if (process.argv.includes("--json")) {
  const { stats, total, unknownLeaves } = collect();
  const overview = Object.fromEntries([...stats.entries()].map(([op, r]) => [op, {
    models: r.models.size, nodes: r.nodes, instances: r.instances,
    nonzero: r.nonzero, matrix: r.matrix, bytes: r.bytes,
    slots: [...r.slots].sort(),
  }]));
  const classes = collectByStructureClass().map(({ cls, modelId, ops, moduleFacts }) => ({
    cls, modelId,
    ops: Object.fromEntries([...ops.entries()].map(([op, row]) => [op, {
      nodes: row.nodes, instances: row.instances,
      prefill: { ...row.prefill, ai: arithmeticIntensity(row.prefill), bound: boundOf(row.prefill) },
      decode: { ...row.decode, ai: arithmeticIntensity(row.decode), bound: boundOf(row.decode) },
      module: moduleFacts.get(op) ?? null,
    }])),
  }));
  console.log(JSON.stringify({
    chip: CHIP,
    overviewPhases: PHASES,
    classPhases: CLASS_PHASES,
    totalModels: total,
    unknownLeaves,
    overview,
    classes,
  }, null, 2));
  process.exit(0);
}

const generated = render(collect());
const existing = fs.existsSync(DOC) ? fs.readFileSync(DOC, "utf8") : "";
const hasMarkers = existing.includes(BEGIN) && existing.includes(END);
const next = hasMarkers
  ? existing.slice(0, existing.indexOf(BEGIN)) + generated + existing.slice(existing.indexOf(END) + END.length)
  : `${existing.trimEnd()}\n\n${generated}\n`;

if (process.argv.includes("--check")) {
  if (next !== existing) {
    console.error("✗ operators_reference.md 机器段与注册表/探针不一致。跑 `node scripts/gen-operators-reference.mjs` 重生成。");
    process.exit(1);
  }
  console.log("operators_reference.md 机器段：与注册表/探针一致");
} else {
  fs.writeFileSync(DOC, next);
  console.log(`written 机器段 -> ${path.relative(repoRoot, DOC)}`);
}
