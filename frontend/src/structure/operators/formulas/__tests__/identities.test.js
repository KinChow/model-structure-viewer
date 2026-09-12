// 融合分解恒等式报表（W1，warn 模式）。
//
// 判据（W4 转 error）：
//   1. fused.matrix|vector|sfu === Σ decompose（整数相等）
//   2. Σ decompose.bytes − fused.bytes === Σ 2·residentIntermediates.elements·b
// W1 只断言「每个注册模块都能取到 decompose 且原子可求值」，其余以报表形式
// 输出到 stderr，作为 W2-W4 的待修清单。
import assert from "node:assert/strict";
import fs from "node:fs";
import test from "node:test";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { evaluateDecomposition } from "../atoms.js";
import { DECOMPOSE_PENDING, MODULES } from "../modules.js";
import { moduleParamsFor } from "../moduleProbeParams.js";
import { normalizeConfig } from "../../../config/normalize.js";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../../../../../..");
const B = 2;

// 16 结构类的代表模型（plan §五）。
const REPRESENTATIVES = [
  ["S01", "Qwen/Qwen3.5-0.8B"],
  ["S02", "Qwen/Qwen3.5-35B-A3B"],
  ["S03", "zai-org/GLM-5"],
  ["S04", "moonshotai/Kimi-K2-Instruct"],
  ["S05", "deepseek-ai/DeepSeek-V4-Pro"],
  ["S06", "moonshotai/Kimi-K2.5"],
  ["S07", "deepseek-ai/DeepSeek-V3.1"],
  ["S08", "zai-org/GLM-5.3-Flash"],
  ["S09", "MiniMaxAI/MiniMax-M3"],
  ["S10", "Qwen/Qwen3.8-2.4T-A95B"],
  ["S11", "Qwen/Qwen3.8-Flash-Next"],
  ["S12", "deepseek-ai/DeepSeek-V3.2"],
  ["S13", "deepseek-ai/DeepSeek-V4-Flash-Vision-Exp"],
  ["S14", "zai-org/GLM-4.7"],
  ["S15", "moonshotai/Kimi-K3"],
  ["S16", "MiniMaxAI/MiniMax-M2.7"],
];

const PHASES = [
  { name: "prefill", tokens: 128, sequence: 128 },
  { name: "decode", tokens: 1, sequence: 4096 },
];

function loadConfigs() {
  const catalog = JSON.parse(fs.readFileSync(path.join(repoRoot, "models/catalog.json"), "utf8"));
  const byId = new Map(catalog.models.map((m) => [m.model_id, m]));
  return REPRESENTATIVES.map(([cls, modelId]) => {
    const entry = byId.get(modelId);
    assert.ok(entry, `代表模型缺失: ${modelId}`);
    const raw = JSON.parse(fs.readFileSync(path.join(repoRoot, "models", entry.config_path), "utf8"));
    return { cls, modelId, normalized: normalizeConfig(raw) };
  });
}

// 每个模块在给定 (config, phase) 下的代表参数。**实现在
// ../moduleProbeParams.js** —— 表的机器段生成器（scripts/gen-operators-reference.mjs）
// 要打印同一工作点的恒等式结果与融合收益，抄两份必然漂移。
function paramsFor(id, c, ph) {
  return moduleParamsFor(id, c, ph, B);
}

const fmt = (n) => (Number.isFinite(n) ? n.toExponential(3) : String(n));
const ratio = (a, e) => (e === 0 ? (a === 0 ? "1.0000" : "inf") : (a / e).toFixed(4));

test("融合分解恒等式报表（W1 warn 模式）", () => {
  const configs = loadConfigs();
  const findings = [];
  let checked = 0;

  for (const id of Object.keys(MODULES)) {
    const entry = MODULES[id];
    assert.equal(typeof entry.decompose, "function", `${id}: 缺 decompose（叶模块必须声明分解）`);
    assert.ok(entry.source?.ref, `${id}: 缺 source.ref（算法出处必须可追溯）`);

    for (const { cls, modelId, normalized } of configs) {
      for (const ph of PHASES) {
        const p = paramsFor(id, normalized, ph);
        if (!p) continue;
        const fused = entry.fused(p);
        const decomposed = evaluateDecomposition(entry.decompose(p));
        // 驻留中间量按各自声明的 bytesPerElement 计（fp32 打分中间量是 4B，
        // 不是激活的 2B）；读+写各省一次 → 2x。
        const resident = (entry.residentIntermediates?.(p) || [])
          .reduce((s, x) => s + 2 * x.elements * (x.bytesPerElement ?? B), 0);
        checked += 1;
        for (const unit of ["matrix", "vector", "sfu"]) {
          if (fused[unit] !== decomposed[unit]) {
            findings.push({ kind: "compute", id, cls, modelId, phase: ph.name, unit, fused: fused[unit], decomposed: decomposed[unit] });
          }
        }
        const fusedBytes = fused.bytes.weights + fused.bytes.actIn + fused.bytes.actOut;
        const decBytes = decomposed.bytes.weights + decomposed.bytes.actIn + decomposed.bytes.actOut;
        void resident;
        // W5 判据改为**两侧夹逼**，而不是「差额恰等于驻留量」。原因：原子的字节
        // 模型按「操作数个数 x 元素数」粗计，对逐元素融合链（norm/gate/激活）
        // 而言 Σ分解 并不是一个物理上有意义的「未融合基线」——主输入会被重复
        // 计入多次。物理上真正成立且可核的是这两条：
        //   ① fused.bytes >= compulsoryBytes（主输入 + 权重 + 输出的强制流量下界）
        //   ② fused.bytes <= Σ decompose.bytes（融合不可能比不融合更费）
        // 越界即公式错，无容差无登记。
        const floor = entry.compulsoryBytes ? entry.compulsoryBytes(p) : null;
        if (floor != null && fusedBytes < floor - 1e-9) {
          findings.push({ kind: "bytes-floor", id, cls, modelId, phase: ph.name, gap: fusedBytes, declared: floor });
        }
        if (fusedBytes > decBytes + 1e-9) {
          findings.push({ kind: "bytes-ceiling", id, cls, modelId, phase: ph.name, gap: fusedBytes, declared: decBytes });
        }
      }
    }
  }

  // ---- 报表 ----
  const byModule = new Map();
  for (const f of findings) {
    const key = `${f.id}|${f.kind}|${f.phase}|${f.unit || "bytes"}`;
    if (!byModule.has(key)) byModule.set(key, []);
    byModule.get(key).push(f);
  }
  console.error(`\n=== W1 融合分解恒等式报表：检查 ${checked} 组 (模块 x 结构类 x 相位)，不闭合 ${byModule.size} 类 ===`);
  for (const [key, list] of [...byModule.entries()].sort()) {
    const [id, kind, phase, unit] = key.split("|");
    const sample = list[0];
    const detail = kind === "compute"
      ? `fused=${fmt(sample.fused)} decompose=${fmt(sample.decomposed)} ratio=${ratio(sample.fused, sample.decomposed)}`
      : kind === "bytes-floor"
        ? `fused=${fmt(sample.gap)} < compulsory 下界 ${fmt(sample.declared)}（少算了强制流量）`
        : `fused=${fmt(sample.gap)} > Σ分解 ${fmt(sample.declared)}（融合反而更费，公式有误）`;
    console.error(`  [${kind}] ${id} ${phase} ${unit}: ${detail}  (${list.length} 个结构类，例 ${sample.cls} ${sample.modelId})`);
  }
  console.error(`\n=== 未声明分解的已登记模块 ${Object.keys(DECOMPOSE_PENDING).length} 条 ===`);
  for (const [id, why] of Object.entries(DECOMPOSE_PENDING)) console.error(`  ${id}: ${why}`);
  console.error("");

  assert.ok(checked > 0, "报表未覆盖任何模块");

  // W5：**计算三分量转 error 模式**。融合不改变计算量，matrix/vector/sfu 必须
  // 与原子分解逐位相等，无容差、无登记。归零路径（每条都是公式修正，不是放宽）：
  //   F2 补 1/sqrt(d) 的 scale（vector 3→4·scores）
  //   F3 均方求和每 token 少一次加法（4·T·H → 4·T·H - T）
  //   F7b 外积/delta matvec/query 三段归 matrix，vector 只剩 decay；次数按
  //       chunked steps（与运行时 stateUpdateCounts 同源）
  //   F8 补归一化求和 T·(k-1)，并补 div 原子承载 sfu
  const computeOffenders = findings.filter((f) => f.kind === "compute");
  assert.deepEqual(
    [...new Set(computeOffenders.map((f) => `${f.id}|${f.phase}|${f.unit}`))].sort(),
    [],
    "融合分解的计算三分量必须逐位相等：改公式或改分解，不允许登记容差",
  );

  // bytes 侧仍是 warn：融合收益（residentIntermediates）尚未逐项列全，
  // 剩余类别见上方报表。转 error 的前置是把每个模块的驻留中间量列清。
  // bytes 侧同样转 error：两条物理不等式，越界即错。
  const byteOffenders = findings.filter((f) => f.kind !== "compute");
  assert.deepEqual(
    [...new Set(byteOffenders.map((f) => `${f.kind}|${f.id}|${f.phase}`))].sort(),
    [],
    "bytes 夹逼越界：fused 必须 >= compulsory 下界且 <= Σ分解",
  );
});
