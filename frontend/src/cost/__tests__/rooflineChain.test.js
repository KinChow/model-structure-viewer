import assert from "node:assert/strict";
import fs from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { normalizeConfig } from "../../structure/config/normalize.js";
import { resolveArchitecture } from "../../structure/registry/resolveArchitecture.js";
import { buildNetwork } from "../../structure/model_executor/models/index.js";
import { createStructureIr } from "../../structure/ir/createStructureIr.js";
import { materializeModelStructure } from "../../structure/materializers/toStructureNode.js";
import { deriveBuildPlan } from "../../structure/model_executor/plan.js";
import { aggregateCost } from "../aggregate.js";
import { planCommunicationBytes } from "../comm.js";
import { classifyRoofline } from "../roofline.js";
import { PUBLIC_CHIPS } from "../chips/public.js";

// 第六 oracle（M11-P0-1）：UI 入口 → roofline 的全链路测试。
// 此前 roofline.test.js 只直接构造入参测函数本身，CostSummary/lens 两个
// 生产入口均未传 actions（五路退化三路）却全绿——"两端都测了，中间没测"。
// 本测试复刻 CostSummary.jsx 的聚合调用形状，对全部内置模型断言：
//   1. counts 通道完整（computeComplete 且 actions 非空）
//   2. bound 可分类（不得为 unknown）
//   3. matrix/memory 两路时间可得
// P0-4 接通 actions 后，此处追加五路时间断言；P0-5 接入 counts.bytes 后
// 追加访存侧来自 counts 通道的断言。

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../../../..");
const MACHINE = PUBLIC_CHIPS[0];

test("all built-in models classify a roofline bound through the aggregate chain", () => {
  const catalog = JSON.parse(fs.readFileSync(path.join(repoRoot, "models/catalog.json"), "utf8"));
  assert.equal(catalog.models.length, 59);
  const bounds = {};
  for (const entry of catalog.models) {
    const rawConfig = JSON.parse(fs.readFileSync(path.join(repoRoot, "models", entry.config_path), "utf8"));
    const normalized = normalizeConfig(rawConfig);
    const resolved = resolveArchitecture(normalized, { modelId: entry.model_id });
    const structure = materializeModelStructure(createStructureIr({
      network: buildNetwork(resolved, normalized),
      normalized,
      resolved,
    }));

    // CostSummary.jsx 聚合调用形状（M11-P0-4 起为 actions 通道）
    const cost = aggregateCost({ graph: structure.graph, config: normalized, phase: "prefill", batch: 1, sequence: 512 });
    assert.equal(cost.computeComplete, true, `${entry.model_id}: ${cost.unknownComputePaths.join(", ")}`);
    assert.ok(cost.actions, `${entry.model_id}: aggregate actions missing`);

    const plan = deriveBuildPlan(normalized.raw ?? normalized);
    const communication = planCommunicationBytes({ graph: structure.graph, config: normalized, plan, batch: 1, tokens: 512 });
    const roofline = classifyRoofline({
      actions: {
        ...cost.actions,
        bytes: {
          weights: cost.memory.weightBytes,
          actIn: cost.actions.actIn,
          actOut: cost.actions.actOut,
        },
        commBytes: communication?.totalBytes || 0,
      },
    }, MACHINE, { dtype: "bf16", efficiency: {} });

    assert.ok(roofline.bound && roofline.bound !== "unknown", `${entry.model_id}: bound unclassified (missing: ${roofline.missing.join(", ")})`);
    assert.ok(roofline.times.matrix != null, `${entry.model_id}: matrix time missing`);
    // v2（P0-4）：五路必须全部可得——vector/sfu 计数来自 counts 通道，
    // 计数为零时时间是精确零，不得再出现 legacy 伪造前的 null
    assert.ok(roofline.times.vector != null, `${entry.model_id}: vector time missing`);
    assert.ok(roofline.times.sfu != null, `${entry.model_id}: sfu time missing`);
    assert.ok(roofline.times.memory != null, `${entry.model_id}: memory time missing`);
    assert.ok(roofline.times.comm != null, `${entry.model_id}: comm time missing`);
    bounds[roofline.bound] = (bounds[roofline.bound] || 0) + 1;
  }
  // bound 分布记录在断言消息里，方便人工审阅（矩阵/访存/通信的分布变化）
  assert.ok(Object.keys(bounds).length >= 1, `bound distribution: ${JSON.stringify(bounds)}`);
});
