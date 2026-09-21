// 只读前端 oracle：以真机实际 config 计算逻辑驻留账本，不把实测值写入产品。
// 用法：node accounting.mjs config.json [neutral|vllm|sglang] [tp] [ep]
// 回填：docs/details/evidence/memory/framework_runtime_validation_20260921.md。
import fs from "node:fs";
import { buildStructureFromConfig } from "../../../frontend/src/structure/buildStructure.js";
import { normalizeConfig } from "../../../frontend/src/structure/config/normalize.js";
import { buildCostAccounting, graphWeightCapacity } from "../../../frontend/src/cost/memory.js";
import { projectPlan } from "../../../frontend/src/cost/parallel.js";
import { resolveFrameworkPlan, expertShardDivisor } from "../../../frontend/src/cost/sharding.js";

const [file, frameworkProfile = "neutral", tp = "1", ep = "1"] = process.argv.slice(2);
const raw = JSON.parse(fs.readFileSync(file, "utf8"));
const config = normalizeConfig(raw);
const structure = buildStructureFromConfig(raw, { frameworkProfile });
if (structure.source?.diagnostics?.resolution === "unsupported") {
  console.error(JSON.stringify({ file, status: "unsupported", diagnostics: structure.source.diagnostics }));
  process.exit(2); // 不允许把 unsupported 空图当作 0-byte 精确对账。
}
const { graph } = structure;
const accounting = buildCostAccounting({
  graph, config, frameworkProfile, tokens: 2048, batch: 1,
  weightBytes: graphWeightCapacity(graph).bytes,
});
const plan = resolveFrameworkPlan({ tp: Number(tp), ep: Number(ep), dp: 1, pp: 1 }, frameworkProfile, config);
const projected = projectPlan({ graph, config, accounting, plan });
console.log(JSON.stringify({
  file, frameworkProfile, plan, expertSharding: expertShardDivisor(plan),
  main: accounting.main, draft: accounting.draft, shared: accounting.shared, total: accounting.total,
  stages: projected.stages, evidence: accounting.evidence, pools: accounting.pools,
}, null, 2));
