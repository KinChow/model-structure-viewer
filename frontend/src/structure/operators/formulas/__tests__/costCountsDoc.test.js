import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";
import { buildCostCountsRoster, currentCostCountsSegment } from "../../../../../../scripts/gen-cost-counts.mjs";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../../../../../..");
const DOC = path.join(repoRoot, "docs/details/cost_counts.md");

// 冻结用例（npm test / CI）：逐条清单机器段必须与 FORMULAS 派生结果逐字节相等。
// 新增/删除算子或改 group/counts 后，跑 `node scripts/gen-cost-counts.mjs` 重生成即可。
test("cost_counts.md 逐条清单机器段与 FORMULAS 不漂移", () => {
  const doc = fs.readFileSync(DOC, "utf8");
  const current = currentCostCountsSegment(doc);
  assert.ok(current, "cost_counts.md 缺少 cost-counts-roster 机器段标记");
  assert.equal(
    current,
    buildCostCountsRoster(),
    "cost_counts.md 逐条清单已过期：运行 `node scripts/gen-cost-counts.mjs` 重生成",
  );
});
