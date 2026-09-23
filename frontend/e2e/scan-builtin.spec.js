import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { expect, test } from "./fixtures.js";

// 全量内置模型端到端展示层巡检（人工验收辅助）：逐个打开内置模型，读取结论条
// （Fit / 总显存 / 瓶颈 / GPU）与 Cost 面板（VRAM / MACs / FLOPs / Roofline
// bound），并在中/英两态各采一次，落一份 JSON 报告供归档与异常分析。
const __dirname = path.dirname(fileURLToPath(import.meta.url));
const REPORT = path.join(__dirname, "..", "..", "docs", "details", "evidence", "frontend", "builtin-scan-report.json");

test.beforeEach(async ({ page }) => {
  await page.route(/https:\/\/(?:www\.)?(?:huggingface\.co|modelscope\.cn)\//, (route) => route.abort());
  await page.goto("/");
});

test("全量内置模型展示层巡检：结论条与成本数据可正确展示", async ({ page }, testInfo) => {
  test.skip(testInfo.project.name !== "desktop-chrome", "全量巡检仅桌面浏览器");
  test.setTimeout(900_000);
  page.setDefaultTimeout(12_000);

  const modelIds = await page.locator("datalist#builtin-models option").evaluateAll((options) => options.map((o) => o.value));
  expect(modelIds.length).toBeGreaterThan(0);

  const readAnswerBar = async () => {
    const items = page.locator(".detail-answer-bar .ab-item");
    const n = await items.count();
    const out = {};
    for (let i = 0; i < n; i += 1) {
      const key = (await items.nth(i).locator("i").textContent())?.trim();
      const val = (await items.nth(i).locator("b").textContent())?.trim();
      if (key) out[key] = val;
    }
    return out;
  };

  const readCost = async () => {
    const costToggle = page.locator(".detail-cost-toggle > button");
    if (!(await costToggle.count())) return null;
    if ((await page.locator(".detail-cost-panel.is-collapsed").count()) > 0) await costToggle.click();
    const metrics = page.locator(".cost-summary .cost-metrics");
    await expect(metrics).toBeVisible();
    const rooflineSpan = metrics.locator("[data-bound]");
    const dataBound = await rooflineSpan.getAttribute("data-bound");
    const rooflineLabel = (await rooflineSpan.locator("b").textContent())?.trim();
    const text = (await metrics.textContent()) || "";
    return { dataBound, rooflineLabel, text };
  };

  const report = [];
  for (const modelId of modelIds) {
    await page.getByLabel("model id").fill(modelId);
    await page.getByRole("button", { name: "打开模型" }).click();
    await expect(page.locator(".detail-page")).toBeVisible();
    await expect(page.locator(".react-flow-diagram")).toHaveAttribute("data-graph-version", "2");
    await expect(page.locator(".detail-answer-bar")).toBeVisible();

    const zhAnswer = await readAnswerBar();
    const zhCost = await readCost();

    await page.getByRole("button", { name: "中 / EN" }).click();
    await expect(page.locator(".detail-answer-bar")).toBeVisible();
    const enAnswer = await readAnswerBar();
    const enCost = await readCost();
    await page.getByRole("button", { name: "EN / 中" }).click();

    const anomalies = [];
    const bad = (s) => s == null || /nan|undefined|null/i.test(s) || s.trim() === "";
    const boundZh = zhAnswer["瓶颈"];
    const boundEn = enAnswer["Bound"];
    const memZh = zhAnswer["总显存"];
    if (bad(boundZh)) anomalies.push(`zh bound empty: ${JSON.stringify(boundZh)}`);
    if (bad(boundEn)) anomalies.push(`en bound empty: ${JSON.stringify(boundEn)}`);
    if (zhCost?.dataBound === "unknown") anomalies.push("roofline data-bound=unknown");
    if (boundEn && /^(matrix|memory|comm|vector|sfu)$/i.test(boundEn)) anomalies.push(`en bound not categorized: ${boundEn}`);
    if (boundZh && /^(矩阵|访存|通信|向量)$/.test(boundZh)) anomalies.push(`zh bound not categorized: ${boundZh}`);
    if (bad(memZh)) anomalies.push(`zh per-card VRAM bad: ${JSON.stringify(memZh)}`);
    if (zhCost && /NaN|undefined/.test(zhCost.text)) anomalies.push("cost metrics contain NaN/undefined");

    report.push({ modelId, zhAnswer, enAnswer, zhCost: zhCost && { dataBound: zhCost.dataBound, rooflineLabel: zhCost.rooflineLabel }, enCost: enCost && { dataBound: enCost.dataBound, rooflineLabel: enCost.rooflineLabel }, anomalies });

    await page.getByRole("button", { name: /Model Structure Viewer v/ }).click();
    await expect(page.getByLabel("model id")).toBeVisible();
  }

  fs.mkdirSync(path.dirname(REPORT), { recursive: true });
  fs.writeFileSync(REPORT, JSON.stringify({ scannedAt: new Date().toISOString(), total: report.length, report }, null, 2));

  const flagged = report.filter((r) => r.anomalies.length > 0);
  await testInfo.attach("builtin-scan-report.json", { path: REPORT, contentType: "application/json" });
  console.log(`\n[builtin-scan] total=${report.length} flagged=${flagged.length}`);
  for (const r of flagged) console.log(`  - ${r.modelId}: ${r.anomalies.join("; ")}`);

  const displayLayerBugs = flagged.filter((r) => r.anomalies.some((a) => /empty|not categorized|VRAM bad|NaN/.test(a)));
  expect(displayLayerBugs, `展示层异常：\n${displayLayerBugs.map((r) => `${r.modelId}: ${r.anomalies.join("; ")}`).join("\n")}`).toEqual([]);
});
