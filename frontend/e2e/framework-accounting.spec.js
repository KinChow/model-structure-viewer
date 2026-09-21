import { expect, test } from "./fixtures.js";
import { writeFile } from "node:fs/promises";

test.beforeEach(async ({ page }) => {
  await page.route(/https:\/\/(?:www\.)?(?:huggingface\.co|hf-mirror\.com|modelscope\.cn)\//, (route) => route.abort());
  await page.goto("/");
});

async function openCost(page, model, profile = "neutral") {
  await page.goto(`/?fw=${profile}`);
  await expect(page.locator(`datalist#builtin-models option[value="${model}"]`)).toHaveCount(1);
  await page.getByLabel("model id").fill(model);
  await page.getByRole("button", { name: "打开模型", exact: true }).click();
  await expect(page.locator(".detail-page")).toBeVisible();
  await page.locator(".detail-cost-toggle > button").click();
  const cost = page.locator(".cost-summary");
  await expect(cost).toHaveAttribute("data-framework", profile);
  await cost.getByRole("button", { name: "展开配置", exact: true }).click();
  return cost;
}

async function number(cost, label, value) {
  const input = cost.getByLabel(label, { exact: true });
  await input.fill(String(value));
  await input.press("Enter");
}

test("V4.1 Provider 日期快照与最新排序", async ({ page }, testInfo) => {
  await page.locator(".provider-card").filter({ hasText: "deepseek-ai" }).click();
  const rows = page.locator(".provider-model-list > button");
  await expect(rows.first().locator("strong")).toHaveText("DeepSeek-V4.1-Flash");
  await expect(rows.first().locator("small")).toHaveText(/2026.*09.*10/);
  await page.screenshot({ path: testInfo.outputPath("release-metadata.png"), fullPage: true });
});

test("vLLM TP4 dense 和 MoE 不产生 invalid；显式 EP 可用", async ({ page }) => {
  test.setTimeout(90_000);
  for (const model of ["Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-35B-A3B"]) {
    const cost = await openCost(page, model, "vllm");
    await number(cost, "TP", 4);
    await expect(cost.locator(".cost-plan-error")).toHaveCount(0);
    if (model.endsWith("A3B")) {
      await number(cost, "EP", 4);
      await expect(cost.locator(".cost-plan-error")).toHaveCount(0);
      // Switching EP off must not persist an injected moeTp=1 in UI state.
      await number(cost, "EP", 1);
      await expect(cost.locator(".cost-plan-error")).toHaveCount(0);
    }
  }
});

test("Qwen draft allocation 与 Total VRAM / Fit / Max Context 同源", async ({ page }, testInfo) => {
  const cost = await openCost(page, "Qwen/Qwen3.5-4B");
  await number(cost, "输入 tokens / request", 2100000);
  const draft = cost.locator('[data-owner="draft"]');
  await expect.poll(async () => Number(await draft.getAttribute("data-bytes"))).toBeGreaterThan(0);
  await expect(cost.locator(".cost-machine-summary .no-fit")).toBeVisible();
  await expect(cost.locator(".cost-metrics").getByText(/单卡适配/)).toContainText("否");
  const rollup = await cost.locator(".cost-rollup > span").evaluateAll((els) => els.map((el) => Number(el.dataset.bytes)));
  expect(Math.abs(rollup[0] + rollup[1] + rollup[2] - rollup[3])).toBeLessThan(0.01);
  const maxContext = cost.locator(".cost-metrics > span").filter({ hasText: "Max context" }).locator("b");
  expect(Number((await maxContext.innerText()).replaceAll(",", ""))).toBeLessThan(2100000);
  await cost.screenshot({ path: testInfo.outputPath("draft-no-fit.png") });
});

test("SGLang fusion opt-in 增加通信量，取消恢复默认", async ({ page }) => {
  const cost = await openCost(page, "Qwen/Qwen3.5-35B-A3B", "sglang");
  await number(cost, "TP", 4);
  await number(cost, "EP", 4);
  const metric = cost.locator(".cost-metrics > span").filter({ hasText: /^通信 / }).locator("b");
  const baseline = await metric.innerText();
  const fusion = cost.getByLabel("共享专家融合（显式启用）", { exact: true });
  await expect(fusion).not.toBeChecked();
  await fusion.check();
  await expect(metric).not.toHaveText(baseline);
  await fusion.uncheck();
  await expect(metric).toHaveText(baseline);
});

test("DSpark 三种 profile 的主/草稿/共享/总 KV 对账", async ({ page }) => {
  test.setTimeout(90_000);
  for (const profile of ["neutral", "vllm", "sglang"]) {
    const cost = await openCost(page, "deepseek-ai/DeepSeek-V4.1-Flash", profile);
    const values = await cost.locator(".cost-kv-ownership > span").evaluateAll((els) => els.map((el) => Number(el.dataset.bytes)));
    expect(values[0]).toBeGreaterThan(0);
    expect(values[1]).toBeGreaterThan(0); // private storage proven by current upstream, not zeroed to match measurements
    expect(values[0] + values[1] + values[2]).toBe(values[3]);
    const draft = values[1];
    await number(cost, "输入 tokens / request", 4096);
    const changed = Number(await cost.locator('[data-owner="draft"]').getAttribute("data-bytes"));
    expect(changed).toBe(profile === "neutral" ? draft * 2 : draft); // runtime private SWA is bounded
  }
});

test("DSA 默认 KV fallback 控件不覆盖显式 dtype", async ({ page }) => {
  const cost = await openCost(page, "deepseek-ai/DeepSeek-V3.2", "sglang");
  const kv = cost.locator('[data-owner="main"]');
  const before = await kv.getAttribute("data-bytes");
  await expect(cost.getByText("仅作为 fallback：", { exact: false })).toBeVisible();
  await cost.getByRole("combobox", { name: "默认 KV bytes / element", exact: true }).selectOption("1");
  await expect(kv).toHaveAttribute("data-bytes", before);
});

test("Chrome 全量 60 模型成本与图扫描（桌面和移动）", async ({ page }, testInfo) => {
  test.setTimeout(600_000);
  const models = await page.locator("datalist#builtin-models option").evaluateAll((els) => els.map((el) => el.value));
  expect(models).toHaveLength(60);
  const results = [];
  for (const model of models) {
    await page.getByLabel("model id").fill(model);
    await page.getByRole("button", { name: "打开模型", exact: true }).click();
    await expect(page.locator(".react-flow__node").first()).toBeVisible();
    await page.locator(".detail-cost-toggle > button").click();
    const cost = page.locator(".cost-summary");
    await expect(cost.locator("[data-bound]")).not.toHaveAttribute("data-bound", "unknown");
    await expect(cost).not.toContainText(/NaN|Infinity|\[object Object\]/);
    await expect(cost.locator(".cost-plan-error")).toHaveCount(0);
    const kv = await cost.locator(".cost-kv-ownership > span").evaluateAll((els) => els.map((el) => Number(el.dataset.bytes)));
    expect(kv.every(Number.isFinite), model).toBe(true);
    expect(kv[0] + kv[1] + kv[2], model).toBe(kv[3]);
    const width = await page.evaluate(() => [document.documentElement.scrollWidth, document.documentElement.clientWidth]);
    expect(width[0], model).toBeLessThanOrEqual(width[1] + 1);
    results.push({ model, kv, bound: await cost.locator("[data-bound]").getAttribute("data-bound") });
    await page.getByRole("button", { name: /Model Structure Viewer v/ }).click();
  }
  const output = testInfo.outputPath("all-model-accounting.json");
  await writeFile(output, JSON.stringify({ project: testInfo.project.name, total: models.length, results }, null, 2));
  await testInfo.attach("all-model-accounting", { path: output, contentType: "application/json" });
});
