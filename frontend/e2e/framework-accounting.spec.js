import { expect, test } from "./fixtures.js";
import { writeFile } from "node:fs/promises";

// 将冷启动浏览器/Vite 的准备时间纳入多模型场景预算；
// 只在 test body 中设置会让 beforeEach 仍受 30 秒限制。
test.describe.configure({ timeout: 90_000 });

test.beforeEach(async ({ page }, testInfo) => {
  if (testInfo.title.includes("全量")) testInfo.setTimeout(600_000);
  await page.route(/https:\/\/(?:www\.)?(?:huggingface\.co|hf-mirror\.com|modelscope\.cn)\//, (route) => route.abort());
  await page.goto("/");
});

async function openCost(page, model, profile = "neutral") {
  await page.goto(`/?fw=${profile}`);
  await expect(page.locator(`datalist#builtin-models option[value="${model}"]`)).toHaveCount(1);
  await page.getByLabel("model id").fill(model);
  await page.getByRole("button", { name: "打开模型", exact: true }).click();
  await expect(page.locator(".detail-page")).toBeVisible({ timeout: 30_000 });
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

// 「成本假设」ConfigSection 默认折叠（openConfigSections.assumptions=false），
// 其内的 KV/inter-node/投机等控件在折叠时 hidden。测这些控件前先展开该分区。
// 用 data-section 定位，避免依赖语言。
async function openAssumptions(cost) {
  await cost.locator('[data-section="assumptions"] > .cost-config-section-toggle').click();
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

test("模型和硬件选择后按单机 1/2/4/8 卡档位给出默认并行策略", async ({ page }) => {
  test.setTimeout(120_000);
  const small = await openCost(page, "Qwen/Qwen3.5-4B");
  await expect(small.getByLabel("TP", { exact: true })).toHaveValue("1");
  await expect(small.getByLabel("GPU / 节点", { exact: true })).toHaveValue("8");
  await expect(small.getByTestId("parallel-default-note")).toContainText("单机");
  await expect(small.getByLabel("节点数", { exact: true })).toHaveValue("1");

  const large = await openCost(page, "deepseek-ai/DeepSeek-V4.1-Flash");
  await expect(large.getByLabel("TP", { exact: true })).toHaveValue("8");
  await expect(large.getByTestId("parallel-default-note")).toContainText("单机");
  await expect(large.getByLabel("GPU / 节点", { exact: true })).toHaveValue("8");
  await expect(large.getByLabel("节点数", { exact: true })).toHaveValue("1");
});

test("切换硬件后自动推荐重新按显存档位计算", async ({ page }) => {
  test.setTimeout(90_000);
  const cost = await openCost(page, "Qwen/Qwen3.5-27B");
  const chip = cost.locator("select").first();
  await expect(chip).toHaveValue("nvidia-a100-80gb-sxm");
  await expect(cost.getByLabel("TP", { exact: true })).toHaveValue("1");
  // 没有编辑的 focus/blur 不能把自动推荐误判为手动配置。
  await cost.getByLabel("TP", { exact: true }).focus();
  await cost.getByLabel("TP", { exact: true }).blur();
  await expect(cost.locator(".cost-default-deployment")).toHaveAttribute("data-mode", "auto");
  await chip.selectOption("nvidia-l40s-48gb");
  await expect(chip).toHaveValue("nvidia-l40s-48gb");
  await expect(cost.getByLabel("TP", { exact: true })).toHaveValue("2");
  await expect(page.locator(".diagram-lens-status")).toContainText("TP2 / PP1 / EP1 / DP1");
  await chip.selectOption("nvidia-a100-80gb-sxm");
  await expect(cost.getByLabel("TP", { exact: true })).toHaveValue("1");
});

test("手动修改并行后切换硬件不会覆盖，并可恢复默认", async ({ page }) => {
  test.setTimeout(90_000);
  const cost = await openCost(page, "Qwen/Qwen3.5-4B");
  await number(cost, "TP", 4);
  await expect(cost.locator('[data-mode="manual"]')).toBeVisible();
  const chip = cost.locator("select").first();
  await chip.selectOption("nvidia-l40s-48gb");
  await expect(cost.getByLabel("TP", { exact: true })).toHaveValue("4");
  await cost.getByRole("button", { name: "恢复默认部署", exact: true }).click();
  await expect(cost.getByLabel("TP", { exact: true })).toHaveValue("1");
  await expect(cost.locator('[data-mode="auto"]')).toBeVisible();
});

test("P/D 默认各单节点，手动策略独立保存，切换模型恢复推荐", async ({ page }, testInfo) => {
  test.setTimeout(120_000);
  const cost = await openCost(page, "Qwen/Qwen3.5-27B");
  await cost.getByRole("button", { name: "PD 分离", exact: true }).click();
  await expect(cost.getByLabel("节点 / prefill", { exact: true })).toHaveValue("1");
  await number(cost, "TP", 2);
  await cost.getByRole("button", { name: "Decode", exact: true }).click();
  await expect(cost.getByLabel("TP", { exact: true })).toHaveValue("1");
  await expect(cost.getByLabel("节点 / decode", { exact: true })).toHaveValue("1");
  await number(cost, "TP", 4);
  await cost.getByRole("button", { name: "Prefill", exact: true }).click();
  await expect(cost.getByLabel("TP", { exact: true })).toHaveValue("2");
  await cost.getByRole("button", { name: "恢复默认部署", exact: true }).click();
  await expect(cost.locator(".pd-deployment-summary")).toContainText("Prefill · 1");
  await expect(cost.locator(".pd-deployment-summary")).toContainText("Decode · 1");
  await expect(cost.getByLabel("TP", { exact: true })).toHaveValue("1");
  await number(cost, "TP", 2);
  await page.getByRole("button", { name: "模型选项", exact: true }).click();
  await page.locator(".drawer .compact-list button").filter({ hasText: /^deepseek-ai\/DeepSeek-V4.1-Flash/ }).click();
  await expect(page.locator(".detail-model-id")).toHaveText("deepseek-ai/DeepSeek-V4.1-Flash", { timeout: 30_000 });
  await expect(cost.getByLabel("TP", { exact: true })).toHaveValue("8");
  await expect(cost.locator(".cost-default-deployment")).toHaveAttribute("data-mode", "auto");
  await cost.screenshot({ path: testInfo.outputPath("single-node-defaults.png") });
});

test("Qwen draft allocation 与 Total VRAM / Fit / Max Context 同源", async ({ page }, testInfo) => {
  const cost = await openCost(page, "Qwen/Qwen3.5-4B");
  await number(cost, "输入 tokens / request", 2100000);
  const draft = cost.locator('[data-owner="draft"]');
  await expect.poll(async () => Number(await draft.getAttribute("data-bytes"))).toBeGreaterThan(0);
  await expect(cost.locator(".cost-machine-summary .no-fit")).toBeVisible();
  // Changing the workload should reveal no-fit, not silently grow the plan.
  await expect(cost.getByLabel("TP", { exact: true })).toHaveValue("1");
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

test("SGLang 显式 speculative workload 进入 scratch、Fit、Max Context，PD 仅传输持久 state", async ({ page }) => {
  const cost = await openCost(page, "Qwen/Qwen3.5-4B", "sglang");
  await openAssumptions(cost);
  const state = cost.locator(".cost-breakdown > span").filter({ hasText: "KDA 状态" }).locator("b");
  const maxContext = cost.locator(".cost-metrics > span").filter({ hasText: "Max context" }).locator("b");
  const initialMaxContext = Number((await maxContext.innerText()).replaceAll(",", ""));
  await expect(cost.getByText("投机 scratch", { exact: true })).toHaveCount(0);
  await number(cost, "投机 draft tokens", 2);
  await expect(cost.locator(".cost-assumptions")).toContainText("投机状态 scratch");
  await number(cost, "投机有效请求容量", 4);
  await expect(cost.getByText("投机 scratch", { exact: true })).toBeVisible();
  expect(Number((await maxContext.innerText()).replaceAll(",", ""))).toBeLessThan(initialMaxContext);
  await expect(cost.locator(".cost-assumptions")).toContainText("sglang");
  await expect(state).toBeVisible();
  await cost.getByRole("button", { name: "PD 分离", exact: true }).click();
  const pd = cost.locator(".pd-summary-modern");
  await expect(pd).toBeVisible();
  await expect(cost.getByText("投机 scratch", { exact: true })).toHaveCount(0);
  await cost.getByRole("button", { name: "Decode", exact: true }).click();
  await expect(cost.getByText("投机 scratch", { exact: true })).toBeVisible();
  const transferWithScratch = await pd.locator("span").first().innerText();
  await number(cost, "投机有效请求容量", 1000);
  await expect(pd).toContainText("Prefill 适配 是 · Decode 适配 否");
  await expect(maxContext).toHaveText("0");
  await expect(pd.locator("span").first()).toHaveText(transferWithScratch);
  await number(cost, "投机 draft tokens", 0);
  await number(cost, "投机有效请求容量", 0);
  await expect(pd.locator("span").first()).toHaveText(transferWithScratch);
  await expect(pd).toContainText("Prefill 适配 是 · Decode 适配 是");
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
  await openAssumptions(cost);
  const kv = cost.locator('[data-owner="main"]');
  const before = await kv.getAttribute("data-bytes");
  await expect(cost.getByText("仅作为 fallback：", { exact: false })).toBeVisible();
  await cost.getByRole("combobox", { name: "默认 KV bytes / element", exact: true }).selectOption("1");
  await expect(kv).toHaveAttribute("data-bytes", before);
});

// 每个用例处理 10 个模型，限制慢机器上的失败/重试范围；两个 Chrome 项目
// 仍覆盖全部 60 个模型，不是抽样。
for (let batchIndex = 0; batchIndex < 6; batchIndex += 1) {
test(`Chrome 全量 60 模型成本与图扫描（桌面和移动） ${batchIndex * 10 + 1}-${batchIndex * 10 + 10}`, async ({ page }, testInfo) => {
  test.setTimeout(600_000);
  await expect(page.locator("datalist#builtin-models option")).toHaveCount(60, { timeout: 60_000 });
  const models = await page.locator("datalist#builtin-models option").evaluateAll((els) => els.map((el) => el.value));
  expect(models).toHaveLength(60);
  const results = [];
  for (const model of models.slice(batchIndex * 10, batchIndex * 10 + 10)) {
    await page.getByLabel("model id").fill(model);
    await page.getByRole("button", { name: "打开模型", exact: true }).click();
    // 冷启动 Vite lazy chunk + ELK layout 以真实可见性作为 readiness，不使用固定 sleep。
    await expect(page.locator(".react-flow__node").first()).toBeVisible({ timeout: 30_000 });
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
    const status = await page.locator(".diagram-lens-status").innerText();
    expect(status, model).toMatch(/TP[1248] \/ PP1 \/ EP1 \/ DP1/);
    results.push({ model, kv, deployment: status, bound: await cost.locator("[data-bound]").getAttribute("data-bound") });
    await page.getByRole("button", { name: /Model Structure Viewer v/ }).click();
  }
  const output = testInfo.outputPath("all-model-accounting.json");
  await writeFile(output, JSON.stringify({ project: testInfo.project.name, catalogTotal: models.length, batchIndex, total: results.length, results }, null, 2));
  await testInfo.attach("all-model-accounting", { path: output, contentType: "application/json" });
});
}
