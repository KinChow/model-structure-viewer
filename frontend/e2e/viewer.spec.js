import { expect, test } from "@playwright/test";

test.beforeEach(async ({ page }) => {
  await page.route("**/api/settings", (route) => route.fulfill({
    json: {
      model_root: "",
      hf_endpoint: "https://huggingface.co",
      cache_policy: "prefer-local",
      offline: false,
    },
  }));
  await page.route("**/api/models", (route) => route.fulfill({ json: [] }));
  await page.route(/https:\/\/(?:www\.)?(?:huggingface\.co|modelscope\.cn)\//, (route) => route.abort());
  await page.goto("/");
});

test("首页只提供明确的模型来源", async ({ page }) => {
  await expect(page).toHaveTitle("Model Structure Viewer");
  await expect(page.getByRole("heading", { name: "理解模型" })).toBeVisible();
  await expect(page.getByLabel("model source")).toHaveValue("huggingface");
  await expect(page.getByLabel("model source").locator('option[value="auto"]')).toHaveCount(0);
});

test("内置模型以 React Flow 图打开并保留成本交互", async ({ page }) => {
  await expect(page.locator('datalist#builtin-models option[value="MiniMaxAI/MiniMax-M3"]')).toHaveCount(1);
  await page.getByLabel("model id").fill("MiniMaxAI/MiniMax-M3");
  await page.getByRole("button", { name: "打开模型" }).click();

  await expect(page.locator(".detail-page")).toBeVisible();
  await expect(page.locator(".detail-brand")).toHaveText("Model Structure Viewer v0.2.0");
  await expect(page.locator(".react-flow__node").first()).toBeVisible();
  await expect(page.locator(".react-flow-diagram")).toHaveAttribute("data-graph-version", "2");
  await expect.poll(() => page.locator(".react-flow__node").count()).toBeGreaterThan(3);
  await expect.poll(() => page.locator(".react-flow__edge").count()).toBeGreaterThan(1);
  await expect(page.locator(".react-flow__minimap")).toBeVisible();

  await page.locator(".detail-cost-toggle > button").click();
  await expect(page.getByText("Total VRAM", { exact: false })).toBeVisible();
  await expect(page.getByText("MACs / forward", { exact: false })).toBeVisible();

  const overflow = await page.evaluate(() => ({
    width: document.documentElement.scrollWidth,
    viewport: document.documentElement.clientWidth,
  }));
  expect(overflow.width).toBeLessThanOrEqual(overflow.viewport + 1);
});

test("多模态模型图包含视觉塔和投影节点", async ({ page }) => {
  await expect(page.locator('datalist#builtin-models option[value="Qwen/Qwen3.6-27B"]')).toHaveCount(1);
  await page.getByLabel("model id").fill("Qwen/Qwen3.6-27B");
  await page.getByRole("button", { name: "打开模型" }).click();

  await expect(page.locator(".detail-page")).toBeVisible();
  await expect(page.locator(".react-flow__node").filter({ hasText: "Vision Tower" })).toBeVisible();
  await expect(page.locator(".react-flow__node").filter({ hasText: "Multi-modal Projector" })).toBeVisible();
});

test("每个内置模型都能展开父节点并保持可计算图", async ({ page }, testInfo) => {
  test.skip(testInfo.project.name !== "desktop-chrome", "全量内置模型回归只在桌面浏览器运行");
  test.setTimeout(300_000);
  const modelIds = await page.locator("datalist#builtin-models option").evaluateAll((options) => options.map((option) => option.value));
  expect(modelIds).toHaveLength(59);

  for (const modelId of modelIds) {
    await page.getByLabel("model id").fill(modelId);
    await page.getByRole("button", { name: "打开模型" }).click();
    await expect(page.locator(".detail-page")).toBeVisible();
    await expect(page.locator(".react-flow-diagram")).toHaveAttribute("data-graph-version", "2");
    const diagram = page.locator(".react-flow-diagram");
    await expect.poll(() => page.locator(".react-flow__node").count()).toBeGreaterThan(3);
    await expect.poll(() => page.locator(".react-flow__edge").count()).toBeGreaterThan(1);
    expect(await diagram.innerText()).not.toMatch(/\bUNKNOWN\b|\bunknown\b/);

    const decoder = page.locator(".rf-node-content").filter({ hasText: /Decoder Layers|Text Decoder Layers/ }).first();
    const expand = decoder.getByRole("button", { name: "展开", exact: true });
    if (await expand.count()) {
      const before = await page.locator(".react-flow__edge").count();
      await expand.click();
      await expect(page.locator(".react-flow__node").filter({ hasText: "Decoder layer group" }).first()).toBeVisible();
      await expect.poll(() => page.locator(".react-flow__edge").count()).toBeGreaterThanOrEqual(before);
    }
    const vision = page.locator(".react-flow__node").filter({ hasText: "Vision Tower" }).first();
    if (await vision.count()) {
      const visionExpand = vision.getByRole("button", { name: "展开", exact: true });
      if (await visionExpand.count()) await visionExpand.click();
      const visionLayer = page.locator('.react-flow__node[data-id="root.0.2"]').first();
      await visionLayer.locator("button").first().click();
      await expect(page.getByTestId("rf__node-root.0.2.3").getByText("vision attention scores", { exact: true })).toBeVisible();
    }
    await page.getByRole("button", { name: /Model Structure Viewer v/ }).click();
    await expect(page.getByLabel("model id")).toBeVisible();
  }
});

test("父节点详情提供子模块和 Shape", async ({ page }) => {
  await page.getByLabel("model id").fill("MiniMaxAI/MiniMax-M3");
  await page.getByRole("button", { name: "打开模型" }).click();
  const decoder = page.locator(".rf-node-content").filter({ hasText: "Decoder Layers" }).first();
  await decoder.getByRole("button", { name: "展开", exact: true }).click();
  const edgesAfterDecoder = await page.locator(".react-flow__edge").count();
  await page.locator(".rf-node-content").filter({ hasText: "Decoder layer group" }).first().getByRole("button", { name: "展开", exact: true }).click();
  await expect.poll(() => page.locator(".react-flow__edge").count()).toBeGreaterThan(edgesAfterDecoder);
  await page.locator(".rf-node-content").filter({ hasText: "GQA Attention" }).first().getByRole("button", { name: "展开", exact: true }).click();

  const operator = page.locator(".rf-node-content").filter({ hasText: "QKV projection" }).first();
  await operator.click();
  await expect(page.locator(".child-modules-section")).toHaveCount(0);
  await expect(page.locator(".inspector-disclosure").filter({ hasText: "Shape / Tensor" })).toBeVisible();
});
