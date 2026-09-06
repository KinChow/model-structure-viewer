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
