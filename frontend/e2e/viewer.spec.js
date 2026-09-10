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

  // W6-2（§2.2）：evidence 数据契约上 DOM。折叠态下只有顶层 module-order 序列边，
  // 展开内层模块后 declared 声明边出现——两类类名互异。
  await expect.poll(async () => page.locator('path[data-evidence="module-order"]').count()).toBeGreaterThan(0);
  await expect.poll(async () => page.locator(".react-flow__edge title").count()).toBeGreaterThan(0);

  await page.locator(".detail-cost-toggle > button").click();
  await expect(page.getByText("Total VRAM", { exact: false })).toBeVisible();
  await expect(page.getByText("MACs / forward", { exact: false })).toBeVisible();

  const overflow = await page.evaluate(() => ({
    width: document.documentElement.scrollWidth,
    viewport: document.documentElement.clientWidth,
  }));
  expect(overflow.width).toBeLessThanOrEqual(overflow.viewport + 1);
});

test("多模态模型图包含视觉塔和视觉投影路径", async ({ page }) => {
  await expect(page.locator('datalist#builtin-models option[value="Qwen/Qwen3.6-27B"]')).toHaveCount(1);
  await page.getByLabel("model id").fill("Qwen/Qwen3.6-27B");
  await page.getByRole("button", { name: "打开模型" }).click();

  await expect(page.locator(".detail-page")).toBeVisible();
  const vision = page.locator(".react-flow__node").filter({ hasText: "Vision Tower" }).first();
  await expect(vision).toBeVisible();
  await vision.locator("button").first().click();
  await expect(page.locator(".react-flow__node").filter({ hasText: "Vision Merger" })).toBeVisible();
});

test("每个内置模型都能展开父节点并保持可计算图", async ({ page }, testInfo) => {
  test.skip(testInfo.project.name !== "desktop-chrome", "全量内置模型回归只在桌面浏览器运行");
  test.setTimeout(600_000);
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
    // 语义断言（M11-P0-1）：图内不得出现空标题占位节点。旧断言
    // not.toMatch(/unknown/i) 惩罚诚实展示的"未知"标签，已废弃。
    const titles = await page.locator(".rf-node-title").allTextContents();
    expect(titles.filter((title) => !title.trim())).toHaveLength(0);

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
      await expect(page.locator(".react-flow__node").filter({ hasText: "SDPA attention" }).first()).toBeVisible();
      const sdpa = page.locator(".rf-node-content").filter({ hasText: "SDPA attention" }).first();
      const sdpaExpand = sdpa.getByRole("button", { name: "展开", exact: true });
      if (await sdpaExpand.count()) await sdpaExpand.click();
      await expect(page.locator(".react-flow__node").filter({ hasText: "vision attention scores" }).first()).toBeVisible();
      const hasInternalMerger = /^(Qwen\/Qwen3\.5|Qwen\/Qwen3\.6|Qwen\/Qwen3\.8-|zai-org\/GLM-5\.3-Flash)/.test(modelId);
      // DeepSeek V4 Flash Vision 用的是**扁平** vision 配置（顶层 vision_*），
      // 视觉塔输出 1024 与文本 hidden 4096 不同宽，必然有一层视觉→文本投影。
      // 此前判成「无投影器」，结构树里整层缺失（权重字节恒等式差 1024×4096）。
      const hasExternalProjector = /^(MiniMaxAI\/|moonshotai\/Kimi|deepseek-ai\/DeepSeek-V4-Flash-Vision)/.test(modelId);
      const hasMerger = await page.getByText("Vision Merger", { exact: true }).count() > 0;
      const hasProjector = await page.getByText("Multi-modal Projector", { exact: true }).count() > 0;
      expect(hasMerger).toBe(hasInternalMerger);
      expect(hasProjector).toBe(hasExternalProjector);
    }
    const costToggle = page.locator(".detail-cost-toggle > button");
    if (await costToggle.count()) {
      await costToggle.click();
      const costPanel = page.locator(".cost-summary");
      await expect(costPanel.getByText("MACs / forward", { exact: false })).toBeVisible();
      // 语义断言（M11-P0-1）：内置模型聚合 bound 必须可分类，不得为 unknown
      // （data-bound 契约，扩展 W6 的 data-evidence 先例）。旧断言
      // not.toMatch(/unknown/i) 惩罚诚实展示，已废弃。
      const roofline = costPanel.locator(".cost-metrics [data-bound]");
      await expect(roofline).toBeVisible();
      expect(await roofline.getAttribute("data-bound")).not.toBe("unknown");
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
  await expect(page.locator(".rf-node-content").filter({ hasText: "SDPA attention" }).first()).toBeVisible();
  await expect(page.locator(".rf-node-content").filter({ hasText: "attention scores" })).toHaveCount(0);
  await page.locator(".rf-node-content").filter({ hasText: "SDPA attention" }).first().getByRole("button", { name: "展开", exact: true }).click();
  await expect(page.locator(".rf-node-content").filter({ hasText: "attention scores" }).first()).toBeVisible();

  // 展开到算子层后，builder 声明的 dataflow 边（declared）在场且与推断边类名互异
  await expect.poll(async () => page.locator('path[data-evidence="declared"]').count()).toBeGreaterThan(0);
  const declaredClass = await page.locator('path[data-evidence="declared"]').first().getAttribute("class");
  const orderClass = await page.locator('path[data-evidence="module-order"]').first().getAttribute("class");
  expect(declaredClass).not.toEqual(orderClass);

  const operator = page.locator(".rf-node-content").filter({ hasText: "QKV projection" }).first();
  await operator.click();
  await expect(page.locator(".child-modules-section")).toHaveCount(0);
  await expect(page.locator(".inspector-disclosure").filter({ hasText: "Shape / Tensor" })).toBeVisible();
});
