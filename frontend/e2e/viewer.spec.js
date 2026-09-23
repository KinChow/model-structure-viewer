import { expect, test } from "./fixtures.js";
import { readFileSync } from "node:fs";

// 品牌版本号跟随 package.json，避免每次版本升级都要手改断言。
const appVersion = JSON.parse(readFileSync(new URL("../package.json", import.meta.url), "utf8")).version;

test.beforeEach(async ({ page }) => {
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
  await expect(page.locator(".detail-brand")).toHaveText(`Model Structure Viewer v${appVersion}`);
  await expect(page.locator(".react-flow__node").first()).toBeVisible();
  await expect(page.locator(".react-flow-diagram")).toHaveAttribute("data-graph-version", "2");
  await expect.poll(() => page.locator(".react-flow__node").count()).toBeGreaterThan(3);
  await expect.poll(() => page.locator(".react-flow__edge").count()).toBeGreaterThan(1);
  if (page.viewportSize().width <= 640) await expect(page.locator(".react-flow__minimap")).toBeHidden();
  else await expect(page.locator(".react-flow__minimap")).toBeVisible();
  await expect(page.getByRole("button", { name: "用 Transformers 校验" })).toHaveCount(0);
  await expect(page.locator(".diagnostics-meta")).toContainText("张量");

  // W6-2（§2.2）：evidence 数据契约上 DOM。顶层主干（embed→decoder→norm→lm_head）
  // 是真实顺序数据流，声明为 declared 实线边（无草稿模型也统一实线，见
  // networkSpecWithDraft）；折叠态下即可在 DOM 上看到 declared 边，evidence 契约生效。
  await expect.poll(async () => page.locator('path[data-evidence="declared"]').count()).toBeGreaterThan(0);
  await expect.poll(async () => page.locator(".react-flow__edge title").count()).toBeGreaterThan(0);

  await page.locator(".detail-cost-toggle > button").click();
  await expect(page.getByText("Total VRAM", { exact: false })).toBeVisible();
  await expect(page.getByText("MACs / forward", { exact: false })).toBeVisible();
  await expect(page.locator('body')).not.toContainText('[object Object]');

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
  await vision.getByRole('button', {name: '展开', exact: true}).click();
  await expect(page.locator(".react-flow__node").filter({ hasText: "Vision Merger" })).toBeVisible();
});

test("每个内置模型都能展开父节点并保持可计算图", async ({ page }, testInfo) => {
  test.skip(testInfo.project.name !== "desktop-chrome", "全量内置模型回归只在桌面浏览器运行");
  test.setTimeout(600_000);
  page.setDefaultTimeout(10_000);
  // 展开不再自动 fit；通过页面实际控件把下一操作目标移回画布后再点击。
  // ref: https://playwright.dev/docs/actionability — 不用 force 跳过遮挡检查。
  const fitCanvas = () => page.locator(".react-flow__controls-fitview").click();
  const inCanvasViewport = (locator) => locator.evaluate((element) => {
    const node = element.getBoundingClientRect();
    const canvas = element.closest(".react-flow")?.getBoundingClientRect();
    return Boolean(canvas && node.width > 0 && node.height > 0
      && node.right > canvas.left && node.left < canvas.right
      && node.bottom > canvas.top && node.top < canvas.bottom);
  });
  const modelIds = await page.locator("datalist#builtin-models option").evaluateAll((options) => options.map((option) => option.value));
  expect(modelIds).toHaveLength(60);

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
      await fitCanvas();
      await expand.click();
      await expect(page.locator(".react-flow__node").filter({ hasText: /\(DecoderLayer\)/ }).first()).toBeVisible();
      await expect.poll(() => page.locator(".react-flow__edge").count()).toBeGreaterThanOrEqual(before);
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
  await page.locator(".rf-node-content").filter({ hasText: /\(DecoderLayer\)/ }).first().getByRole("button", { name: "展开", exact: true }).click();
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
  await expect(operator).toBeVisible();
  await page.locator('.react-flow__controls-fitview').click();
  await expect(operator).toBeInViewport();
  // React Flow pans with CSS transform. Playwright's default click then
  // scrollIntoViewIfNeeded() fights that pane and reports "outside of the viewport".
  // locator.click({force: true}) is the documented bypass for actionability.
  await operator.click({force: true});
  await expect(page.locator(".child-modules-section")).toHaveCount(0);
  await expect(page.locator(".inspector-disclosure").filter({ hasText: "Shape / Tensor" })).toBeVisible();
});

test("桌面对比模式保留两张可见 React Flow 画布", async ({ page }, testInfo) => {
  test.skip(testInfo.project.name !== "desktop-chrome", "桌面对比画布回归");
  await page.getByLabel("model id").fill("deepseek-ai/DeepSeek-V3.1");
  await page.getByRole("button", { name: "打开模型" }).click();
  await page.locator(".detail-page").waitFor();
  await page.locator(".detail-cost-toggle > button").click();
  const cost = page.locator(".cost-summary");
  await cost.getByRole("button", { name: "展开配置", exact: true }).click();
  for (const label of ["芯片", "方案"]) {
    await cost.locator(".cost-segmented button").filter({ hasText: label }).click();
    const panes = page.locator(".diagram-compare-pane .diagram-frame");
    await expect(panes).toHaveCount(2);
    await expect.poll(() => panes.nth(0).evaluate((element) => element.getBoundingClientRect().height)).toBeGreaterThanOrEqual(400);
    await expect.poll(() => panes.nth(1).evaluate((element) => element.getBoundingClientRect().height)).toBeGreaterThanOrEqual(400);
    await expect.poll(() => page.locator(".diagram-compare-pane .react-flow__node").count()).toBeGreaterThan(0);
  }
});

test("短桌面窗口展开公式索引时保留完整控件高度", async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 800 });
  await page.getByLabel("model id").fill("deepseek-ai/DeepSeek-V3.1");
  await page.getByRole("button", { name: "打开模型" }).click();
  await page.locator(".detail-page").waitFor();
  const strip = page.locator(".formula-strip");
  await strip.getByRole("button", { name: "公式索引" }).click();
  await expect.poll(() => strip.boundingBox().then((box) => box?.height || 0)).toBeGreaterThanOrEqual(100);
  await expect(strip.locator(".formula-strip-links")).toBeVisible();
});

test("短桌面视口保留画布最小高度且不产生横向溢出", async ({ page }) => {
  // 短屏时应整页纵向滚动而不是把画布压扁：校验画布 min-height 与无横向溢出，
  // 而非旧断言的“文档不超过视口高度”（该旧行为正是被压扁的 bug）。
  for (const viewport of [{ width: 1000, height: 750 }, { width: 1280, height: 800 }]) {
    await page.setViewportSize(viewport);
    await page.getByLabel("model id").fill("deepseek-ai/DeepSeek-V3.1");
    await page.getByRole("button", { name: "打开模型" }).click();
    await page.locator(".diagram-frame").waitFor();
    const size = await page.evaluate(() => ({
      clientWidth: document.documentElement.clientWidth,
      scrollWidth: document.documentElement.scrollWidth,
      frameHeight: Math.round(document.querySelector(".diagram-frame").getBoundingClientRect().height),
    }));
    expect(size.scrollWidth).toBeLessThanOrEqual(size.clientWidth + 1);
    expect(size.frameHeight).toBeGreaterThanOrEqual(360);
    await page.getByRole("button", { name: /Model Structure Viewer v/ }).click();
  }
});

test("Cost 展开后 Formula 仍可点击，桌面移动端都不覆盖内容", async ({ page }, testInfo) => {
  await page.getByLabel("model id").fill("deepseek-ai/DeepSeek-V3.1");
  await page.getByRole("button", { name: "打开模型" }).click();
  await page.locator(".diagram-frame").waitFor();
  await page.locator(".detail-cost-toggle > button").click();
  const formula = page.locator(".formula-strip");
  await expect(formula).toBeVisible();
  await formula.getByRole("button", { name: "公式索引" }).click();
  await expect(formula.locator(".formula-strip-links")).toBeVisible();
  const size = await page.evaluate(() => ({ viewport: innerHeight, document: document.documentElement.scrollHeight }));
  if (testInfo.project.name === "mobile-chrome") expect(size.document).toBeGreaterThanOrEqual(size.viewport);
});

test("成本并行度 TP 输入可整段清空，失焦回退到 1 且合法输入即时生效", async ({ page }, testInfo) => {
  test.skip(testInfo.project.name !== "desktop-chrome", "成本输入交互回归只在桌面浏览器运行");
  await page.getByLabel("model id").fill("deepseek-ai/DeepSeek-V3.1");
  await page.getByRole("button", { name: "打开模型" }).click();
  await page.locator(".detail-page").waitFor();
  await page.locator(".detail-cost-toggle > button").click();
  const cost = page.locator(".cost-summary");
  await cost.getByRole("button", { name: "展开配置", exact: true }).click();
  const tp = cost.locator("label").filter({ hasText: /^TP/ }).locator("input").first();
  await tp.scrollIntoViewIfNeeded();
  // 编辑期允许整段清空（旧行为会立刻回填成 1，用户无法直接输入两位数）。
  await tp.click();
  await tp.press("ControlOrMeta+a");
  await tp.press("Backspace");
  await expect(tp).toHaveValue("");
  // 直接输入两位数不再被前导 1 污染。
  await tp.type("16");
  await expect(tp).toHaveValue("16");
  // 清空后失焦回退到最小值 1，不产生非法态。
  await tp.press("ControlOrMeta+a");
  await tp.press("Backspace");
  await tp.blur();
  await expect(tp).toHaveValue("1");
  // 合法输入提交后派生 World size 实时更新。
  await tp.click();
  await tp.press("ControlOrMeta+a");
  await tp.press("Backspace");
  await tp.type("8");
  await tp.blur();
  await expect(tp).toHaveValue("8");
  await expect(cost.locator("label").filter({ hasText: /World size|世界大小/ }).locator("output")).toHaveText("8");
});

test("宽屏桌面按视口比例撑开画布且只有单一滚动区域", async ({ page }, testInfo) => {
  test.skip(testInfo.project.name !== "desktop-chrome", "宽屏布局回归只在桌面浏览器运行");
  await page.setViewportSize({ width: 1920, height: 1080 });
  await page.getByLabel("model id").fill("deepseek-ai/DeepSeek-V3.1");
  await page.getByRole("button", { name: "打开模型" }).click();
  await page.locator(".detail-page").waitFor();
  await page.locator(".diagram-frame").first().waitFor();
  const metrics = await page.evaluate(() => {
    const layout = document.querySelector(".detail-layout").getBoundingClientRect();
    const frame = document.querySelector(".diagram-frame").getBoundingClientRect();
    const scrollers = [];
    document.querySelectorAll("*").forEach((el) => {
      const s = getComputedStyle(el);
      if ((s.overflowY === "auto" || s.overflowY === "scroll") && el.scrollHeight > el.clientHeight + 2) {
        scrollers.push(el.className?.toString?.().slice(0, 40) || el.tagName);
      }
    });
    return {
      layoutWidth: Math.round(layout.width),
      frameWidth: Math.round(frame.width),
      docScroll: document.documentElement.scrollHeight > document.documentElement.clientHeight + 2,
      innerScrollers: scrollers,
      overflowX: document.documentElement.scrollWidth - document.documentElement.clientWidth,
    };
  });
  // 布局按视口比例撑开（不再固定收缩到 ~1009px），画布主区拿到大部分宽度。
  expect(metrics.layoutWidth).toBeGreaterThan(1500);
  expect(metrics.frameWidth).toBeGreaterThan(1000);
  // 无横向溢出。
  expect(metrics.overflowX).toBeLessThanOrEqual(1);
  // 单一滚动区域：高视口下 app-shell 锁高，既无整页滚动条也无并存的内部滚动条。
  expect(metrics.docScroll).toBe(false);
  expect(metrics.innerScrollers.length).toBeLessThanOrEqual(1);
});

test("高视口下 Inspector 详情不被裁到视口外且可内部滚动查看全部属性", async ({ page }, testInfo) => {
  test.skip(testInfo.project.name !== "desktop-chrome", "Inspector 裁剪回归只在桌面浏览器运行");
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.getByLabel("model id").fill("deepseek-ai/DeepSeek-V3.1");
  await page.getByRole("button", { name: "打开模型" }).click();
  await page.locator(".detail-page").waitFor();
  await page.locator(".diagram-frame").first().waitFor();
  // 跳到一个属性较多的算子节点（公式索引里的 matmul），填满 Inspector。
  await page.locator(".formula-strip").getByRole("button", { name: "公式索引" }).click();
  const links = page.locator(".formula-strip-links button");
  const count = await links.count();
  let jumped = false;
  for (let i = 0; i < count; i++) {
    if ((await links.nth(i).textContent())?.includes("matmul")) { await links.nth(i).click(); jumped = true; break; }
  }
  if (!jumped && count) await links.first().click();
  await page.waitForTimeout(300);
  const metrics = await page.evaluate(() => {
    const panel = document.querySelector(".detail-inspector-slot .detail-panel")
      || document.querySelector(".detail-inspector-slot .model-inspector-summary");
    const r = panel.getBoundingClientRect();
    return {
      bottom: Math.round(r.bottom),
      viewportH: window.innerHeight,
      overflowsInternally: panel.scrollHeight > panel.clientHeight + 2,
      overflowY: getComputedStyle(panel).overflowY,
    };
  });
  // 高视口 app-shell 锁高时，面板底边不得越过视口（否则底部内容不可达）。
  expect(metrics.bottom).toBeLessThanOrEqual(metrics.viewportH + 1);
  // 内容超长时面板自身可滚动，保证全部属性可查看。
  expect(metrics.overflowY).toBe("auto");
});
