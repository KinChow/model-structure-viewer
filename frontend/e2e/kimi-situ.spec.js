import { expect, test, waitForLayout } from "./fixtures.js";
import fs from "node:fs";
import { buildStructureFromArtifacts } from "../src/structure/buildStructure.js";

test("K3 dense/shared/routed SiTU mechanisms and costs render", async ({ page }, testInfo) => {
  test.setTimeout(180000);
  const raw = JSON.parse(fs.readFileSync(new URL("../../models/moonshotai/Kimi-K3/config.json", import.meta.url)));
  const { graph } = buildStructureFromArtifacts({ config: raw, modelId: "moonshotai/Kimi-K3" });
  const nodes = new Map(graph.nodes.map(n => [n.canonical_id, n]));
  const ids = ["layers.0.mlp.situ_glu", "layers.1.block_sparse_moe.shared_experts.situ_glu", "layers.1.block_sparse_moe.expert_mlp"];
  await page.route(/https:\/\/(?:www\.)?(?:huggingface\.co|hf-mirror\.com|modelscope\.cn)\//, route => route.abort());
  await page.goto("/");
  await page.getByLabel("model id").fill("moonshotai/Kimi-K3");
  await page.getByRole("button", { name: "打开模型", exact: true }).click();
  await expect(page.locator(".react-flow-diagram")).toHaveAttribute("data-graph-version", "2", { timeout: 30000 });
  await page.getByRole("button", { name: "展开全部", exact: true }).click();
  await expect(page.locator(".react-flow-diagram"))
    .toHaveAttribute("data-layout-ready", "true", { timeout: 60_000 });
  for (const id of ids.slice(0, 2)) {
    for (const from of ["gate_proj", "up_proj"]) {
      const source = id.replace(/situ_glu$/, from);
      const edge = graph.edges.find(e => e.source_canonical_id === source && e.target_canonical_id === id);
      expect(edge).toBeTruthy();
      await expect(page.locator(`.react-flow__edge[data-id="${edge.id}"] path.react-flow__edge-path`))
        .toHaveAttribute("d", /^M/, { timeout: 30000 });
    }
  }
  for (const id of ids) {
    const selected = page.locator(`.react-flow__node[data-id="${nodes.get(id).id}"] .rf-model-node`);
    await selected.focus();
    await page.keyboard.press("Enter");
    await expect(selected).toHaveAttribute("aria-selected", "true");
    await expect(page.locator(".formula-section")).toContainText(/SiTU/);
    // KaTeX (pure activation) and ASCII override (fused MLP) both preserve
    // tanh and optional up softcap, never display canonical SwiGLU instead.
    const formula = page.locator(".formula-section");
    const math = formula.locator('[role="math"]');
    if (await math.count()) await expect(math).toHaveAttribute("aria-label", /tanh.*linear_beta/);
    else await expect(formula.locator("code")).toContainText("tanh");
  }
  const shared = page.locator(`.react-flow__node[data-id="${nodes.get(ids[1]).id}"]`);
  for (let step = 0; step < 12 && (await shared.boundingBox()).width < 110; step++) {
    const width = (await shared.boundingBox()).width;
    await page.locator(".react-flow__controls-zoomin").click();
    await expect.poll(async () => (await shared.boundingBox()).width).toBeGreaterThan(width + .01);
  }
  await shared.locator(".rf-model-node").focus();
  await page.keyboard.press("Enter");
  await page.locator(".react-flow-diagram").scrollIntoViewIfNeeded();
  await page.screenshot({ path: testInfo.outputPath("kimi-situ-shared.png") });
  await page.getByRole("button", { name: "收起全部", exact: true }).click();
  await waitForLayout(page);
  await expect(shared).toHaveCount(0);
  await page.getByRole("button", { name: "展开全部", exact: true }).click();
  await expect(page.locator(".react-flow-diagram"))
    .toHaveAttribute("data-layout-ready", "true", { timeout: 60_000 });
  await expect(shared).toHaveCount(1, { timeout: 30000 });
  const search = page.getByPlaceholder("搜索节点名称 / 类型 / class...");
  await search.fill("SiTU-GLU");
  await expect(page.getByRole("option").first()).toBeVisible();
  await search.fill("");
  await shared.locator(".rf-model-node").focus();
  await page.keyboard.press("Enter");
  await page.getByTestId("language-toggle").click();
  await expect(page.locator(".formula-section")).toContainText("hyperparameters, not trainable weights");
  await page.locator(".detail-cost-toggle > button").click();
  await expect(page.locator(".cost-summary")).toContainText("MACs / forward");
  const download = page.waitForEvent("download");
  await page.getByRole("button", { name: "SVG", exact: true }).click();
  await (await download).saveAs(testInfo.outputPath("kimi-situ.svg"));
});
