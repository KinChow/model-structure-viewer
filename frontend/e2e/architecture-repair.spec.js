import { expect, test } from "./fixtures.js";
import fs from "node:fs";
import { buildStructureFromArtifacts } from "../src/structure/buildStructure.js";

const read = url => fs.existsSync(url) ? JSON.parse(fs.readFileSync(url, "utf8")) : null;
for (const model of ["Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.8-Flash-Next"]) {
  test(`Qwen gate fan-in rendered through expansion and collapse: ${model}`, async ({ page }, testInfo) => {
    test.setTimeout(120_000);
    const dir = new URL(`../../models/${model}/`, import.meta.url);
    const structure = buildStructureFromArtifacts({ modelId: model,
      config: read(new URL("config.json", dir)),
      checkpointTruth: read(new URL("skeleton-truth.json", dir)) || read(new URL("header-truth.json", dir)),
      sourceRef: read(new URL("source-ref.json", dir)) });
    await page.route(/https:\/\/(?:www\.)?(?:huggingface\.co|hf-mirror\.com|modelscope\.cn)\//, route => route.abort());
    await page.goto("/");
    await page.getByLabel("model id").fill(model);
    await page.getByRole("button", { name: "打开模型", exact: true }).click();
    await expect(page.locator(".react-flow-diagram")).toHaveAttribute("data-graph-version", "2");
    const graph = structure.graph;
    const labels = new Set(graph.edges.filter(e => e.label).map(e => e.label));
    expect(labels.has("z") || labels.has("gate")).toBeTruthy();
    const before = await page.locator(".react-flow__edge").count();
    await page.getByRole("button", { name: "展开全部", exact: true }).click();
    await expect.poll(() => page.locator(".react-flow__edge").count()).toBeGreaterThanOrEqual(before);
    const titles = page.locator(".react-flow__edge title");
    for (const label of labels) await expect(titles.filter({ hasText: label }).first()).toContainText(label);
    await page.screenshot({ path: testInfo.outputPath("expanded-gate-inputs.png") });
    await page.getByRole("button", { name: "收起全部", exact: true }).click();
    await expect.poll(() => page.locator(".react-flow__edge").count()).toBeLessThanOrEqual(before);
    await page.getByRole("button", { name: "展开全部", exact: true }).click();
    for (const label of labels) await expect(page.locator(".react-flow__edge title").filter({ hasText: label }).first()).toContainText(label);
    await page.getByRole("button", { name: "中 / EN" }).click();
    for (const label of labels) await expect(page.locator(".react-flow__edge title").filter({ hasText: label }).first()).toContainText(label);
    await page.locator(".detail-cost-toggle > button").click();
    await expect(page.locator(".cost-summary")).toContainText("MACs / forward");
  });
}

test("GLM IndexShare cross-layer top-k relation is rendered: zai-org/GLM-5.2", async ({ page }, testInfo) => {
  test.setTimeout(120_000);
  const model = "zai-org/GLM-5.2";
  const dir = new URL(`../../models/${model}/`, import.meta.url);
  const structure = buildStructureFromArtifacts({
    modelId: model,
    config: read(new URL("config.json", dir)),
    checkpointTruth: read(new URL("header-truth.json", dir)),
    sourceRef: read(new URL("source-ref.json", dir)),
  });
  expect(structure.graph.edges.filter(edge => edge.relation === "index-reuse")).toHaveLength(19);
  await page.route(/https:\/\/(?:www\.)?(?:huggingface\.co|hf-mirror\.com|modelscope\.cn)\//, route => route.abort());
  await page.goto("/");
  await page.getByLabel("model id").fill(model);
  await page.getByRole("button", { name: "打开模型", exact: true }).click();
  await expect(page.locator(".react-flow-diagram")).toHaveAttribute("data-graph-version", "2");
  await page.getByRole("button", { name: "展开全部", exact: true }).click();
  await expect.poll(() => page.locator(".react-flow__edge").count()).toBeGreaterThan(0);
  await expect(page.locator(".react-flow__edge title").filter({ hasText: "IndexShare" }).first())
    .toContainText("IndexShare");
  await page.screenshot({ path: testInfo.outputPath("expanded-index-share.png") });
  await page.getByRole("button", { name: "中 / EN" }).click();
  await expect(page.locator(".react-flow__edge title").filter({ hasText: "IndexShare" }).first())
    .toContainText("IndexShare");
});
