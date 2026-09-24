import { expect, test } from "./fixtures.js";
import fs from "node:fs";
import { buildStructureFromArtifacts } from "../src/structure/buildStructure.js";

const read = url => JSON.parse(fs.readFileSync(url, "utf8"));
const models = ["deepseek-ai/DeepSeek-V3.2", "zai-org/GLM-5", "zai-org/GLM-5.1",
  "zai-org/GLM-5.2", "zai-org/GLM-5.2-FP8", "zai-org/GLM-5.3", "zai-org/GLM-5.3-BF16",
  "zai-org/GLM-5.3-Flash", "zai-org/GLM-5.3-Flash-BF16"];

for (const modelId of models) {
  test(`physical DSA indexer fan-in renders in SVG: ${modelId}`, async ({ page }, testInfo) => {
    test.setTimeout(120_000);
    const dir = new URL(`../../models/${modelId}/`, import.meta.url);
    const { graph } = buildStructureFromArtifacts({
      modelId, config: read(new URL("config.json", dir)),
      checkpointTruth: read(new URL("header-truth.json", dir)),
      sourceRef: read(new URL("source-ref.json", dir)),
    });
    const indexer = graph.nodes.find(n =>
      ["dsa_indexer", "dsa_kpool_indexer"].includes(n.attributes?.operator_id)
      && n.canonical_id.includes(".self_attn."));
    expect(indexer).toBeTruthy();
    const prefix = indexer.canonical_id;
    const pairs = [
      [`${prefix}.wk`, `${prefix}.k_norm`],
      [`${prefix}.wq_b`, prefix],
      [`${prefix}.k_norm`, prefix],
      [`${prefix}.weights_proj`, prefix],
    ];
    await page.route(/https:\/\/(?:www\.)?(?:huggingface\.co|hf-mirror\.com|modelscope\.cn)\//,
      route => route.abort());
    await page.goto("/");
    await page.getByLabel("model id").fill(modelId);
    await page.getByRole("button", { name: "打开模型", exact: true }).click();
    await expect(page.locator(".react-flow-diagram")).toHaveAttribute("data-graph-version", "2");
    const before = await page.locator(".react-flow__edge").count();
    await page.getByRole("button", { name: "展开全部", exact: true }).click();
    for (const [from, to] of pairs) {
      const edge = graph.edges.find(e => e.source_canonical_id === from && e.target_canonical_id === to);
      expect(edge, `${from} -> ${to}`).toBeTruthy();
      await expect(page.locator(`.react-flow__edge[data-id="${edge.id}"] path.react-flow__edge-path`))
        .toHaveAttribute("d", /^M/, { timeout: 60_000 });
    }
    await expect.poll(() => page.locator(".react-flow__edge").count()).toBeGreaterThan(before);
    if (modelId === "zai-org/GLM-5.2") {
      await expect(page.locator(".react-flow__edge title").filter({ hasText: "IndexShare" }).first())
        .toContainText("IndexShare");
    }
    await page.screenshot({ path: testInfo.outputPath("dsa-physical-indexer.png") });
    await page.getByRole("button", { name: "收起全部", exact: true }).click();
    await page.locator(".detail-cost-toggle > button").click();
    await expect(page.locator(".cost-summary")).toContainText("MACs / forward");
    await page.getByRole("button", { name: "中 / EN" }).click();
    await expect(page.locator(".react-flow-diagram")).toHaveAttribute("data-graph-version", "2");
  });
}
