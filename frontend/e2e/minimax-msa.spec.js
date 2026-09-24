import { expect, test } from "./fixtures.js";
import fs from "node:fs";
import { buildStructureFromArtifacts } from "../src/structure/buildStructure.js";

for (const variant of ["MiniMax-M3", "MiniMax-M3-MXFP8"]) {
  test(`${variant} MSA group selector and sparse attention render`, async ({ page }, testInfo) => {
    test.setTimeout(180000);
    const modelId = `MiniMaxAI/${variant}`;
    const dir = new URL(`../../models/${modelId}/`, import.meta.url);
    const read = file => JSON.parse(fs.readFileSync(new URL(file, dir)));
    const { graph } = buildStructureFromArtifacts({ config: read("config.json"), modelId,
      checkpointTruth: read("header-truth.json"), sourceRef: read("source-ref.json") });
    const errors = [];
    page.on("pageerror", error => errors.push(error.message));
    await page.route(/https:\/\/(?:www\.)?(?:huggingface\.co|hf-mirror\.com|modelscope\.cn)\//, route => route.abort());
    await page.goto("/");
    await page.getByLabel("model id").fill(modelId);
    await page.getByRole("button", { name: "打开模型", exact: true }).click();
    await expect(page.locator(".react-flow-diagram")).toHaveAttribute("data-graph-version", "2", { timeout: 30000 });
    await page.getByRole("button", { name: "展开全部", exact: true }).click();
    const indexer = graph.nodes.find(n => n.attributes?.operator_id === "minimax_sparse_indexer");
    const sparse = graph.nodes.find(n => n.attributes?.operator_id === "minimax_sparse_attention");
    for (const node of [indexer, sparse]) {
      const card = node === indexer
        ? page.locator(`.react-flow__node[data-id="frame-${node.id}"] .rf-group-frame`)
        : page.locator(`.react-flow__node[data-id="${node.id}"] .rf-model-node`);
      await expect(card).toBeVisible({ timeout: 30000 });
      await card.focus(); await page.keyboard.press("Enter");
      await expect(page.locator(".formula-section")).toBeVisible();
      if (node === sparse) await expect(page.locator(".formula-section")).toContainText(/GQA|block|块/);
    }
    const selected = graph.nodes.find(n => n.canonical_id === `${indexer.canonical_id}.valid_block_ids`);
    const edge = graph.edges.find(e => e.source === selected.id && e.target === sparse.id);
    expect(edge).toBeTruthy();
    await expect(page.locator(`.react-flow__edge[data-id="${edge.id}"] path.react-flow__edge-path`))
      .toHaveAttribute("d", /^M/, { timeout: 30000 });
    for (const step of ["group_scores", "block_max", "local_boost", "group_topk"]) {
      const node = graph.nodes.find(n => n.canonical_id === `${indexer.canonical_id}.${step}`);
      await expect(page.locator(`.react-flow__node[data-id="${node.id}"] .rf-model-node`))
        .toBeVisible({ timeout: 30000 });
    }
    await page.getByRole("button", { name: "中 / EN" }).click();
    await page.locator(".detail-cost-toggle > button").click();
    await expect(page.locator(".cost-summary")).toContainText("MACs / forward");
    const download = page.waitForEvent("download");
    await page.getByRole("button", { name: "SVG", exact: true }).click();
    await (await download).saveAs(testInfo.outputPath("msa.svg"));
    await page.screenshot({ path: testInfo.outputPath("msa.png") });
    expect(errors).toEqual([]);
  });
}
