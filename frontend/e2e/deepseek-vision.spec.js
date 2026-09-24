import fs from "node:fs";
import { expect, test } from "./fixtures.js";
import { buildStructureFromArtifacts } from "../src/structure/buildStructure.js";

const root = new URL("../../models/", import.meta.url);
for (const model of ["DeepSeek-V4-Flash-Vision-Exp", "DeepSeek-V4.1-Flash"]) {
  test(`released ViT topology renders in Chrome: ${model}`, async ({ page }, testInfo) => {
    test.setTimeout(120_000);
    const modelId = `deepseek-ai/${model}`;
    const dir = new URL(`${modelId}/`, root);
    const read = file => JSON.parse(fs.readFileSync(new URL(file, dir), "utf8"));
    const graph = buildStructureFromArtifacts({
      modelId, config: read("config.json"),
      checkpointTruth: read("header-truth.json"), sourceRef: read("source-ref.json"),
    }).graph;
    const node = id => graph.nodes.find(n => n.canonical_id === id);
    await page.goto("/");
    await page.getByLabel("model id").fill(modelId);
    await page.getByRole("button", { name: "打开模型", exact: true }).click();
    await expect(page.locator(".detail-model-id")).toContainText(modelId);
    await expect(page.locator(`.react-flow__node[data-id="${node("vision").id}"]`)).toBeVisible();
    await page.getByRole("button", { name: "展开全部", exact: true }).click();
    for (const id of ["vision.patch_embed.proj", "vision.blocks.0.attn.rope",
      "vision.blocks.0.mlp.w1", "vision.blocks.0.mlp.w2", "vision.norm"]) {
      await expect(page.locator(`.react-flow__node[data-id="${node(id).id}"]`)).toHaveCount(1);
    }
    for (const [from, to] of [
      ["vision.blocks.0.attn.qkv_split", "vision.blocks.0.attn.rope"],
      ["vision.blocks.0.block_input", "vision.blocks.0.residual_attn"],
      ["vision.blocks.0.mlp.w2", "vision.blocks.0.residual_mlp"],
    ]) {
      const edge = graph.edges.find(e => e.source_canonical_id === from && e.target_canonical_id === to);
      expect(edge).toBeTruthy();
      await expect(page.locator(`.react-flow__edge[data-id="${edge.id}"] path.react-flow__edge-path`))
        .toHaveAttribute("d", /^M/);
    }
    await page.locator(".react-flow-diagram").screenshot({
      path: testInfo.outputPath(`${model}-${testInfo.project.name}-graph.png`),
    });
  });
}
