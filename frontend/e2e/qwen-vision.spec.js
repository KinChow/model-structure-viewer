import fs from "node:fs";
import { expect, test } from "./fixtures.js";
import { buildStructureFromArtifacts } from "../src/structure/buildStructure.js";

for (const variant of ["Qwen3.8-27B", "Qwen3.8-Flash-Next"]) {
  test(`Qwen published visual.blocks path renders in Chrome: ${variant}`, async ({ page }, testInfo) => {
    test.setTimeout(120_000);
    const modelId = `Qwen/${variant}`;
    const dir = new URL(`../../models/${modelId}/`, import.meta.url);
    const read = file => JSON.parse(fs.readFileSync(new URL(file, dir), "utf8"));
    const graph = buildStructureFromArtifacts({
      modelId, config: read("config.json"), sourceRef: read("source-ref.json"),
    }).graph;
    const node = id => graph.nodes.find(n => n.canonical_id === id);
    await page.goto("/");
    await page.getByLabel("model id").fill(modelId);
    await page.getByRole("button", { name: "打开模型", exact: true }).click();
    await expect(page.locator(".detail-model-id")).toContainText(modelId);
    await page.getByRole("button", { name: "展开全部", exact: true }).click();
    for (const id of [
      "visual.patch_embed.proj",
      "visual.pos_embed",
      "visual.blocks.0.norm1",
      "visual.blocks.0.attn.qkv",
      "visual.blocks.0.attn.rope",
      "visual.blocks.0.mlp.linear_fc1",
      "visual.merger.linear_fc1",
      "visual.merger.linear_fc2",
    ]) await expect(page.locator(`.react-flow__node[data-id="${node(id).id}"]`)).toHaveCount(1);
    const edge = graph.edges.find(e =>
      e.source_canonical_id === "visual.rotary_pos_emb"
      && e.target_canonical_id === "visual.blocks.0");
    expect(edge?.relation).toBe("index-control");
    await expect(page.locator(`.react-flow__edge[data-id="${edge.id}"] path.react-flow__edge-path`))
      .toHaveAttribute("d", /^M/);
    await page.locator(".react-flow-diagram").screenshot({
      path: testInfo.outputPath(`${variant}-${testInfo.project.name}-vision.png`),
    });
  });
}
