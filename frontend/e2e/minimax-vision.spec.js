import fs from "node:fs";
import { expect, test } from "./fixtures.js";
import { buildStructureFromArtifacts } from "../src/structure/buildStructure.js";

for (const variant of ["MiniMax-M3", "MiniMax-M3-MXFP8"]) {
  test(`MiniMax published separate-QKV visual path renders in Chrome: ${variant}`, async ({ page }, testInfo) => {
    test.setTimeout(120_000);
    const modelId = `MiniMaxAI/${variant}`;
    const dir = new URL(`../../models/${modelId}/`, import.meta.url);
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
    await page.getByRole("button", { name: "展开全部", exact: true }).click();
    for (const id of [
      "vision_tower.embeddings.proj", "vision_tower.pre_layrnorm",
      "vision_tower.layers.0.self_attn.q_proj",
      "vision_tower.layers.0.self_attn.k_proj",
      "vision_tower.layers.0.self_attn.v_proj",
      "vision_tower.layers.0.self_attn.rope",
      "vision_tower.layers.0.residual_attn",
      "vision_tower.layers.0.residual_mlp",
    ]) {
      await expect(page.locator(`.react-flow__node[data-id="${node(id).id}"]`)).toHaveCount(1);
    }
    const edge = graph.edges.find(e =>
      e.source_canonical_id === "vision_tower.layers.0.self_attn.v_reshape"
      && e.target_canonical_id === "vision_tower.layers.0.self_attn.sdpa");
    expect(edge).toBeTruthy();
    await expect(page.locator(`.react-flow__edge[data-id="${edge.id}"] path.react-flow__edge-path`))
      .toHaveAttribute("d", /^M/);
    await page.locator(".react-flow-diagram").screenshot({
      path: testInfo.outputPath(`${variant}-${testInfo.project.name}-vision.png`),
    });
  });
}
