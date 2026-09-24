import fs from "node:fs";
import { expect, test } from "./fixtures.js";
import { buildStructureFromArtifacts } from "../src/structure/buildStructure.js";

test("Kimi-K3 published MoonViT path renders in Chrome", async ({ page }, testInfo) => {
  test.setTimeout(120_000);
  const modelId = "moonshotai/Kimi-K3";
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
    "vision_tower.patch_embed.proj",
    "vision_tower.patch_embed.pos_emb",
    "vision_tower.encoder.blocks.0.wqkv",
    "vision_tower.encoder.blocks.0.rope",
    "vision_tower.encoder.blocks.0.residual_attn",
    "vision_tower.encoder.blocks.0.mlp.fc0",
    "vision_tower.encoder.final_layernorm",
  ]) {
    await expect(page.locator(`.react-flow__node[data-id="${node(id).id}"]`)).toHaveCount(1);
  }
  const edge = graph.edges.find(e =>
    e.source_canonical_id === "vision_tower.encoder.rope_2d"
    && e.target_canonical_id === "vision_tower.encoder.blocks.0");
  expect(edge?.relation).toBe("index-control");
  await expect(page.locator(`.react-flow__edge[data-id="${edge.id}"] path.react-flow__edge-path`))
    .toHaveAttribute("d", /^M/);
  await page.locator(".react-flow-diagram").screenshot({
    path: testInfo.outputPath(`kimi-k3-vision-${testInfo.project.name}.png`),
  });
});
