import { expect, test } from "./fixtures.js";
import fs from "node:fs";
import { buildStructureFromArtifacts } from "../src/structure/buildStructure.js";

const variants = ["DeepSeek-V4-Flash", "DeepSeek-V4-Flash-0731",
  "DeepSeek-V4-Flash-Vision-Exp", "DeepSeek-V4-Pro", "DeepSeek-V4-Pro-0813",
  "DeepSeek-V4.1-Flash"];
const read = url => fs.existsSync(url) ? JSON.parse(fs.readFileSync(url, "utf8")) : undefined;

for (const variant of variants) {
  test(`V4 compressor forward dependency renders in SVG: ${variant}`, async ({ page }, testInfo) => {
    test.setTimeout(120_000);
    const modelId = `deepseek-ai/${variant}`;
    const dir = new URL(`../../models/${modelId}/`, import.meta.url);
    const { graph } = buildStructureFromArtifacts({
      modelId,
      config: read(new URL("config.json", dir)),
      checkpointTruth: read(new URL("skeleton-truth.json", dir)) || read(new URL("header-truth.json", dir)),
      sourceRef: read(new URL("source-ref.json", dir)),
    });
    await page.route(/https:\/\/(?:www\.)?(?:huggingface\.co|hf-mirror\.com|modelscope\.cn)\//,
      route => route.abort());
    await page.goto("/");
    await page.getByLabel("model id").fill(modelId);
    await page.getByRole("button", { name: "打开模型", exact: true }).click();
    await expect(page.locator(".react-flow-diagram")).toHaveAttribute("data-graph-version", "2");
    const before = await page.locator(".react-flow__edge").count();
    await page.getByRole("button", { name: "展开全部", exact: true }).click();
    const compressor = graph.nodes.find(n =>
      /layers\.(?:(?:encoder|decoder)\.)?\d+\.self_attn\.compressor$/.test(n.canonical_id || ""));
    expect(compressor).toBeTruthy();
    const prefix = compressor.canonical_id.replace(/\.compressor$/, "");
    const indexer = graph.nodes.find(n =>
      n.attributes?.operator_id === "dsv4_indexer" && /\.self_attn\.indexer$/.test(n.canonical_id || ""));
    const pairs = variant.includes("V4.1")
      ? [[`${prefix}.compressor`, `${prefix}.attention`]]
      : [
        [`${prefix}.compressor.norm`, `${prefix}.compressor`],
        [`${prefix}.compressor`, `${prefix}.attention`],
        [`${indexer.canonical_id}.compressor.wkv_gate`, `${indexer.canonical_id}.compressor.norm`],
        [`${indexer.canonical_id}.compressor.norm`, indexer.canonical_id],
      ];
    for (const [from, to] of pairs) {
      const edge = graph.edges.find(e => e.source_canonical_id === from && e.target_canonical_id === to);
      expect(edge, `${from} -> ${to}`).toBeTruthy();
      await expect(page.locator(`.react-flow__edge[data-id="${edge.id}"] path.react-flow__edge-path`))
        .toHaveAttribute("d", /^M/, { timeout: 60_000 });
    }
    await expect.poll(() => page.locator(".react-flow__edge").count()).toBeGreaterThan(before);
    await page.screenshot({ path: testInfo.outputPath("compressor-expanded.png") });
    await page.getByRole("button", { name: "收起全部", exact: true }).click();
    await page.locator(".detail-cost-toggle > button").click();
    await expect(page.locator(".cost-summary")).toContainText("MACs / forward");
    await page.getByRole("button", { name: "中 / EN" }).click();
    await expect(page.locator(".react-flow-diagram")).toHaveAttribute("data-graph-version", "2");
  });
}
