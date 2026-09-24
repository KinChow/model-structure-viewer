import { expect, test } from "./fixtures.js";
import fs from "node:fs";
import { buildStructureFromArtifacts } from "../src/structure/buildStructure.js";
for (const variant of ["Qwen3.8-Flash-Next", "Qwen3.8-Flash-Next-FP8"]) {
  test(`${variant} GR gates remain connected on expansion`, async ({ page }, testInfo) => {
    test.setTimeout(180000);
    const modelId = `Qwen/${variant}`;
    const dir = new URL(`../../models/${modelId}/`, import.meta.url);
    const read = file => JSON.parse(fs.readFileSync(new URL(file, dir)));
    const { graph } = buildStructureFromArtifacts({ config: read("config.json"), modelId,
      checkpointTruth: read("header-truth.json"), sourceRef: read("source-ref.json") });
    const pageErrors = [];
    page.on("pageerror", error => pageErrors.push(error.message));
    await page.route(/https:\/\/(?:www\.)?(?:huggingface\.co|hf-mirror\.com|modelscope\.cn)\//, route => route.abort());
    await page.goto("/");
    await page.getByLabel("model id").fill(modelId);
    await page.getByRole("button", { name: "打开模型", exact: true }).click();
    await expect(page.locator(".react-flow-diagram")).toHaveAttribute("data-graph-version", "2", { timeout: 30000 });
    await page.getByRole("button", { name: "展开全部", exact: true }).click();
    for (const [from, to] of [
      ["layers.0.attn_hyper_connection.write_gate", "layers.0.attn_residual_add"],
      ["layers.0.mlp_hyper_connection.write_gate", "layers.0.ffn_residual_add"],
      ["layers.1.ple", "layers.1.ple_residual_add"], ["mtp.fc_hidden", "mtp.input_add"],
      ["hyper_connection_mixer", "lm_head"],
    ]) {
      const edge = graph.edges.find(e => e.source_canonical_id === from && e.target_canonical_id === to);
      expect(edge).toBeTruthy();
      await expect(page.locator(`.react-flow__edge[data-id="${edge.id}"] path.react-flow__edge-path`))
        .toHaveAttribute("d", /^M/, { timeout: 30000 });
    }
    const write = graph.nodes.find(n => n.canonical_id === "layers.0.attn_residual_add");
    const card = page.locator(`.react-flow__node[data-id="${write.id}"] .rf-model-node`);
    await card.focus(); await page.keyboard.press("Enter");
    await expect(card).toHaveAttribute("aria-selected", "true");
    await expect(page.locator(".formula-section")).toContainText("block_output");
    await page.screenshot({ path: testInfo.outputPath("gr-write.png") });
    await page.getByRole("button", { name: "收起全部", exact: true }).click();
    await expect(card).toHaveCount(0);
    await page.getByRole("button", { name: "展开全部", exact: true }).click();
    await expect(card).toHaveCount(1, { timeout: 30000 });
    const search = page.getByPlaceholder("搜索节点名称 / 类型 / class...");
    await search.fill("Gated Residual final read");
    await expect(page.getByRole("option").first()).toBeVisible();
    await search.fill("");
    await page.getByRole("button", { name: "中 / EN" }).click();
    await page.locator(".detail-cost-toggle > button").click();
    await expect(page.locator(".cost-summary")).toContainText("MACs / forward");
    const pending = page.waitForEvent("download");
    await page.getByRole("button", { name: "SVG", exact: true }).click();
    await (await pending).saveAs(testInfo.outputPath("gr.svg"));
    expect(pageErrors).toEqual([]);
  });
}
