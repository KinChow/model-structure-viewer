import { expect, test, waitForLayout } from "./fixtures.js";
import fs from "node:fs";
import { buildStructureFromArtifacts } from "../src/structure/buildStructure.js";

test("K3 AttnRes depth states, block boundary and final output survive folding", async ({ page }, testInfo) => {
  test.setTimeout(240000);
  const config = JSON.parse(fs.readFileSync(new URL("../../models/moonshotai/Kimi-K3/config.json", import.meta.url)));
  const { graph } = buildStructureFromArtifacts({ config, modelId: "moonshotai/Kimi-K3" });
  const byCanonical = new Map(graph.nodes.map(n => [n.canonical_id, n]));
  const edgeOf = (from, to) => graph.edges.find(e => e.source_canonical_id === from && e.target_canonical_id === to);
  const rendered = edge => page.locator(`.react-flow__edge[data-id="${edge.id}"] path.react-flow__edge-path`);
  await page.route(/https:\/\/(?:www\.)?(?:huggingface\.co|hf-mirror\.com|modelscope\.cn)\//, route => route.abort());
  await page.goto("/");
  await page.getByLabel("model id").fill("moonshotai/Kimi-K3");
  await page.getByRole("button", { name: "打开模型", exact: true }).click();
  await expect(page.locator(".react-flow-diagram")).toHaveAttribute("data-graph-version", "2", { timeout: 30000 });
  const finalEdge = edgeOf("layers.92.bank_out", "output_attn_residual");
  await expect(rendered(finalEdge)).toHaveAttribute("d", /^M/, { timeout: 30000 });
  await page.getByRole("button", { name: "展开全部", exact: true }).click();
  await waitForLayout(page);
  const pairs = [
    ["layers.11.bank_out", "layers.12.bank_in"],
    ["layers.11.prefix_out", "layers.12.layer_in"],
    ["layers.12.layer_in", "layers.12.bank_out"],
    ["layers.12.attn_res_pre", "layers.12.input_layernorm"],
    ["layers.12.self_attn", "layers.12.prefix_after_attn"],
    ["layers.12.bank_out", "layers.12.attn_res_mlp"],
    ["layers.12.prefix_after_attn", "layers.12.attn_res_mlp"],
    ["layers.12.attn_res_mlp", "layers.12.post_attention_layernorm"],
    ["output_attn_res_norm", "output_attn_res_proj"],
    ["output_attn_residual.probabilities", "output_attn_residual.weighted_sum"],
    ["output_attn_residual", "norm"],
  ];
  for (const [from, to] of pairs) {
    expect(edgeOf(from, to)).toBeTruthy();
    await expect(rendered(edgeOf(from, to))).toHaveAttribute("d", /^M/, { timeout: 60000 });
  }
  await expect(rendered(finalEdge)).toHaveAttribute("d", /^M/);
  const bank = page.locator(`.react-flow__node[data-id="${byCanonical.get("layers.12.bank_out").id}"]`);
  // AttnRes snapshots use the reference dense token-major torch.cat path.
  // The node must expose a concrete resident snapshot size; the old `V -`
  // assertion described the pre-fusion-cost contract and is no longer valid.
  await expect(bank).toContainText(/V\s+(?!-)\d/);
  const target = byCanonical.get("layers.12.attn_res_mlp");
  const node = page.locator(`.react-flow__node[data-id="frame-${target.id}"]`);
  await node.locator(".rf-group-frame").focus();
  await page.keyboard.press("Enter");
  await expect(page.locator(".formula-section")).toContainText("候选状态");
  await expect(page.locator('.formula-section [role="math"]')).toHaveAttribute("aria-label", /softmax.*depth/);
  for (let step = 0; step < 12 && (await node.boundingBox()).width < 110; step++) {
    const width = (await node.boundingBox()).width;
    await page.locator(".react-flow__controls-zoomin").click();
    await expect.poll(async () => (await node.boundingBox()).width).toBeGreaterThan(width + .01);
  }
  await node.locator(".rf-group-frame").focus();
  await page.keyboard.press("Enter");
  await page.locator(".react-flow-diagram").scrollIntoViewIfNeeded();
  await page.screenshot({ path: testInfo.outputPath("attnres-block-boundary.png") });
  await page.getByRole("button", { name: "收起全部", exact: true }).click();
  await waitForLayout(page);
  await expect(node).toHaveCount(0);
  await expect(rendered(finalEdge)).toHaveAttribute("d", /^M/, { timeout: 30000 });
  await page.getByRole("button", { name: "展开全部", exact: true }).click();
  await waitForLayout(page);
  await expect(node).toHaveCount(1, { timeout: 60000 });
  const search = page.getByPlaceholder("搜索节点名称 / 类型 / class...");
  await search.fill("Output Attention Residual");
  await expect(page.getByRole("option").first()).toBeVisible();
  await search.fill("");
  await page.getByRole("button", { name: "中 / EN" }).click();
  await page.locator(".detail-cost-toggle > button").click();
  await expect(page.locator(".cost-summary")).toContainText("MACs / forward");
  const pending = page.waitForEvent("download");
  await page.getByRole("button", { name: "SVG", exact: true }).click();
  await (await pending).saveAs(testInfo.outputPath("attnres.svg"));
});
