import { expect, test, waitForLayout } from "./fixtures.js";
import fs from "node:fs";
import { buildStructureFromArtifacts } from "../src/structure/buildStructure.js";

const read = url => JSON.parse(fs.readFileSync(url, "utf8"));
for (const modelId of ["Qwen/Qwen3.8-Flash-Next", "Qwen/Qwen3.8-Flash-Next-FP8"]) {
  test(`QSA complete blocks, tail and gate render: ${modelId}`, async ({ page }, testInfo) => {
    test.setTimeout(150_000);
    const dir = new URL(`../../models/${modelId}/`, import.meta.url);
    const structure = buildStructureFromArtifacts({ modelId,
      config: read(new URL("config.json", dir)), checkpointTruth: read(new URL("header-truth.json", dir)),
      sourceRef: read(new URL("source-ref.json", dir)) });
    const byCanonical = new Map(structure.graph.nodes.map(node => [node.canonical_id, node]));
    const node = suffix => byCanonical.get(`layers.3.self_attn.${suffix}`);
    await page.route(/https:\/\/(?:www\.)?(?:huggingface\.co|hf-mirror\.com|modelscope\.cn)\//, route => route.abort());
    await page.goto("/");
    await page.getByLabel("model id").fill(modelId);
    await page.getByRole("button", { name: "打开模型", exact: true }).click();
    await expect(page.locator(".react-flow-diagram")).toHaveAttribute("data-graph-version", "2", { timeout: 30000 });
    await page.getByRole("button", { name: "展开全部", exact: true }).click();
    await waitForLayout(page);
    const pairs = [
      ["indexer.key_block_mean_pool", "indexer.k_layernorm"],
      ["indexer.k_layernorm", "indexer.k_rope"],
      ["indexer.q_rope", "indexer.score"],
      ["indexer.k_rope", "indexer.score"],
      ["indexer.block_select", "indexer.block_expand"],
      ["indexer.block_expand", "indexer.tail_append"],
      ["q_gate_split", "output_gate"], ["sparse_attention", "output_gate"],
    ];
    for (const [from, to] of pairs) {
      const edge = structure.graph.edges.find(edge => edge.source === node(from).id && edge.target === node(to).id);
      expect(edge, `${from} -> ${to}`).toBeTruthy();
      // Verify the actual rendered SVG path, not merely the in-memory graph.
      await expect(page.locator(`.react-flow__edge[data-id="${edge.id}"] path.react-flow__edge-path`))
        .toHaveAttribute("d", /^M/, { timeout: 30000 });
    }
    await expect(page.locator(".react-flow__edge title").filter({ hasText: "visible tail" }).first()).toHaveCount(1);
    const pool = page.locator(`.react-flow__node[data-id="${node("indexer.key_block_mean_pool").id}"]`);
    const norm = page.locator(`.react-flow__node[data-id="${node("indexer.k_layernorm").id}"]`);
    const poolBox = await pool.boundingBox(), normBox = await norm.boundingBox();
    expect(poolBox).toBeTruthy();
    expect(normBox).toBeTruthy();
    expect(normBox.y).toBeGreaterThan(poolBox.y);
    await page.screenshot({ path: testInfo.outputPath("qsa-expanded.png") });
    // Full-model fit is intentionally tiny. Use real UI zoom controls to
    // inspect the mechanism at a readable scale before taking local evidence.
    for (let step = 0; step < 12; step++) {
      const width = (await pool.boundingBox()).width;
      if (width >= 110) break;
      await page.locator(".react-flow__controls-zoomin").click();
      await expect.poll(async () => (await pool.boundingBox()).width).toBeGreaterThan(width + 0.01);
    }
    // Select via accessible keyboard activation; production selection focuses
    // this subgraph even when the full model is much larger than the viewport.
    await pool.locator(".rf-model-node").focus();
    await page.keyboard.press("Enter");
    await expect(pool.locator(".rf-model-node")).toHaveAttribute("aria-selected", "true");
    await expect(pool).toContainText("V -");
    await expect.poll(async () => (await pool.boundingBox()).width).toBeGreaterThan(80);
    await page.locator(".react-flow-diagram").scrollIntoViewIfNeeded();
    await expect.poll(async () => {
      const box = await pool.boundingBox(), pane = await page.locator(".react-flow-diagram").boundingBox();
      return box.x >= pane.x && box.x < pane.x + pane.width && box.y >= pane.y && box.y < pane.y + pane.height;
    }).toBe(true);
    await page.screenshot({ path: testInfo.outputPath("qsa-pool-focus.png") });
    await page.getByRole("button", { name: "收起全部", exact: true }).click();
    await waitForLayout(page);
    await expect(pool).toHaveCount(0);
    await page.getByRole("button", { name: "展开全部", exact: true }).click();
    await waitForLayout(page);
    await expect(pool).toHaveCount(1, { timeout: 30000 });
    await page.getByRole("button", { name: "中 / EN" }).click();
    await expect(page.locator(".react-flow__edge title").filter({ hasText: "visible tail" }).first()).toHaveCount(1);
    await page.locator(".detail-cost-toggle > button").click();
    await expect(page.locator(".cost-summary")).toContainText("MACs / forward");
    const pending = page.waitForEvent("download");
    await page.getByRole("button", { name: "SVG", exact: true }).click();
    const download = await pending;
    await download.saveAs(testInfo.outputPath("qsa-export.svg"));
  });
}

test("local header binding does not render a bound tensor again as a checkpoint gap", async ({ page }, testInfo) => {
  test.setTimeout(90000);
  const config = read(new URL("../../models/Qwen/Qwen3.8-Flash-Next/config.json", import.meta.url));
  const dir = testInfo.outputPath("local-header-fixture");
  fs.mkdirSync(dir, { recursive: true });
  fs.writeFileSync(`${dir}/config.json`, JSON.stringify(config));
  // Synthetic values, real known norm path/shape. An intentionally unmatched
  // module tests that genuine gaps still survive; no downloaded weight data.
  const boundName = "model.language_model.layers.3.self_attn.indexer.q_layernorm.weight";
  const extraName = "model.language_model.fixture_extra.weight";
  const header = Buffer.from(JSON.stringify({
    [boundName]: { dtype: "BF16", shape: [128], data_offsets: [0, 256] },
    [extraName]: { dtype: "BF16", shape: [4], data_offsets: [256, 264] },
  }));
  const prefix = Buffer.alloc(8);
  prefix.writeBigUInt64LE(BigInt(header.length));
  fs.writeFileSync(`${dir}/model.safetensors`, Buffer.concat([prefix, header, Buffer.alloc(264)]));
  await page.goto("/");
  await page.getByRole("button", { name: "打开本地模型目录", exact: true }).click();
  const choosing = page.waitForEvent("filechooser");
  await page.getByRole("button", { name: "打开文件夹", exact: true }).click();
  await (await choosing).setFiles(dir);
  await expect(page.locator(".detail-page")).toBeVisible({ timeout: 30000 });
  await page.getByRole("button", { name: "展开全部", exact: true }).click();
  await waitForLayout(page);
  await expect(page.locator(".rf-node-title").filter({ hasText: "index query zero-centered RMSNorm" }).first()).toHaveCount(1);
  await page.locator(".detail-aux-actions").getByRole("button", { name: "导出", exact: true }).click();
  const panel = page.locator(".export-panel");
  await panel.locator("select").selectOption("json");
  const exported = JSON.parse(await panel.locator("textarea").inputValue());
  expect(exported.graph.nodes.filter(node => node.tensor_names?.includes(boundName))).toHaveLength(1);
  expect(exported.graph.nodes.filter(node => node.tensor_names?.includes(extraName))).toHaveLength(1);
  expect(exported.graph.nodes.filter(node => node.type === "checkpoint-gaps")).toHaveLength(1);
  await page.screenshot({ path: testInfo.outputPath("local-header-export.png") });
});
