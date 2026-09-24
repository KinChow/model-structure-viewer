import { expect, test } from "./fixtures.js";
import fs from "node:fs";
import { buildStructureFromArtifacts } from "../src/structure/buildStructure.js";

test("K3 NoPE, gate fan-in and vision 2D RoPE render in the production view", async ({ page }, testInfo) => {
  test.setTimeout(180000);
  const config = JSON.parse(fs.readFileSync(new URL("../../models/moonshotai/Kimi-K3/config.json", import.meta.url), "utf8"));
  // K3 currently has no offline header/skeleton sidecar: the real built-in
  // route is config-backed. Header binding is tested separately, not fabricated here.
  const { graph } = buildStructureFromArtifacts({ config, modelId: "moonshotai/Kimi-K3" });
  const nodes = new Map(graph.nodes.map(n => [n.canonical_id, n]));
  const prefix = "layers.3.self_attn";
  await page.route(/https:\/\/(?:www\.)?(?:huggingface\.co|hf-mirror\.com|modelscope\.cn)\//, route => route.abort());
  await page.goto("/");
  await page.getByLabel("model id").fill("moonshotai/Kimi-K3");
  await page.getByRole("button", { name: "打开模型", exact: true }).click();
  await expect(page.locator(".react-flow-diagram")).toHaveAttribute("data-graph-version", "2", { timeout: 30000 });
  await page.getByRole("button", { name: "展开全部", exact: true }).click();
  const pairs = [
    [`${prefix}.g_proj`, `${prefix}.output_gate`],
    [`${prefix}.sdpa`, `${prefix}.output_gate`],
    [`${prefix}.output_gate`, `${prefix}.o_proj`],
    [`${prefix}.kv_split`, `${prefix}.sdpa`],
    [`${prefix}.kv_b_proj`, `${prefix}.sdpa`],
    ["vision_tower.encoder.blocks.0.qkv_reshape", "vision_tower.encoder.blocks.0.rope"],
    ["vision_tower.encoder.blocks.0.rope", "vision_tower.encoder.blocks.0.sdpa"],
    ["vision_tower.encoder.blocks.0.qkv_reshape", "vision_tower.encoder.blocks.0.sdpa"],
  ];
  const checkEdges = async () => {
    for (const [from, to] of pairs) {
      const edge = graph.edges.find(e => e.source_canonical_id === from && e.target_canonical_id === to);
      expect(edge, `${from} -> ${to}`).toBeTruthy();
      await expect(page.locator(`.react-flow__edge[data-id="${edge.id}"] path.react-flow__edge-path`))
        .toHaveAttribute("d", /^M/, { timeout: 30000 });
    }
  };
  await checkEdges();
  await expect(page.locator(".react-flow__edge title").filter({ hasText: "gate logits" }).first()).toHaveCount(1);
  await expect(page.locator(".react-flow__edge title").filter({ hasText: "shared key channels (NoPE)" }).first()).toHaveCount(1);
  const gate = page.locator(`.react-flow__node[data-id="${nodes.get(`${prefix}.output_gate`).id}"]`);
  const projection = page.locator(`.react-flow__node[data-id="${nodes.get(`${prefix}.g_proj`).id}"]`);
  const out = page.locator(`.react-flow__node[data-id="${nodes.get(`${prefix}.o_proj`).id}"]`);
  await expect(gate).toHaveCount(1);
  const g = await gate.boundingBox(), p = await projection.boundingBox(), o = await out.boundingBox();
  expect(g).toBeTruthy(); expect(p).toBeTruthy(); expect(o).toBeTruthy();
  expect(g.y).toBeGreaterThanOrEqual(p.y + p.height);
  expect(o.y).toBeGreaterThanOrEqual(g.y + g.height);
  for (let step = 0; step < 12 && (await gate.boundingBox()).width < 110; step++) {
    const width = (await gate.boundingBox()).width;
    await page.locator(".react-flow__controls-zoomin").click();
    await expect.poll(async () => (await gate.boundingBox()).width).toBeGreaterThan(width + 0.01);
  }
  await gate.locator(".rf-model-node").focus();
  await page.keyboard.press("Enter");
  await expect(gate.locator(".rf-model-node")).toHaveAttribute("aria-selected", "true");
  await page.locator(".react-flow-diagram").scrollIntoViewIfNeeded();
  await page.screenshot({ path: testInfo.outputPath("kimi-mla-gate.png") });
  await page.getByRole("button", { name: "收起全部", exact: true }).click();
  await expect(gate).toHaveCount(0);
  await page.getByRole("button", { name: "展开全部", exact: true }).click();
  await checkEdges();
  const search = page.getByPlaceholder("搜索节点名称 / 类型 / class...");
  await search.fill("MLA gate projection");
  await expect(page.getByRole("option").first()).toBeVisible();
  await search.fill("");
  await page.getByRole("button", { name: "中 / EN" }).click();
  await page.locator(".detail-cost-toggle > button").click();
  await expect(page.locator(".cost-summary")).toContainText("MACs / forward");
  const pending = page.waitForEvent("download");
  await page.getByRole("button", { name: "SVG", exact: true }).click();
  await (await pending).saveAs(testInfo.outputPath("kimi-mla-export.svg"));
});
