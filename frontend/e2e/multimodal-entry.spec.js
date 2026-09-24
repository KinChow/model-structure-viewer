import fs from "node:fs";
import { expect, test } from "./fixtures.js";
import { normalizeConfig } from "../src/structure/config/normalize.js";
import { buildStructureFromArtifacts } from "../src/structure/buildStructure.js";

const root = new URL("../../models/", import.meta.url);
const read = url => fs.existsSync(url) ? JSON.parse(fs.readFileSync(url, "utf8")) : null;
const entries = read(new URL("catalog.json", root)).models.filter(entry =>
  normalizeConfig(read(new URL(entry.config_path, root))).hasVision);
const families = [...new Map(entries.map(entry => {
  const type = normalizeConfig(read(new URL(entry.config_path, root))).modelType;
  return [type, entry];
})).values()];

async function openAndCheck(page, entry, testInfo, deep = false) {
  const config = read(new URL(entry.config_path, root));
  const dir = new URL("./", new URL(entry.config_path, root));
  const structure = buildStructureFromArtifacts({ config, modelId: entry.model_id,
    checkpointTruth: read(new URL("header-truth.json", dir)), sourceRef: read(new URL("source-ref.json", dir)) });
  const graph = structure.graph;
  const get = canonical => graph.nodes.find(node => node.canonical_id === canonical);
  const embed = get("embed_tokens"), fusion = get("multimodal_fusion");
  const vision = graph.nodes.find(node => node.type === "projector")
    || graph.nodes.find(node => node.type === "vision-encoder");
  await page.goto("/");
  await page.getByLabel("model id").fill(entry.model_id);
  await page.getByRole("button", { name: "打开模型", exact: true }).click();
  await expect(page.locator(".detail-model-id")).toContainText(entry.model_id, { timeout: 30000 });
  await expect(page.locator(".react-flow-diagram")).toHaveAttribute("data-graph-version", "2");
  // 边在同步临时布局中也会出现；真实布局由异步 ELK 完成，不能把 fitView
  // 的 260ms 过渡帧误判为最终节点重叠或以临时连线冒充浏览器验收。
  await expect(page.locator(".react-flow-diagram")).toHaveAttribute("data-layout-ready", "true", { timeout: 30000 });
  const tile = n => page.locator(`.react-flow__node[data-id="${n.id}"]`);
  const assertNoOverlap = async (left, right, message) => {
    await expect.poll(async () => {
      const a = await tile(left).boundingBox(), b = await tile(right).boundingBox();
      if (!a || !b) return true;
      return a.x < b.x + b.width && a.x + a.width > b.x
        && a.y < b.y + b.height && a.y + a.height > b.y;
    }, { message, timeout: 10_000, intervals: [100, 200, 300] }).toBe(false);
  };
  for (const source of [embed, vision]) {
    const edge = graph.edges.find(e => e.source === source.id && e.target === fusion.id);
    await expect(page.locator(`.react-flow__edge[data-id="${edge.id}"] path.react-flow__edge-path`))
      .toHaveAttribute("d", /^M/, { timeout: 30000 });
  }
  if (entry.model_id.startsWith("MiniMaxAI/MiniMax-M3")) {
    const first = get("multi_modal_projector"), merge = get("patch_merge_mlp");
    expect(first).toBeTruthy(); expect(merge).toBeTruthy();
    const link = graph.edges.find(e => e.source === first.id && e.target === merge.id);
    expect(link).toBeTruthy();
    await expect(page.locator(`.react-flow__edge[data-id="${link.id}"] path.react-flow__edge-path`))
      .toHaveAttribute("d", /^M/, { timeout: 30000 });
    await assertNoOverlap(first, merge, "the separate patch merge stage must not overlap the projector");
  }
  await assertNoOverlap(embed, vision, `${entry.model_id}: independent branches overlap`);
  await page.locator(".detail-cost-toggle > button").click();
  await expect(page.locator(".cost-summary")).not.toContainText(/NaN|undefined/);
  await expect(page.locator('.cost-domain-item[data-group="memory"] b')).toHaveText(/未知|unknown/i);
  await page.screenshot({ path: testInfo.outputPath(`${entry.model_id.replaceAll("/", "__")}-overview.png`) });
  if (deep) {
    const search = page.getByPlaceholder("搜索节点名称 / 类型 / class...");
    await search.fill("text / vision fusion");
    await page.getByRole("option").filter({ hasText: "text / vision fusion" }).first().click();
    await expect(tile(fusion).locator(".rf-model-node")).toHaveAttribute("aria-selected", "true");
    await search.fill("");
    await page.getByRole("button", { name: "展开全部", exact: true }).click();
    await expect(page.locator(".react-flow-diagram")).toHaveAttribute("data-layout-ready", "true", { timeout: 30000 });
    if (entry.model_id === "moonshotai/Kimi-K3") {
      const postNorm = get("mm_projector.post_norm");
      expect(postNorm).toBeTruthy();
      expect(get("mm_projector.pre_norm")).toBeFalsy();
      await expect(tile(postNorm)).toHaveCount(1);
      const projection = get("mm_projector.proj.2");
      const edge = graph.edges.find(e => e.source === projection.id && e.target === postNorm.id);
      expect(edge).toBeTruthy();
      await expect(page.locator(`.react-flow__edge[data-id="${edge.id}"] path.react-flow__edge-path`))
        .toHaveAttribute("d", /^M/, { timeout: 30000 });
    }
    await expect(page.locator(".react-flow__edge title").filter({ hasText: "visual features" }).first()).toHaveCount(1);
    await expect(page.locator(".react-flow__edge title").filter({ hasText: "placeholder positions" }).first()).toHaveCount(1);
    await tile(fusion).locator(".rf-model-node").focus();
    await page.keyboard.press("Enter");
    await expect(tile(fusion).locator(".rf-model-node")).toHaveAttribute("aria-selected", "true");
    await page.getByRole("button", { name: "收起全部", exact: true }).click();
    await expect(page.locator(".react-flow-diagram")).toHaveAttribute("data-layout-ready", "true", { timeout: 30000 });
    await expect(tile(fusion)).toHaveCount(1);
    await page.getByRole("button", { name: "中 / EN" }).click();
    await expect(page.locator(".react-flow__edge title").filter({ hasText: "visual features" }).first()).toHaveCount(1);
    await page.screenshot({ path: testInfo.outputPath(`${entry.model_id.replaceAll("/", "__")}-english.png`) });
    const downloading = page.waitForEvent("download");
    await page.getByRole("button", { name: "SVG", exact: true }).click();
    await (await downloading).saveAs(testInfo.outputPath(`${entry.model_id.replaceAll("/", "__")}.svg`));
    await page.getByRole("button", { name: "EN / 中" }).click();
  }
  return { model_id: entry.model_id, fusion_semantics: fusion.attributes.fusion_semantics, rendered: true, deep };
}

test.beforeEach(async ({ page }) => {
  await page.route(/https:\/\/(?:www\.)?(?:huggingface\.co|hf-mirror\.com|modelscope\.cn)\//, route => route.abort());
});

test("desktop production loading of all 39 multimodal entries", async ({ page }, testInfo) => {
  test.skip(testInfo.project.name !== "desktop-chrome");
  test.setTimeout(1200_000);
  const results = [];
  for (const entry of entries) results.push(await openAndCheck(page, entry, testInfo,
    families.some(family => family.model_id === entry.model_id)));
  expect(results).toHaveLength(39);
  fs.writeFileSync(testInfo.outputPath("multimodal-entry-report.json"), JSON.stringify(results, null, 2));
});

test("desktop final unknown-traffic and fusion-search presentation", async ({ page }, testInfo) => {
  test.skip(testInfo.project.name !== "desktop-chrome");
  test.setTimeout(150_000);
  await openAndCheck(page, entries.find(entry => entry.model_id === "Qwen/Qwen3.5-0.8B"), testInfo, true);
});

test("desktop MiniMax released projector and patch merge remain separate visible stages", async ({ page }, testInfo) => {
  test.skip(testInfo.project.name !== "desktop-chrome");
  test.setTimeout(150_000);
  await openAndCheck(page, entries.find(entry => entry.model_id === "MiniMaxAI/MiniMax-M3"), testInfo, true);
});

test("desktop Kimi-K3 V2 projector ends at post-norm", async ({ page }, testInfo) => {
  test.skip(testInfo.project.name !== "desktop-chrome");
  test.setTimeout(150_000);
  await openAndCheck(page, entries.find(entry => entry.model_id === "moonshotai/Kimi-K3"), testInfo, true);
});

for (const entry of families) {
  test(`mobile fusion family: ${entry.model_id}`, async ({ page }, testInfo) => {
    test.skip(testInfo.project.name !== "mobile-chrome");
    test.setTimeout(150_000);
    await openAndCheck(page, entry, testInfo, true);
  });
}
