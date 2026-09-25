#!/usr/bin/env node
/**
 * Browser audit for the focused-module detail viewport.
 *
 * Every built-in model opens its first expandable module, enters focused
 * detail, and checks that the focused graph has a ready layout, visible
 * boundary edges, no page errors, and no sampled edge occlusion.
 */
import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../../..");
const playwright = await import(pathToFileURL(path.join(repoRoot, "frontend/node_modules/playwright-core/index.mjs")));
const catalog = JSON.parse(await fs.readFile(path.join(repoRoot, "models/catalog.json"), "utf8"));
const baseUrl = process.env.MSV_BASE_URL || "http://127.0.0.1:4173";
const outputPath = process.argv[2] || path.join(repoRoot, "artifacts/architecture-repair/focused-module-audit.json");

const browser = await playwright.chromium.launch({
  headless: true,
  executablePath: process.env.CHROME_BIN || "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
});
const page = await browser.newPage({ viewport: { width: 1600, height: 1200 } });
const findings = [];

async function auditOcclusion() {
  return page.evaluate(() => {
    const rect = (element) => {
      const box = element.getBoundingClientRect();
      return { left: box.left, top: box.top, right: box.right, bottom: box.bottom };
    };
    const inside = (point, box) => point.x > box.left + 1 && point.x < box.right - 1
      && point.y > box.top + 1 && point.y < box.bottom - 1;
    const tiles = [...document.querySelectorAll(".react-flow__node")]
      .filter((element) => element.querySelector(".rf-model-node"))
      .map((element) => ({ id: element.getAttribute("data-id"), box: rect(element) }));
    const hits = [];
    for (const edgeElement of document.querySelectorAll(".react-flow__edge")) {
      const pathElement = edgeElement.querySelector(".react-flow__edge-path");
      if (!pathElement) continue;
      const length = pathElement.getTotalLength();
      const transform = pathElement.getScreenCTM();
      if (!transform) continue;
      const edgeId = edgeElement.getAttribute("data-id") || "";
      const parts = edgeId.split("=>");
      const endpointIds = new Set([parts[0], parts[1], `frame-${parts[0]}`, `frame-${parts[1]}`]);
      const samples = Math.max(20, Math.ceil(length / 8));
      const obscuredBy = new Set();
      for (let index = 1; index < samples - 1; index += 1) {
        const point = pathElement.getPointAtLength((length * index) / samples);
        const screenPoint = {
          x: point.x * transform.a + point.y * transform.c + transform.e,
          y: point.x * transform.b + point.y * transform.d + transform.f,
        };
        for (const tile of tiles) {
          if (!endpointIds.has(tile.id) && inside(screenPoint, tile.box)) obscuredBy.add(tile.id);
        }
      }
      if (obscuredBy.size) hits.push({ id: edgeId, obscuredBy: [...obscuredBy] });
    }
    return {
      tileCount: tiles.length,
      edgeCount: document.querySelectorAll(".react-flow__edge").length,
      hits,
    };
  });
}

for (const entry of catalog.models) {
  const pageErrors = [];
  const onPageError = (error) => pageErrors.push(String(error));
  page.on("pageerror", onPageError);
  try {
    await page.goto(`${baseUrl}/?model=${encodeURIComponent(entry.model_id)}&source=builtin`, {
      waitUntil: "domcontentloaded",
      timeout: 30_000,
    });
    await page.locator(".react-flow-diagram[data-layout-ready='true']").waitFor({ timeout: 30_000 });
    const group = page.locator(".rf-model-node.closed-group:has(.layer-group-toggle)").first();
    if (!(await group.count())) throw new Error("no expandable module");
    await group.click();
    const open = page.getByRole("button", { name: /打开模块详情|Open module detail/ });
    await open.waitFor({ timeout: 5000 });
    await open.click();
    await page.locator(".react-flow-diagram[data-focus-path]:not([data-focus-path=''])").waitFor({ timeout: 30_000 });
    await page.waitForFunction(() => document.querySelectorAll(".react-flow__edge").length > 0, null, { timeout: 30_000 });
    const result = await auditOcclusion();
    findings.push({
      model_id: entry.model_id,
      focus_path: await page.locator("[data-focus-path]").getAttribute("data-focus-path"),
      ...result,
      page_errors: pageErrors,
    });
    console.log(`${entry.model_id}: ${result.hits.length} focused occlusion(s)`);
  } catch (error) {
    findings.push({ model_id: entry.model_id, error: String(error), page_errors: pageErrors });
    console.error(`${entry.model_id}: ERROR ${error}`);
  } finally {
    page.removeListener("pageerror", onPageError);
  }
}

await fs.mkdir(path.dirname(outputPath), { recursive: true });
await fs.writeFile(outputPath, JSON.stringify({
  generated_at: new Date().toISOString(),
  base_url: baseUrl,
  findings,
}, null, 2));
await browser.close();

const errors = findings.filter((item) => item.error);
const withHits = findings.filter((item) => item.hits?.length);
console.log(JSON.stringify({
  models: findings.length,
  errors: errors.length,
  models_with_potential_occlusion: withHits.length,
  potential_occlusions: findings.reduce((sum, item) => sum + (item.hits?.length || 0), 0),
  page_errors: findings.reduce((sum, item) => sum + (item.page_errors?.length || 0), 0),
  output: outputPath,
}, null, 2));
