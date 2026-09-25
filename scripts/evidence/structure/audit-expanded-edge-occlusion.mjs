#!/usr/bin/env node
/**
 * Open every built-in model in the browser, expand every visible group, and
 * detect dataflow paths that pass through a non-endpoint model tile.
 * This is a diagnostic audit only: it does not mutate the graph or repair paths.
 */
import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../../..");
const catalogPath = path.join(repoRoot, "models/catalog.json");
const playwright = await import(pathToFileURL(path.join(repoRoot, "frontend/node_modules/playwright/index.mjs")));
const catalog = JSON.parse(await fs.readFile(catalogPath, "utf8"));
const baseUrl = process.env.MSV_BASE_URL || "http://127.0.0.1:4173";
const outputPath = process.argv[2] || path.join(repoRoot, "docs/details/evidence/structure/generated/expanded-edge-occlusion-audit.json");

const browser = await playwright.chromium.launch({ headless: true });
const page = await browser.newPage({ viewport: { width: 1600, height: 1200 } });
const findings = [];

for (const entry of catalog.models) {
  const url = `${baseUrl}/?model=${encodeURIComponent(entry.model_id)}&source=builtin`;
  try {
    await page.goto(url, { waitUntil: "domcontentloaded", timeout: 30_000 });
    await page.waitForTimeout(700);
    const expand = page.getByRole("button", { name: "Expand all", exact: true });
    if (await expand.count()) {
      await expand.click();
      await page.waitForTimeout(600);
    }
    const result = await page.evaluate(() => {
      const rect = (element) => {
        const box = element.getBoundingClientRect();
        return { left: box.left, top: box.top, right: box.right, bottom: box.bottom };
      };
      const inside = (point, box) => point.x > box.left + 1 && point.x < box.right - 1
        && point.y > box.top + 1 && point.y < box.bottom - 1;
      const tiles = [...document.querySelectorAll(".react-flow__node")]
        .filter((element) => element.querySelector(".rf-model-node"))
        .map((element) => ({ id: element.getAttribute("data-id"), box: rect(element) }));
      const frameIds = new Set([...document.querySelectorAll(".react-flow__node.groupFrame")]
        .map((element) => element.getAttribute("data-id")));
      const hits = [];
      for (const edgeElement of document.querySelectorAll(".react-flow__edge")) {
        const id = edgeElement.getAttribute("data-id") || "";
        const [rawSource, rawTarget] = id.split("=>");
        const pathElement = edgeElement.querySelector(".react-flow__edge-path");
        if (!pathElement || !rawTarget) continue;
        const sourceId = frameIds.has(`frame-${rawSource}`) ? `frame-${rawSource}` : rawSource;
        const targetId = frameIds.has(`frame-${rawTarget}`) ? `frame-${rawTarget}` : rawTarget;
        const endpointIds = new Set([sourceId, targetId]);
        const length = pathElement.getTotalLength();
        const transform = pathElement.getScreenCTM();
        if (!transform) continue;
        const pointAt = (distance) => {
          const point = pathElement.getPointAtLength(distance);
          return {
            x: point.x * transform.a + point.y * transform.c + transform.e,
            y: point.x * transform.b + point.y * transform.d + transform.f,
          };
        };
        const hitTiles = new Set();
        const samples = Math.max(20, Math.ceil(length / 8));
        for (let index = 1; index < samples - 1; index += 1) {
          const point = pointAt((length * index) / samples);
          for (const tile of tiles) {
            if (!endpointIds.has(tile.id) && inside(point, tile.box)) hitTiles.add(tile.id);
          }
        }
        if (hitTiles.size) {
          hits.push({
            id,
            className: edgeElement.className.baseVal || "",
            obscuredBy: [...hitTiles].slice(0, 20),
          });
        }
      }
      return { tileCount: tiles.length, edgeCount: document.querySelectorAll(".react-flow__edge").length, hits };
    });
    findings.push({ model_id: entry.model_id, ...result });
    console.log(`${entry.model_id}: ${result.hits.length} potential occlusion(s)`);
  } catch (error) {
    findings.push({ model_id: entry.model_id, error: String(error) });
    console.error(`${entry.model_id}: ERROR ${error}`);
  }
}

await fs.mkdir(path.dirname(outputPath), { recursive: true });
await fs.writeFile(outputPath, JSON.stringify({ generated_at: new Date().toISOString(), base_url: baseUrl, findings }, null, 2));
await browser.close();
const errors = findings.filter((item) => item.error);
const withHits = findings.filter((item) => item.hits?.length);
console.log(JSON.stringify({
  models: findings.length,
  errors: errors.length,
  models_with_potential_occlusion: withHits.length,
  potential_occlusions: findings.reduce((sum, item) => sum + (item.hits?.length || 0), 0),
  zero_tile_models: findings.filter((item) => item.tileCount === 0).map((item) => item.model_id),
  output: outputPath,
}, null, 2));
