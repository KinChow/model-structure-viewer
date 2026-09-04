import fs from "node:fs/promises";

const pageUrl = process.env.MSV_PAGE_URL || "http://127.0.0.1:5174/";
const debugHost = process.env.MSV_CHROME_DEBUG_HOST || "127.0.0.1";
const debugPort = process.env.MSV_CHROME_DEBUG_PORT || "9223";
const screenshotPath = process.env.MSV_SCREENSHOT || "";

async function json(url, options) {
  const response = await fetch(url, options);
  const text = await response.text();
  if (!response.ok) throw new Error(`${url}: HTTP ${response.status} ${text.slice(0, 160)}`);
  return text ? JSON.parse(text) : null;
}

class Cdp {
  constructor(url) {
    this.ws = new WebSocket(url);
    this.id = 0;
    this.pending = new Map();
    this.ws.addEventListener("message", (event) => {
      const message = JSON.parse(event.data);
      const request = this.pending.get(message.id);
      if (!request) return;
      this.pending.delete(message.id);
      if (message.error) request.reject(new Error(message.error.message));
      else request.resolve(message.result);
    });
  }
  async open() {
    if (this.ws.readyState === WebSocket.OPEN) return;
    await new Promise((resolve, reject) => {
      this.ws.addEventListener("open", resolve, { once: true });
      this.ws.addEventListener("error", reject, { once: true });
    });
  }
  send(method, params = {}) {
    const id = ++this.id;
    this.ws.send(JSON.stringify({ id, method, params }));
    return new Promise((resolve, reject) => this.pending.set(id, { resolve, reject }));
  }
  close() { this.ws.close(); }
}

async function page(cdp, body, timeout = 15000) {
  const result = await cdp.send("Runtime.evaluate", { expression: `(async()=>{${body}})()`, awaitPromise: true, returnByValue: true, timeout });
  if (result.exceptionDetails) throw new Error(result.exceptionDetails.text || "Runtime evaluation failed");
  return result.result.value;
}

async function wait(cdp, body, label, timeout = 15000) {
  const started = Date.now();
  while (Date.now() - started < timeout) {
    if (await page(cdp, body, 1000)) return;
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
  throw new Error(`Timed out waiting for ${label}`);
}

async function navigate(cdp) {
  await cdp.send("Page.navigate", { url: pageUrl });
  await wait(cdp, "return Boolean(document.querySelector('.model-entry-page'));", "model entry");
}

async function clickText(cdp, selector, text) {
  return page(cdp, `const item=Array.from(document.querySelectorAll(${JSON.stringify(selector)})).find((node)=>node.textContent.trim()===${JSON.stringify(text)}); item?.click(); return Boolean(item);`);
}

async function openBuiltin(cdp) {
  if (!await clickText(cdp, ".entry-quick button", "MiniMax-M3")) throw new Error("MiniMax-M3 quick entry not found");
  await wait(cdp, "return Boolean(document.querySelector('.diagram-frame'));", "detail workspace");
}

async function smoke(cdp) {
  await navigate(cdp);
  const entry = await page(cdp, `return {title:document.title, input:Boolean(document.querySelector('input[aria-label="model id"]')), providers:document.querySelectorAll('.provider-card').length};`);
  await openBuiltin(cdp);
  const graph = await page(cdp, `const toggle=document.querySelector('.formula-strip-toggle'); toggle?.click(); await new Promise((r)=>setTimeout(r,0)); return {svg:Boolean(document.querySelector('svg[aria-label="Model architecture diagram"]')), nodes:document.querySelectorAll('g[data-node-path]').length, formulas:document.querySelectorAll('.formula-strip-links button[data-node-path]').length, layersButton:Array.from(document.querySelectorAll('button')).some((b)=>b.textContent.trim()==='Layers')};`);
  const analysis = await clickText(cdp, ".toolbar-actions > button", "分析配置") || await clickText(cdp, ".toolbar-actions > button", "Analysis");
  const compare = await page(cdp, `const button=Array.from(document.querySelectorAll('.lens-mode-switch button')).find((b)=>['方案','Plan'].includes(b.textContent.trim())); button?.click(); await new Promise((r)=>setTimeout(r,0)); return {analysis:${analysis},panes:document.querySelectorAll('.diagram-compare .diagram-frame').length,compareTp:document.body.innerText.includes('对比 TP')||document.body.innerText.includes('Compare TP')};`);
  const formula = await page(cdp, `const button=document.querySelector('.formula-strip-links button[data-node-path]'); button?.click(); await new Promise((r)=>setTimeout(r,0)); return {path:button?.dataset.nodePath||'',inspector:Boolean(document.querySelector('.detail-panel')),expanded:Boolean(document.querySelector('.formula-strip-toggle[aria-expanded="true"]'))};`);
  await navigate(cdp);
  await openBuiltin(cdp);
  const cost = await page(cdp, `document.querySelector('.detail-cost-toggle > button')?.click(); await new Promise((r)=>setTimeout(r,350)); document.querySelector('.cost-expand-button')?.click(); await new Promise((r)=>setTimeout(r,100)); Array.from(document.querySelectorAll('.cost-segmented button')).find((b)=>b.textContent.includes('PD'))?.click(); await new Promise((r)=>setTimeout(r,0)); return {lens:Boolean(document.querySelector('.cost-lens-row')),pd:Boolean(document.querySelector('.pd-deployment-summary')),kv:document.body.innerText.includes('KV Transfer')||document.body.innerText.includes('KV 传输')};`);
  await navigate(cdp);
  await openBuiltin(cdp);
  const auxiliary = await page(cdp, `const buttons=Array.from(document.querySelectorAll('.detail-aux-actions button')); const exportButton=buttons.find((b)=>/Export|导出/.test(b.textContent)); exportButton?.click(); await new Promise((r)=>setTimeout(r,0)); document.querySelector('.export-panel button')?.click(); await new Promise((r)=>setTimeout(r,0)); const mermaid=document.querySelector('.export-panel textarea')?.value||''; const rawButton=buttons.find((b)=>/Raw config|原始配置/.test(b.textContent)); rawButton?.click(); await new Promise((r)=>setTimeout(r,0)); return {mermaid:mermaid.includes('flowchart TD'),raw:Boolean(document.querySelector('.export-panel textarea')?.value)};`);
  return {entry, graph, analysis, compare, formula, cost, auxiliary};
}

const wsUrl = (await json(`http://${debugHost}:${debugPort}/json/new?${encodeURIComponent(pageUrl)}`, { method: "PUT" })).webSocketDebuggerUrl;
if (!wsUrl) throw new Error("Chrome DevTools did not return a websocket URL");
const cdp = new Cdp(wsUrl);
await cdp.open();
try {
  await cdp.send("Page.enable");
  await cdp.send("Runtime.enable");
  await cdp.send("Emulation.setDeviceMetricsOverride", { width: 1440, height: 1000, deviceScaleFactor: 1, mobile: false });
  const result = await smoke(cdp);
  if (screenshotPath) {
    const shot = await cdp.send("Page.captureScreenshot", { format: "png", captureBeyondViewport: true });
    await fs.writeFile(screenshotPath, Buffer.from(shot.data, "base64"));
  }
  console.log(JSON.stringify({ pageUrl, result }, null, 2));
  const ok = result.entry.title === "Model Structure Viewer" && result.entry.input && result.entry.providers > 0 && result.graph.svg && result.graph.nodes > 0 && result.graph.formulas > 0 && !result.graph.layersButton && result.compare.panes === 2 && result.formula.path && result.formula.inspector && result.cost.lens && result.cost.pd && result.cost.kv && result.auxiliary.mermaid && result.auxiliary.raw;
  if (!ok) process.exit(1);
} finally {
  cdp.close();
}
