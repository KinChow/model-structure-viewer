import fs from "node:fs/promises";

const pageUrl = process.env.MSV_PAGE_URL || "http://127.0.0.1:4173/";
const debugHost = process.env.MSV_CHROME_DEBUG_HOST || "127.0.0.1";
const debugPort = process.env.MSV_CHROME_DEBUG_PORT || "9223";
const screenshotPath = process.env.MSV_SCREENSHOT || "";

async function requestJson(url, options) {
  const response = await fetch(url, options);
  const text = await response.text();
  if (!response.ok) {
    throw new Error(`${url} returned HTTP ${response.status}: ${text.slice(0, 200)}`);
  }
  return text ? JSON.parse(text) : null;
}

async function createPage() {
  const target = await requestJson(`http://${debugHost}:${debugPort}/json/new?${encodeURIComponent(pageUrl)}`, {
    method: "PUT",
  });
  if (!target.webSocketDebuggerUrl) {
    throw new Error("Chrome DevTools did not return a page websocket URL");
  }
  return target.webSocketDebuggerUrl;
}

class CdpSession {
  constructor(wsUrl) {
    this.ws = new WebSocket(wsUrl);
    this.nextId = 1;
    this.pending = new Map();
    this.events = [];
    this.ws.addEventListener("message", (event) => this.handleMessage(event));
  }

  async open() {
    if (this.ws.readyState === WebSocket.OPEN) return;
    await new Promise((resolve, reject) => {
      this.ws.addEventListener("open", resolve, { once: true });
      this.ws.addEventListener("error", reject, { once: true });
    });
  }

  handleMessage(event) {
    const message = JSON.parse(event.data);
    if (message.id && this.pending.has(message.id)) {
      const { resolve, reject } = this.pending.get(message.id);
      this.pending.delete(message.id);
      if (message.error) reject(new Error(`${message.error.message}: ${message.error.data || ""}`.trim()));
      else resolve(message.result);
      return;
    }
    if (message.method) this.events.push(message);
  }

  send(method, params = {}) {
    const id = this.nextId++;
    this.ws.send(JSON.stringify({ id, method, params }));
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
    });
  }

  close() {
    this.ws.close();
  }
}

function expression(body) {
  return `(async () => { ${body} })()`;
}

async function evaluate(cdp, body, timeoutMs = 15000) {
  const result = await cdp.send("Runtime.evaluate", {
    expression: expression(body),
    awaitPromise: true,
    returnByValue: true,
    timeout: timeoutMs,
  });
  if (result.exceptionDetails) {
    throw new Error(result.exceptionDetails.text || "Runtime evaluation failed");
  }
  return result.result.value;
}

async function waitFor(cdp, conditionBody, label, timeoutMs = 15000) {
  const started = Date.now();
  while (Date.now() - started < timeoutMs) {
    const ok = await evaluate(cdp, conditionBody, 1000);
    if (ok) return true;
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
  throw new Error(`Timed out waiting for ${label}`);
}

async function runPageSmoke(cdp) {
  await cdp.send("Page.navigate", { url: pageUrl });
  await waitFor(cdp, "return Boolean(document.querySelector('button.primary'));", "application shell");

  const basics = await evaluate(cdp, `
    const catalogUrl = new URL('models/catalog.json', document.baseURI).toString();
    const catalog = await fetch(catalogUrl).then((response) => response.json());
    return {
      title: document.title,
      catalogCount: catalog.models.length,
      sourceOptions: Array.from(document.querySelectorAll('select[aria-label="source"] option')).map((option) => option.value),
      hasModelInput: Boolean(document.querySelector('input[aria-label="model id"]')),
    };
  `);

  await evaluate(cdp, `
    document.querySelector('button.primary').click();
    return true;
  `);
  await waitFor(cdp, `
    return document.body.innerText.includes('Source') &&
      document.body.innerText.includes('built-in config') &&
      document.body.innerText.includes('DeepseekV3ForCausalLM');
  `, "default model generation");

  const architecture = await evaluate(cdp, `
    return {
      hasSvg: Boolean(document.querySelector('svg[aria-label="Model architecture diagram"]')),
      text: document.body.innerText,
    };
  `);

  await evaluate(cdp, `
    const tab = Array.from(document.querySelectorAll('[role="tab"]')).find((button) => button.textContent.trim() === 'Layers');
    tab.click();
    return true;
  `);
  await waitFor(cdp, "return document.body.innerText.includes('Layers');", "Layers tab");
  await evaluate(cdp, `
    const button = Array.from(document.querySelectorAll('.layers-panel button')).find((item) => item.textContent.trim() === 'Expand all');
    button.click();
    return true;
  `);
  await waitFor(cdp, `
    const text = document.body.innerText;
    return text.includes('Decoder Layers') && text.includes('Routed MoE') && text.includes('formula');
  `, "expanded Layers content");
  const layersText = await evaluate(cdp, "return document.body.innerText;");

  await evaluate(cdp, `
    const tab = Array.from(document.querySelectorAll('[role="tab"]')).find((button) => button.textContent.trim() === 'Export');
    tab.click();
    return true;
  `);
  await waitFor(cdp, "return Boolean(document.querySelector('.export-panel textarea'));", "Export tab");
  await evaluate(cdp, `
    const button = Array.from(document.querySelectorAll('.export-panel button')).find((item) => item.textContent.trim() === 'Export');
    button.click();
    return true;
  `);
  await waitFor(cdp, `
    const output = document.querySelector('.export-panel textarea')?.value || '';
    return output.includes('flowchart TD') && output.includes('DeepseekV3ForCausalLM');
  `, "Mermaid export");
  const exportPreview = await evaluate(cdp, "return document.querySelector('.export-panel textarea').value.slice(0, 160);");

  await evaluate(cdp, `
    const tab = Array.from(document.querySelectorAll('[role="tab"]')).find((button) => button.textContent.trim() === 'Raw Config');
    tab.click();
    return true;
  `);
  await waitFor(cdp, `
    const raw = document.querySelector('.export-panel textarea')?.value || '';
    return raw.includes('deepseek_v3') && raw.includes('DeepseekV3ForCausalLM');
  `, "Raw Config content");
  const rawPreview = await evaluate(cdp, "return document.querySelector('.export-panel textarea').value.slice(0, 160);");

  return {
    basics,
    architecture: {
      hasSvg: architecture.hasSvg,
      hasExpectedModel: architecture.text.includes("DeepseekV3ForCausalLM"),
      hasExpectedSource: architecture.text.includes("built-in config"),
      hasFrontendTemplateStatus: architecture.text.includes("Frontend template"),
    },
    layers: {
      hasDecoderLayers: layersText.includes("Decoder Layers"),
      hasRoutedMoe: layersText.includes("Routed MoE"),
      hasFormula: layersText.includes("formula"),
    },
    export: {
      hasMermaid: exportPreview.includes("flowchart TD"),
      preview: exportPreview,
    },
    rawConfig: {
      hasDeepseekConfig: rawPreview.includes("deepseek_v3") || rawPreview.includes("DeepseekV3ForCausalLM"),
      preview: rawPreview,
    },
  };
}

async function verifyAllBuiltinModels(cdp) {
  await cdp.send("Page.navigate", { url: pageUrl });
  await waitFor(cdp, "return Boolean(document.querySelector('button.primary'));", "application shell");

  const entries = await evaluate(cdp, `
    if (!window.__msvOriginalFetch) {
      window.__msvOriginalFetch = window.fetch.bind(window);
      window.__msvFetchUrls = [];
      window.fetch = (...args) => {
        const target = typeof args[0] === 'string' ? args[0] : args[0]?.url;
        if (target) {
          window.__msvFetchUrls.push(decodeURIComponent(new URL(target, location.href).pathname));
        }
        return window.__msvOriginalFetch(...args);
      };
    }
    const catalogUrl = new URL('models/catalog.json', document.baseURI).toString();
    const catalog = await fetch(catalogUrl).then((response) => response.json());
    return catalog.models.map((entry) => ({
      model_id: entry.model_id,
      config_path: entry.config_path,
      architecture: Array.isArray(entry.architectures) ? entry.architectures[0] : '',
    }));
  `);

  const failures = [];
  const passed = [];
  for (const entry of entries) {
    const result = await evaluate(cdp, `
      const entry = ${JSON.stringify(entry)};
      window.__msvFetchUrls = [];
      const sourceSelect = document.querySelector('select[aria-label="source"]');
      const modelInput = document.querySelector('input[aria-label="model id"]');
      const nativeSelectSetter = Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, 'value').set;
      const nativeInputSetter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set;
      nativeSelectSetter.call(sourceSelect, 'builtin');
      sourceSelect.dispatchEvent(new Event('change', { bubbles: true }));
      nativeInputSetter.call(modelInput, entry.model_id);
      modelInput.dispatchEvent(new Event('input', { bubbles: true }));
      modelInput.dispatchEvent(new Event('change', { bubbles: true }));
      document.querySelector('button.primary').click();
      const expectedConfigSuffix = \`/models/\${entry.config_path}\`;
      const started = Date.now();
      while (Date.now() - started < 5000) {
        await new Promise((resolve) => setTimeout(resolve, 50));
        const text = document.body.innerText;
        const error = document.querySelector('.error')?.innerText || '';
        const loadedExpectedConfig = (window.__msvFetchUrls || []).some((url) => url.endsWith(expectedConfigSuffix));
        if (error) return { ok: false, model_id: entry.model_id, error };
        if (
          loadedExpectedConfig &&
          text.includes(entry.architecture || entry.model_id) &&
          text.includes('built-in config') &&
          Boolean(document.querySelector('svg[aria-label="Model architecture diagram"]'))
        ) {
          return {
            ok: true,
            model_id: entry.model_id,
            architecture: entry.architecture,
          };
        }
      }
      return {
        ok: false,
        model_id: entry.model_id,
        error: 'Timed out waiting for built-in page generation',
        text: document.body.innerText.slice(0, 300),
      };
    `, 7000);
    if (result.ok) passed.push(result);
    else failures.push(result);
  }
  return { total: entries.length, passed: passed.length, failed: failures.length, failures };
}

async function main() {
  const wsUrl = await createPage();
  const cdp = new CdpSession(wsUrl);
  await cdp.open();
  try {
    await cdp.send("Page.enable");
    await cdp.send("Runtime.enable");
    await cdp.send("Emulation.setDeviceMetricsOverride", {
      width: 1440,
      height: 1000,
      deviceScaleFactor: 1,
      mobile: false,
    });

    const smoke = await runPageSmoke(cdp);
    const models = await verifyAllBuiltinModels(cdp);

    if (screenshotPath) {
      const shot = await cdp.send("Page.captureScreenshot", { format: "png", captureBeyondViewport: true });
      await fs.writeFile(screenshotPath, Buffer.from(shot.data, "base64"));
    }

    const report = {
      pageUrl,
      smoke,
      models,
    };
    console.log(JSON.stringify(report, null, 2));

    const sourceOptions = new Set(smoke.basics.sourceOptions);
    const smokeOk =
      smoke.basics.title === "Model Structure Viewer" &&
      smoke.basics.catalogCount === models.total &&
      sourceOptions.has("auto") &&
      sourceOptions.has("builtin") &&
      smoke.basics.hasModelInput &&
      smoke.architecture.hasSvg &&
      smoke.architecture.hasExpectedModel &&
      smoke.architecture.hasExpectedSource &&
      smoke.layers.hasDecoderLayers &&
      smoke.layers.hasRoutedMoe &&
      smoke.layers.hasFormula &&
      smoke.export.hasMermaid &&
      smoke.rawConfig.hasDeepseekConfig;

    if (!smokeOk || models.failed > 0) {
      process.exit(1);
    }
  } finally {
    cdp.close();
  }
}

main().catch((error) => {
  console.error(error.stack || error.message);
  process.exit(1);
});
