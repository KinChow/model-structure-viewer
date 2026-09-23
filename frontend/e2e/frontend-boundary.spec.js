import { test, expect } from "./fixtures.js";
import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";

const forbiddenUi = /Backend unavailable|后端不可用|Verify with Transformers|用 Transformers 校验|Backend Local Models|后端本地模型|Model root|模型根目录|Save settings|保存设置|Open path|打开路径/;

for (const language of ["zh", "en"]) {
  test(`纯前端入口与模型选项 (${language})`, async ({ page }) => {
    const english = language === "en";
    await page.addInitScript((locale) => localStorage.setItem("msv-language", locale), language);
    await page.route(/https:\/\/(?:www\.)?(?:huggingface\.co|modelscope\.cn)\//, (route) => route.abort());
    await page.goto("/");
    await expect(page.locator("body")).not.toContainText(forbiddenUi);
    await page.getByRole("button", { name: english ? "Open local model directory" : "打开本地模型目录", exact: true }).click();
    await expect(page.getByRole("button", { name: english ? "Choose folder" : "打开文件夹", exact: true })).toBeVisible();
    await expect(page.getByLabel("local model path")).toHaveCount(0);
    await page.getByRole("button", { name: english ? "Enter / choose model" : "输入 / 选择模型", exact: true }).click();
    await page.getByLabel("model id").fill("deepseek-ai/DeepSeek-V3.1");
    await page.getByRole("button", { name: english ? "Open model" : "打开模型", exact: true }).click();
    await expect(page.locator(".detail-page")).toBeVisible();
    await expect(page.locator(".diagnostics-meta")).toContainText(english ? "tensors" : "张量");
    await expect(page.locator("body")).not.toContainText(forbiddenUi);
    await page.getByRole("button", { name: english ? "Model options" : "模型选项", exact: true }).click();
    const options = page.getByRole("dialog", { name: english ? "Model options" : "模型选项", exact: true });
    await expect(options).toBeVisible();
    await options.getByLabel("Revision", { exact: true }).fill("test-revision");
    await expect(options.getByLabel("Revision", { exact: true })).toBeFocused();
    await expect(options.getByLabel(english ? "Hugging Face search" : "Hugging Face 搜索", { exact: true })).toBeVisible();
    await expect(options.locator(".compact-list button")).toHaveCount(60);
    await expect(options).not.toContainText(forbiddenUi);
    await options.getByRole("button", { name: english ? "Close" : "关闭", exact: true }).click();
    await expect(options).toBeHidden();
    await page.screenshot({ path: test.info().outputPath(`frontend-${language}.png`), fullPage: true });
  });
}

const remoteConfig = { model_type: "qwen3", architectures: ["Qwen3ForCausalLM"], num_hidden_layers: 1, hidden_size: 64, num_attention_heads: 4, num_key_value_heads: 2, intermediate_size: 128, vocab_size: 256 };

for (const language of ["zh", "en"]) {
  test(`远程读取与搜索直连，失败不回退 (${language})`, async ({ page }) => {
    const english = language === "en";
    const openLabel = english ? "Open model" : "打开模型";
    const optionsLabel = english ? "Model options" : "模型选项";
    const searchLabel = english ? "Hugging Face search" : "Hugging Face 搜索";
    const remoteRequests = [];
    await page.addInitScript((locale) => localStorage.setItem("msv-language", locale), language);
    await page.route(/https:\/\/(?:www\.)?(?:huggingface\.co|modelscope\.cn)\//, (route) => {
      const url = new URL(route.request().url());
      remoteRequests.push(url.href);
      if (url.pathname.endsWith("/config.json") && url.pathname.includes("/org/remote/")) return route.fulfill({ json: remoteConfig });
      if (url.origin === "https://huggingface.co" && url.pathname === "/api/models" && url.searchParams.get("search") === "remote") {
        return route.fulfill({ json: [{ id: "org/remote", pipeline_tag: "text-generation" }] });
      }
      if (url.pathname.endsWith("/config.json")) return route.fulfill({ status: 404, json: { error: "not found" } });
      return route.abort();
    });
    await page.goto("/");
    await page.getByLabel("model id").fill("org/missing");
    await page.getByRole("button", { name: openLabel, exact: true }).click();
    await expect(page.getByRole("alert")).toContainText("org/missing");
    await expect(page.getByRole("alert")).toContainText("modelscope");
    await page.getByLabel("model id").fill("org/remote");
    await page.getByRole("button", { name: openLabel, exact: true }).click();
    await expect(page.locator(".detail-page")).toBeVisible();
    await expect(page.locator(".detail-model-id")).toHaveText("org/remote");
    await expect(page.locator(".react-flow__node").first()).toBeVisible();
    await page.getByRole("button", { name: optionsLabel, exact: true }).click();
    const options = page.getByRole("dialog", { name: optionsLabel, exact: true });
    await options.getByLabel(searchLabel, { exact: true }).fill("remote");
    await options.getByRole("button", { name: english ? "Search" : "搜索", exact: true }).click();
    await expect(options.getByRole("button", { name: "org/remote text-generation" })).toBeVisible();
    await options.getByRole("button", { name: "org/remote text-generation" }).click();
    await expect(options).toBeHidden();
    await expect(page.locator(".detail-page")).toBeVisible();
    await page.getByRole("button", { name: optionsLabel, exact: true }).click();
    await options.getByLabel(searchLabel, { exact: true }).fill("failed-search");
    await options.getByRole("button", { name: english ? "Search" : "搜索", exact: true }).click();
    await expect(page.locator(".detail-error")).toContainText(english ? "Hugging Face search failed" : "Hugging Face 搜索失败");
    expect(remoteRequests.some((url) => url.includes("/org/missing/"))).toBe(true);
    expect(remoteRequests.some((url) => url.includes("/api/models?"))).toBe(true);
  });

  test(`旧本地路径链接提示重新选择目录 (${language})`, async ({ page }) => {
    await page.addInitScript((locale) => localStorage.setItem("msv-language", locale), language);
    await page.goto("/?source=local&config_path=%2Fold%2Fmodel");
    await expect(page.getByRole("alert")).toContainText(language === "en" ? "choose the model folder again" : "重新选择文件夹");
    await expect(page.getByLabel("model id")).toBeVisible();
    await expect(page.locator("body")).not.toContainText(forbiddenUi);
  });
}

for (const language of ["zh", "en"]) {
  test(`浏览器目录读取、取消、错误及导出 (${language})`, async ({ page }, testInfo) => {
    test.setTimeout(60_000);
    const english = language === "en";
    await page.addInitScript((locale) => localStorage.setItem("msv-language", locale), language);
    await page.route(/https:\/\/(?:www\.)?(?:huggingface\.co|modelscope\.cn)\//, (route) => route.abort());
    const makeDirectory = async (name, files) => {
      const dir = testInfo.outputPath("folders", name);
      await mkdir(dir, { recursive: true });
      for (const [file, content] of Object.entries(files)) await writeFile(path.join(dir, file), content);
      return dir;
    };
    const choose = async (dir) => {
      const pending = page.waitForEvent("filechooser");
      await page.getByRole("button", { name: english ? "Choose folder" : "打开文件夹", exact: true }).click();
      await (await pending).setFiles(dir);
    };
    await page.goto("/");
    await page.getByRole("button", { name: english ? "Open local model directory" : "打开本地模型目录", exact: true }).click();
    const input = page.locator('input[type="file"]');
    // 标准取消事件与空 FileList 都不能触发构建或报错；不用模拟系统文件对话框的按键。
    await input.dispatchEvent("cancel");
    expect(await input.evaluate((element) => element.files.length)).toBe(0);
    await input.dispatchEvent("change");
    await expect(page.locator(".entry-loading")).toHaveCount(0);
    await expect(page.getByRole("alert")).toHaveCount(0);
    const missing = await makeDirectory("missing", { "tokenizer_config.json": "{}" });
    await choose(missing);
    await expect(page.getByRole("alert")).toContainText(english ? "No config.json" : "没有找到 config.json");
    const invalid = await makeDirectory("invalid", { "config.json": "{" });
    await choose(invalid);
    await expect(page.getByRole("alert")).toContainText(english ? "Unable to read config.json" : "无法读取所选文件夹中的 config.json");
    const array = await makeDirectory("array", { "config.json": "[]" });
    await choose(array);
    await expect(page.getByRole("alert")).toContainText(english ? "JSON object" : "JSON 对象");
    const json = JSON.stringify(remoteConfig);
    const configOnly = await makeDirectory("config-only", { "config.json": json });
    await choose(configOnly);
    await expect(page.locator(".detail-page")).toBeVisible();
    await expect(page.locator(".react-flow__node").first()).toBeVisible();
    await expect(page.locator("body")).not.toContainText(forbiddenUi);
    await page.getByRole("button", { name: /Model Structure Viewer v/ }).click();
    await page.getByRole("button", { name: english ? "Open local model directory" : "打开本地模型目录", exact: true }).click();
    const header = Buffer.from(JSON.stringify({ "model.embed_tokens.weight": { dtype: "BF16", shape: [256, 64], data_offsets: [0, 32768] } }));
    const prefix = Buffer.alloc(8);
    prefix.writeBigUInt64LE(BigInt(header.length));
    const withHeader = await makeDirectory("with-header", {
      "config.json": json,
      "model.safetensors": Buffer.concat([prefix, header, Buffer.alloc(32768)]),
    });
    await choose(withHeader);
    await expect(page.locator(".detail-page")).toBeVisible();
    await expect(page.locator(".diagnostics-meta")).toContainText(english ? "tensors" : "张量");
    await page.locator(".detail-aux-actions").getByRole("button", { name: english ? "Export" : "导出", exact: true }).click();
    const panel = page.locator(".export-panel");
    for (const format of ["json", "mermaid", "dot"]) {
      await panel.locator("select").selectOption(format);
      await expect(panel.locator("textarea")).not.toHaveValue("");
      const text = await panel.locator("textarea").inputValue();
      if (format === "json") {
        const exported = JSON.parse(text);
        expect(exported.summary.parameters_total).toBe(16384);
        expect(exported.source.checkpoint_truth).toBe("available");
        expect(exported.graph.nodes.length).toBeGreaterThan(0);
      } else expect(text).toContain(format === "dot" ? "digraph" : "flowchart");
    }
    await page.screenshot({ path: testInfo.outputPath(`local-export-${language}.png`), fullPage: true });
  });
}
