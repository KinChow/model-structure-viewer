import { test, expect } from "./fixtures.js";

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
    await expect(options.locator(".compact-list button")).toHaveCount(59);
    await expect(options).not.toContainText(forbiddenUi);
    await options.getByRole("button", { name: english ? "Close" : "关闭", exact: true }).click();
    await expect(options).toBeHidden();
    await page.screenshot({ path: test.info().outputPath(`frontend-${language}.png`), fullPage: true });
  });
}
