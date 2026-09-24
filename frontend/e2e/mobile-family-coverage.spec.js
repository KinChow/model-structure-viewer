import { expect, test } from "./fixtures.js";

// One representative from each built-in model_type. This is intentionally a
// mobile-only smoke matrix: it checks the rendered product path, not just the
// in-memory graph builder, while keeping the full 60-model desktop scan
// separate from the family coverage gate.
const representatives = [
  ["minimax_m2", "MiniMaxAI/MiniMax-M2.7"],
  ["minimax_m3_vl", "MiniMaxAI/MiniMax-M3"],
  ["qwen3_5", "Qwen/Qwen3.5-0.8B"],
  ["qwen3_5_moe", "Qwen/Qwen3.5-122B-A10B"],
  ["qwen3_5_moe_text", "Qwen/Qwen3.8-2.4T-A95B"],
  ["qwen4_exp", "Qwen/Qwen3.8-Flash-Next"],
  ["deepseek_v3", "deepseek-ai/DeepSeek-R1"],
  ["deepseek_v32", "deepseek-ai/DeepSeek-V3.2"],
  ["deepseek_v4", "deepseek-ai/DeepSeek-V4-Flash"],
  ["deepseek_v41", "deepseek-ai/DeepSeek-V4.1-Flash"],
  ["kimi_k2", "moonshotai/Kimi-K2-Base"],
  ["kimi_k25", "moonshotai/Kimi-K2.5"],
  ["kimi_k3", "moonshotai/Kimi-K3"],
  ["glm4_moe", "zai-org/GLM-4.7"],
  ["glm_moe_dsa", "zai-org/GLM-5"],
  ["glm5_next", "zai-org/GLM-5.3-Flash"],
];

test("mobile Chrome renders every built-in architecture family", async ({ page }) => {
  test.skip(test.info().project.name !== "mobile-chrome", "family matrix is mobile-only");
  test.setTimeout(900_000);
  page.setDefaultTimeout(15_000);
  await page.route(/https:\/\/(?:www\.)?(?:huggingface\.co|modelscope\.cn)\//, route => route.abort());
  await page.goto("/");

  for (const [modelType, modelId] of representatives) {
    await test.step(`${modelType}: ${modelId}`, async () => {
      await page.getByLabel("model id").fill(modelId);
      await page.getByRole("button", { name: "打开模型", exact: true }).click();
      await expect(page.locator(".detail-page")).toBeVisible();
      await expect(page.locator(".detail-model-id")).toHaveText(modelId);
      await expect(page.locator(".react-flow-diagram")).toHaveAttribute("data-graph-version", "2");
      await expect(page.locator(".detail-answer-bar")).toBeVisible();

      const expand = page.getByRole("button", { name: "展开全部", exact: true });
      if (await expand.count()) {
        await expand.click();
        await expect(page.locator(".react-flow__node").first()).toBeVisible();
        await page.getByRole("button", { name: "收起全部", exact: true }).click();
      }

      const costToggle = page.locator(".detail-cost-toggle > button");
      if (await costToggle.count()) {
        await costToggle.click();
        await expect(page.locator(".cost-summary")).toBeVisible();
      }
    });
  }
});
