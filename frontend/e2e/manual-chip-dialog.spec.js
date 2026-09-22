import { expect, test } from "./fixtures.js";

// 在吸顶栏和弹窗重叠的位置做实际命中测试，不能只检查 z-index 数值。
for (const theme of ["dark", "light"]) {
  for (const short of [false, true]) {
    test(`自定义芯片顶层弹窗 ${theme} ${short ? "短视口" : "标准视口"}`, async ({ page }, testInfo) => {
      test.setTimeout(60_000);
      const mobile = testInfo.project.name === "mobile-chrome";
      if (short) await page.setViewportSize({ width: mobile ? 412 : 1400, height: 600 });
      await page.route(/https:\/\/(?:www\.)?(?:huggingface\.co|hf-mirror\.com|modelscope\.cn)\//, route => route.abort());
      await page.goto("/");
      await page.getByLabel("model id").fill("deepseek-ai/DeepSeek-V3.1");
      await page.getByRole("button", { name: "打开模型", exact: true }).click();
      await expect(page.locator(".detail-page")).toBeVisible();
      if (theme === "light") await page.getByRole("button", { name: "切换到浅色主题", exact: true }).click();
      const costToggle = page.locator(".detail-cost-toggle > button");
      await costToggle.click();
      await page.getByRole("button", { name: "展开配置", exact: true }).click();
      // 把吸顶栏置于滚动后的顶部，以重现用户截图。
      await page.locator(".detail-main").evaluate(el => {
        if (getComputedStyle(el).overflowY === "auto") el.scrollTop = 100;
        else window.scrollTo(0, el.getBoundingClientRect().top + window.scrollY);
      });
      const trigger = page.getByRole("button", { name: "自定义 GPU", exact: true });
      await trigger.click();
      const overlay = page.locator("body > .manual-chip-backdrop");
      const dialog = page.getByRole("dialog", { name: "添加自定义芯片", exact: true });
      await expect(overlay).toHaveClass(new RegExp(`theme-${theme}`));
      await expect(dialog).toBeVisible();
      const geometry = await page.evaluate(() => {
        const overlay = document.querySelector("body > .manual-chip-backdrop");
        const dialog = overlay.querySelector(".manual-chip-form");
        const d = dialog.getBoundingClientRect();
        const bar = document.querySelector(".detail-cost-toggle").getBoundingClientRect();
        const y = Math.max(d.top + 2, bar.top + 2);
        const hit = document.elementFromPoint(d.left + d.width / 2, y);
        return {
          overlap: y < Math.min(d.bottom, bar.bottom),
          hitDialog: dialog.contains(hit),
          bounds: d.top >= 0 && d.left >= 0 && d.bottom <= innerHeight && d.right <= innerWidth,
          background: getComputedStyle(dialog).backgroundColor,
          viewportCovered: overlay.getBoundingClientRect().height === innerHeight,
        };
      });
      expect(geometry.bounds).toBe(true);
      expect(geometry.viewportCovered).toBe(true);
      expect(geometry.background).toBe(theme === "light" ? "rgb(255, 255, 255)" : "rgb(21, 30, 41)");
      if (short) {
        expect(geometry.overlap).toBe(true);
        expect(geometry.hitDialog).toBe(true);
      }
      const name = dialog.getByLabel("名称", { exact: true });
      await name.fill("Stacking regression GPU");
      await name.click();
      await expect(name).toBeFocused();
      await dialog.getByLabel("SFU TOPS", { exact: true }).fill("5");
      const submit = dialog.getByRole("button", { name: "加入芯片", exact: true });
      const close = dialog.getByRole("button", { name: "关闭", exact: true });
      await submit.focus();
      await page.keyboard.press("Tab");
      await expect(close).toBeFocused();
      await page.keyboard.press("Shift+Tab");
      await expect(submit).toBeFocused();
      await close.scrollIntoViewIfNeeded();
      await page.screenshot({ path: testInfo.outputPath("dialog-top-layer.png") });
      await page.keyboard.press("Escape");
      await expect(dialog).toHaveCount(0);
      await expect(trigger).toBeFocused();
      await expect(costToggle).toHaveAttribute("aria-expanded", "true");
      await trigger.click();
      await expect(name).toHaveValue("Stacking regression GPU");
      await dialog.getByRole("button", { name: "取消", exact: true }).click();
      await expect(trigger).toBeFocused();
      await trigger.click();
      await overlay.click({ position: { x: 3, y: 3 } });
      await expect(dialog).toHaveCount(0);
      await expect(trigger).toBeFocused();
      expect(await page.locator("body").evaluate(el => el.style.overflow)).not.toBe("hidden");
    });
  }
}
