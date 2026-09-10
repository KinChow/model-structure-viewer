import { defineConfig, devices } from "@playwright/test";

const baseURL = "http://127.0.0.1:4173";
const browserChannel = process.env.CI ? undefined : "chrome";

export default defineConfig({
  testDir: "./e2e",
  fullyParallel: false,
  workers: 1,
  // P0 后收尾：59 模型重用例存在偶发失败（2026-09-10 实测一次，同日同配置两次
  // 全量通过，根因无日志可查）。Playwright 官方重试惯例兜底：CI 两次、本地一次；
  // retry 仍失败的用例照红，不掩盖真实回归。
  retries: process.env.CI ? 2 : 1,
  timeout: 30_000,
  expect: { timeout: 8_000 },
  reporter: "line",
  use: {
    baseURL,
    ...(browserChannel ? { channel: browserChannel } : {}),
    headless: true,
    locale: "zh-CN",
    screenshot: "only-on-failure",
    trace: "retain-on-failure",
  },
  projects: [
    {
      name: "desktop-chrome",
      use: { ...devices["Desktop Chrome"], viewport: { width: 1440, height: 1000 } },
    },
    {
      name: "mobile-chrome",
      use: { ...devices["Pixel 7"] },
    },
  ],
  webServer: {
    command: "npm run dev -- --host 127.0.0.1 --port 4173",
    url: baseURL,
    // P0 收尾实证（2026-09-10）：长时运行 + 多轮 HMR 的 dev server 会让 React
    // Flow 节点布局持续不稳定（"element is not stable" 耗尽 click 超时，全量
    // 连续两轮同用例失败；杀掉 4173 旧 server 后 9/9 全绿）。每轮强制全新
    // server，消除该变量——代价仅 ~2s 启动。
    reuseExistingServer: false,
    timeout: 30_000,
  },
});
