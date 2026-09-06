import { defineConfig, devices } from "@playwright/test";

const baseURL = "http://127.0.0.1:4173";
const browserChannel = process.env.CI ? undefined : "chrome";

export default defineConfig({
  testDir: "./e2e",
  fullyParallel: false,
  workers: 1,
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
    reuseExistingServer: true,
    timeout: 30_000,
  },
});
