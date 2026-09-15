import { test as base, expect } from "@playwright/test";

// ref: https://playwright.dev/docs/network — 同时记录和拦截产品源 /api 请求。
// Hugging Face 自身的 /api/models 属于公开模型源，不是 MSV 后端。
export const test = base.extend({
  frontendBoundary: [async ({ context, baseURL }, use) => {
    const apiRequests = [];
    const errors = [];
    const isBackendRequest = (url) => {
      const target = new URL(url);
      return target.origin === new URL(baseURL).origin && /^\/api(?:\/|$)/.test(target.pathname);
    };
    context.on("request", (request) => {
      if (isBackendRequest(request.url())) apiRequests.push(request.url());
    });
    context.on("page", (page) => page.on("pageerror", (error) => errors.push(error.message)));
    await context.route(isBackendRequest, (route) => route.abort());
    await use();
    expect(apiRequests, "产品前端不得访问 MSV 后端").toEqual([]);
    expect(errors, "浏览器不得出现未捕获异常").toEqual([]);
  }, { auto: true }],
});

export { expect };
