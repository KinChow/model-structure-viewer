# 前端全量布局后续系统验证（2026-09-26）

## 验证范围

- 浏览器：Google Chrome stable，通过 Playwright `channel: "chrome"`。
- 全量模型：`models/catalog.json` 的 60 个内置模型。
- 移动视口：Pixel 7，412×915；每个模型执行展开全部、布局检查、收起恢复。
- 短桌面视口：1000×750；每个模型执行展开全部、布局检查、收起恢复。
- 补充 UI：深/浅色主题、模型选项抽屉、60 个模型列表、英文公式索引、搜索框、面包屑换行、内置模型 Verify 控件边界、横向溢出。
- 回归套件：桌面 Chrome 82 passed、移动 Chrome 82 passed；每轮 10 skipped 为项目条件跳过，不是失败。

## 后续全量结果

| 项目 | 结果 |
|---|---:|
| 移动 Chrome 全量模型 | 60/60 |
| 移动展开后残留折叠模块 | 0 |
| 移动坏尺寸节点 | 0 |
| 移动同级重叠模型 | 0 |
| 移动横向溢出模型 | 0 |
| 移动收起恢复 | 60/60 |
| 短桌面全量模型 | 60/60 |
| 短桌面展开后残留折叠模块 | 0 |
| 短桌面坏尺寸节点 | 0 |
| 短桌面同级重叠模型 | 0 |
| 短桌面横向溢出模型 | 0 |
| 短桌面收起恢复 | 60/60 |
| UI 补充检查 | 16/16 |
| page error | 0 |

## 发现的问题

### 已修复：共享静态资源 `/favicon.ico` 返回 404

- 已新增 `frontend/public/favicon.ico`，并在 `frontend/index.html` 显式声明 ICO 与 SVG。
- Chrome preview 验证 `/favicon.ico` 返回 HTTP 200，未再记录 favicon 资源错误。

### 已修复：视觉模型 Roofline / 融合流量建模缺口

- 新增 `multimodalFusionCounts`，按 `batch × (sequence × hidden + visionTokens × visionHidden)` 估算融合输入搬运，按 `batch × sequence × hidden` 估算融合输出搬运；decode 不重复执行视觉路径。
- 39 个视觉模型的视觉塔、投影器、融合节点仍保持独立建模；Kimi-K3 的 AttnRes snapshot 已按参考实现的 dense token-major `torch.cat` 路径计入搬运。
- 全量成本扫描已由 `flagged=39` 变为 `flagged=0`；当前没有模型因成本 Roofline unknown 被豁免。

## 修复后验证（2026-09-27）

- 60 个模型在 Pixel 7 和 1000×750 短桌面视口中都能完整展开。
- 没有残留折叠模块、坏尺寸、同级重叠、横向溢出或 page error。
- 深/浅色主题切换、模型选项抽屉、60 个内置模型列表、英文公式索引、搜索框和深层面包屑均通过补充检查。
- 内置模型路径没有暴露后端 Verify 控件或后端专属 UI。
- `npm test`：628/628 通过。
- `npm run build`：通过。
- `npm run docs:check`：通过。
- 桌面 `framework-accounting.spec.js`：18/18 通过。
- 桌面 `scan-builtin.spec.js`：60/60，`flagged=0`。
- 全量成本静态审计：60/60 模型、prefill/decode 两相位均无 `weights/actIn/actOut` null，Roofline unknown 数量为 0。
- 本轮未提交或 push；保留原有 ELK 未提交改动。

## 证据文件

- `docs/details/evidence/frontend/mobile-full-expand-followup-2026-09-26.json`
- `docs/details/evidence/frontend/short-desktop-full-expand-followup-2026-09-26.json`
- `docs/details/evidence/frontend/ui-followup-2026-09-26.json`
- `docs/details/evidence/frontend/builtin-scan-report.json`
- `docs/details/evidence/frontend/full-layout-system-audit-2026-09-26.json`
- `frontend/public/favicon.ico`
- `frontend/src/structure/operators/formulas/counts.js`
