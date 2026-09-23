# 内置模型 Chrome 全量端到端验证报告

日期：2026-09-23 · 范围：`models/catalog.json` 全部 **60** 个内置模型 · 结论：**60/60 通过，0 展示层问题，0 非展示层待办**。

## 方法
- 引擎：Chromium（Playwright 1.63，`@playwright/test` 自带浏览器；与 Chrome 同内核），对本地 dev server（Vite，`127.0.0.1`）逐模型加载 `/?model=<model_id>&source=builtin`。
- 全量扫描（60 个）：每模型独立浏览器上下文（空 localStorage → 默认 `viewPreset=custom`、`activeLenses={vram}`、成本面板折叠），加载后打开成本面板，抽取并断言：header model_id、模型摘要（参数量）、顶层模块数、架构图节点数、成本指标（Total VRAM / Fit / Max context / MACs·token / MACs·forward / FLOPs·forward / Roofline bound / Communication）、显存分解（Weights/Buffers/KV/State + KV ownership）、默认 VRAM lens 角标数、`pageerror`/console error。
- 代表深检（14 个，覆盖全部架构族）：切换 **Compute** 与 **Memory traffic** lens 读取逐节点角标（`.diagram-lens-value.lens-compute/.lens-memory`）、选中节点读取节点详情 Cost Lens 行（MACs/FLOPs/Roofline/Memory traffic/Communication）与真值/公式区。
- 交互行为（视图预设、自定义面板胶囊、配置分区折叠摘要、lens 三轴命名、KV 归入 VRAM）此前已在真实 Chrome（Codex CUA）逐一目视确认；这些为模型无关的 UI 行为，不需 60× 重复。
- 脚本：`/tmp/e2e-scan.cjs`（全量）、`/tmp/e2e-scan2.cjs`（深检），结果 `/tmp/e2e-results*.json`。均为临时文件，未纳入仓库。

## 覆盖结果（全部通过）
| Provider | 模型数 | 结果 |
|---|---|---|
| MiniMaxAI | 3 | 3/3 ok |
| Qwen | 31 | 31/31 ok |
| deepseek-ai | 9 | 9/9 ok |
| moonshotai | 8 | 8/8 ok |
| zai-org | 9 | 9/9 ok |
| **合计** | **60** | **60/60 ok** |

全 60 个模型逐项断言均通过（每项均"ALL 60 ok"）：Total VRAM、Max context、MACs·forward、FLOPs·forward、Roofline bound（无 unknown）、Communication、Fit（yes/no，非 unknown）、显存分解 Weights+KV、摘要参数量、顶层模块>0、架构图节点>0、默认 VRAM lens 角标>0。示例（Qwen3.5-9B）：`Total VRAM 18.10 GiB · Fit yes · Max context 1,645,026 · MACs/forward 16.97 T · FLOPs/forward 33.93 T · Roofline matrix · checkpoint truth`。

代表深检 14 个（Qwen 稠密/MoE/Flash-Next/GPTQ-Int4、DeepSeek V3.1/V3.2/V4.1-Flash/V4-Vision、MiniMax-M3/-MXFP8、Kimi-K2/-K3、GLM-5.3/-5.3-Flash）：**Compute** 与 **Memory traffic** lens 逐节点角标均渲染（4–7 个/模型，值非空）；节点详情 Cost Lens 5 行（MACs/FLOPs/Roofline/Memory traffic/Communication）均有值；真值/公式区存在。

## 展示层问题（当场修复）
无。本次未发现展示层缺陷，因而无代码改动产生于本次验证。

## 非展示层待后续（落档）
无。60 个模型的结构与成本数据全部正确展示，无 config 解析、缺证据、unsupported、coverage 缺口或口径异常。

## 两处误报说明（非缺陷）
- `moonshotai/Kimi-K3` 全量扫描报 `pageerror:8`：实为构建 1920 节点大图期间的 `net::ERR_CONNECTION_RESET` 资源加载重置（console 资源错误，非 JS 异常）；其数据正常（顶层模块 7、图节点 15、Total VRAM 1448.74 GiB、Roofline matrix、lens 角标 7）。
- 深检脚本对 14 个模型报 `vram-lens-empty`：系测试方法问题——VRAM lens 默认即开启，脚本先点 Compute/Memory 再点 VRAM，末次点击把默认开启的 VRAM lens 又切回关闭。全量扫描已证实默认 VRAM lens 在 60/60 均渲染角标，VRAM lens 正常。

## 复现
- URL：`http://127.0.0.1:<port>/?model=<model_id>&source=builtin`（dev：`npm run dev`）。
- 基线：`npm run verify:models` 60/60 结构通过；`npm test` 468/468；`npm run build` 通过。

## 附录 · 2026-09-23 结论条打磨复验
- 打磨点：结论条（`.detail-answer-bar`）瓶颈项从原始单位键改为友好归类——`matrix/vector/sfu → Compute-bound / 算力受限`、`memory → Memory-bound / 访存受限`、`comm → Communication-bound / 通信受限`；Cost 面板 Roofline 明细行仍显示原始 `矩阵/访存/…`，精确单位不丢。
- 展示层问题（当场修复，1 处）：结论条显存项标签原为「单卡显存 / Per-card VRAM」，但取值实为整模型 `Total VRAM`（与 Cost 面板 `cost.totalVram` 同值，例：MiniMax-M3 795.77 GiB 远超单卡 74.51 GiB）——标签与数据口径不符，已改为「总显存 / Total VRAM」。属纯标签修复，计算层无改动。
- 自动化复验：新增 `frontend/e2e/scan-builtin.spec.js`，逐个打开全部 60 个内置模型，中/英双语各采一次结论条（Fit / 总显存 / 瓶颈 / GPU）与 Cost 面板（Roofline `data-bound`），落 `builtin-scan-report.json`。结果 `total=60 flagged=0`：瓶颈归类中/英均落地、`data-bound` 无 `unknown`、无 `NaN/undefined/空值`、无未捕获异常。瓶颈分布 39 Compute-bound / 21 Memory-bound。
- 非展示层待后续（落档）：无。
- 基线复跑：`npm test` 469/469（新增 1 项 bound 归类单测）；`npm run build` 通过。
