# 全量内置模型前端布局系统验证（2026-09-26）

## 验证口径

- 浏览器：Google Chrome stable，通过 Playwright `channel: "chrome"`，不是 Chromium fallback。
- 视口：1440×1000。
- 模型：`models/catalog.json` 全部 60 个内置模型。
- 每个模型都在 Chrome 中打开 builtin route，点击页面真实的“展开全部”，等待所有真实 `react-flow__node-msvNode` 的“展开”按钮消失；随后检查节点、边、容器、尺寸、同级重叠、横向溢出、剩余折叠模块和 page error；最后点击“收起全部”并验证节点/边恢复。
- 另有逐级单模块展开抽查，覆盖 MiniMax、Qwen、DeepSeek、Kimi、GLM 的代表性模型和特殊模块。

## 总结果

| 指标 | 结果 |
|---|---:|
| 全量模型 | 60/60 |
| 展开后仍有未展开真实模块 | 0 |
| 展开后坏尺寸节点 | 0 |
| 展开后同级模块重叠 | 0 |
| 展开后横向溢出 | 0 |
| page error | 0 |
| 收起后节点/边恢复 | 60/60 |
| 模型专属布局问题 | 0 个模型 |

展开前合计 459 个真实模块节点、488 条边；展开后合计 39,543 个真实模块节点、45,745 条边。

## 发现的问题

### 共享资源问题：`/favicon.ico` 返回 404

- Kimi-K3 批次的 Chrome console 记录了两条资源 404。
- 单独请求 `http://127.0.0.1:4176/favicon.ico` 得到 HTTP 404；首页本身 HTTP 200。
- 这不是 Kimi-K3 的模型布局问题，而是前端共享静态资源问题；会在某些新 Chrome context 的控制台中出现。
- 建议补充 favicon，或在 HTML 中明确引用现有 favicon 资源。

### 未发现模型专属布局故障

60 个模型在“展开全部”后均满足：没有残留折叠的真实模块、没有检测到坏尺寸节点、没有检测到同级模块矩形重叠、没有横向溢出；“收起全部”后节点/边计数均恢复。

## 代表性外部/源码对照

联网读取了 Sebastian Raschka 的 LLM Architecture Gallery 页面。Gallery 页面本身列出 DeepSeek V3/V3.2/V4、Qwen3/Qwen3.5/Qwen3.8、Kimi K2/K2.5/K3、GLM-5/5.2/5.3、MiniMax M2/M3 等模型，并以架构图和事实卡片描述 decoder、MoE、MLA/DSA、混合注意力、视觉路径等结构特征。

仓库的 gallery alignment projection 和本地模型配置与这些结构类别可对应：

- DeepSeek-V3.1：MLA、AttnRes、Decoder-only；
- Qwen3.8-Flash-Next：DeltaNet/QSA、视觉塔、gated residual；
- Kimi-K3：KDA/Gated MLA、视觉塔、Attention Residual；
- GLM-5.2：DSA/MLA、IndexShare；
- MiniMax-M3：视觉塔、Multi-modal Projector、Patch Merge、稀疏注意力。

该对照用于验证显示层的族/模块命名，不替代各模型论文或官方源码真值。

## 证据文件

- 完整机器报告：`/tmp/msv-full-layout-system-audit-2026-09-26.json`
- 工作区机器报告：`docs/details/evidence/frontend/full-layout-system-audit-2026-09-26.json`
- 分批原始报告：`/tmp/msv-global-batch1.json` 至 `/tmp/msv-global-batch4.json`
- 全量结论条/成本扫描：`docs/details/evidence/frontend/builtin-scan-report.json`

## 工作区边界

本轮没有修改产品代码，也没有提交或 push；原有 ELK 未提交改动保持不变。本报告只记录验证结果。
