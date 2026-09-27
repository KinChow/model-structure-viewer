# 全量内置模型前端布局审计（2026-09-26）

## 范围

- 入口：Chrome channel，`http://127.0.0.1:4173/`
- 目录：`models/catalog.json` 当前 60 个内置模型
- 验证方向：顶层模块、同级模块、可展开父节点、子模块/算子层、成本面板、公式索引、对比画布、短桌面视口
- 数据与实现对照：
  - `frontend/src/structure/` 的 Graph IR、架构 builder、模型配方和 gallery terminology projection
  - `docs/architectures_reference.md`
  - `README.md`
- 外部资料：本次尝试检索 Sebastian Raschka 的 LLM Architecture Gallery 与论文，但检索工具未返回可引用正文；因此本报告不把外部资料逐条核验标记为完成，也不以外部检索结果替代仓库源码和本地模型配置证据。

## 已完成且通过

### 1. 模型目录和静态组网

命令：

```bash
node scripts/verify-builtin-models.mjs
```

结果：`total=60, passed=60, failed=0`。60 个模型均有非空 Graph IR、顶层模块和已注册算子公式。

### 2. Chrome 桌面端 E2E

命令：

```bash
npm run test:e2e -- --project=desktop-chrome
```

结果：`92 tests`，`82 passed`，`10 skipped`，`0 failed`，耗时约 15.3 分钟。

覆盖到的关键路径包括：

- 60 个内置模型分批打开、React Flow 节点/边和成本扫描
- 顶层父节点展开、Decoder Layer 展开、子模块和 Shape 详情
- MiniMax、Qwen、DeepSeek、Kimi、GLM 的特化结构
- 视觉塔、投影器、patch merge、DSA/DSV4/QSA/IndexShare 等路径
- SVG 导出、公式索引、Cost 展开、TP 输入编辑
- 芯片/方案双画布对比
- 短桌面视口、宽屏视口、Inspector 内部滚动

其中双画布回归已在当前 checkout 通过：

```bash
npm run test:e2e -- --project=desktop-chrome viewer.spec.js \
  -g '桌面对比模式保留两张可见 React Flow 画布'
```

结果：`1 passed`。

### 3. 全量结论条和成本展示扫描

命令：

```bash
npm run test:e2e -- --project=desktop-chrome scan-builtin.spec.js
```

结果：60/60 模型打开成功，测试通过；没有发现空结论、NaN/undefined、空显存或未分类 bound。

注意：该扫描的报告逻辑把 `data-bound=unknown` 作为 advisory anomaly，但最终失败门只拦截空值、未分类值、显存坏值和 NaN。因此“测试通过”不等于每个模型都有可计算 Roofline bound。

### 4. 构建和布局单测

```bash
npm test -- --test-name-pattern='ELK-only layout keeps descendant-to-compound output|edge to its ancestor terminates'
npm run build
```

结果：ELK 端点聚合方向测试通过；生产构建通过。构建仍有既有的大 chunk warning（ELK bundle 和 DetailWorkspace 超过 500 kB），不是本轮新增布局故障。

## 发现的问题

### P1：39/60 个模型的 Roofline bound 仍显示 `unknown`

本次全量扫描中，以下模型族出现 `roofline data-bound=unknown`：

- MiniMax-M3 两个变体
- Qwen3.5、Qwen3.6、Qwen3.8 的多组模型
- DeepSeek-V4-Flash-Vision-Exp、DeepSeek-V4.1-Flash
- Kimi-K2.5、Kimi-K2.6、Kimi-K2.7-Code、Kimi-K3
- GLM-5.3-Flash 两个变体

当前实现把部分多模态激活/图像流量无法由文本 workload 唯一确定的情况保留为 `unknown`，这在成本模型语义上可能是有意的保守状态；但从前端用户体验看，结论条同时显示“瓶颈：未知”，没有明确说明“未知来自视觉 token/融合流量未给定”，容易被误解为模型成本计算失败。

建议后续：

1. 在结论条和 Cost 面板直接显示 unknown 的来源；
2. 区分“模型结构存在但 workload 不足”和“公式/成本覆盖缺失”；
3. 对每个 unknown 模型族补一条配置/论文/源码证据和预期状态。

### P2：全量“逐级打开”自动化证据仍不完整

已有 E2E 覆盖了每个内置模型的父节点展开和多个家族的深层展开，但本轮额外编写的通用“逐级点击第一个可展开节点”巡检脚本没有形成可靠的全量结果，原因是：

- React Flow 同时渲染 `frame-root` 结构容器和真实模块节点，通用脚本一度把容器 frame 当成展开目标；
- 大图展开后目标按钮可能在当前画布 viewport 外，或被 MiniMap 命中测试拦截；
- 长时间逐模型运行时 Vite dev server 曾出现 `ERR_CONNECTION_REFUSED`，属于服务生命周期证据，不应当当作模型布局失败。

因此，不能把“所有模型的每一级子模块都已由人工/自动化逐层点开”标记为完成。当前结论是：全量打开、顶层结构和关键家族深层路径已通过；任意模型任意深度的通用逐级覆盖仍是 pending。

建议后续把深层巡检固定为：

- 每次展开前先点击 `Fit View`；
- 只选择真实 `msvNode`，排除 `groupFrame`；
- 记录每一级的 node/edge 数、目标 path、viewport 几何和 page error；
- 将超大模型按家族拆批，使用持久 Vite 服务，避免把服务重启误报成产品问题。

## 当前未复现为问题 / 已被当前证据覆盖

以下历史审计项在当前 checkout 的 Chrome E2E 中已得到正向证据，不再作为当前已确认故障：

- 桌面芯片/方案对比双画布高度为 0
- 短桌面公式索引不可点击
- 短视口画布过度压扁或横向溢出
- Cost/Formula 打开后覆盖内容
- 高视口 Inspector 被裁剪

以下历史审计项本轮没有单独重新取证，不能宣称已修复，也不能直接宣称仍然存在：深色 Settings Drawer、浅色主题 Summary 按钮、英文公式解释、后端不可用时 Verify 按钮、搜索结果截断、favicon 404、深层 breadcrumb 换行。

## 工作区边界

- 本轮没有提交代码，也没有 push。
- 审计开始前已经存在的未提交源码改动仍保留：
  - `/Users/zhouzijian01/Desktop/workspace/code/kinchow/model-structure-viewer/frontend/src/diagram/elkHierarchyEdges.js`
  - `/Users/zhouzijian01/Desktop/workspace/code/kinchow/model-structure-viewer/frontend/src/diagram/elkHierarchyEdges.test.js`
  - `/Users/zhouzijian01/Desktop/workspace/code/kinchow/model-structure-viewer/frontend/src/diagram/elkOnlyLayout.js`
  - `/Users/zhouzijian01/Desktop/workspace/code/kinchow/model-structure-viewer/frontend/src/diagram/elkOnlyLayout.test.js`
- 本报告只记录验证结果，没有把上述源码改动归因于本轮审计。
