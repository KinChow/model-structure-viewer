# 待实现计划

本文只记录当前仍待实现或需要继续维护的事项。已经完成的能力以代码、测试和 [`CHANGELOG.md`](../CHANGELOG.md) 为准；历史版本通过 Git 提交记录追溯。

## 当前基线

以下能力已经落地，不再作为待实现任务：

- 前端静态结构生成、内置模型 catalog、HF/ModelScope 配置读取和 safetensors header 真值接入。
- 模型 registry、通用和专用 builder、layers、ops、公式、IR、materializer 和诊断链路。
- 结构图、Layers、Inspector、JSON/Mermaid/DOT 导出、芯片 Cost Lens、并行投影和 PD 分析。
- Python API/CLI、local cache、transformers meta-device 验证和 GitHub Pages 构建流程。
- Graph IR v2 节点事实、graph-to-legacy-tree projection 和前后端 parity 测试。
- Graph-first layout、compute/aggregate、通信、PP/PD projection、导出和 canonical node identity。

## P0：验证流程收口（已完成）

Playwright 已替换旧版 SVG/CDP 验收路径。`npm --prefix frontend run test:e2e`
使用隔离 Vite 服务和桌面/移动 Chrome，覆盖入口来源、React Flow 节点与显式边、成本交互、窄屏布局和多模态视觉节点；`verify:page` 保留为兼容别名。

## P1：原则收口重构

[`principles.md`](principles.md) 已定为强制约束，当前代码存在多处存量偏离（登记在该文 §10）。
收口路线见 [`refactor_plan.md`](refactor_plan.md)：自底向上分 W0–W6（含 W4.5）+ 一条可并行旁路，
每波含范围、入口、依赖、验收命令、不包含项与回退方式，全部可用差分测试验收（行为不变）。

未排期项（需先决策，不属于任一波次）：

- **§6.4 来源解析归并**：endpoint fallback / revision 默认值 / auto 降级顺序目前前后端各一套且已不一致，需确定统一到前端 `model/loadModelArtifacts.js`。
- **§7 国产芯片条目**：每字段必须有公开来源，缺项保持 unknown。此项决定 W5 第 4 步（ERT 分离）的优先级。
- **§2.5 残差与跨层级边**：需先扩展 IR 才能表达，暂缓；不得用并列节点伪装。

### 统一结构协议的生成或契约测试

当前前后端 schema 与前端 materializer 仍由两边维护。优先增加跨端契约样例和字段兼容测试；只有重复维护成本继续上升时，才引入 schema-first 生成，避免为了工具本身扩大构建复杂度。

`schemas.py:13-50` 的 `StructureNode` 与 `StructureGraphNode` 逐字重复 13 个字段，`materializeStructureGraph.js:503-522` 在 JS 侧再抄一遍；抽公共基模型属本项范围。

### 模型 catalog 维护自动化

继续保持模型清单、配置、发布时间来源和验证报告可追溯。新增模型时应同时更新 catalog、来源记录、模型专项说明和验证结果，避免只增加配置却没有来源和验收记录。

## P2：条件性需求

以下事项只有在明确需求出现时才启动：

- 公开国产芯片规格扩展：每个字段必须有公开来源，缺项保持 unknown。
- 后端生产化：补充路径约束、remote code 沙箱、鉴权、请求限制、日志脱敏和会话级 settings。
- 更细的模型专项模块或公式：先确认现有 IR 能否表达，优先扩展 `details/modules.md` 对应的 registry、builder、layers、ops 和 formulas。

## 明确不做

- plan 搜索、自动推荐最优并行配置、吞吐/延迟/TTFT 预测。
- 训练内存规划、运行时 trace、调度仿真和 KV transfer overlap 仿真。
- 前端引入 torch 或下载权重。

## 任务规则

每项待实现任务必须同时写清楚范围、代码入口、依赖、验收命令和不包含的内容。完成后从本文移入 changelog 或实现细节文档，不在本文件长期保留已完成事项。
