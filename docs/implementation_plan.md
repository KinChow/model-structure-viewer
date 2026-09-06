# 待实现计划

本文只记录当前仍待实现或需要继续维护的事项。已经完成的能力以代码、测试和 [`CHANGELOG.md`](../CHANGELOG.md) 为准；历史版本通过 Git 提交记录追溯。

## 当前基线

以下能力已经落地，不再作为待实现任务：

- 前端静态结构生成、内置模型 catalog、HF/ModelScope 配置读取和 safetensors header 真值接入。
- 模型 registry、通用和专用 builder、layers、ops、公式、IR、materializer 和诊断链路。
- 结构图、Layers、Inspector、JSON/Mermaid/DOT 导出、芯片 Cost Lens、并行投影和 PD 分析。
- Python API/CLI、local cache、transformers meta-device 验证和 GitHub Pages 构建流程。

## P0：验证流程收口

### 浏览器验证脚本适配 React Flow

现状：`npm --prefix frontend run verify:page` 仍包含旧版 SVG 和页面选择器，当前不能作为发布闸门。

实现范围：

- 使用当前 React Flow DOM 结构验证节点、边、布局和图面板。
- 将首页生成、方案对比、公式联动、Layers、Export 和 Raw Config 检查改为稳定的 `role`、`aria-*` 或专用 data 属性。
- 保留静态页面与前后端页面两种验证模式，不依赖快捷模型排序。

验收：页面验证脚本在干净构建产物上通过；所有内置模型仍由 `verify:models` 覆盖。

## P1：协议和维护性

### 统一结构协议的生成或契约测试

当前前后端 schema 与前端 materializer 仍由两边维护。优先增加跨端契约样例和字段兼容测试；只有重复维护成本继续上升时，才引入 schema-first 生成，避免为了工具本身扩大构建复杂度。

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
