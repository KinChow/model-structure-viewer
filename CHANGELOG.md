# 更新日志

这里记录 Model Structure Viewer 的主要变化。

项目按语义化版本管理：

- `MAJOR`：API、命令行或缓存格式有不兼容变化。
- `MINOR`：向后兼容的新功能、模型适配、界面改进或诊断能力。
- `PATCH`：向后兼容的修复、文档更新和测试调整。

## [未发布]

### 新增

- 开始使用这份更新日志管理版本变化。
- 前端增加状态诊断，可以区分 `meta-introspect`、前端组网和 config fallback。
- 前端可以从粘贴的 config JSON 生成结构，结果里包含模块、算子和公式信息。
- 新增 `frontend/src/structure` 结构目录，拆分为 config 归一化、registry、模型构建、layers、ops、formulas、IR、materializers、diagnostics 和 catalog。
- 新增仓库内置模型配置目录 `models/`，方便直接用 `--root ./models` 启动和验证，也能用于静态部署。
- 新增 `models/catalog.json` 和前端 `builtin` 来源，GitHub Pages 这类静态站点也可以直接读取内置配置并在前端组网。
- 新增内置模型验证脚本 `npm run verify:models`，用于检查 catalog 中的配置能否完成前端组网。
- 新增页面级验证脚本 `npm run verify:page`，通过 Chrome 打开静态页面，逐个模型验证 `builtin` 生成链路。
- 新增 `/api/local/config`，网页端可以直接读取本地 `config.json`，不必先调用后端模型 introspection。
- 新增中文架构说明：`docs/frontend_structure_architecture.md`。
- 前端支持 JSON、Mermaid 和 DOT 导出，config-only 场景不再依赖 `/api/export`。
- 增加前端单测，覆盖架构图默认 fit/center 和结构状态文案。

### 修复

- 对相同结构请求增加进程内缓存，减少重复的后端 introspection 开销。
- 后端结构生成改成分层路径：优先 config 输出，超预算时直接 fallback，必要时再用隔离 worker 做 introspection。
- 改进嵌套 `text_config` 的 fallback 输出，让 decoder layer 数量能挂到对应的 text 节点下。
- 改进 config fallback 的 decoder layers，能展示来自 config 的 Dense/MoE 分组，并保留可展开的 attention、MLP/MoE 和 norm 骨架。
- Architecture 图默认居中并适配窗口，缩放按钮仍然基于 fit 后的视图工作。
- Layers 支持折叠重复 pattern，例如 `A x3 + B + A x3 + B` 可以合成一个 pattern group，同时不隐藏尾部不完整结构。
- 调整 loading 和诊断文案，生成中显示 `Generating...`，fallback 原因用状态标签展示。

### 已知问题

- 不支持或超出预算的模型仍可能退回到 config-derived 视图。现在界面会把这件事明示为 warning，不再悄悄隐藏。

## [0.1.0] - 2026-07-04

### 新增

- 支持从本地模型目录和 Hugging Face model id 查看模型结构。
- 提供 FastAPI 服务，包含本地模型列表、Hugging Face 查询、结构生成、导出和设置接口。
- 提供 React 网页端，包含数据来源选择、本地缓存抽屉、summary 标签、Architecture、Layers、Export 和 Raw Config 标签页。
- 使用 `$MODEL_ROOT/<org>/<model>/config.json` 作为本地模型缓存布局。
- Hugging Face 只缓存 `config.json`、`README.md`、`configuration_*.py`、`modeling_*.py` 和 `tokenization_*.py`，不会缓存权重。
- 支持 meta-device 模型 introspection，并提供 config-only fallback 诊断。
- 针对部分 remote-code/config 兼容问题加入修复逻辑，包括 DeepSeek import 兼容和 MiniMax-M3 config 适配。
- 支持 JSON、Mermaid 和 DOT 导出。
- 增加测试，覆盖 resolver、API 结构响应、修复策略、折叠逻辑、fallback 构造和导出。

### 说明

- 这个工具用于从模型配置和元数据查看结构。它不下载权重，也不运行推理。
