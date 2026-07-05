# 更新日志

这里记录 Model Structure Viewer 的主要变化。

项目按语义化版本管理：

- `MAJOR`：API、命令行或缓存格式有不兼容变化。
- `MINOR`：向后兼容的新功能、模型适配、界面改进或诊断能力。
- `PATCH`：向后兼容的修复、文档更新和测试调整。

## [未发布]

### 新增

- 新增测试 Skill 文档，固定单测、后端 transformers/API 验证和浏览器页面验证流程。
- 开始使用这份更新日志管理版本变化。
- Layers 卡片和详情面板增加输入/输出维度，使用 `batch`、`sequence`、`hidden size` 这类完整名字，避免缩写看不懂。
- 前端结构生成器增加统一的 shape 推导，覆盖 embedding、decoder、attention、MLP、MoE、norm、lm head、vision tower 和 projector。
- 页面级验证增加 readable shape 检查，确认 Layers 展开后能看到输入/输出维度。
- 前端增加状态诊断，可以区分前端组网、后端 meta introspection 和修复后的 meta introspection。
- 前端可以从粘贴的 config JSON 生成结构，结果里包含模块、算子和公式信息。
- 新增 `frontend/src/structure` 结构目录，拆分为 config 归一化、registry、模型构建、layers、ops、formulas、IR、materializers、diagnostics 和 catalog。
- 新增仓库内置模型配置目录 `models/`，方便直接用 `--root ./models` 启动和验证，也能用于静态部署。
- 新增 `models/catalog.json` 和前端 `builtin` 来源，GitHub Pages 这类静态站点也可以直接读取内置配置并在前端组网。
- 新增内置模型验证脚本 `npm run verify:models`，用于检查 catalog 中的配置能否完成前端组网。
- 新增页面级验证脚本 `npm run verify:page`，通过 Chrome 打开静态页面，逐个模型验证 `builtin` 生成链路。
- 新增 `/api/local/config`，网页端可以直接读取本地 `config.json`，不必先调用后端模型 introspection。
- 新增后端 transformers 验证入口：`msv verify` 和 `POST /api/verify`。它们只检查 meta-device 下 `AutoModel.from_config` 是否成功，不使用 config fallback。
- Transformers 验证遇到本机缺少 FlashAttention2 时，会临时切到 `sdpa` 重新做结构验证，并在 diagnostics 中记录 `attention_backend_retry`。
- Transformers 验证增加 MiniMax-M2.7 `rope_parameters` 兼容、Kimi-K2.5/Kimi-K2.6 `is_torch_fx_available` 和 `tie_weights` 签名兼容。
- 后端结构接口复用同一套 transformers 兼容逻辑，避免 `msv verify` 通过但 `/api/structure` 失败。
- 新增中文架构说明：`docs/frontend_structure_architecture.md`。
- 前端支持 JSON、Mermaid 和 DOT 导出，config-only 场景不再依赖 `/api/export`。
- 增加前端单测，覆盖架构图默认 fit/center 和结构状态文案。

### 修复

- 补齐 config 归一化里的 `head_dim`、`intermediate_size`、`moe_intermediate_size` 和 `vocab_size`，让维度展示能带上具体数值。
- 对相同结构请求增加进程内缓存，减少重复的后端 introspection 开销。
- 移除后端 config 兜底结构。`/api/structure` 和 `msv inspect` 现在只返回 transformers introspection 的真实结果；不支持的模型会直接报错。
- Architecture 图默认居中并适配窗口，缩放按钮仍然基于 fit 后的视图工作。
- Layers 支持折叠重复 pattern，例如 `A x3 + B + A x3 + B` 可以合成一个 pattern group，同时不隐藏尾部不完整结构。
- 调整 loading 和诊断文案，生成中显示 `Generating...`，结构来源和修复策略用状态标签展示。
- 后端 worker 会屏蔽第三方 transformers 代码写到 stdout/stderr 的日志，避免 `msv inspect --format json` 输出被污染。
- 后端默认 worker 超时调到 90 秒，避免 Kimi-K2.5 这类多次 transformers 兼容重试的结构检查被过早杀掉。
- Hugging Face tree 查询是元数据补全的 best-effort 探测，网络失败时不再向 CLI stderr 打 warning。

### 已知问题

- 后端结构接口依赖 transformers 和本地 remote code。新模型如果还没有被 transformers 支持，会返回明确错误；静态页面的 `builtin` 和 `config` 来源仍然走前端组网。

### 验证

- 前端单测：`npm --prefix frontend test`，26 个用例通过。
- 前端内置模型结构验证：`npm --prefix frontend run verify:models`，44/44 通过。
- 前端构建：`npm --prefix frontend run build` 通过。
- 浏览器页面验证：`npm --prefix frontend run verify:page`，44/44 通过，并确认 readable shape。
- 后端单测：`.venv/bin/pytest -q`，115 个用例通过。
- 后端 transformers 严格验证：仓库内置 44 个模型 `msv verify --root ./models --offline` 全部通过。
- 后端 HTTP 验证：真实启动 `msv serve`，`/api/models` 返回 44 个模型，`/api/structure` 44/44 通过，结构策略只出现 `meta-introspect` 和 `repaired-meta-introspect`。

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
