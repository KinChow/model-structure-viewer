# Changelog

Model Structure Viewer 的重要变更记录。

本文件遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/) 格式，版本号遵循 [Semantic Versioning](https://semver.org/lang/zh-CN/)。

项目按语义化版本管理：

- `MAJOR`：API、命令行或缓存格式有不兼容变化。
- `MINOR`：向后兼容的新功能、模型适配、界面改进或诊断能力。
- `PATCH`：向后兼容的修复、文档更新和测试调整。

## [Unreleased]

- 后端 introspection 改为通过 `GraphDraft` 直接生成 Graph IR v2，`StructureNode` 只作为兼容投影和旧调用入口。
- 前端搜索、节点选择、祖先展开、面包屑和顶层模块列表优先从 Graph IR 稳定路径读取，新增 graph selector 单测。
- 本阶段验证：后端 `147 passed`，前端 `195 passed`，内置模型 `59/59`，生产构建通过，Playwright 桌面/移动 `6/6` 通过。

## [0.2.0] - 2026-09-06

### 新增

- 前端可直连 Hugging Face / ModelScope 获取公开配置与 safetensors header；无模板模型可直接用 checkpoint trie 生成含参模块树。
- 新增权重、KV、激活、运行时和通信缓冲显存分解，以及 Prefill/Decode 的逐模块 MACs 与三类 roofline bound。
- 新增 TP/PP/EP/DP given-plan 投影、逐 stage fit、最大上下文、通信量和 PD 分离分析。
- 新增互斥的芯片对比与方案对比模式，只突出 bound 翻转节点；公式、Architecture 和 Layers 使用同一路径双向联动。
- 公开芯片目录增加 A100、H100、L40S 官方规格及字段级来源；保留 `chips.local.json` 与会话手动录入入口。
- 内部结构 IR 升级为 `version: 2`；GitHub Pages 官方工作流增加前端单测和 59 模型验证闸门。
- 新增测试 Skill 文档，固定单测、后端 transformers/API 验证和浏览器页面验证流程。
- 建立基于 Keep a Changelog 和 Semantic Versioning 的版本管理流程。
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

- 修复 MLA KV 被误算为两份 latent、GQA/DP-attention KV 切分、PP fold 重复倍数和首尾 stage 平均摊薄问题。
- 修复 EP all-to-all 被多个节点重复归因、专家路径未按 EP 投影，以及 TP/EP 方案对比仍使用未切分节点成本的问题。
- 修复 checkpoint 真值参与前端构建时仍显示 `Frontend template` 的状态错误。
- 移动端页头不再粘性遮挡 Architecture 控件。
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
- 页面验证脚本 `npm --prefix frontend run verify:page` 仍需适配现版 React Flow 页面选择器，当前未将其作为发布闸门。
- 国产芯片公开规格目录按当前计划暂缓；非公开或未完整公开的数据仍通过本地配置或手动录入，不在仓库中填估算值。

### 验证

- 前端单测：`npm --prefix frontend test`，172 个用例通过；F3–F17 中要求测试的公式均有对应用例。
- 前端内置模型结构验证：`npm --prefix frontend run verify:models`，59/59 通过。
- 前端构建：`npm --prefix frontend run build` 通过。
- 浏览器手动验证：SmolLM2-135M checkpoint 节点账本与模型参数均为 134,515,008；Qwen 低互联 Decode TP1→TP8 出现 2 个 `memory→comm` 翻转；后端 local/settings 路径通过。
- 后端单测：`.venv/bin/pytest -q`，130 个用例通过。

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

[Unreleased]: https://github.com/KinChow/model-structure-viewer/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/KinChow/model-structure-viewer/releases/tag/v0.2.0
[0.1.0]: https://github.com/KinChow/model-structure-viewer/releases/tag/v0.1.0
