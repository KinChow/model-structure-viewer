# Model Structure Viewer

这是一个模型结构查看工具。它读取 `config.json` 和少量模型元数据，生成结构图、层列表和导出内容。

工具只看结构，不下载权重，也不跑推理。当前网页端主要在前端完成组网：

```text
config.json -> registry -> model builder -> layers -> ops/formulas -> IR -> UI/export
```

后端主要负责本地配置读取、Hugging Face 配置读取，以及保留兼容用的结构接口。
仓库内置模型配置可以直接作为静态资源使用，适合部署到 GitHub Pages 这类静态站点。

## 路径

- 开发仓库：`/Users/zhouzijian01/Desktop/workspace/code/kinchow/model-structure-viewer`
- 仓库内置配置：`./models`
- 默认模型目录：`/Users/zhouzijian01/Desktop/workspace/models`
- 本地模型布局：`/Users/zhouzijian01/Desktop/workspace/models/<org>/<model>/config.json`
- MiniMax-M3 示例：`/Users/zhouzijian01/Desktop/workspace/models/MiniMaxAI/MiniMax-M3/config.json`

## 安装

```bash
cd /Users/zhouzijian01/Desktop/workspace/code/kinchow/model-structure-viewer
python3 -m venv .venv
.venv/bin/pip install -e '.[test]'
cd frontend && npm install
```

## 命令行

```bash
# 列出本地 config.json
.venv/bin/msv list --root /Users/zhouzijian01/Desktop/workspace/models

# 列出仓库内置配置
.venv/bin/msv list --root ./models

# 搜索 Hugging Face
.venv/bin/msv search MiniMax-M3 --endpoint https://huggingface.co

# 通过 Hugging Face model id 查看结构，并把配置元数据缓存到 MODEL_ROOT
.venv/bin/msv inspect --model MiniMaxAI/MiniMax-M3 --source hf --format json

# 优先用本地缓存，本地没有再访问 Hugging Face
.venv/bin/msv inspect --model MiniMaxAI/MiniMax-M3 --source auto --cache-policy prefer-local --format mermaid

# 直接查看一个 config 文件
.venv/bin/msv inspect --config tests/fixtures/minimax_m3/config.json --format dot

# 验证 transformers 是否能构造 meta 模型；失败会直接返回错误
.venv/bin/msv verify --root ./models --offline --model Qwen/Qwen3.5-0.8B --source local --cache-policy offline
```

## 网页

启动 API：

```bash
.venv/bin/msv serve --root /Users/zhouzijian01/Desktop/workspace/models --port 8000
```

如果只想用仓库里已经适配的配置，可以直接指定 `./models`：

```bash
.venv/bin/msv serve --root ./models --port 8000
```

启动前端：

```bash
cd frontend
npm run dev
```

打开 `http://localhost:5173`。Vite dev server 会把 API 请求转发到 `http://localhost:8000`。

## API

- `GET /api/models`
- `GET /api/local/config?model_id=MiniMaxAI/MiniMax-M3`
- `GET /api/hf/search?q=MiniMax-M3`
- `GET /api/hf/config?model_id=MiniMaxAI/MiniMax-M3&revision=main`
- `POST /api/structure`
- `POST /api/verify`
- `POST /api/export`
- `GET /api/settings`
- `POST /api/settings`

## 数据来源

- `local`：只读取 `$MODEL_ROOT/<org>/<model>/config.json`
- `builtin`：读取仓库内置 `models/<org>/<model>/config.json`，不需要后端
- `hf`：从 Hugging Face 拉取 `config.json` 和允许缓存的元数据
- `auto`：网页端优先读 `builtin`，再读后端本地缓存，最后才走后端兼容接口
- `config`：使用粘贴或上传的 JSON

允许缓存的 Hugging Face 元数据只有 `config.json`、`README.md`、`configuration_*.py`、`modeling_*.py` 和 `tokenization_*.py`。

下面这些权重或模型文件不会被这个工具缓存：

```text
.safetensors
.bin
.gguf
.pt
.onnx
```

网页端会优先从静态 `/models/catalog.json` 和 `/models/<org>/<model>/config.json` 读取内置配置，然后在前端生成结构。如果使用后端本地缓存或 Hugging Face 查询，才会访问 `/api/local/config` 和 `/api/hf/config`。`/api/structure` 仍然保留，主要用于命令行、兼容旧调用和后端诊断。

新模型适配时，后端验证优先使用 `msv verify` 或 `/api/verify`。它只检查 transformers meta-device 构造是否成功，不会在失败时生成一份看似可用的 config 结构。

## 已验证支持模型

当前 `models/catalog.json` 中有 44 个内置模型配置，已经全部通过两类验证：

- 配置组网验证：读取仓库内置 `config.json`，调用前端 `buildStructureFromConfig`，能够生成带 `summary`、`root` 和子节点的结构。
- 页面级验证：打开构建后的静态页面，逐个模型在页面里切到 `builtin`、填入 model id、点击 Generate，并确认页面读取了对应的 `models/<org>/<model>/config.json`。同时抽测 Architecture、Layers 展开、Export 和 Raw Config。

这不是权重验证，也不包含推理验证。

### Qwen

- `Qwen/Qwen3.5-0.8B`
- `Qwen/Qwen3.5-0.8B-Base`
- `Qwen/Qwen3.5-2B`
- `Qwen/Qwen3.5-2B-Base`
- `Qwen/Qwen3.5-4B`
- `Qwen/Qwen3.5-4B-Base`
- `Qwen/Qwen3.5-9B`
- `Qwen/Qwen3.5-9B-Base`
- `Qwen/Qwen3.5-27B`
- `Qwen/Qwen3.5-27B-FP8`
- `Qwen/Qwen3.5-27B-GPTQ-Int4`
- `Qwen/Qwen3.5-35B-A3B`
- `Qwen/Qwen3.5-35B-A3B-Base`
- `Qwen/Qwen3.5-35B-A3B-FP8`
- `Qwen/Qwen3.5-35B-A3B-GPTQ-Int4`
- `Qwen/Qwen3.5-122B-A10B`
- `Qwen/Qwen3.5-122B-A10B-FP8`
- `Qwen/Qwen3.5-122B-A10B-GPTQ-Int4`
- `Qwen/Qwen3.5-397B-A17B`
- `Qwen/Qwen3.5-397B-A17B-FP8`
- `Qwen/Qwen3.5-397B-A17B-GPTQ-Int4`
- `Qwen/Qwen3.6-27B`
- `Qwen/Qwen3.6-27B-FP8`
- `Qwen/Qwen3.6-35B-A3B`
- `Qwen/Qwen3.6-35B-A3B-FP8`

### DeepSeek

- `deepseek-ai/DeepSeek-R1`
- `deepseek-ai/DeepSeek-V3.1`
- `deepseek-ai/DeepSeek-V3.2`
- `deepseek-ai/DeepSeek-V4-Flash`
- `deepseek-ai/DeepSeek-V4-Pro`

### GLM

- `zai-org/GLM-4.7`
- `zai-org/GLM-5`
- `zai-org/GLM-5.1`
- `zai-org/GLM-5.2`

### Kimi

- `moonshotai/Kimi-K2-Base`
- `moonshotai/Kimi-K2-Instruct`
- `moonshotai/Kimi-K2-Instruct-0905`
- `moonshotai/Kimi-K2-Thinking`
- `moonshotai/Kimi-K2.5`
- `moonshotai/Kimi-K2.6`
- `moonshotai/Kimi-K2.7-Code`

### MiniMax

- `MiniMaxAI/MiniMax-M2.7`
- `MiniMaxAI/MiniMax-M3`
- `MiniMaxAI/MiniMax-M3-MXFP8`

## 静态部署

前端构建会先生成 `models/catalog.json`，再把仓库根目录的 `models/` 复制到 `frontend/dist/models`：

```bash
cd frontend
npm run build
```

部署 `frontend/dist` 到 GitHub Pages 后，`builtin` 和 `config` 两种来源可以在没有后端的情况下工作。`local`、`hf`、后端设置保存和 Hugging Face 搜索仍然需要 API 服务。

如果站点部署在子路径，例如 `https://kinchow.github.io/model-structure-viewer/`，需要设置 Vite base：

```bash
cd frontend
npm run build -- --base /model-structure-viewer/
```

## 错误

API 错误统一返回 `{"detail": "<message>"}`。常见状态码如下：

- `400 Bad Request`（`ConfigError`）：请求不合法，比如缺少 `model_id`、`source` 不支持、config JSON 格式错误，或者离线请求却需要访问 Hugging Face。
- `404 Not Found`（`NotFoundError`）：`MODEL_ROOT` 下没有对应模型目录或 `config.json`。
- `502 Bad Gateway`（`RemoteError`）：Hugging Face 访问失败，比如 HTTP 错误、网络错误或返回内容不是 JSON。
- `500 Internal Server Error`：其他没有单独映射的 `ViewerError`。

## 结构状态

每个 `ModelStructure` 的 `summary` 都会带上结构数字和诊断字段。它们用来说明当前结构是怎么生成的。

- `summary.strategy`：结构生成策略。
  - `frontend-architecture-template`：前端根据 config、registry 和模型 builder 生成结构，适合静态部署和手动验证。
  - `meta-introspect`：后端成功通过 `nn.Module` 做了结构检查。
  - `repaired-meta-introspect`：后端先遇到 transformers 兼容问题，套用很窄的修复策略后构建成功。
- `source.diagnostics`：更细的机器可读诊断信息，比如 `failure_kind`、`repair_strategy`、worker 超时和退出码。
- 后端结构接口不再提供 config 兜底结构。transformers 或 remote code 不支持时，`/api/structure` 和 `msv inspect` 会直接报错，避免把不可靠的结构展示成正确结果。

## Transformers 验证

`msv verify` 和 `POST /api/verify` 是后端验证入口。它们只认 transformers meta-device 构造成功：

- 成功：返回 `ok=true`、`status=passed`。
- 失败：返回 `ok=false`、`status=failed`，并带上 `error` 和 `diagnostics.failure_kind`。
- 失败时不生成 config-derived 结构，因此可以作为“transformers 是否支持这个模型结构”的判断依据。
- 如果失败原因只是本机没有 FlashAttention2，验证会把 attention backend 临时切到 `sdpa` 后重试。这个过程只影响结构验证，不改 `config.json`。
- 对个别官方 remote code 和当前 transformers API 的差异，验证会做很窄的运行时兼容，例如 MiniMax-M2.7 的 `rope_parameters`、Kimi-K2.5/Kimi-K2.6 的旧 `tie_weights` 签名。
- 默认 worker 超时是 90 秒。特别慢的 remote code 可以用 `MSV_STRUCTURE_WORKER_TIMEOUT_SECONDS` 临时调大。

示例：

```bash
curl -s http://127.0.0.1:8000/api/verify \
  -H 'Content-Type: application/json' \
  -d '{"source":"local","model_id":"Qwen/Qwen3.5-0.8B","cache_policy":"offline","offline":true}'
```

## 测试

```bash
.venv/bin/pytest -q
cd frontend
npm test
npm run verify:models
npm run build
```

页面级验证需要一个已经启动的静态页面和一个带 DevTools 端口的 Chrome，例如：

```bash
cd /Users/zhouzijian01/Desktop/workspace/code/kinchow/model-structure-viewer
python3 -m http.server 4183 --directory frontend/dist
/Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome \
  --headless=new \
  --remote-debugging-port=9223 \
  --user-data-dir=/tmp/msv-chrome-profile \
  --disable-gpu \
  --no-first-run \
  --no-default-browser-check \
  about:blank
MSV_PAGE_URL=http://127.0.0.1:4183/ MSV_CHROME_DEBUG_PORT=9223 npm --prefix frontend run verify:page
```

## 文档

- `CHANGELOG.md`
- `docs/frontend_structure_architecture.md`
- `docs/reverse_original_space.md`
- `docs/minimax_m3_mapping.md`
- `docs/model_source_resolution.md`
- `models/README.md`
