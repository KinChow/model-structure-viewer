# Model Structure Viewer

Configuration-driven model structure viewer for local model folders and Hugging Face model ids.

The MVP is generic, with a high-fidelity template for `MiniMaxAI/MiniMax-M3`. It reads model configuration and metadata only. It does not download weights, instantiate models, or execute remote model code.

## Paths

- Development repo: `/Users/zhouzijian01/Desktop/workspace/code/kinchow/model-structure-viewer`
- Default model root: `/Users/zhouzijian01/Desktop/workspace/models`
- Local model layout: `/Users/zhouzijian01/Desktop/workspace/models/<org>/<model>/config.json`
- MiniMax-M3 local path: `/Users/zhouzijian01/Desktop/workspace/models/MiniMaxAI/MiniMax-M3/config.json`

## Install

```bash
cd /Users/zhouzijian01/Desktop/workspace/code/kinchow/model-structure-viewer
python3 -m venv .venv
.venv/bin/pip install -e '.[test]'
cd frontend && npm install
```

## CLI

```bash
# list local config.json files
.venv/bin/msv list --root /Users/zhouzijian01/Desktop/workspace/models

# search Hugging Face
.venv/bin/msv search MiniMax-M3 --endpoint https://huggingface.co

# inspect by Hugging Face model id, cache config metadata under MODEL_ROOT
.venv/bin/msv inspect --model MiniMaxAI/MiniMax-M3 --source hf --format json

# prefer local cache, then Hugging Face
.venv/bin/msv inspect --model MiniMaxAI/MiniMax-M3 --source auto --cache-policy prefer-local --format mermaid

# inspect a config file directly
.venv/bin/msv inspect --config tests/fixtures/minimax_m3/config.json --format dot
```

## Web

Run the API:

```bash
.venv/bin/msv serve --root /Users/zhouzijian01/Desktop/workspace/models --port 8000
```

Run the frontend:

```bash
cd frontend
npm run dev
```

Open `http://localhost:5173`. The Vite dev server proxies API calls to `http://localhost:8000`.

## API

- `GET /api/models`
- `GET /api/hf/search?q=MiniMax-M3`
- `GET /api/hf/config?model_id=MiniMaxAI/MiniMax-M3&revision=main`
- `POST /api/structure`
- `POST /api/export`
- `GET /api/settings`
- `POST /api/settings`

## Source Modes

- `local`: read only `$MODEL_ROOT/<org>/<model>/config.json`
- `hf`: fetch `config.json` and allowed metadata from Hugging Face
- `auto`: prefer local cache, then Hugging Face
- `config`: use pasted or uploaded JSON

Allowed HF metadata cache files are `config.json`, `README.md`, `configuration_*.py`, `modeling_*.py`, and `tokenization_*.py`. Weight files such as `.safetensors`, `.bin`, `.gguf`, `.pt`, and `.onnx` are never cached by this tool.

When `config.json` declares `auto_map` (e.g. DeepSeek-V3, MiniMax-M3) and the matching `modeling_*.py` / `configuration_*.py` files are missing in the local cache, the resolver auto-downloads them from the same HF revision so the structure builder can introspect the live `nn.Module` tree (Plan A). Disable with `MSV_AUTO_FETCH_REMOTE_CODE=0`, the `--no-auto-fetch-remote-code` CLI flag, or `auto_fetch_remote_code: false` in the settings payload. Auto-fetch is always skipped in offline mode and for `source=config`.

### API error semantics

All API errors flow through a single `ViewerError` hierarchy and map to HTTP status codes via a FastAPI exception handler:

- `400 Bad Request` (`ConfigError`): invalid request payload — missing `model_id`, unsupported `source`, malformed config JSON, or an offline-only request that requires HF.
- `404 Not Found` (`NotFoundError`): the local config file or model directory does not exist under `MODEL_ROOT`.
- `502 Bad Gateway` (`RemoteError`): Hugging Face fetch failed (HTTP error, network error, or non-JSON response).
- `500 Internal Server Error`: any other `ViewerError` subclass without a more specific status.

Error responses use the shape `{"detail": "<message>"}`.

### Summary fields

The `summary` block on every `ModelStructure` payload includes diagnostic fields in addition to the model-shape numbers:

- `summary.strategy`: `"meta-introspect"` when live `nn.Module` introspection succeeded, `"config-fallback"` when builder-level introspection failed, `"budget-config-fallback"` when generic resource-budget gating skipped introspection, or `"worker-config-fallback"` when the isolated introspection worker failed or timed out.
- `summary.fallback_reason`: present for fallback strategies; a short human-readable reason such as `"AutoModel.from_config failed: ..."`, `"resource budget exceeded for meta introspection"`, or `"worker exited with code -9"`. The same value is mirrored on `source.fallback_reason` for backward compatibility.
- `source.diagnostics`: machine-readable fallback context such as `failure_kind`, `execution_mode`, budget estimates, worker timeout, or worker exit code.

## Test

```bash
.venv/bin/pytest -q
cd frontend && npm run build
```

## Docs

- `CHANGELOG.md`
- `docs/reverse_original_space.md`
- `docs/minimax_m3_mapping.md`
- `docs/model_source_resolution.md`
