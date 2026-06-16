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

Allowed HF metadata cache files are `config.json`, `README.md`, and `configuration_*.py`. Weight files such as `.safetensors`, `.bin`, `.gguf`, `.pt`, and `.onnx` are never cached by this tool.

## Test

```bash
.venv/bin/pytest -q
cd frontend && npm run build
```

## Docs

- `docs/reverse_original_space.md`
- `docs/minimax_m3_mapping.md`
- `docs/model_source_resolution.md`
