# Model Source Resolution

## Product frontend

The browser reads built-in static assets, public Hugging Face/ModelScope resources, or user-selected directory files. It never calls MSV `/api/*`. Legacy `auto` tries builtin → hf; legacy `local` asks the user to choose a folder again. Local directory files are read in the browser and are not uploaded. Transformers verification remains a developer CLI/API tool.

## Python developer tools

The filesystem and cache rules below apply only to Python CLI/API.

Default model root:

```text
/Users/zhouzijian01/Desktop/workspace/models
```

Model ids map to local directories by path segments:

```text
MiniMaxAI/MiniMax-M3
=> /Users/zhouzijian01/Desktop/workspace/models/MiniMaxAI/MiniMax-M3/config.json
```

## Resolution Modes

| Mode | Behavior |
|---|---|
| `local` | Only read local `config.json`; fail if missing. |
| `hf` | Fetch remote `config.json` from the configured HF endpoint and cache allowed metadata. |
| `auto` | CLI/API compatibility mode: prefer built-in and local cache, then use HF unless offline. |
| `config` | Use uploaded or pasted JSON directly. |

## Cache Policy

| Policy | Behavior |
|---|---|
| `prefer-local` | Use local config when present; otherwise fetch HF. |
| `refresh` | Fetch HF even if local cache exists. |
| `offline` | Do not access HF. |

`offline=true` is stronger than cache policy. If offline is enabled, `hf` lookup and HF search fail immediately.

Contract (shared endpoint/Graph fields, separate frontend and Python source rules): [`source_contract.json`](source_contract.json). Unique key = `repo_id + revision + cache_dir`. `auto` fallback = builtin → local → hf. ModelScope empty/`main` revision maps to `master`.

## Hugging Face Endpoint

Default:

```text
https://huggingface.co
```

Mirror example:

```text
https://hf-mirror.com
```

CLI example:

```bash
.venv/bin/msv inspect \
  --model MiniMaxAI/MiniMax-M3 \
  --source hf \
  --endpoint https://hf-mirror.com
```

## Cached Files

The built-in static bundle contains only:

- `config.json`
- `catalog.json`
- optional `header-truth.json`, `skeleton-truth.json`, and `source-ref.json`

The backend may cache remote-code helpers when `auto_fetch_remote_code` is enabled. The resolver never downloads or caches weight files, including:

- `.safetensors`
- `.bin`
- `.gguf`
- `.pt`
- `.pth`
- `.onnx`
- `.h5`

This boundary is intentional. The viewer is for structure inspection from model configuration, not local inference.
