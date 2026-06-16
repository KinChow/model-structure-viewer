# Model Source Resolution

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
| `auto` | Prefer local cache, then use HF unless offline. |
| `config` | Use uploaded or pasted JSON directly. |

## Cache Policy

| Policy | Behavior |
|---|---|
| `prefer-local` | Use local config when present; otherwise fetch HF. |
| `refresh` | Fetch HF even if local cache exists. |
| `offline` | Do not access HF. |

`offline=true` is stronger than cache policy. If offline is enabled, `hf` lookup and HF search fail immediately.

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

The resolver caches only:

- `config.json`
- `README.md`
- `configuration_*.py`

The resolver never downloads or caches weight files, including:

- `.safetensors`
- `.bin`
- `.gguf`
- `.pt`
- `.pth`
- `.onnx`
- `.h5`

This boundary is intentional. The viewer is for structure inspection from model configuration, not local inference.
