# 测试 Skill

这份文档用于后续系统测试。目标是把“单测、后端测试、浏览器验证”拆成固定流程，避免每次只凭印象跑几条命令。

## 适用场景

- 修改前端结构解析、layers、ops、公式、导出或页面交互。
- 修改后端 API、CLI、transformers 验证或模型配置读取。
- 新增或更新 `models/` 里的内置模型配置。
- 发布前需要确认静态部署页面是否可用。

## 前置条件

在仓库根目录执行：

```bash
cd /Users/zhouzijian01/Desktop/workspace/code/kinchow/model-structure-viewer
```

Python 依赖应已经安装：

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[test]'
```

前端依赖应已经安装：

```bash
cd frontend
npm install
cd ..
```

## 1. 单测

先跑后端单测：

```bash
.venv/bin/pytest -q
```

再跑前端单测：

```bash
npm --prefix frontend test
```

通过标准：

- pytest 没有 failure 或 error。
- Node test 没有 failure。
- 如果修改的是结构字段、shape、layers 展示或导出，不能只跑后端单测，必须继续跑前端验证。

## 2. 前端内置模型验证

这一步不启动浏览器。它直接读取 `models/catalog.json` 和 `models/<org>/<model>/config.json`，检查所有内置模型能否完成前端组网。

```bash
npm --prefix frontend run verify:models
```

通过标准：

- `passed` 等于 catalog 模型总数。
- `failed` 为 0。
- 每个模型都能生成 `summary`、`root` 和子节点。

如果新增模型配置，先重新生成 catalog：

```bash
npm --prefix frontend run catalog
```

## 3. 前端构建

```bash
npm --prefix frontend run build
```

通过标准：

- Vite build 成功。
- `frontend/dist/models/catalog.json` 存在。
- `frontend/dist/models/<org>/<model>/config.json` 能随静态站点一起发布。

## 4. 后端 transformers 验证

后端验证只检查 transformers 能否在 meta device 上构造模型。它不下载权重，也不跑推理。

单模型验证：

```bash
.venv/bin/msv verify \
  --root ./models \
  --offline \
  --model Qwen/Qwen3.5-0.8B \
  --source local \
  --cache-policy offline \
  --format json
```

通过标准：

- `ok` 为 `true`。
- `status` 为 `passed`。
- 不出现 config fallback。当前后端失败时应直接返回错误，不生成不可靠结构。

发布前或模型配置变更后，建议对 catalog 全量验证：

```bash
python3 - <<'PY'
import json
import subprocess
import sys
from pathlib import Path

root = Path("models")
catalog = json.loads((root / "catalog.json").read_text())
failed = []

for entry in catalog["models"]:
    model_id = entry["model_id"]
    cmd = [
        ".venv/bin/msv",
        "verify",
        "--root",
        str(root),
        "--offline",
        "--model",
        model_id,
        "--source",
        "local",
        "--cache-policy",
        "offline",
        "--format",
        "json",
    ]
    result = subprocess.run(cmd, text=True, capture_output=True)
    if result.returncode != 0:
        failed.append((model_id, result.stderr.strip() or result.stdout.strip()))
        print(f"FAIL {model_id}", file=sys.stderr)
        continue
    payload = json.loads(result.stdout)
    if not payload.get("ok"):
        failed.append((model_id, payload.get("error", "verify returned ok=false")))
        print(f"FAIL {model_id}", file=sys.stderr)
    else:
        print(f"PASS {model_id}")

print(f"passed={len(catalog['models']) - len(failed)} failed={len(failed)} total={len(catalog['models'])}")
if failed:
    for model_id, error in failed:
        print(f"\n{model_id}\n{error[:1000]}", file=sys.stderr)
    raise SystemExit(1)
PY
```

## 5. 后端 API 验证

启动后端：

```bash
.venv/bin/msv --root ./models --offline serve --host 127.0.0.1 --port 8000
```

另开一个终端验证 `/api/models` 和 `/api/structure`：

```bash
python3 - <<'PY'
import json
import urllib.request

base = "http://127.0.0.1:8000"
models = json.load(urllib.request.urlopen(f"{base}/api/models"))
failed = []

for item in models:
    model_id = item["model_id"]
    body = json.dumps({
        "source": "local",
        "model_id": model_id,
        "cache_policy": "offline",
        "offline": True,
    }).encode()
    request = urllib.request.Request(
        f"{base}/api/structure",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        payload = json.load(urllib.request.urlopen(request, timeout=180))
        strategy = payload.get("summary", {}).get("strategy")
        if strategy not in {"meta-introspect", "repaired-meta-introspect"}:
            failed.append((model_id, f"unexpected strategy: {strategy}"))
        else:
            print(f"PASS {model_id} {strategy}")
    except Exception as exc:
        failed.append((model_id, str(exc)))
        print(f"FAIL {model_id}")

print(f"passed={len(models) - len(failed)} failed={len(failed)} total={len(models)}")
if failed:
    for model_id, error in failed:
        print(f"\n{model_id}\n{error[:1000]}")
    raise SystemExit(1)
PY
```

通过标准：

- `/api/models` 能返回 catalog 中的所有模型。
- `/api/structure` 全部成功。
- 策略只出现 `meta-introspect` 或 `repaired-meta-introspect`。

结束后停止后端，并确认端口没有残留：

```bash
lsof -iTCP:8000 -sTCP:LISTEN -n -P || true
```

## 6. 浏览器验证

这一步验证真实页面，不只验证代码函数。它适合静态部署前、UI 改动后、layers 展开和导出功能改动后执行。

先构建静态页面：

```bash
npm --prefix frontend run build
```

启动静态服务：

```bash
python3 -m http.server 4183 --directory frontend/dist
```

启动带 DevTools 端口的 Chrome：

```bash
/Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome \
  --headless=new \
  --remote-debugging-port=9223 \
  --user-data-dir=/tmp/msv-chrome-profile \
  --disable-gpu \
  --no-first-run \
  --no-default-browser-check \
  about:blank
```

运行页面验证：

```bash
MSV_PAGE_URL=http://127.0.0.1:4183/ \
MSV_CHROME_DEBUG_PORT=9223 \
npm --prefix frontend run verify:page
```

通过标准：

- 页面能打开并完成默认模型生成。
- Architecture 有 SVG 图。
- Layers 能展开，能看到 `Decoder Layers`、公式、输入维度和输出维度。
- Export 能生成 Mermaid。
- Raw Config 能展示原始配置。
- 所有内置模型逐个切到 `builtin` 后都能生成结构。

结束后停止静态服务和 Chrome，并确认端口没有残留：

```bash
lsof -iTCP:4183 -sTCP:LISTEN -n -P || true
lsof -iTCP:9223 -sTCP:LISTEN -n -P || true
```

## 推荐执行顺序

普通代码改动：

```bash
.venv/bin/pytest -q
npm --prefix frontend test
npm --prefix frontend run verify:models
npm --prefix frontend run build
```

模型配置、后端验证或发布前：

```bash
.venv/bin/pytest -q
npm --prefix frontend test
npm --prefix frontend run verify:models
npm --prefix frontend run build
# 再执行后端 transformers 全量验证、后端 API 验证和浏览器验证
```

## 结果记录

每次系统测试建议记录：

- 当前 commit。
- 执行时间。
- catalog 模型数量。
- 单测结果。
- 后端验证通过数量。
- 浏览器验证通过数量。
- 失败模型和失败原因。

如果失败来自 transformers 不支持模型结构，记录 `diagnostics.failure_kind`、模型 id 和 transformers 版本。不要把失败模型改成前端 config fallback 后就算通过。
