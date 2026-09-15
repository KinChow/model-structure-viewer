# 测试和发布验证

本文定义当前仓库的验证层级和发布检查。每层验证回答不同问题，不能用低层通过代替高层验证。

## 环境

在仓库根目录执行：

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[test]'
npm --prefix frontend install
```

提交闸另装 `.[dev]` 并执行 `pre-commit install`，见 [`CONTRIBUTING.md`](../CONTRIBUTING.md)。后端默认使用仓库内置模型目录时传入 `--root ./models`。

## 1. 单元测试

```bash
.venv/bin/pytest -q
npm --prefix frontend test
```

后端测试覆盖 resolver、API、repair、结构 introspection、验证和导出。前端测试覆盖配置归一化（golden 冻结件）、算子公式注册表与 counts、四条恒等式（容差 0）、per-op golden、truth（skeleton/路径绑定）、cost（含 sharding/parallelPlan 接缝）、diagram、导出、hooks、入口辅助逻辑与 operators 声明执法。

通过标准：命令退出码为 0，不存在 failure、error、unexpected skip。

### 基线记录

重构（[`refactor_plan.md`](refactor_plan.md)）的差分替换以本表为参照。每完成一个里程碑后应刷新一次。

| 日期 | 命令 | 结果 |
|---|---|---|
| 2026-09-07 | `npm --prefix frontend run test` | 208/208 pass，0 fail |
| 2026-09-07 | `.venv/bin/python -m pytest -q` | 148 passed（11.7s） |
| 2026-09-07 | `npm --prefix frontend run verify:models` | 59 个内置模型全过，`"failed": 0`（离线跑通，不依赖外网） |
| 2026-09-07 | `npm --prefix frontend run test:e2e` | 9 passed，1 skipped（设计使然：`e2e/viewer.spec.js:66` 将"全量内置模型回归"限定为仅 desktop-chrome） |
| 2026-09-10 | `npm --prefix frontend test` | 341 pass，0 fail |
| 2026-09-10 | `.venv/bin/python -m pytest -q` | 169 passed（含 evidence 对账） |
| 2026-09-10 | `npm --prefix frontend run test:e2e` | 9 passed，1 skipped（playwright `retries`: CI 2/本地 1；`reuseExistingServer: false` 每轮全新 dev server——长时 server 状态降级实证修复） |

## 2. 内置模型组网验证

```bash
npm --prefix frontend run verify:models
```

该命令读取 `models/catalog.json` 和每个内置 `config.json`，调用前端结构生成链路。通过标准：

- `total` 等于 catalog 条目数。
- `passed` 等于 `total`，`failed` 为 0。
- 每个模型生成非空 `summary`、`root` 和子节点。

新增或删除模型后先重新生成 catalog：

```bash
npm --prefix frontend run catalog
```

## 3. 生产构建

```bash
npm --prefix frontend run build
```

通过标准：

- Vite 构建成功。
- `frontend/dist/index.html` 存在。
- `frontend/dist/models/catalog.json` 与各模型 `config.json` 被复制到构建产物；后端专用 Python 文件不进入静态产物。
- 构建没有修改需要人工维护的源码文件。

chunk size warning 不等于构建失败，但应在影响首屏加载时单独处理。

## 4. Python 包和版本

```bash
.venv/bin/python -m pip wheel --no-deps --no-build-isolation . --wheel-dir /tmp/msv-wheel-check
.venv/bin/python -c 'import model_structure_viewer; from model_structure_viewer.api import app; print(model_structure_viewer.__version__, app.version)'
```

通过标准：wheel 文件名、包 `__version__` 和 FastAPI `app.version` 一致。前端版本以 `frontend/package.json` 为手写源，`package-lock.json` 由 npm 同步。

## 5. 后端 transformers 验证

单模型验证：

```bash
.venv/bin/msv --root ./models --offline verify \
  --model Qwen/Qwen3.5-0.8B \
  --source local \
  --cache-policy offline \
  --format json
```

该验证只判断 transformers 是否能在 meta device 构造模型，不下载权重、不运行推理。通过标准：`ok=true`，并记录 `strategy`、repair steps 和失败分类。

上行前端 Graph（CLI `msv verify --graph <file>` 或 API `msv_graph` 字段）时，响应额外携带 per-module evidence 对账：`evidence.modules`（path/class/params/weight_shapes/dtype）与 `evidence.diff` 三分类（only_transformers / only_msv / mismatch），并区分「构造通过」与「结构一致」两态——diff 经四桶 triage（renaming / nonparam_drop / fold_frontend_suffixes / known_divergences，规则随包分发于 `verification/fixtures/canonical_path_contract.json`），`unclassified` 为空才算 `structurally_consistent=true`。缺省上行时 diff 为空并标注 "msv_graph not provided"，空 diff 不等于对账通过。

仓库目前没有单条 CLI 命令执行全部内置模型的后端验证；全量执行时必须记录遍历脚本、模型总数和逐模型结果，不能把单模型命令描述成全量验证。

## 6. API 验证

启动本地 API：

```bash
.venv/bin/msv --root ./models serve --port 8000
```

至少验证：

```text
GET  /api/models
GET  /api/local/config
GET  /api/hf/search
GET  /api/hf/config
POST /api/structure
POST /api/verify
GET  /api/settings
POST /api/settings
GET  /api/health
```

`POST /api/verify` 的对账语义见第 5 节（上行 `msv_graph` 时返回 evidence diff）。

API 验证必须包含正常请求和错误请求；响应应为合法 JSON，错误状态码和 `detail` 可读。结束后关闭服务并确认 8000 端口释放。

## 7. 浏览器验证

`npm --prefix frontend run test:e2e`（`verify:page` 为兼容别名）会启动隔离 Vite 服务，并在桌面/移动 Chrome 中验证当前 React Flow 页面。全量 59 模型展开回归在桌面运行，移动端覆盖代表模型和全部纯前端边界用例。

浏览器验收至少覆盖：

- 入口页加载 catalog、Provider 和模型快捷入口。
- 所有产品流程记录并禁止同源 `/api/*` 请求；不 mock 后端成功响应，不出现验证按钮或服务端设置。
- 直连模型源的成功/失败夹具、模型选项搜索、旧 local 链接、目录取消/缺配置/无效 JSON/header 和导出。
- 内置模型生成后出现 `.react-flow-diagram` 和 React Flow 节点。
- 搜索、节点选择、breadcrumb、公式索引和 Inspector 联动。
- 展开/折叠结构组、fit、缩放和 minimap。
- Cost Lens、集中式/PD、芯片/方案对比和 fit 状态。
- Mermaid、DOT、JSON、SVG 和 raw config 输出。
- 中文/英文、深色/浅色和窄屏布局。

浏览器检查若启动临时 HTTP 服务或 Chrome DevTools 端口，结束后必须确认相关进程已停止。

## 8. 发布检查

发布前按顺序执行：

```bash
.venv/bin/pytest -q
npm --prefix frontend test
npm --prefix frontend run verify:models
npm --prefix frontend run test:e2e
npm --prefix frontend run build
.venv/bin/python -m pip wheel --no-deps --no-build-isolation . --wheel-dir /tmp/msv-wheel-check
git diff --check
```

然后确认：

- Python 与前端版本源一致，lockfile 已同步。
- `CHANGELOG.md` 的 `[Unreleased]` 已归档到带日期的版本节。
- README 和 `docs/` 没有旧版本号、旧模型数量或失效路径；页面验收描述与 React Flow 实现一致。
- 当前已知验证缺口被明确记录，没有把未执行或未通过的测试写成通过。
- 提交后创建 `vX.Y.Z` tag，并使用对应 changelog 创建 GitHub Release。

## 结果记录

发布记录至少包含 commit、执行时间、catalog 模型数、测试数量、构建结果、wheel 版本、未执行项和失败原因。配置组网、transformers 验证、HTTP 验证和浏览器验证必须分别记录，不能合并成一个“模型已验证”结论。
