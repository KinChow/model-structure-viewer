# Contributing

## 发布与版本管理

项目版本遵循 [Semantic Versioning](https://semver.org/lang/zh-CN/)，变更记录遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)。日常开发先把变更写入 `CHANGELOG.md` 的 `[Unreleased]`，发布时再归档到 `## [X.Y.Z] - YYYY-MM-DD`。

发布检查清单：

1. 更新 `src/model_structure_viewer/__init__.py` 和 `frontend/package.json` 两个版本源；Python 包元数据和 FastAPI 会自动读取前者，运行 `npm install --package-lock-only` 同步前端 lockfile。
2. 把 changelog 的 `[Unreleased]` 内容归档到对应版本，并确认 README 不包含需要手工同步的版本号。
3. 运行后端和前端测试、构建及模型验证，确认 `git diff --check` 通过。
4. 提交版本变更后创建 `vX.Y.Z` Git tag，并通过 GitHub Release 发布对应 changelog。

## 芯片规格数据

公开芯片条目放在 `frontend/src/cost/chips/public.js`，每条必须包含可核实的公开 `source` 和 `confidence`。字段来自不同资料时，请同时填写 `field_sources`。不要用估算值补齐厂商没有公开的字段。

非公开或内部规格只放在本机的 `frontend/public/chips.local.json`，该文件已被 Git 忽略。格式参考 `frontend/src/cost/chips/chips.local.example.json`。已有公开芯片允许仅覆盖需要修改的字段；新增本地芯片需要提供 `id`、`vendor`、`name` 和 `source`。

提交公开规格前运行：

```bash
cd frontend
npm test
npm run build
```
