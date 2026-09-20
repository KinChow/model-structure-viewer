# _lib —— 复现脚本共享工具

放置多个探针复用的工具，避免每个脚本各写一份：

- 减层器：由真实 config 缩层/缩宽产出可离线构造的小模型（transformers AutoConfig 路线，产物落 `../_fixtures/`）。
- dump helper：`named_modules()` / `named_buffers()` 的结构化导出。
- 对账比对器：把探针 dump 与前端 spec（`frontend/src/structure`）逐点比对。

约定：新增共享逻辑先落这里再被各维度脚本 import；探针自身只保留“构造 + 调用 + 回填哪份 doc”。

> 现状：各探针仍为自包含（未在无法在本机端到端验证的情况下强行抽取，避免引入不可验证的破坏）。
> 后续在具备运行环境时按上表抽取。
