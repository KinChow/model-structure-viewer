# Contributing

## 芯片规格数据

公开芯片条目放在 `frontend/src/cost/chips/public.js`，每条必须包含可核实的公开 `source` 和 `confidence`。字段来自不同资料时，请同时填写 `field_sources`。不要用估算值补齐厂商没有公开的字段。

非公开或内部规格只放在本机的 `frontend/public/chips.local.json`，该文件已被 Git 忽略。格式参考 `frontend/src/cost/chips/chips.local.example.json`。已有公开芯片允许仅覆盖需要修改的字段；新增本地芯片需要提供 `id`、`vendor`、`name` 和 `source`。

提交公开规格前运行：

```bash
cd frontend
npm test
npm run build
```
