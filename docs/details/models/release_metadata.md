# 内置模型发布时间来源

本次 `release_time` 查询时间：2026-09-06（Asia/Shanghai）。

2026-09-22 补录：`deepseek-ai/DeepSeek-V4.1-Flash` 使用 Hugging Face Hub
镜像 API 的 `createdAt = 2026-09-10T02:17:58.000Z`，快照 revision
`dba1be0a40aa45a94ad051997016db3960a90277`。这是新增条目的来源例外，
其余既有条目保留下面的来源，不重写历史快照。

- `Qwen/*`、`deepseek-ai/*`、`moonshotai/*`：使用 ModelScope 官方模型 API 对应条目的 `Data.CreatedTime`。
- `MiniMaxAI/*`、`zai-org/*`：ModelScope 对应仓库返回 404，因此使用 Hugging Face 镜像 API 对应条目的 `createdAt`。
- 没有使用 `LastUpdatedTime` 或 `lastModified`，因为它们表示仓库更新时间，不是仓库创建时间。

查询接口模板：

- `https://www.modelscope.cn/api/v1/models/{model_id}`
- `https://hf-mirror.com/api/models/{model_id}`

`catalog.json` 中的时间统一保存为 ISO 8601 UTC 字符串。这里的 `release_time` 是公开模型仓库的创建时间代理值，不等同于论文、官方公告或模型首次对外发布的精确时间。

catalog 生成脚本和单元测试要求全部内置条目具有合法 UTC 时间。
页面只读取构建时快照，Provider 沿用既有排序和日期格式化逻辑，不运行时查询 Hub。
