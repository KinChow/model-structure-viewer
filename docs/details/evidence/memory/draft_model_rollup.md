# 显存分项：主模型 / 草稿模型 / 总共（口径 + 抽检，2026-09-21）

> 对应改动 commit `c8d0e9e`（前端功能，非运行时验证）。把 Cost 面板显存从平铺 5 项改为在**有草稿（MTP/DSpark）时**给出三项 roll-up：主模型 / 草稿模型 / 总共。

## 口径

- 新增 `cost/memory.js: draftWeightBytes(graph, weightBytesTotal)`：与 `draftKvBytesPerToken` **同源**（按 parent 祖先链判定 mtp/dspark 子树），用 `nodeWeightCapacityBytes` 逐叶累加草稿子树权重；给了实际总权重（checkpoint/override）时**按草稿子树在图权重里的占比摊到该总量**（避免「图声明字节」与「checkpoint 总量」两种口径直接相减双计），无草稿子树返回 0。
- 三项定义（仅当 `draftKvBytes>0` 显示）：
  - **主模型** = `totalBytes − draftWeight`（= 主干权重 + buffers + 主干 KV + KDA state）
  - **草稿模型** = `draftWeight + draftKV`（草稿常驻权重 + 草稿常驻 KV；草稿 state 当前恒 0）
  - **总共** = `totalBytes + draftKV`（草稿权重已含在 `totalBytes` 里，故只再加草稿 KV）
- 「Total VRAM」指标同步计入草稿 KV，与「总共」一致。非草稿模型（8/60）界面与数值逐字节不变。
- **草稿 state**：全 catalog 60 模型（52 个含 mtp/dspark 子树）草稿注意力层实测均为 KV 型（`draftStateRecurrent=0`），故草稿 state 恒 0、**不新增 state 池**；未来若出现线性注意力草稿层 → 进 state 池（按序列固定）**不进 KV**。

## 抽检（草稿权重占比合理）

| 模型 | 草稿权重(绝对) | 图总权重 | 草稿占比 |
|---|---|---|---|
| Qwen3.5-4B（MTP 1 层） | 241.2 MB | 9,314 MB | 2.59% |
| DeepSeek-V4.1-Flash（DSpark） | 28,478 MB | 1,526,232 MB | 1.87% |

占比与「单草稿层 / 数十主干层」的量级吻合。

## 验证

- `node --test` 438/438（含新增 `draftWeightBytes`：占比拆分 + total 摊分 + 无草稿返回 0）；`vite build`、`docs:check`、`verify:models` 60/60 全绿。
- 复现：`draftWeightBytes` 单测见 `frontend/src/cost/__tests__/memory.test.js`；抽检用 `verify-builtin-models.mjs --dump-graphs` 落图后按 mtp/dspark 子树统计。
