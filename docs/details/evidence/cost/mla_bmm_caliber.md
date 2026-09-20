# MLA 注意力 bmm "0.30×" 归因：减层配置 head_dim 注入 artifact（非前端错误）

> Part1 遗留：DeepSeek-V3 MLA 注意力 bmm 前端/torch=0.30×（异常，非标准因果 0.5×）。本文定性。

## 根因

前端 `normalize.js:attentionHeadDim` 取值优先级：**`HEAD_DIM_KEYS`（head_dim）优先**，其次
`qk_nope_head_dim + qk_rope_head_dim`（normalize.js:66-73）。

- **原始 catalog 配置**（DeepSeek-V3.1）**无 head_dim** → 前端用 qk_nope+qk_rope = 128+64 = **192**（正确 MLA 打分宽度）。
- **减层配置**经 transformers `DeepseekV3Config.to_json_file` **被注入 head_dim=64**（框架默认，=qk_rope）
  → 前端 HEAD_DIM_KEYS 优先取 64，MLA sdpa matrix = scores×(64+128)=scores×192，**少算 QK^T 的 nope 段**。

## 验证（去掉注入的 head_dim 后）

| 配置 | 前端 headDim | sdpa matrix/层 | fe×2 / torch bmm |
|---|---|---|---|
| 减层(含注入 head_dim=64) | 64 | 202,899,456 | **0.302（artifact）** |
| 去 head_dim（=原始 catalog 口径） | 192 (qk_nope+qk_rope) | 338,165,760 | **0.5039（纯因果）** |

torch eager MLA bmm/层 = 671,090,688 MACs（full square：QK^T heads×S²×192 + PV heads×S²×128）。
前端去 head_dim 后 = scores(因果)×320 = 338M/层，**比值 0.504 = 纯因果**（S(S+1)/2÷S²=8256/16384）。

## 结论

- **"MLA bmm 0.30×" 是减层配置 artifact，非前端错误**。生产（原始 catalog 无 head_dim）下 MLA 注意力 matrix
  用 qk_nope+qk_rope=192，与 eager torch **在纯因果 0.504× 上一致——与标准 MHA 同口径**（operator_cost.md Qwen3 已验因果）。
- **caveat（配置卫生）**：`attentionHeadDim` 让显式 head_dim 优先于 qk_nope+qk_rope；若某 MLA 配置携带
  与 qk_nope+qk_rope 不一致的 head_dim（如 transformers 注入的 rope-only 64），MLA 打分宽度会被算错。
  60 个 catalog 模型均无此冲突（不触发），故**不改前端优先级**（避免为假设修复破坏其它模型）；减层 harness
  须剥离框架注入的 head_dim（`scripts/evidence/_fixtures/deepseek_v3_nohd.json`）。
