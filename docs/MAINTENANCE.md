# 维护基线（MAINTENANCE.md）

2026-09-08，W0-W6 七波重构收官后的维护态基准。**任何代码改动前先跑本清单，
改动后再跑一遍——两侧全绿才是安全变更。** 棘轮数字只允许向好（下降/收紧）。

## 五重 oracle（按耗时排序，全跑约 1 分钟）

| # | 命令（cwd） | 守护什么 | 基线 |
|---|---|---|---|
| 1 | `bash scripts/check_principles.sh`（根） | 原则护栏：§8.1 家族名棘轮、§3.2 显示名全禁、§3.1 counts 完整性 | §8.1 ≤14/16 |
| 2 | `cd frontend && npm test` | 246 例单测：恒等式 2% 容差、per-op golden、plan parity、normalize/树/边哈希基线、声明执法、role 绑定 | 全绿 |
| 3 | `cd frontend && npm run verify:models` | 59 内置模型结构可构建 | `"failed": 0` |
| 4 | `../.venv/bin/python -m pytest -q`（根） | 后端 transformers 对照 | 148 passed |
| 5 | `cd frontend && npm run test:e2e` | 浏览器端：图渲染、边 evidence 契约、成本交互 | 9 passed |

哈希基线文件（有意变更时重生成并人工审阅 diff，流程见各测试头注释）：
- `ops-spec-tree.golden.json` + `ops-edge.golden.json`（spec 树 + 边集，含 evidence）
- `normalize.golden.json`（normalizeConfig 输出；diff 只允许字段删除）
- `normalize.plan-fixture.json`（方案字段冻结件，**永不重写**）

## 棘轮数字（只许向好）

| 指标 | 当前 | 说明 |
|---|---|---|
| §8.1 家族名文件 | 14 / 16 | 配方表接管 plan.js 后应 <14 |
| 恒等式容差 | 2% | 超差须先归因再放宽或修 bug |
| `graph_ambiguous_truth_matches` | 0 | 任何模型非 0 即绑定回归 |
| 未知算子（unknown 叶子） | 0 | 新模型接入时允许临时 >0，须登记 |

## 变更纪律

1. **新算子**：先在 `formulas/index.js` 注册 counts（§3.1 唯一注册点），禁止在
   cost/ 加分派；
2. **新模型家族**：触发 refactor_plan.md 带债项——先做家族知识收口
   （archs/ 声明化 + plan.js 收缩），再接模型；
3. **新芯片**：条目进 `chips/public.js` 或本地 `chips.local.json`，必须带
   `field_sources`；无 sfu 规格的架构（昇腾）用 `sfu_rate_source: "vector"`
   语义映射；
3b. **新家族接入（证据三源齐备）**：先 `node scripts/fetch-evidence.mjs
   <org>/<id> --probe <modeling 文件名列表> <index.json>` 取证至
   `models/<org>/<id>/`（HF hub 单模型仓库惯例：config/modeling 源码/
   index 摘要与 config.json 同仓；index 原件 gitignore），并在
   evidence-manifest.json 登记来源 URL。公式级校准按
   /details/identity_calibration.md 的域拆分账本方法执行；
3c. **公式来源标注**：新算子进 `formulas/` 必须带来源注释——一等
   （aten 锚点）/二等（modeling 源码对照，引用 `models/<org>/<id>/` 内入库源码证据）/
   三等（分解声明），并写明单位换算（FLOPs↔MACs 2× 等），样式照
   counts.js F1 注释；
4. **哈希基线 diff 审阅**：有意变更 → 重生成 → diff 中只允许出现该变更
   声称的字段类型，任何其他差异 = 回归；
5. **恒等式超差**：先归因（counts/期望侧/建模边界三选一），建模边界写入
   REGISTERED 并同步 `details/cost_counts.md`。归因方法（域拆分账本、
   四样东西、已排除假设纪律）见 `details/identity_calibration.md`。

## 已知登记残差（不阻塞，详见 principles.md §10 与 details/cost_counts.md）

- kv_b 宽度（MLA，counts 侧 −255.9M/token，M8-V1 批次修）
- GLM-5 / Qwen3.8 恒等式 +0.5% 正向残差未归因
- embedding gather / 残差加法流量不可见（结构级缺口）
- vision 域绑定与恒等式（M8 清账中）
