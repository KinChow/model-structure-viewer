# 维护基线（MAINTENANCE.md）

2026-09-08，W0-W6 七波重构收官后的维护态基准。**任何代码改动前先跑本清单，
改动后再跑一遍——两侧全绿才是安全变更。** 棘轮数字只允许向好（下降/收紧）。

## 五重 oracle（按耗时排序，全跑约 1 分钟）

| # | 命令（cwd） | 守护什么 | 基线 |
|---|---|---|---|
| 1 | `bash scripts/check_principles.sh`（根） | 原则护栏：§8.1 家族名棘轮、§3.2 显示名全禁、§3.1 counts 完整性、§3.1b fromNode 查表、§3.1d 来源标注、§3.5b /tmp 引用棘轮、P0 root 复活棘轮 | §8.1 ≤8/16 |
| 2 | `cd frontend && npm test` | 前端单测：四条恒等式**容差 0**、N2-4 锚 1/锚 2、S3 图声明对 header 逻辑元素（量化解包后）、量化 per-matrix、原子 exact、golden、全链路 | 全绿 |
| 3 | `cd frontend && npm run verify:models` | 59 内置模型结构可构建 | `"failed": 0` |
| 4 | `.venv/bin/python -m pytest -q`（根） | 后端 transformers 对照、params numel 对账、FlopCounterMode 独立算子抽查 | 184 passed |
| 5 | `cd frontend && npm run test:e2e` | 浏览器端：图渲染、边 evidence 契约、成本交互（全量内置模型回归仅桌面跑） | 9 passed + 1 skipped |

e2e 注：2026-09-10 出现两轮全量单用例失败（不同用例/project，隔离单跑均过、
重试仍红），实证根因 = **长时运行 + 多轮 HMR 的 dev server 状态降级**（节点
布局持续不稳定 → click "element is not stable" 耗尽超时；杀掉旧 server 后
9/9 全绿）。处置：`reuseExistingServer: false`（每轮全新 server）+ `retries`
（CI 2 / 本地 1）双保险；再红先查是否复用了旧 4173 进程，再定性为回归。

哈希基线文件（有意变更时重生成并人工审阅 diff，流程见各测试头注释）：
- `ops-spec-tree.golden.json` + `ops-edge.golden.json`（spec 树 + 边集，含 evidence）
- `normalize.golden.json`（normalizeConfig 输出；diff 只允许字段删除）
- `normalize.plan-fixture.json`（方案字段冻结件，**永不重写**）

## 棘轮数字（只许向好）

| 指标 | 当前 | 说明 |
|---|---|---|
| §8.1 家族名文件 | 8 / 16 | 删 plan.js + 共享 layer 家族分派后 10→8 |
| 四条恒等式容差 | **0** | 权重字节/KV/形状连续性/融合分解全部 error 模式；超差用 checkpoint header 或锚 1（声明 vs 叶 counts）归因，禁止放宽 |
| `/tmp` 取证引用（operators_reference） | 15（§3.5b 棘轮） | 只许下降；新证据落 models/<org>/<id>/ 证据库 |
| DECOMPOSE_PENDING | **0** | 新模块进 formulas/modules.js 必须同时声明 decompose |
| 语义边登记（形状连续性） | 只许缩短 | 未登记的不连续边即失败 |
| `graph_ambiguous_truth_matches` | 0 | 任何模型非 0 即绑定回归 |
| 未知算子（unknown 叶子） | 0 | 新模型接入时允许临时 >0，须登记 |
| N2-4 声明单源（锚 1） | 18399 声明叶（dtype-aware），违例 **0** | 声明元素 × (param_dtype ? paramDtypes 字节宽 : 2B) == 叶 counts.bytes.weights，全目录逐叶容差 0；embedding 走登记例外（声明=驻留，gather 流量按行计） |
| N2-4 EP 组合自洽（锚 2） | M2.7 三方一致 | 专家块÷moe_ep + 其余÷tp 与聚合投影、expertWeightRange 闭式互证（DP 切专家 / EP+DP attention / 混合 ETP） |
| P2 声明覆盖（缺声明带权叶） | **0** / 带权叶 18399 | 判据：counts.bytes.weights>0 或 weight_shapes 非空或 type=embedding ⇒ 必须有 weightMatrices。**只许下降**；新增带权算子不声明即顶破基线。缺口台账见 details/sharding_matrix.md 附录；归零后方可删 WEIGHT_PROJECTION_RULES |

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
   四样东西、已排除假设纪律）见 `details/identity_calibration.md`；
6. **疑似冗余处置（2026-09-08 用户裁决）**：看起来没人用的代码先查是否
   "未完成系统接入"（案例：counts.bytes 算完无消费者、extractor 31 条手搓
   分支绕开注册表 counts、generic-config 兜底路径必崩）——**补全接入优先于
   删除**；只有确认无意图、无消费者的代码才删，且区分"真死函数"（删函数体）
   与"多余 export"（只删关键字）两种处置；
7. **接缝测试（2026-09-08 四路审计教训）**：两端各有测试不等于链路被覆盖
   （案例：graphTruth 键名两侧各自通过、接缝不匹配致歧义面板生产死亡；
   roofline 函数有测试、UI 入口未传 actions 致五路退化三路）。新增任何
   跨模块字段，必须在**消费侧真实入口**补一条贯通测试；新增任何计算产出，
   必须同时指明它的消费者——无消费者的产出不得合入。

## 已知登记残差（不阻塞，详见 principles.md §10 与 details/cost_counts.md）

- ~~GLM-5/5.1/5.2/5.3 +0.3%~+0.5% 正向残差未归因~~（已销：根因 = derivedWeights
  的 DSA 分支 model_type 白名单漏 glm5_next + ops 模板 KDA 宽度错，
  2026-09-09 修正后全类 1.0000）
- embedding gather 流量已于 M11 计入 counts.bytes；残差加法流量已由
  residual_add 叶覆盖（2026-09-09）
- ~~恒等式容差：全局 2%，REGISTERED 9 项~~（已收至**容差 0、REGISTERED 空**：
  matrix 0.005（浮点求和误差）/ 权重字节与 KV 与形状连续性精确；
  identity_calibration.md 的案例即归因记录）
