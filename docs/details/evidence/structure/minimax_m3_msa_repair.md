# MiniMax-M3 MSA：GQA group 选块证据与修复

状态：**已修复结构与费用机制，最终门禁进行中，尚未提交**。日期：2026-09-24。
Graph IR 保持 v2。

## 外部证据

- MiniMax MSA 官方仓库 README 与论文读取快照保存在
  `artifacts/architecture-repair/C6-minimax-msa/`，URL、读取时间和 SHA256 见
  `minimax_m3_msa_sources.json`。
- 论文 §2.3、§3.1 Eq.(5–7) 明确：每个 KV head 对应一个 GQA group；index branch
  为每个 GQA group 生成一个 index query head，并由所有该组 query heads 共享选出的
  block 集合；index key 是跨组单头共享。每个 query 还必须保留 local block。
- 入库 Transformers forward 的 `MiniMaxM3VLIndexer` 与论文一致：index Q 为
  `index_n_heads=4`、index K 为单头，token score 按 block `amax`，再 top-k；关闭
  index value 时没有 value/output 分支。

## 当前问题与修复

原图已有 4 个 index heads 和单头 index K 的数值配置，但没有把“4 个 heads 是
4 个 KV/GQA group selector、而不是 64 个 query-head selector”写入结构语义，
也没有明确主 sparse attention 使用 group-shared selection。现已补充：

- `index_selection_scope=gqa_group`、`index_group_count=4`、
  `query_heads_per_group=16`、`selection_shared_by_query_heads=true`；
- `index_key_heads=1`、`block_score_reduction=max`、
  `local_block_always_included=true`、`index_value_path=disabled`；
- 主注意力的 `selection_scope=gqa_group`、每组 block budget 和禁用 index value。
- 展开步骤 `group_scores → block_max → local_boost → group_topk →
  valid_block_ids`，其中解释节点不持有 checkpoint 权重；父 composite 单独计费，
  子步骤不重复计费。

费用同时修正为：MSA 不计 QSA/DSA 的 ReLU 或跨 index-head 求和；块 max 按
每个 group 独立归约；local block 已在 Top-k 名额内，不额外增加选中 token。
没有把 MSA 改成 Flash-Next QSA 的每头选块，也没有改变现有 4×128 index
projection、单头 key cache、KV group cache。indexer composite 的矩阵计算仍按
4 个 group selector 计，主 attention 按 64 个 query heads 计。

## 独立测试

`frontend/src/structure/models/minimaxM3Msa.test.js` 覆盖两变体的 config-only、
production artifacts、group 语义、展开阶段、父子计费和小尺寸费用；旧实现先失败，
修复后当前定向结果为 6/6。浏览器覆盖原版/FP8、桌面/移动端 **4/4**，检查实际
indexer→sparse 边、展开阶段、公式、成本面板、SVG 和 pageerror。

最终全量门禁和受影响 golden 审阅通过后提交。本页不声称完整 checkpoint 逐模块审计。

## 本批验收（最终代码）

- MiniMax M3 / MXFP8，config-only + production artifacts：独立机制 6/6；
  前端全量 545/545，后端 184/184，内置模型基础检查 60/60。
- docs check、构建、principles 检查通过；真实桌面/移动浏览器 4/4，验证
  indexer 内部展开阶段、selected block ids → 主 GQA 的 SVG 边、公式、
  成本面板、中英切换、SVG 导出及 pageerror。
- 与 `a3d7288` 基线（B=1,S=4096）逐模型/逐模块对账保存在
  `artifacts/architecture-repair/C6-minimax-msa/reconcile-final.json`。原版及
  MXFP8 四条路径的参数容量均不变；prefill 理论 MAC
  108580721590272→108328198275072，decode MAC
  5534752899072→5534633361408。原因是 local block 纳入已配置的
  16 个 Top-k 块，不再额外加一个；indexer 分组 max、跨组归约及流量同步修正。
  config-only 与 artifacts 图节点数不同（后者受已发布 MTP truth 抑制），
  不能把 config-only 图当生产图。
- 仅 MiniMax-M3 和 MXFP8 的边/spec golden 更新，其他 58 条保持不变。
  这些是理论动作与逻辑流量，不代表 GPU 性能或完整逐模块 checkpoint 审计。
