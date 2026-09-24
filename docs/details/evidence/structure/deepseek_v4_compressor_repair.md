# DeepSeek-V4 compressor / indexer 结构修复

状态：**结构与全量门禁已验证，已在本地 main 提交**。日期：2026-09-24。
Graph IR 保持 v2。

## 证据

- DeepSeek V4 Flash、Flash-0731、Flash-Vision-Exp、Pro、Pro-0813 的
  config/source-ref/header truth 已在仓库；画廊审计此前标记 V4 的
  compressorApe 内部节点悬空。
- pinned SGLang `compressor.py` / `indexer.py` 前向实现、发布模型卡与 SHA256
  记录在 `deepseek_v4_compressor_sources.json`；前向实现属于框架参照，
  不等同于 DeepSeek 官方发布权重；五个变体分别用本地 config/source-ref/header
  和生产 artifacts 加载验证。
- `Compressor` 的 forward 路径是 `wkv_gate -> compressed state -> norm/rope`；
  其中 `norm` 在 fused compressor 内部应用，不能画成 compressor 输出之后的第二次归一化；
  `C4Indexer` 内嵌 Compressor，索引路径是 `indexer.compressor.wkv_gate ->
  indexer.compressor.norm -> indexer`，同时 `q_proj` 与 `weights_proj` 汇入 indexer。
  这些是前向数据依赖，不是仅凭节点名称推断。

## 修复

V4 旧图已声明 compressor/indexer 节点，但没有把嵌套 compressor 的关键输入输出
连到消费者：

- 主 compressor：`compressor.norm -> compressor -> attention`（前一条是复合算子内部参数/子步骤依赖，
  后一条是压缩状态进入注意力）；
- indexer compressor：`indexer.compressor.wkv_gate -> indexer.compressor.norm ->
  indexer`；
- 保留 `indexer.q_proj`、`indexer.weights_proj` 到 indexer 的融合输入；
- 可选节点仍由当前层 ratio / compressorApe 发射条件控制，不存在的节点不生成
  边，避免 V4.1 规则错误套入 V4。

没有为不存在的 ratio=128/0 子节点强行补边，也没有把 V4.1 CSA2 的跨层 source/reuse
规则改写成 V4 规则。新增边只表达已有节点的真实依赖，不新增权重矩阵或计费叶。
特别是 Pro 的首层为 ratio=128，indexer 需要定位到独立的 ratio=4 层，不得
把首层 HCA 误接到 C4 的嵌套投影。

## 形状与费用边界

- V4 ratio=4 的主 `wkv_gate` 打包宽度为 `2 * 2 * head_dim`，
  ratio=128 为 `2 * head_dim`；C4 indexer 的打包宽度为
  `4 * index_head_dim`。打包投影经窗口压缩后形成 `head_dim` 状态，
  然后由 fused kernel 使用 norm 权重归一化。不能把原始投影输出
  与 norm 输入强求同宽，更不能把复合 compressor 输出重新 norm 一次。
- W5 全模型形状测试对这三种**限定 canonical 路径**执行投影宽度、
  ratio、head_dim、hidden_size 的精确断言，没有泛化放行
  `linear -> rmsnorm` 或所有 `norm -> indexer`。
- 本批只改变已存在节点的边，未修改权重/计算/驻留声明。五个变体在
  `batch=1, sequence=4096` 的 prefill/decode 共 10 组逐节点动作和总量
  均不变；机器对账见本地
  `artifacts/architecture-repair/C7-deepseek-v4/reconcile.json`。
  **这不证明原有压缩核成本已完整**：当前 `mla_kv_compress` 公式
  只计投影，缺少 fused 窗口压缩、APE、norm/rope 的精确动作拆解；
  在有可核算公式前保留为待办，不能据此声称压缩执行成本精确。

## 测试与范围

`frontend/src/structure/models/deepseekV4Compressor.test.js` 覆盖五个 V4 发布条目、
config-only 与 production artifacts，并额外覆盖 ratio-128 HCA 和 V4.1 CSA2，
共 11 条机制用例；旧实现的 V4 连接断言全部失败，修复后 11/11。

- 前端单测 **556/556**、后端 pytest **184/184**、内置模型 **60/60**；
  docs 生成与检查、build、principles 均通过。
- 只有五个 V4 条目的 edge/spec golden 哈希变化；V4.1 保持原值。
- Playwright 桌面与移动 **12/12**（五个 V4 + V4.1），逐项检查真实
  SVG 路径、展开/折叠、成本面板和语言切换；另有本地
  `artifacts/architecture-repair/C7-deepseek-v4/browser-deep.json`
  记录两种视口各六次的 SVG 边与控制台异常。

这批不代表 V4.1 被改成 V4，也不声称获得 GPU 实测性能结论。完整的
压缩核成本/缓存状态仍需结合更细粒度实现与 checkpoint 真值另行核对。
