# Flash-Next QSA：机制、checkpoint 与费用修复

## 范围和结论

本批只覆盖 `Qwen/Qwen3.8-Flash-Next` 与 `Qwen/Qwen3.8-Flash-Next-FP8` 的
QSA 主干/草稿模板，不代表完整 C5、全部模型审计或 GPU 性能验收完成。
Graph IR 保持 version/schema_version 2。

本地取证读取时间：2026-09-24（Asia/Shanghai）。基线：
`030e89d46699960d04c1b18993452b3b263ae202`。

### 外部证据

| 来源 | 定位 | 支持范围 / 限制 |
|---|---|---|
| [Raschka 架构图库](https://sebastianraschka.com/llm-architecture-gallery/) | Qwen3.8-Flash-Next 卡片 | 36 GDN + 12 QSA、4路 GR、第2层 PLE；不是完整视觉/MTP 图 |
| [官方技术报告](https://raw.githubusercontent.com/QwenLM/Qwen3.8-Flash-Next/4f58f4ddd855bedaf7eadcd53bbbb1b3362cecff/tech_report.pdf) | §2.1.2，Fig.3，Eq.12–16 | 原始 K 块 mean pool → RMSNorm → partial RoPE；跨 index 头汇总 ReLU；选块展开+尾部；图中有输出 sigmoid 门 |
| [官方模型卡](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) / [FP8 卡](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8) | Model Overview / Technical Report | 发布的512块、每块4 token；独立查阅两个变体 |
| 已入库 Transformers `modeling_qwen4_exp.py` | `Qwen4ExpTextQSAIndexer` 671–777；`Qwen4ExpTextAttention` 817–899 | 前向顺序、共享单 K、每头 Q/gate 打包、零中心 RMSNorm。是库参考实现，不冒称模型仓库自带代码 |
| 两个发布仓库的 shard82 safetensors header | 下表 revision / fixture | 实装张量路径、dtype、shape；仅局部证据，不代表全模型张量审计 |

报告 PDF SHA256：
`04f263446d74a35cb7cea368574e0c561f3b05c133be2c777ac884404063655d`。
下载快照、HTTP 206 / Content-Range 响应与完整差异报告在本地
`artifacts/architecture-repair/C5/`。只读取两个分片的长度前缀与 JSON header
（合计150,560字节），未下载权重数据区。

| 变体 | checkpoint revision | shard82 header SHA256 |
|---|---|---|
| 原版 | `de4b8e4d43b917e7706784d8bb445c9af86a3540` | `ce840d1d64f034886f1774712a1f0c16d89560f5ac8f5194624b9c9f5249eb95` |
| FP8 | `236dfdf285828023ca3bcd3f37366c58a3469b13` | `a4885cf829c44cf4305ceaf1bd2435b762a50194d99fe9ce6cfdd9a26b85ff62` |

### 冻结的实装断言

每个变体各9条 layer3 attention 张量保存在
`frontend/src/structure/models/__fixtures__/qwen-flash-next-qsa-header.json`。
两者在此局部均为 BF16；不能由此推断 FP8 发布的全部张量都是 BF16。

| 模块 | 权重 shape |
|---|---|
| `q_proj` | `[12288, 2560]`，不是无门的 `[6144, 2560]` |
| `k_proj` / `v_proj` | 各 `[512, 2560]` |
| `o_proj` | `[2560, 6144]` |
| `q_norm` / `k_norm` | 各 `[256]` |
| `indexer.index_qk_proj` | `[640, 2560]`，4个Q头+1个共享K头 |
| `indexer.q_layernorm` / `indexer.k_layernorm` | 各 `[128]` |

## 修复

1. QSA indexer 改为复合计费父节点；内部投影、Q/K切片、原始K cache、
   块池化、双 norm/RoPE、跨头打分、Top-k、块展开、tail补入均可展开。
   **先 pool 再 K norm**，禁止相反连线；块首位置与 query 位置分别标注。
2. 保留512个完整块预算；可见长度2051时允许2048+3，2052时回到2048。
   不使用 MSA 每 GQA group 的独立选块规则。
3. 保留 checkpoint 的 Q/K/V/O canonical 路径，Q projection 包含门宽度。
   输出门明确接收 sparse attention output 与 Q/gate split 两路输入，
   仅计 sigmoid×output，不重复声明投影权重。
4. indexer 父节点拥有执行计算，三个真实子模块拥有参数容量；
   父子不会重复计费。停用 MTP 的执行倍率为0，但容量仍保留。
   内部阶段的真实激活物化未知，Lens不把 `complete_blocks` 误当全文
   sequence 推导显存；显示 unknown，而不是虚构容量。
5. 生产费用与生成文档统一遵守复合父节点计费边界，不能只遍历叶节点。
   KV 对账也纳入复合 indexer，避免其 raw-key 读取被遗漏。
6. 真实张量 fixture 暴露公共绑定问题：truth 图使用 `root.0...` 布局ID，
   skeleton 使用真实模块ID，旧的 gap 判断混用了两者，导致已绑定张量
   又被追加为 gap。现在传递 canonical ID；未绑定的真实额外模块仍保留。

## 费用差异与假设

两变体、config-only 与现有 artifacts 加载路径的变化相同：

| 项目 | 基线 | 修复后 |
|---|---:|---:|
| Graph 节点 / 边 | 841 / 718 | 1062 / 952 |
| 模板声明参数 | 179,771,089,536 | 179,996,864,384 |
| prefill MACs，B=1 S=2051 | 14,004,559,796,224 | 14,435,321,724,928 |
| decode MACs，B=1 S=2051 | 284,312,090,624 | 284,520,931,328 |

参数增加225,774,848，精确等于13个驻留QSA（12主干+1MTP）乘以：
遗漏的 gate projection 15,728,640 + index Q/K projection 1,638,400
+ 两个 index norm 256。主干执行只计12层。主 KV/index cache 容量定义不变；
改变的是执行读取和可见尾 token 的计数。

驻留对账（字节；同一 B=1、S=2051）：

| 路径 | 基线权重 | 修复后权重 |
|---|---:|---:|
| 原版 config-only | 359,542,185,984 | 359,993,735,680 |
| 原版 artifacts / header总量 | 359,999,963,128 | 359,999,963,128 |
| FP8 config-only | 233,547,660,288 | 234,390,691,984 |
| FP8 artifacts / header总量 | 185,502,232,570 | 185,502,232,570 |

四条路径的 KV 均保持61,431,552字节、GDN state 均保持115,458,048字节。
FP8 config-only 的变化包含真实 Q/K/V/O 路径恢复后，命中发布配置的
`modules_to_not_convert`（该局部 header 确为 BF16）。FP8模板和header全模型
容量仍有显著差异，**不声称量化全模型已对账**；生产header总容量不受模板增加影响。

成本为**理论动作与逻辑流量估算**，不是实测：

- 按 query 可见长度闭式累计完整块/选中 token pairs，分别处理 batch、
  prefill/decode；独立枚举小输入验证闭式公式。
- indexer 按每次 forward 对每个完整 K 块池化一次估算；
  不模拟参考 Python 循环的重复池化、不声称真实 fused kernel 的 HBM流量。
- 索引按紧凑 int32 逻辑流量；不估算分配器 padding、kernel tiling 重读。
- Top-k 为扫描比较的一阶估算，不是精确排序算法复杂度。
- sparse core 补齐 scale/softmax 的 vector/SFU；KV 读取仍沿用逻辑足迹口径，
  不代表每个 query 无复用地重新读取所有 selected KV。

逐模型/逐模块差异可复现：

```sh
# BASELINE_ROOT 为上述 SHA 的只读 git archive 解包目录；
# 不 checkout / reset 当前 main，不改变用户工作树。
node scripts/evidence/structure/qwen-qsa-reconcile.mjs \
  BASELINE_ROOT artifacts/architecture-repair/C5/reconcile.json
```

## 冲突、unknown 与未完成范围

- 报告 Eq.16 对非整除预算使用 ceil，再展开截断；入库参考代码使用
  floor。当前发布2048/4完全整除，两者一致。合成非整除配置明确按参考
  floor实现，**不标记论文/实现普遍等价**。
- 新的全模型模板参数与既有 header 总量仍有差异，不用“小比例”掩盖；
  本批只完成上述9模块×2变体局部核对，其他差异仍待逐模块调查。
- GR widened/read/write/final contraction、PLE完整实装、MTP索引复用、
  多模态入口及全量60模型浏览器验收仍属于后续阶段。
- 本批 fixture 是局部张量清单，不是完整模型 truth；不能用其缺失张量
  推断其他模块不存在。没有量化 packed 形状的外推或性能 benchmark。
- 后续公共成本审计项：`roofline.js` 还会将已包含在 `actIn` 的
  `kvRead/indexRead` 诊断子桶再相加。本批表格报告 MAC/action ledger，
  **不把当前 roofline 的访存时间作为已对账结论**；该公共重复流量问题需
  独立修复并回归其他家族。

## 独立验证入口

- `qwenFlashNextQsa.test.js`：两路径/两变体、机制边、禁止反向池化边、
  输出门双输入、真实header绑定一次、复合计费/容量不重复。
- `formulas/__tests__/qsa.test.js`：独立因果枚举、边界尾部、手算 MAC/容量/流量。
- `truth/__tests__/graphTruth.test.js`：已绑定与真实未绑定张量共存时，
  无重复 gap；规范ID与布局ID分离。
- `qwenGateInputs.test.js`：31个Qwen变体两路径门控回归。
- `e2e/qwen-qsa.spec.js`：两发布条目的真实 SVG路径、池化/norm位置、
  展开/折叠、选择、缩放、中英文、成本面板和SVG下载。
- 旧 HEAD 上新增4个机制用例均失败，输出门与truth gap另有先失败日志。
  只有两个Qwen条目的结构/边 golden发生变化，其他58条目未更新。

### 本批最终门禁

2026-09-24，本次启动的测试进程，无复用/终止用户服务：

| 门禁 | 结果 | 本地日志（`artifacts/architecture-repair/C5/`） |
|---|---|---|
| 前端完整单测 | 509/509 | `full-unit-validated.log` |
| 内置模型检查 | 60/60 | `models-final.log` |
| 后端完整 pytest | 184/184 | `pytest-final.log` |
| docs check / build / 原则护栏 | 通过 | `docs-validated.log`、`build-final.log`、`principles.log` |
| Chrome桌面 | 3/3 | `browser-validated.log` |
| Chrome移动端 | 3/3 | 同上 |

浏览器包括两变体及本地合成safetensors的绑定/真实gap共存用例；
检查了实际SVG边、展开折叠、位置、选择、unknown容量、中英文、费用面板、
4个SVG导出。人工查看桌面与移动截图；这是本批覆盖，不是60条目全量验收。

首次并发运行后端为183通过/1失败（inline BERT API返回500）；
该用例单跑通过，完整重跑184通过。保留 `pytest.log` 和
`pytest-api-rerun.log`，未通过跳过测试或延长产品超时掩盖失败。
构建仍有既有的大chunk警告，不声明零警告。
