# Kimi-K3 AttnRes：双聚合、不可变深度状态、末端输出

日期 2026-09-24；基线 `a12d33a`，本地 main。保留 Graph IR v2。

## 证据与边界

- 重新联网读取 MoonshotAI 官方 K3 报告 §2.2 式 (8)–(10)：
  8 个 residual blocks、每 block 12 层、部分末 block，加上 embedding 共 9 个末端候选。
- 重新读取 Hugging Face 发布 revision
  `f831ab66814297da540d832a5235f8e904f29d06` 的 `modeling_kimi_linear.py`：
  `_forward_attn_residual`、`_apply_attn_res`、model bank 初始化及末端输出。
  URL、时间、SHA-256、冻结断言见 `kimi-k3-attnres-sources.json`。
- 复用同 revision 第 4 个 shard header 的第 3 层 4 个真实 norm/proj 张量，
  并新读取第 94 个 shard 的 **576-byte header**，确认两个真实
  `output_attn_res_norm/proj`；仅读 header，不读取权重数据。
  fixture：`frontend/src/structure/models/__fixtures__/kimi-k3-attnres-header.json`。

## 纠正旧图

旧图在普通两次 residual add **之后**追加一个 AttnRes 模块，且两个打分分支
由 aggregate 向外输出，既没有真实 pre-attention/pre-MLP 输入，也没有加权汇总末端；
历史数按 floor(index/12) 错算，第一 block 第 1–11 层被当作无历史。

现在每一物理层的顺序：

```text
prefix_in, bank_in → pre-attention aggregate → input norm → attention
prefix_in, bank_in → bank_out（仅 block 边界 append；否则引用）
attention output + 可选 prefix_in → prefix_after_attn
prefix_after_attn, bank_out → pre-MLP aggregate → post-attention norm → FFN
prefix_after_attn + FFN output → prefix_out
prefix_out / bank_out → 下一层的 prefix_in / bank_in
最后一层 prefix_out + bank_out → output aggregate → final norm
```

- 第 0 层 pre-attention 跳过：history 为空；norm/proj 权重仍真实存在，不能因跳过执行删容量。
- pre-attention 在写快照**之前**。边界层把旧 prefix 写入 bank，然后清空 prefix；
  attention 输出开启新 prefix，不能再加旧 prefix。
- 每层 pre-MLP 在 attention prefix 更新之后，使用本层 bank_out。
- bank_in/bank_out 是不可变的逐阶段状态，图中无伪循环。
- 第 i 层进入前 snapshots=`ceil(i/12)`；退出后=`floor(i/12)+1`。
  最后 bank 有 8 份历史（含 embedding），再加当前 prefix，总共 9 候选。
- 原始值不经过 norm 后再加权；norm 只用于 key/score，aggregate 内没有额外 output norm。

真值路径为 `layers.i.self_attention_res_norm/proj`、`layers.i.mlp_res_norm/proj`
和根级 `output_attn_res_norm/proj`，不再伪造 `.attn_residual.*` checkpoint 路径。
它们在视图上归入对应聚合容器，canonical ID 不随显示层次改变。

## 折叠与展示

折叠签名纳入 block index、写边界、有效历史数和物理层端点。
因为每一 prefix_out 都被下一物理层引用，本批将 K3 全部 93 层独立定位，
避免把层 N 的状态指向另一层的 repeat 代表。层内默认仍可折叠，不展开
评分细节就能看两处聚合和状态关系。

全图从 1901 nodes/1707 edges 变为 **4312 nodes/4254 edges**。
这是结构精确化带来的真实密度变化，不是新的协议或布局器。
深度跨层边在视图折叠后投影到真实层容器，保留原端点和 `depth-state` 关系；
全展开恢复真实 bank/prefix 端点。原始 Graph IR 不随折叠修改。

## 参数、计算、状态和未知值

每个聚合是一个复合父算子，**父节点统一计执行**；
两个真实 norm/proj 子节点各拥有 H 个参数，只计驻留，不重复计父节点已经涵盖的执行。
解释子步骤包括 candidates、score norm/proj、depth softmax、weighted sum。

令 T 为此次 batch×query tokens，H 为 hidden，C 为候选数：

- variance：square、sum、乘 `1/H`、加 eps、rsqrt；
- normalized keys，再乘预先合成的 `norm.weight*proj.weight`，归约 hidden 得 scores；
- softmax 沿 C，不是 H；
- `[T,1,C] @ [T,C,H]` 汇总。

按现有动作约定：

```text
MAC    = T C H
vector = T C (5H+3) + H
SFU    = 3 T C
weight reads = 2 H b
candidate reads = T C H b
output writes = T H b
```

内部参考计算为 FP32，边界 dtype 不变；动作分解逐位检查。
这不是运行时内核或设备精度吞吐校准，不能据此声称获得实测延迟。

93 层汇总：187 聚合节点，首层跳过后 **186 次聚合**；
有效候选总数 **1002**；prefix 加法共 **178 次**（8 个边界不加旧 attention prefix）。
所有 norm/proj 驻留保持；原先误执行的首层 score 权重不再读取，weight traffic 减少
`2×7168×2=28,672 bytes/forward`，但容量不减。

深度 history **不是自回归 KV，也不是 KDA recurrent state**：

- bank 节点记录 `depth_state_elements_per_token=snapshot_count×H`，寿命为当前 forward 的深度遍历。
- 不把每一状态版本的逻辑 footprint 相加当成常驻副本。
- 参考实现明确使用 `torch.cat([block_residual, prefix_sum.unsqueeze(1)], dim=1)`；
  8 个写边界使用独立 `attn_res_snapshot` 节点，零参数/零计算，但按
  `旧 bank + 新 prefix` 读取、扩容 bank 写出计入 `actIn/actOut`。
- 逻辑 bank 的形状、当前 forward 生命周期和参考路径搬运已知；仅优化实现的
  allocator 复用、alias/copy 消除和瞬时峰值不计入，不能把整个 snapshot storage 标成 unknown。
- 无写入层的 bank 只表示同一快照引用，不新增缓存容量。

工作点 batch=1、sequence=16：

| 指标 | 基线 | 修复后 |
|---|---:|---:|
| prefill MAC | 2,169,543,933,952 | 2,169,637,404,672 |
| prefill vector | 2,527,507,920 | 2,974,111,072 |
| prefill SFU | 961,333,776 | 940,046,912 |
| decode MAC | 645,231,686,656 | 645,237,528,576 |
| decode vector | 2,263,320,861 | 2,292,483,478 |
| decode SFU | 938,613,159 | 937,282,730 |

参数元素 **2,779,927,645,792**、量化权重驻留 **1,552,951,241,248 bytes**、
KV **442,368 bytes**、KDA state **449,372,160 bytes** 均保持。
逐模块前后差异：`artifacts/architecture-repair/C1-attnres/reconciliation.json`；
脚本 `scripts/evidence/structure/kimi-attnres-reconcile.mjs` 覆盖 60×2=120 组加载、
prefill/decode 两相位，使用独立发布循环期望，不从 builder 倒推。
其余 59 型号不变；golden 仅 K3 变化。

## 验证记录与未完成边界

`artifacts/architecture-repair/C1-attnres/`：

- `red.log`：旧实现机制/费用测试 2/2 失败。
- `targeted2.log`：机制、参数/形状恒等式、bytes 完整性定向通过。
- `kimiAttnRes.test.js`：93 层真实状态、边界、首层、末端、DAG 无环、
  小尺寸精确成本、父子无双计、局部真实 header tensor/skeleton 绑定、折叠端点投影。
- `unit.log` / `models.log` / `docs.log` / `build-final.log` / `pytest.log`：完整门禁。
- `browser-fixed.log` / `browser-fixed/`：桌面/移动的实际 SVG 边、block 边界、
  末端关系折叠保留、公式、选择搜索、中英文、成本和 SVG 导出。
  首轮浏览器测试误用叶卡片 selector 选择展开容器而超时，修正为真实
  `.rf-group-frame` 后重跑；没有放宽产品断言或改超时掩盖产品错误。

最终代码门禁：`unit-final.log` **530/530**；`models-final.log` **60/60**；
后端 `pytest.log` **184/184**；`docs-final.log`、`build-validated.log`、
`principles.log` 通过。构建仅保留既有 chunk 大小提示。
最终 `browser-validated.log` **2/2**，无重试；桌面/移动截图均已查看，
确认内部展开解释步骤仍对优化物化峰值显示 unknown，而 reference `torch.cat`
路径的 snapshot 搬运已经进入成本计算，不再把整条 snapshot 写入标成 unknown。

仍不得外推：

- K3 生产目录没有全量 header/skeleton sidecar；局部 header 不能证明全模型权重均正确。
- 全量 60 型号证据台账、其他 C 阶段机制及最终 60 桌面/16 家族移动验收仍未闭环。
- 未执行 GPU/inference benchmark。默认大图缩放与更深浏览体验须由最终全量验收继续复核，
  不因本批单模型通过即宣称全站布局没有问题。
