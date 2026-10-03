# Kimi-K3：NoPE、输出门与视觉 RoPE 修复

日期：2026-09-24。基线：`1452fe3`，本地 `main`。这是 C1 的 **MLA 子批次**，不是 K3 整体架构验收。

## 证据范围

- Raschka architecture gallery 的 `card-kimi-k3`：3:1 KDA/Gated MLA、NoPE、69 KDA + 24 MLA。图库用于发现和交叉核对，不替代发布实现。
- Moonshot 官方 `k3_tech_report.pdf`，§2.1.2、式 (7)，PDF 第 5 页：
  文本 MLA 无显式位置编码，输出为 `Wo(sigmoid(Wg x) ⊙ O)`。
- Hugging Face 发布 revision `f831ab66814297da540d832a5235f8e904f29d06`：
  - `modeling_kimi_linear.py` 的 `KimiMLAAttention.__init__/forward`。
  - `modeling_kimi_k3.py` 的 `MoonViTEncoderLayer.attention_qkvpacked`、`MoonViT3dEncoder.rope_2d`。
  - 网络读取的两个实现文件与仓库现有文件逐字节一致；不是仅使用本地副本推断。
- 同 revision 的 `model-00004-of-000096.safetensors`：仅读取 8 字节长度前缀和 **817,608 字节 JSON header**，HTTP 206；没有读取张量内容。提取第 3 层 MLA 的 8 个真实张量作为小型 fixture。

精确 URL、读取时间、SHA-256 和断言编号见 `kimi-k3-mla-sources.json`；真实 header fixture 为
`frontend/src/structure/models/__fixtures__/kimi-k3-mla-header.json`。
报告 URL 使用上游 `main`，本次内容以 SHA-256 冻结，不冒充已固定的 Git revision。

## 问题与修复

| 问题 | 修复与独立判据 |
|---|---|
| 文本 MLA 无条件发射 RoPE | 归一化读取 `mla_use_nope`，只取消文本执行型 RoPE；Q/K 仍为 `128+64=192`，KV down 输出仍为 `512+64=576` |
| 无 RoPE 后共享 K 分量容易失联 | Q、展开的 content K/V、原始 shared K 分量分别接入 SDPA；为后两条依赖标注语义，不按末维相等推断连接 |
| `g_proj` 被画成无权重门乘，另有悬空 `mla_gate` 投影 | 删除虚构 `mla_gate`；真实 `g_proj` 为 Linear `[12288,7168]`，独立 `output_gate` 做 sigmoid×O；两路输入接齐，再接 `o_proj` |
| 门投影按 query 宽度计参数 | 真实宽度为 `96×128=12288`，不是 `96×192=18432`；header 确认 |
| norm/down projection 名称无法精确绑定 | K3 配方声明真实 `q_a_layernorm`、`kv_a_layernorm`、`kv_a_proj_with_mqa`；节点与边共用路径表，不靠显示名匹配 |
| 可选 query 压缩不存在时仍声明 `q_a_proj` | 按 `q_lora_rank` 发射 `q_proj` 或压缩链；8 种 NoPE/gate/query-compression 组合均检查端点 |
| 视觉塔原先漏画逐层二维旋转 | 独立的 K3 视觉配方补 Q/K 二维 RoPE，V 旁路保持；保留 patch position additive 路径。视觉动作不会因文本 NoPE 消失 |
| 门乘只读取一个输入的流量 | MLA 门节点显式计两路输入、一路输出；投影费用归真实 Linear，不在门乘重复计权重 |

Graph IR 仍为 v2。没有引入推理框架依赖或新图协议，也没有修改模型原始配置。
SDPA 继续沿用现有复合父算子计费，展开的 score/softmax/context 不重复计入模型总量。
`output_gate` 和视觉 RoPE 都标记为非 checkpoint 模块。

## 参数与动作对账

工作点：batch=1，sequence=16；prefill 16 个文本 token，decode 1 个文本 token。
以下数字均是本工具的静态估算，不是性能实测。

### 单层 MLA

- 旧虚构 `mla_gate`：`18432×7168 = 132,120,576` 参数。
- 正确 `g_proj`：`12288×7168 = 88,080,384` 参数。
- 每层减少 **44,040,192** 参数，BF16 权重容量减少 **88,080,384 bytes**。
- 门乘每 token：0 MAC、12,288 vector FLOPs、24,576 SFU；0 权重。
- 门乘输入流量：`2×12288×2 = 49,152 bytes/token`；输出 `24,576 bytes/token`。
- 三处真实路径重命名不改变矩阵大小、MAC 或驻留。
- 被去掉的文本 RoPE：原估算每 token `2×96×192×3 = 110,592` vector FLOPs；真实 NoPE 为无此操作。

### 模型合计

24 个执行 MLA 层（当前折叠图为 23 个代表模块，最后 91–92 层 repeat=2）：

| 指标 | 修复前 | 修复后 | 原因 |
|---|---:|---:|---|
| 模板参数元素 | 2,780,984,610,400 | 2,779,927,645,792 | 减少 1,056,964,608 个虚增门投影参数 |
| 模板量化权重驻留 bytes | 1,555,065,170,464 | 1,552,951,241,248 | 非量化 MLA 门权重差 2,113,929,216 bytes |
| prefill MAC | 2,186,455,367,680 | 2,169,543,933,952 | `−24×44,040,192×16` |
| decode MAC | 646,288,651,264 | 645,231,686,656 | `−24×44,040,192` |
| prefill vector | 1,987,425,744 | 2,199,762,384 | 新增视觉旋转，去掉文本旋转 |
| decode vector | 1,990,687,005 | 2,242,836,765 | 同上；现有视觉工作负载口径见下方限制 |
| KV 驻留 bytes | 442,368 | 442,368 | 不因 NoPE 删除共享 64 维通道 |
| KDA state bytes | 449,372,160 | 449,372,160 | 未改 KDA |

视觉新增旋转按 27 层、1024 视觉 token、12 头、128 维计：
`27×1024×2×12×128×3 = 254,803,968` vector FLOPs。
原有全模型 SFU 不变，门 sigmoid 从误名 `g_proj` 移到 `output_gate`，没有新增一份 sigmoid。

逐模块前后差异由 `scripts/evidence/structure/kimi-mla-reconcile.mjs` 生成：
`artifacts/architecture-repair/C1-mla/reconciliation.json`。
覆盖 **60 模型×config/artifacts = 120 组**、prefill/decode 两相位；
其余 59 模型的节点数、边数、参数、动作及缓存对账不变。
归一化差异仅新增 `mlaUseNope`；另外 9 个 MLA 型号的 spec golden 变化仅来自显式位置编码属性和 score 公式文字更正，未改拓扑。

## 验证与证据目录

`artifacts/architecture-repair/C1-mla/`：

- `red-confirmed.log`：把新用例放入基线只读 archive，4/4 在旧实现失败；归因是错误 RoPE、无效可选端点、真实 header 绑定失败，而非依赖安装失败。
- `focused-after.log`：机制、形状/参数恒等式、声明和归一化定向测试。
- `unit.log`、`models.log`、`docs.log`、`build.log`、`pytest.log`：本批完整门禁。
- `browser.log`、`browser/`：K3 桌面/移动端真实浏览器。检查实际 SVG 路径、gate fan-in、NoPE 共享 K 边、视觉 Q/K 与 V 旁路、节点相对位置、展开折叠、选择、搜索、中英文、成本与 SVG 导出。两端截图已人工查看。

本批结果：定向 **37/37**，前端完整单测 **519/519**，内置模型 **60/60**，
后端 pytest **184/184**；docs check、生产构建通过；真实浏览器桌面/移动 **2/2**，
无重试。构建仅保留原有 chunk >500 kB 提示。

审阅结构和逐模块差异后才重生成 golden 与参考文档。原始模板测试并不作为官方语义正确的证据。

## 仍未完成 / 不得外推

1. **本批不覆盖 AttnRes 与 SiTU**：两者已分别由
   `kimi_k3_attnres_repair.md` 和 `kimi_k3_situ_repair.md` 独立修复并回归；
   本页仍只对 MLA、NoPE、输出门和视觉 RoPE 负责，不把其他子批次的证据
   重复计入本页。
2. K3 生产目录现在有 `skeleton-truth.json` 与 `source-ref.json`。该 truth
   文件覆盖 safetensors header 的完整 tensor inventory（折叠保存），并将原生
   MXFP4 的 packed storage units 与 logical parameters 分开；它不等价于下载
   权重数据或完成运行时前向审计。
3. 8 个 header 张量只证明第 3 层 MLA 的实装；其余层使用发布配置、报告和前向机制。全模型 packed MoE 参数及 KDA 路径不在这份 header 证明范围内。
4. 现有成本模型在 decode 仍计整个视觉塔；本批保留其工作负载约定，未声称这是“视觉已缓存的实际 decode”。需要公共视觉执行生命周期专项核验，不能用新增 RoPE 数字冒充性能提升或退化。
5. 其他门控算子的输入流量仍须逐一分清融合/独立输入口径；本批只明确 MLA 独立门的两路 compulsory reads，不机械套到复合算子。
6. NoPE 下 Q/K 拼接、广播的具体物化策略没有新造 GEMM 或额外缓冲。SDPA 仍使用既有 MLA 缓存/执行估算；参考 forward 的 expanded KV 与 absorbed 实现的差异需保留为实现选择。
7. 其他视觉家族的二维位置编码、norm、残差、真实路径仍需全量证据审计；不能因为这次公共 vision builder 增加一个配方开关，就将它们全部标记正确。
8. 全量 60 桌面 + 16 家族移动最终浏览器验收、A 阶段全变体证据闭环及其他 C 阶段剩余问题仍未结束。
