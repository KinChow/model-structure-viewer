# Kimi-K3 SiTU-GLU：公式、三条 FFN 路径与动作计数

日期：2026-09-24。基线 `6a1782f`，本地 `main`。C1 的 SiTU 子批次；不代表 AttnRes 已修复。

## 外部证据

本批重新联网读取官方技术报告与发布实现，不只看本地配置：

1. `MoonshotAI/Kimi-K3/k3_tech_report.pdf`，§2.3.2、式 (12)、图 4；附录 §B 的 bounded output 分析。
2. Hugging Face `moonshotai/Kimi-K3` revision
   `f831ab66814297da540d832a5235f8e904f29d06` 的 `modeling_kimi_linear.py`：
   - `SituAndMul.forward`：gate softcap、sigmoid 原始 gate、可选 up softcap、最终乘。
   - `_get_situ_activation_params`：beta 缺省回退 1，linear_beta 缺省不截幅 up。
   - `KimiMLP` 与 `KimiBlockSparseMLP`：dense/shared 和 routed 都使用此激活。
3. Raschka gallery 的 K3 卡片继续作为结构交叉检查入口，不作为数学公式的唯一来源。
4. 复用上一子批次已经读取的同 revision 第 4 个 safetensors shard header，
   提取第 3 层共享专家的 3 个真实 BF16 投影张量到
   `frontend/src/structure/models/__fixtures__/kimi-k3-shared-header.json`。
   没有下载张量内容，也没有把 packed routed 权重的存储形状冒充逻辑形状。

URL、读取时间、内容 SHA-256、断言状态见 `kimi-k3-situ-sources.json`。
报告的 main URL 内容以 SHA-256 冻结，不能当作 Git revision。

## 机制与实现

发布配置：`hidden_act=situ`，`activation_situ_beta=4`，
`activation_situ_linear_beta=25`。

```text
g = gate_proj(x)
u = up_proj(x)
a = beta * tanh(g / beta) * sigmoid(g)
u_cap = linear_beta * tanh(u / linear_beta)  # 若未设置则 u_cap=u
y = a * u_cap
out = down_proj(y)
```

重要边界：

- sigmoid 输入是 **原始 g**，不是 tanh/softcap 之后的值。
- beta/linear_beta 是标量超参数，**零可训练权重**。
- 参考实现把激活转为 FP32，计算完成转回输入 dtype。节点属性保留这一语义。
- 用明确 `hidden_act` 选择激活，不依赖模型名、显示名或 shape 猜测。
- dense/shared 使用独立 `situ_glu` 节点，两个投影都接入它；真实投影 canonical 路径不变。
- routed 仍为 `fused_moe_mlp`，内部激活由结构属性选择 SiTU，三段 GEMM 只计一次；
  不新增一份独立计费激活节点。
- 保留 LatentMoE 的 down projection → routed experts → combine → RMSNorm → up projection，
  以及独立的全宽 shared experts 支路。
- 未使用 SiTU 的 59 个条目不更改其算子或费用；现有其他激活近似不是本批验收结论。
- `situ_glu` 是语义节点，标记 `checkpoint_module=false`；同名但矛盾的 checkpoint
  张量留作 gap，不偷偷变为激活权重。

新增一个公式身份和一个 tanh 原子，沿用现有注册表、分解与费用链路，不另建激活框架，
Graph IR 仍为 v2。

## 动作和流量约定

令 `E=T×I`。常量 `1/beta` 和 `1/linear_beta` 预计算；每次 softcap
包含 2 次 scale 和 1 次 tanh。sigmoid 延续现有 2 SFU 约定。

| 激活 | MAC | vector | SFU | 边界读取 / 写出 |
|---|---:|---:|---:|---|
| 旧 SwiGLU | 0 | 2E | 2E | 2Eb / Eb |
| SiTU，不截幅 up | 0 | 4E | 3E | 2Eb / Eb |
| SiTU，同时截幅 up（K3 发布） | 0 | 6E | 4E | 2Eb / Eb |

tanh 计 **一个语义超越函数动作**，不是保证所有硬件只执行一条 SFU 指令，
更不是延迟测量。独立的 `scale/tanh/scale/sigmoid/mul/...` 分解与闭式计数
在小尺寸用例中逐位核对。

流量是融合激活的 compulsory 边界读取/写出，内部 FP32 临时数组的实际物化取决于实现。
没有把参考 Python 的所有临时张量假定为 HBM 常驻，也不声称获得了融合性能收益。
激活超参数不进入权重、量化、KV 或 recurrent state 账本。

## 全模型对账

冻结的发布结构期望：1 个 dense FFN、92 个 MoE FFN，
每 token 16 个 routed experts、shared 宽 6144；
dense 中间宽 33792，routed 中间宽 3072。

```text
activation elements per text token
  = 33792 + 92 × (16×3072 + 6144)
  = 5,121,024
Δvector = 4 × T × 5,121,024
ΔSFU    = 2 × T × 5,121,024
```

batch=1、sequence=16：

| 指标 | 修复前 | 修复后 |
|---|---:|---:|
| prefill vector | 2,199,762,384 | 2,527,507,920 |
| prefill SFU | 797,461,008 | 961,333,776 |
| decode vector | 2,242,836,765 | 2,263,320,861 |
| decode SFU | 928,371,111 | 938,613,159 |

参数元素 **2,779,927,645,792**、权重驻留、所有 GEMM MAC、
compulsory 输入/输出流量、KV 和 KDA state **均不变**。
prefill 融合入口的流量仍保留 unknown，没有为了对账改成零。

`scripts/evidence/structure/kimi-situ-reconcile.mjs` 对基线只读 archive 运行：
60 条目×config/artifacts=**120 组**、每组 prefill/decode。
只有 K3 的上述 vector/SFU 变化，其余 59 个模型完全不变。
逐模块前后数据见 `artifacts/architecture-repair/C1-situ/reconciliation.json`。

## 测试与证据

证据目录：`artifacts/architecture-repair/C1-situ/`。

- `red.log`：新机制用例在旧实现上 4/4 失败。
- `kimiSitu.test.js`：dense/shared/routed、可选 linear_beta、beta 缺省、
  小尺寸 exact 动作、独立原子分解、prefill/decode、真实共享投影 tensor/skeleton
  绑定、矛盾同名张量不绑定语义节点。
- `atoms.test.js`：新增 tanh 手算动作；已有恒等式继续约束费用不重复。
- `unit.log`、`models.log`、`docs.log`、`build.log`、`pytest.log`：完整门禁输出。
- `browser.log`、`browser/`：桌面/移动真实生产页面，验证 SVG 两路输入、
  dense/shared/routed 正确公式、KaTeX 与融合 ASCII 公式、展开折叠、选择搜索、
  中英文、成本和 SVG 导出。

本批实测：前端完整单测 **525/525**，内置模型 **60/60**，
后端 pytest **184/184**；docs check 与生产构建通过。
桌面/移动浏览器 **2/2**，无重试；两端截图已查看。
构建保留既有 >500 kB chunk 提示，没有将该提示隐藏或修改阈值。

更新 golden 前已审阅逐模块差异：仅 K3 的 spec golden 改变；
边的布局 ID/order 未变，因此 ops-edge golden 不变。归一化 golden 变化来自新增三个
明确配置字段，冻结的 schedule fixture 未修改。

## 未完成项

- K3 **AttnRes** 仍未修复，不能将 K3 整体标记为架构验收通过。
- K3 生产目录没有 header/skeleton sidecar，生产离线加载仍 config-backed；
  本批 header fixture 是局部实装证据，不是全模型参数真值。
- routed packed 权重、视觉生命周期、其他激活变体的费用精化及全量 60/16 浏览器验收
  仍属于整体计划中的后续工作。
- 未运行 GPU、推理框架或性能 benchmark；不提供实测性能结论。
