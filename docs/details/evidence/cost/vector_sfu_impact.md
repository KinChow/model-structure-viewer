# vector/sfu 对算子的影响（Qwen3-0.6B, A100-80GB）

> 问题：前端算子的 vector（逐元素 FLOPs）/ sfu（超越函数 ops）通道对 roofline 有多大影响？
> 方法：① 用 A100 有效费率对每个算子做五路时间分解，看 vector/sfu 是否/何时成为 bound；
> ② ncu 采 XU（SFU/transcendental）流水线指令，交叉核对 MSV 的 sfu 计数量级。
> 费率：matrix 109.2 TMACs/s、vector 19.5 TFLOP/s、sfu **4.875 TOPS**（最慢单元）、mem 1835 GB/s。

## 一、五路时间分解：vector/sfu 从不 bound

| 算子（prefill S=512） | matrix | vector | sfu | memory | bound | vec 占 bound | sfu 占 bound |
|---|---|---|---|---|---|---|---|
| linear（q/k/v/o/gate/up/down/lm_head） | 主导 | 0 | 0 | — | **matrix** | 0% | 0% |
| rmsnorm | 0 | 3.0µs | 0.003µs | 32µs | memory | 9.4% | ~0% |
| rope | 0 | 6.8µs | 0（A3 缓存 sin/cos） | 144µs | memory | 4.7% | 0% |
| **sdpa 注意力** | 138µs | 12µs | **24µs** | 160µs | memory | 7.5% | **15.1%** |
| **swiglu** | 0 | 4.5µs | **18µs** | 144µs | memory | 3.1% | **12.5%** |
| residual_add | 0 | 0.75µs | 0 | 48µs | memory | 1.6% | 0% |

decode 相位所有算子 memory-bound，vector/sfu 占比更低（sdpa sfu 0.1%、swiglu sfu 12.5%）。

**结论**：Qwen3-0.6B 在 A100 上**没有任何算子是 vector-bound 或 sfu-bound**——bound 恒为
matrix（prefill GEMM）或 memory（其余 + 全 decode）。vector/sfu 只占 bounding 路径的 0–15%。

## 二、sfu 影响最大的是"超越函数"算子

- **softmax（注意力）sfu 24µs = bound 的 15%**、**SiLU（swiglu）sfu 18µs = 12.5%**——这两个含
  exp/reciprocal，是 A100 最慢单元（sfu 4.875 TOPS，比 matrix 慢 ~45×、比 vector 慢 4×）的负载。
- rmsnorm 的 sfu（rsqrt）与 rope（A3 sin/cos 缓存）几乎为 0；GEMM 无 vector/sfu。

## 三、ncu 交叉核对：MSV sfu 量级正确、非硬件精确

XU（SFU/transcendental）流水线实测（thread≈warp-inst×32）vs MSV sfu 逻辑计数：

| 算子 | MSV sfu | ncu XU(thread) | ncu/MSV | 解读 |
|---|---|---|---|---|
| swiglu | 3,145,728 | 4,718,592 | **1.5×** | sigmoid 实测 ~3 MUFU/元素；MSV A5 记 2（exp+rcp）——量级对、略低估 |
| 注意力 softmax | 4,202,496 | 14,704,640 | **3.5×** | flash 在线 softmax 的 running max/sum 重标定多耗 MUFU；MSV 记 2/score |
| rmsnorm | 512 | 1,024 | 2.0× | rsqrt 逐行；绝对量极小，占比~0 |
| rope | 0（A3） | ~3.4M（微基准 bf16 处理） | — | 真实 fused rope 用预算 sin/cos 缓存，A3(SFU≈0) 成立；朴素微基准非代表 |

**MSV 的 vector/sfu 是逻辑算子数模型（A5 约定：sigmoid=2、exp/rsqrt/div=1），与硬件 MUFU 指令数
非 1:1（差 1.5–3.5×）**——因为真实 kernel 有范围规约、在线 softmax 重标定、类型转换。这与 matrix
通道的"逐位相等"性质不同（matrix 是 aten 级精确真值）。

## 四、对验证结论的影响 & 何时 sfu 会变重要

- **对本模型无实质影响**：sfu 从不 bound，且量级正确（1.5–3.5×）；即便 sfu 计数有 3.5× 误差，
  最大只把 sfu 路径从 15% 抬到 ~50%（注意力），仍 < memory/matrix，**不改任何 roofline bound 判定**。
  故 operator_cost.md 的对账结论（matrix 精确、bound 分类正确）不受 sfu 计数精度影响。
- **sfu 会变重要的边界**：注意力在 **flash 融合**下 memory 塌缩（见 flash_kernel_caliber.md：scores 不落 HBM，160µs→约 Q/K/V/O
  量级）后，sfu（MSV 24µs、实测 ~84µs）相对 matrix（因果 138µs）显著上升，趋于**可竞争 bound**。
  这正是 MSV 坚持在五路里显式建 sfu 的意义——在融合/长上下文/弱 SFU 芯片下它可能翻转 bound；
  省略它会让 roofline 上界失真。
- **建议（非本轮改动）**：若要 sfu 通道从"量级正确"升到"硬件贴合"，可把 A5 的 sigmoid/exp MUFU
  当量按芯片校准（如 sigmoid 3、flash 在线 softmax 额外重标定项），并在 flash 口径下同步下调注意力
  memory 路径——两者配套才能让注意力的 bound 反转在 roofline 里如实呈现。

## 复现

`python3 -c` 五路分解读前端算子 dump（`frontend_ops.json` 由 `scripts/evidence/cost/operator_cost.mjs` 重生）+ 上述费率；ncu：
`ncu --profile-from-start off --metrics sm__inst_executed_pipe_xu.sum,sm__inst_executed_pipe_fma.sum python3 scripts/evidence/cost/operator_bench.py <op>`。
