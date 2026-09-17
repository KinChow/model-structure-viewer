# 权重归属声明与分片响应矩阵（N2-4 完整方案）

2026-09-09 定稿。状态：**三波收官（W-A/W-B/W-C 全部落地，§7 已销案）**。
W-A：fused_moe_mlp 独立 id + 层 1 声明 + 锚 1 全目录 5233 叶容差 0。
W-B：moe_tp/moe_ep 计划轴 + sharding.js 组合语义纯函数 + 三消费者接线，
锚 2 三方一致。W-C：量化 MoE 家族声明逐模型核实（13 家族 / 30 模型）+
compressed-tensors 第四量化方案（w4a16 / mxfp4）+ quantizationConfigOf
的 raw.text_config 漏检修复；容量修正：M2.7 4.767e11→2.521e11、
V4-Pro 3.237e12→1.689e12、Qwen3.8-2.4T 4.851e12→2.480e12、
K2-Thinking 2.053e12→5.940e11、K2.5 系 5.134e11→5.948e11、
K3 1.390e12→1.555e12。上文问题链：量化容量逐矩阵化（N2-3）→ 显形
swiglu 携带专家 GEMM 未枚举（N2-4 登记）→ EP 维度补刀（用户）→
联网调研定稿（本文件，`188679b`/`c3ed249` 两轮迭代）。

## 一、问题总账（全部实证，无推测项）

| # | 问题 | 影响面 | 实证 |
|---|---|---|---|
| 1 | 量化枚举只认 linear 族叶，**swiglu 携带的专家 GEMM**（MoE 模板把 gate/up/down 三矩阵融合进 swiglu 叶的 counts，3·E·EH·EI）不在枚举范围 | 全部 25 个量化 MoE 模型；V4-Pro 1.55e12 / Qwen3.8-2.4T 2.37e12 / Kimi-K2 系 1.02e12 参数未枚举（探针实测） | operators_reference §7 N2-4 条 |
| 2 | EP/TP 逐卡投影的权重源 `nodeWeightBytes` 依赖叶 weight_shapes，**派生路径全零** → 专家/非专家的 EP/TP 切分塌缩（0/0），只剩外部 targetWeightBytes 整体缩放，不分轴 | 全部无 checkpoint 模型（即全部内置模型的默认路径） | parallel.js:213/230 `nodeWeightBytes` |
| 3 | 容量分桶把**排除矩阵**（modules_to_not_convert / dynamic 命中）落在标量量化宽（fp8 1B）而非其真实运行精度 bf16 | M2.7 实测 -150,048 反常下降 | N2-3 提交后自查 |
| 4 | 组合语义缺失：EP_SIZE=TP×DP、无 EP 时 DP 切专家、混合 ETP、MLA KV ×tp 复制 | 并行投影与容量分桶的 DP/EP 组合 | 联网调研（下文出处） |
| 5 | fp32 / 量化 per-tensor dtype | 已修（paramDtypes.js + quantBytes.js），纳入本方案声明体系统一 | 9382c77 |

其中 #1+#2 使量化 MoE 模型的「放得下吗」答案在派生路径（全部内置模型的默认路径）上偏差接近 2×——这是当前影响面最大的正确性问题。

## 二、根因与设计原则

**根因不是公式，是同一份知识（每个权重矩阵的 [out, in, 数量, 精度, 分片亲和]）
没有单一住址**，散在四处互不一致的载体里：counts 闭式公式（有恒等式锚定）、
nodeWeightBytes 的 weight_shapes（派生路径缺席）、quantBytes 的路径匹配规则、
parallel.js 的路径正则规则表。每处都对各自消费者负责，彼此漂移不可见。

**设计原则**（项目既有纪律的延伸）：
- 算子身份按算法出处区分（QSA/DSA/MSA 先例）——融合专家 MLP 与纯激活
  不得共用 `swiglu` id；
- 一份知识一个住址、多个消费者（paramDtypes / canonicalModulePath 先例）；
- 锚定测试代替手推数字（负容量模拟器的教训）。

对标成熟方案：
- vLLM/SGLang 量化：按**模块**应用（FusedMoE 模块持打包 w13/w2，ignore
  list 按模块名）——模块即单位，模块自描述权重；
- HF / safetensors 类显存工具：per-tensor checkpoint 元数据（safetensors
  index/header）——每个张量天然可见，无需按算子类型推断。msv 的
  checkpoint 路径（parameters_by_dtype per-dtype）已是该方案且正确；
  缺口仅在无 checkpoint 的派生路径。

## 三、三层设计

### 层 1：叶子声明（张量侧）

builder 产出叶子时写入机器可读声明：

```js
attributes: {
  weightMatrices: [
    // 每组 = 共享同一量化处理与分片亲和的矩阵集合
    { class: "ep",          out: 2048, in: 7168, count: 256, matrices: 3 }, // 路由专家 gate/up/down
    { class: "tp",          out: 6144, in: 3072, count: 1,   matrices: 3 }, // shared/dense（若融合）
  ],
}
```

- `class`（分片亲和，四类）：`tp`（attention/dense/shared GEMM，÷tp）、
  `ep`（路由专家，÷ep 轴、含不均衡区间）、`vocab`（lm_head/embed，视
  vocabParallel）、`replicated`（norm）；
- `count` × `matrices` × [out,in] = 该组全部元素（与叶 counts.bytes.weights
  逐位可对账）；
- dtype 不进声明：未量化参数走 paramDtypes，量化参数的字节由层 2 消费者
  按 quant 方案计算；
- **终态**：无声明叶子不得进入权重分片或量化 fallback。协议未覆盖时返回
  `unknown` 并生成诊断；完成覆盖和迁移后删除
  `WEIGHT_PROJECTION_RULES`、路径正则和其他同义推导逻辑。

builder 侧改动清单：MoE 模板（moe.js）的 expert_mlp 叶改独立 operator_id
`fused_moe_mlp`（对标 vLLM FusedMoE，终结与纯激活共用 swiglu id 的身份
过载）并携带声明；dense/attention 的 linear 叶声明由统一助手
（layer base 或 operatorSpec 内）按现有形状自动产出，不逐模板手写。

**fused shared expert 的方案更正（P3，2026-09-10 取证结论）**：本文原写
"shared 融合形态同叶声明两组（ep 组 + tp 组）"，该假设与 checkpoint 事实不符，
已作废。取证：`models/moonshotai/Kimi-K3/index-summary.json`（派生自 gitignore 的 `index.json`）每个 MoE 层只有
`shared_experts.{gate,up,down}_proj.weight` 各一个（92 层 × 3 = 276 个张量），
`modeling_kimi_linear.py:797-801` 先 `intermediate_size = moe_intermediate_size
× num_shared_experts` 再实例化**单个** `KimiMLP`。即"融合"= 一个更宽的 MLP，
不是打包张量，也不涉及 ep 亲和（shared expert 唯一分片语义 = ÷tp，见
`parallel_protocol.md` Q5）。现有三叶形态（out = 模块宽）与 checkpoint 1:1
对应，无需独立 operator_id、无需双组声明。P3 的实际内容 = fused 判定单源化
（删 normalize 的 model_type 子串第二判定源，归 archs 配方）+ 语义锁测试。

### 层 2：计划侧（轴与策略）

plan schema 扩展（validatePlan 同步）：
- `moe_tp` / `moe_ep` 分离（借鉴 SGLang 的显式逻辑轴，同时吸收 vLLM 的
  有效专家域和 local expert ownership 语义）。默认值、物理 rank 约束和
  attention/MoE rank 复用关系以 `details/parallel_strategies.md` 调研结论和
  M12 定稿为准，在定稿前不得继续扩大现有默认语义；
- `attnMode`（"dp"，已有）、`vocabParallel`（已有）保留；
- **组合语义纯函数**（出处：AMD vLLM playbook / TRT-LLM / vLLM DP 文档）：
  - EP 启用：ep_size = tp × dp（DP attention + EP，DeepSeek 系标准部署），
    路由专家每卡 = E/(tp×dp)；
  - 无 EP：DP 也把路由专家 ÷dp 切（专家 TP 切分语义）——「DP=复制」只对
    attention 成立；
  - shared expert ÷tp；仅 EP+TP+DP 全开 + 特定 all2all backend 时复制
    （AMD playbook 实证）；
  - ZeRO/FSDP ÷dp 分片：**范围外**（推理工具，vLLM DP=复制语义正确；
    SiDP 类方案登记不展开）。

### 层 3：内存类响应（每类对轴的响应不同）

| 内存类 | tp | ep/moe 轴 | dp | pp | 现状 |
|---|---|---|---|---|---|
| weights（tp 组） | ÷tp；attnMode=dp 且 attention 叶复制 | — | MLP 仍 ÷tp | stage | 声明化；DP-attention 权重复制已接 `declaredClassDivisor` |
| weights（ep 组） | ÷moe_tp | ÷moe_ep（不均衡区间见下） | 无 EP 时 ÷dp | stage | 缺声明 → 本方案 |
| KV cache | ÷min(tp, kv_heads)；MLA/MQA 复制 | — | attnMode=dp 复制 | stage | **已实现**（kvBytesPerCard isMla/attnMode 分支） |
| KDA state | ÷tp | — | attnMode=dp 复制 | stage | 已实现（stateBytesPerCard） |
| activations | TP/SP/CP | — | — | — | **范围外**，登记不展开 |

EP 不均衡区间沿用 `expertWeightRange`（average = total/ep；worst =
total/experts × ceil(experts/ep)），声明体的 count 即 experts 语义。

## 四、消费者接线（三处，一处不落即死功能——counts.bytes 教训）

1. **量化枚举**（cost/quantBytes.js `quantizedMatrixBytes`）：除 linear 族
   外，消费叶声明逐组计算 `count × matrices × quantLinearWeightBytes`；
   无声明叶子保持现状；
2. **逐卡投影**（cost/parallel.js `weightBytesPerCard` / `accountNode`）：
   声明组按 class 除数投影；`isRoutedExpertPath` 路径正则降级为无声明
   回退；`expertWeightRange` 接入声明的 count；
3. **容量分桶**（aggregate.js `quantCapacityBytes`）：排除矩阵回 bf16 桶
   （已修）由声明取代路径正则判定；分桶与枚举同源。

## 五、验收锚（四条，全部机械化）

- **锚 1（单源）**：声明元素数 × 2B == 叶 counts.bytes.weights，逐叶断言
  （权重字节恒等式已锚定叶 counts，因此声明错误立即红）；
- **锚 2（EP 自洽）**：EP 计划下 M2.7 每卡权重 = 专家块÷moe_ep + 其余÷tp，
  与聚合容量、`expertWeightRange` 三方一致；
- **锚 3（回退不变）**：无声明叶子走路径规则表，行为逐位不变（现有
  parallel/roofline 测试全绿）；
- **锚 4（golden 纪律）**：哈希基线重生成 + 人工审 diff，diff 中只允许
  出现本方案声称的字段类型。

## 六、执行波次（每波独立可回退，基线先行）

| 波 | 内容 | 验收 |
|---|---|---|
| **W-A** | 层 1：weightMatrices 声明 schema + operatorSpec 统一产出助手 + MoE 模板改 id（fused_moe_mlp）+ linear 叶自动声明 | 312+ 全绿（golden diff 仅 id/attributes）；锚 1 逐叶断言绿 |
| **W-B** | 层 2+3：plan schema 扩展（moe_tp/moe_ep）+ sharding.js 纯函数 + 三个消费者接线（声明优先、规则表回退） | 锚 2/3 绿；EP 语义测试（DP 切专家、混合 ETP）手算断言 |
| **W-C** | 覆盖推进：25 个量化模型的声明逐模型核实（家族差异表落地）+ golden 全量重生成审阅 + 文档（§7 销案 N2-4、MAINTENANCE 棘轮 +1） | 锚 4；六 oracle 全绿；M2.7 容量回落 ≈2.4e11 量级 |

回退方式：每波单提交；W-B 的消费者接线以 feature flag（声明存在才走新
路径）实现原子回退。

## 七、范围外（显式登记，不做）

- ZeRO/FSDP ÷dp 推理分片（SiDP 类方案，登记出处）；
- activations 的 CP/SP 响应；
- 计划搜索/推荐（五支柱越界，Vidur 指引不变）；
- EP 负载不均衡的动态建模（保留 average/worst 区间即可）。

## 附录：声明覆盖缺口台账（P2 护栏首跑，2026-09-10）

判据 = `counts.bytes.weights > 0` 或 `weight_shapes` 非空或 `type === "embedding"`
的叶必须有 `weightMatrices` 声明。护栏测试
`modelIdentities.test.js` 「P2 护栏：带权重叶的 weightMatrices 声明覆盖」，
棘轮基线登记在 `MAINTENANCE.md`（只许下降）。

首跑：**带权叶 18399，已声明 5233（linear tp 组 4271 + fused_moe_mlp ep 组 962），
缺声明 13166**。

P4 提交 2 后：**已声明 18399 / 带权叶 18399，缺声明 0（棘轮归零）**。长尾族
（MLA/KDA/conv1d/mHC/HC/PLE/embedding）的声明组与 counts 组成逐项同源，分片
亲和全部按 vLLM 源码取证：

- **replicated（无并行包装 = 每卡完整持有）**：mHC fn/base/scale（裸
  nn.Parameter，`deepseek_v4/amd/model.py:712-753`）+ 融合 norm；HyperConnection
  的 raw nn.Linear（`hyperconnection.py:176-193` 注释原文 "raw Linear weights
  (checkpoint-compatible)"）；PLE/conv1d/embedding（custom kernel /
  ParallelEmbedding 无 quant_method 包装，本就不被量化 → quantizable: false）；
- **tp（有分片取证）**：KDA/GDN 衰减参数 dt_bias/A_log（
  `kimi_gdn_linear_attn.py:241,268` `sharded_weight_loader(0)` /
  `a_log_weight_loader(0)` 沿头维切）；
- **锚 1 升级为 dtype-aware**：声明组带 `param_dtype` 键（引用 paramDtypes
  登记表，dtype 知识不进声明），fp32 组按 4B 对账；embedding 走登记例外
  （声明=驻留容量，gather 流量按行计入 actIn）。

下表为首跑缺口台账（历史记录，供追溯）：

| operator_id / type | 缺声明叶（首跑） | P4 提交 1 后 | 典型 path | 目标 class |
|---|---|---|---|---|
| linear | 5966 | **0** | `lm_head.linear`、`moe.router`、`self_attn.qkv_proj`、projector | vocab（lm_head/output）/ replicated（router）/ tp（其余） |
| gemma_rmsnorm | 2402 | **0** | `norm.rmsnorm` | replicated |
| rmsnorm | 2027 | **0** | `input_layernorm` | replicated |
| gated_rmsnorm | 433 | **0** | `self_attn.output_gate_norm` | replicated（逐头宽度取最后一维） |
| mla_kv_compress | 475 | 475 | `self_attn.kv_a_proj` | tp |
| gated_delta_attention | 433 | 433 | `self_attn.state_update`（KDA 衰减/门参数） | tp |
| causal_conv1d | 433 | 433 | `self_attn.short_conv`（卷积核） | tp |
| mhc_fused_post_pre | 299 | 299 | `mhc_ffn_pre.fused_post_pre` | tp（fn 为 fp32，见 paramDtypes） |
| mhc_pre | 299 | 299 | `mhc_attn_pre.pre` | tp |
| mla_query_compress | 230 | 230 | `self_attn.q_a_proj` | tp |
| hyper_connection | 110 | 110 | `hyper_connection_mixer.final`（down/up/inject） | tp |
| embedding | 57 | 57 | `embed_tokens` | vocab |
| ple | 2 | 2 | `decoder.1.ple.inject` | tp |

P4 的四处口径修正（都由锚 1/锚 2/golden 三条护栏抓出，非事后发现）：
1. **router 是 replicated 而非 ÷tp**：vLLM `qwen3_moe.py:167`
   `self.gate = ReplicatedLinear(...)`、`fused_moe/router/gate_linear.py:18`
   `class GateLinear(ReplicatedLinear)`。声明前 router 落在规则表第 4 条
   （tp>1 即 ÷tp），M2.7 每卡权重因此少算 7.3e7 B。
2. **lm_head 是 vocab 类**：`ParallelLMHead`（vLLM `qwen3_moe.py:571`）——
   受 `vocabParallel` 开关支配，与普通 tp 叶不同（关闭时复制而非切分）。
3. **声明需要 `quantizable` 标记**：量化方案只作用于 Linear 权重矩阵
   （vLLM/SGLang quant config `targets: ["Linear"]`），norm scale 与 bias 不在
   其中。**不能用"维度>1"当判据** —— K3 的 `attn_residual.res_proj` 是
   out=1 的真 GEMM（[1, 7168] 打分投影），会被误伤（实测 K3 容量差 1.97e6 B）。
