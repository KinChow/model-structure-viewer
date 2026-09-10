# 恒等式校准方法：域拆分账本

来源：T4（R1 收敛）与 M8-V2（vision 域拆分）两轮校准的实践沉淀。
恒等式本身见 `extractor.identity.test.js` 头注释；本文档写**方法**——
当 ratio 超差时如何定位，而不是猜。

## 四样东西（必须同时持有）

校准期间任何一个时刻，四样输入缺一样就会退化成猜谜：

1. **config 字段语义**——每个参与公式的字段"在 modeling 源码里是什么"。
   依据源阶梯（`principles.md` §4.3）：checkpoint 自带 modeling_*.py →
   transformers → 原始仓库。警惕：同名字段跨家族语义不同（例：
   Kimi 的 vision_config 无 `in_channels`，缺省 RGB=3；
   `num_expert_group` 等新字段语义未知时必须查源码，不能按名字猜）。
2. **公式两侧**——counts 侧（`formulas/` 注册表 + extractor 分派）与
   期望侧（identity 测试的 `textExpectedSide`/`visionExpectedSide`）各自的
   公式和来源。两侧是**两本独立手工账**，超差 = 至少一本错，也可能都错
   （两侧照抄同一份错误理解时，只有外部锚点能救，见 R1 的 k/E 案例）。
3. **官方数（外部锚点）**——官方参数总量 / active 参数 / 已知层宽。
   这是唯一独立于两本账的裁判。缺锚点时先去 modeling 源码立锚，
   再归因（K3 案例的教训：两侧 1.30 倍差异若没有 32B active 官方数，
   根本无法判断哪侧错）。
4. **探针输出（域拆分账本）**——分域、分叶子、分单位的计数与期望对照表。

## 工作流

```
冻结口径（T、V、phase 固定）
  → 域拆分账本（text/vision 分域：counts 与 expected 各自求和、求比）
  → 差异归因三选一：counts 错 / 期望侧错 / 建模边界（登记 REGISTERED）
  → 修复后先看方向（ratio 是否向 1 收敛、量值是否吻合预测）
  → 外部锚点闭合（总数对上官方）
  → 断言收紧、报告制行转入断言
```

## 账本形态（探针脚本模式）

用 `node --input-type=module -e` 内联：`buildStructureFromConfig` + 树遍历
（`childRepeatMultiplier`）+ `countsForNode`，按域（node.id 含 vision）
分桶累加；期望侧调用测试文件的共享构建器。输出四行：

```
counts  text=…  vision=…
expect  text=…  vision=…
域比:   text x.xxxx  vision x.xxxx
```

域比 ≈1 的域先排除，集中攻偏离域；域内再按叶子拆
（`matrix/T` = 该叶的参数当量，可与 derived 公式逐项对照）。

## 已排除假设登记（必写）

每轮校准要记录"排除了什么、怎么排除的"——防止下轮重猜：

- K3 乘子 4× 假设：探针 `childRepeatMultiplier(decoder,1)→layer0` = 1，
  且 qkv 354M 与 derived `hidden·4·proj` 吻合 → 排除。
- K3 state_update 公式假设：F7b 3·vh·vd·kd = 4.72e6/token 与探针
  6.04e8/128 精确吻合 → 排除。
- K2.5 vision 7× 假设：**成立**——vision_config 无 `in_channels` →
  channels 守卫一票否决整个推导（已修，channels 缺省 3）。

## 案例一（已闭合）：R1 逐项账本

超差 0.4786（W1 时期）→ 闭合到"每一个参数"：

```
+20,434M  shared expert（derived 误用 dense intermediate 18432，真值 2048）
− 319M    score matmul（counts 有、nEff 无——测试期望侧独立项）
+ 255M    kv_b 宽度（counts 少算，2026-09-08 修复清账）
+ 0.86M   norms 会计口径差
─────────
+20,371M  = nEff − counts（账本逐项闭合，无未解释残差）
```

教训：账本闭合后每一项都有名字；当时闭合不了 kv_b 项，登记后
M8-V1 修复，账本预测的量值与修复后实测一致。

## 案例二（进行中）：K3 双侧偏离——锚点已立（2026-09-08）

**外部锚点（官方模型卡，confidence 高）**：total **2.8T**、active **104B**。
源码已入库：[`models/moonshotai/Kimi-K3/modeling_kimi_linear.py`](../../models/moonshotai/Kimi-K3/modeling_kimi_linear.py)（文本解码器真身）。

**已澄清事实**（agent 取证，含行号）：
- 层型混合：**69 KDA + 24 MLA**（每 4 层 3 KDA + 1 MLA，末层 93 为 MLA；
  `kda_layers`/`full_attn_layers` 显式 1-based 清单，非正则）；
- 仅 layer0 dense MLP（intermediate 33792），其余 MoE；
- latent MoE：`routed_expert_down_proj(7168→3584)` → 896 experts
  （w1/w3: 3584→**3072**，w2: 3072→3584）→ RMSNorm(3584) →
  `routed_expert_up_proj(3584→7168)`；shared ×2（intermediate 6144）；
  router 896×7168 sigmoid top-16；
- KDA 层权重：q/k/v 各 7168→12288 + conv 4；decay 低秩
  f_a(7168→128)→f_b(128→12288)；full-rank gate g_proj 7168→12288
  （use_full_rank_gate=true）；b_proj 7168→96；z FusedRMSNormGated(128)
  逐头门控；o_proj 12288→7168；
- 字段语义：`num_expert_group=1` = 关闭分组路由；`attn_res_block_size=12`
  = AttnRes 存档周期（layer_idx%12==0）；`routed_expert_hidden_size=3584`
  = latent 宽，**不是** expert intermediate（那是 3072）。

**当前差距**（2026-09-08 KDA 模板修复后）：counts 106.5B/token vs 官方
active 104B（**残差 2.4%**）。两处 counts 侧错误已修复：① 独立全宽
decay 叶与融合内 f_a 重复计数；② output_gate_norm/out_proj 数值输入
误用融合宽 49376（源码 o_norm 逐头门控后 o_proj 输入 = 12288）。
剩余 2.4% 疑点：24 个 MLA 层计数（K3 MLA 需 q_lora_rank 等 config
值核对）、shared expert 6144 语义。GLM-5.3-Flash（1.0964）依赖案例二
追加的四发现落地（hc 超连接等）。

**index.json 已下载解析**（59.7MB，total_size 1.561T ✓ 与 2.8T 官方口径
在量化平均下吻合）。首轮发现：**全部 93 层都有 KDA 张量**
（b_proj/g_proj/f_b_proj 等 ×93），其中 **24 层另有 MLA 张量**
（q_a_proj/kv_b_proj）——"69 KDA + 24 MLA"的清单语义需要重审：
可能是"统一 KDA 注意力 + 24 层叠加 MLA 组件"的混合结构，而非二选一。
这是 counts 侧 26B 多计的头号嫌疑（我们的模板按二选一建模，
24 个 MLA 层的 KDA 部分可能被漏算或 MLA 层被双算——逐张量对账定案）。

**待办（M8-V2 收尾，方法已就绪）**：按 index 逐层做
`每层张量清单 × shape` 精确权重表 → 与 counts/T 按叶对照 →
差异张量定位到模板叶子 → 修模板或期望侧 → 账本闭合（目标：
counts 参数当量 ≈ 104B active + state/score 项）。

**注意**：此 HF 源码为推理专用实现（MoE forward 有 Training not
supported 断言），参数量公式可用，勿当训练图逐算子真值。

### 案例二追加：GLM-5.3-Flash 权重清单对账（2026-09-08，index.json 逐张量）

GLM 的 `model.safetensors.index.json`（8.4MB，76108 张量）已本地解析
（`/tmp/m8v2/glm-flash-index.json`），四发现：

1. **46 层 vs config 45 层**：index 中 input_layernorm/o_proj 各 ×46
   （层号 0..45），本地 glm-flash-config.json 写 45——需核实是 config
   陈旧还是存在额外层；
2. **hyper-connection 未建模**：每层有 `hc_attn_base/fn/scale`、
   `hc_ffn_base/fn/scale` ×45——GLM-5.3-Flash 用 HC（超连接），我们的
   normalize 只认 `hc_count`/`mhc` 字段，GLM 的 hc_* 张量既不在结构图
   里也不在参数推导里（预期侧缺口）；
3. **KDA 层真实权重构成**（34 层）：q/k/v **三个独立投影**（非 fused
   qkvz）、b_proj、**f_a/f_b 低秩 decay**、**g_a/g_b 低秩 gate**、
   **k/q/v 三个独立 conv1d**（非单 conv）、o_norm、dt_bias、A_log——
   derived 的 glm5Next 公式已按 g_a/g_b 重写，但 conv 结构（3 个独立
   conv）与 A_log 未覆盖；
4. **DSA 层 12 个**（非 11）：indexer 子模块含 kpool_compress/ape/
   weights_proj 等张量，我们的 qsa 模板需对照。

另：MoE 43 层（3 dense ✓ 与 first_k_dense_replace=3 吻合）、experts
带 weight_scale_inv（FP8 量化分组尺度）。

**结论**：GLM-5.3-Flash 的恒等式闭合依赖：① hyper-connection 组件
（结构+派生）；② KDA 层 conv/decay/gate 构成对齐（g_a/g_b 已修）；
③ 层号核实。全部登记，随 M11 并行层对齐或独立小波处理。

### 案例二追加二：GLM 分片头实锤（2026-09-08，62 分片 header 全取）

1. **46 层实锤**（层号 0-45 连续无空洞）——config 45 陈旧，模板层号以
   index 为准待修；
2. **hc 超连接实锤**：非每层张量，是**全局张量**（model.language_model.
   hc_*）：hc_attn_fn/hc_ffn_fn 各 [24, 16384]（1.77e7×2）、base 各 [24]
   （1080×2）、scale 各 [3]（135×2）——合计 ≈35.4M 参数；24 = 融合边界
   数（待源码确认），16384 = 2×hidden（双流？待源码确认）。结构图完全
   未建模，预期侧缺口 35.4M（占总参数 ~1%）；
3. **KDA conv 无差异**：q/k/v 三独立 conv [8192, 1, 4] ✓ 与我们
   3·qkv_dim·kernel 一致；
4. **DSA indexer 实锤**：wq_b [75.5M]、wk [6.29M]、weights_proj
   [1.57M] ≈ **83.4M/DSA 层** × 12 层 ≈ 1B 参数——预期侧
   dsaAttentionParameters 需核对是否含 indexer 权重（GLM 9.6% 缺口的
   重要嫌疑，1B/总参数方向与量级吻合）。

## M8-V2 案例三（进行中）：Kimi K3/K2.5 vision 塔逐层对账（源码已核实）

一手源码已入库（HF hub 单模型仓库惯例，与 config.json 同仓）：
[`models/moonshotai/Kimi-K3/`](../../models/moonshotai/Kimi-K3/)（modeling_kimi_k3.py、modeling_kimi_linear.py、configuration_kimi_k3.py、kimi-linear-analysis.md、k3-layer-tensors.json）、
[`models/moonshotai/Kimi-K2.5/modeling_kimi_k25.py`](../../models/moonshotai/Kimi-K2.5/modeling_kimi_k25.py)、
[`models/zai-org/GLM-5.3-Flash/modeling_glm5_next.py`](../../models/zai-org/GLM-5.3-Flash/modeling_glm5_next.py)（glm5_next 文本层待解析）。

**K3 vision 塔真值**（MoonViT3dEncoder，27 层，源码核实）：
- 每层：norm0/norm1（各 1024）、wqkv = Linear(1024 → **4608**)（qkv_hidden_size
  1536×3）、wo = Linear(**1536** → 1024)、MLP2 [1024, 4096, 1024]（fc0/fc1 无 gate）；
- 头：12 头 × qkv_head_dim 128（qkv_hidden_size/heads）；
- tower 尾：final RMSNorm(1024)；patch embed：3×14² → 1024；
- merger（patchmergerv2）：Linear(4096→4096) + GELU + Linear(4096→**7168**)
  （in = mm_hidden_size 1024 × merge_kernel 2×2）+ post RMSNorm(7168)；
- 合计 ≈ 443M（与 counts 反推 3.9-4.75e8 吻合）。

**模板 gap（已定位待修，M8-V2 收尾）**：
1. `qkv_proj` 模板 1024→3072，源码 1024→**4608**——normalize 缺
   `visionQkvHiddenSize`（qkv_hidden_size 字段），vision.js 用 3×hidden 近似；
2. `out_proj` 模板输入出现 `12 × 85.33` 坏维度（1024/12 的推导残渣）——
   应为 qkv_hidden_size 1536；
3. 修复顺序：normalize 加 `visionQkvHiddenSize` → vision.js 用它算 qkv/out
   宽度 → derivedVisionParameters 加 Kimi 分支（上表公式）→ 域账本重跑。

**K2.5 塔真值**（patchmerger V1、qkv 未指定 → 3×hidden）：wqkv 1152→3456、
wo 1152→1152、MLP2 [1152, 4304, 1152]、merger Linear(4608→4608)+
Linear(4608→7168)（带 bias）+ pre_norm。合计 ≈ 465M。

## 与其他文档的关系

- 恒等式公式与容差：`extractor.identity.test.js` 头注释
- 残差登记：`cost_counts.md`（本文档的案例由它链接）
- oracle 命令与纪律：`MAINTENANCE.md`
