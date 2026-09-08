# 恒等式校准方法：域拆分账本

来源：T4（R1 收敛）与 M8-V2（vision 域拆分）两轮校准的实践沉淀。
恒等式本身见 `extractor.identity.test.js` 头注释；本文档写**方法**——
当 ratio 超差时如何定位，而不是猜。

## 四样东西（必须同时持有）

校准期间任何一个时刻，四样输入缺一样就会退化成猜谜：

1. **config 字段语义**——每个参与公式的字段"在 modeling 源码里是什么"。
   依据源阶梯（`principles.md` §5）：checkpoint 自带 modeling_*.py →
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
源码已入库：`details/models/kimi-k3/modeling_kimi_linear.py`（文本解码器真身）。

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

**当前差距**：counts 130B vs 官方 active 104B（counts 多 ~26B）；
expected 97.6B（少 ~6.4B）。归因方向：counts 侧优先（多计量大）——
重点排查 69 KDA 层的 KDA 叶子计数与 MLA 层 24 层的 MLA 计数在
93 层混合调度下的乘子；再用 index.json（59MB，每张量精确 shape，
agent 已定位）做第三重印证。expected 侧缺 ~6.4B 待对账。

**注意**：此 HF 源码为推理专用实现（MoE forward 有 Training not
supported 断言），参数量公式可用，勿当训练图逐算子真值。

## M8-V2 案例三（进行中）：Kimi K3/K2.5 vision 塔逐层对账（源码已核实）

一手源码已入库：`details/models/kimi-k3/{modeling_kimi_k3,configuration_kimi_k3}.py`、
`kimi-k25/`、`glm5-next/`（modeling_glm5_next.py，glm5_next 文本层待解析）。

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
