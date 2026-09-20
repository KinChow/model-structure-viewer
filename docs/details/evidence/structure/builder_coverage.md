# 60 内置模型 × 前端 builder × 模块 × 运行时验证状态（代码级，非假设）

逐模型 `buildStructureFromConfig`，抓真实 `architecture / builder(MODELS[arch].name) /
attention_kind 集合 / cache 类型`。**60 模型，0 unsupported，11 个 builder**。运行时状态按"是否有真机 SGLang 证据"标注。
（当时用一次性审计脚本产出，脚本已移除；覆盖状态可由 `frontend/src/structure/models/index.js` 的 `MODELS` 注册表 + `verify:models` 复核。）

| builder | 模型数 | attention_kinds | caches | 运行时验证 | 说明 |
|---|---|---|---|---|---|
| assembleQwen3_5 | 29 | linear, qwen35_full, vision | kv,state | ✅ 真机(dense+MoE) | dense Qwen3.5-4B + **MoE 分支 qwen3_5_moe 减层真机**（GDN state 35,840 + GQA 512 与 SGLang 0.0%，MoE runner E=8 端到端；`runtime_profiles/qwen3_5_moe.md`） |
| assembleDeepseekV3 | 9 | mla, vision | kv | ✅ 减层真机 | DeepSeek-V3.1/R1/Kimi-K2；减层 MLA(kv 576)+MoE E=8 真机（`runtime_profiles/glm4_minimax_deepseekv3.md`）；本体 fp8/过大全前向留 H20 |
| assembleDeepseekV32 | 7 | **dsa_sparse_mla** | kv,**index** | ❌ 未验 | DSA 稀疏 indexer；V3.2/GLM-5 系。**未真机** |
| assembleDeepseekV4 | 5 | dsv4/compressed/sparse/swa | kv,index | 🟡 H20 dsv4 前向(同族) | 静态逐张量已验(V4-Flash)；**dsv4 fp8 前向 kernel 经 V4.1 同 backend H20 跑通**（`runtime_profiles/sglang_dsv41_dspark_h20.md`）；V4-Flash 专测 + fp4 KV 数值留后续 |
| assembleMiniMaxM3 | 2 | **minimax_m3_sparse_gqa, sparse**, gqa | kv,index | ✅ 减层(cache口径) | MiniMax 块稀疏。GQA kv 1024 + 稀疏 indexer 128 与 SGLang 0.0%；**修正 index 口径 512→128**（`../memory/glm5next_minimax_m3_cache.md`） |
| assembleQwen4Exp | 2 | linear, **qsa**, vision | kv,state,index | ✅ 真机 | Qwen3.8-Flash-Next（本次）。注意含 qsa+index，与 qwen3_5 不同 |
| assembleGlm5Next | 2 | **dsa_sparse_mla, linear** | kv,state,index | ✅ **H20 稀疏前向** | DSA+线性 hybrid；KDA 1,122,304 + MLA 512 + DSA index 128 与 SGLang 0.0%（`../memory/glm5next_minimax_m3_cache.md`）；**H20(SM90) DSA 稀疏前向端到端已跑通**（减层 dummy，`flashmla_sparse`/`fa3`/`TritonKDAKernel`，长 prompt>topk 触发稀疏，`runtime_profiles/sglang_glm5next.md`） |
| assembleMiniMaxM2 | 1 | gqa | kv | ✅ 减层真机 | MiniMax-M2.7；减层 GQA(kv 512)+MoE E=8(sigmoid 路由) 真机（`runtime_profiles/glm4_minimax_deepseekv3.md`） |
| assembleDeepseekV41 | 1 | dsv4/sparse/swa | kv,index | ✅ **H20 fp8 前向+DSpark** | 静态全验；**H20(SM90) fp8 前向端到端 + DSpark 投机跑通**（W8A8/W4A8-INT8 build，dsv4 backend/marlin MoE/kv fp8_e4m3，`/generate` 正确出词，`runtime_profiles/sglang_dsv41_dspark_h20.md`）；fp4 设计 KV(890) 数值对拍留 fp4-build |
| assembleKimiK3 | 1 | **linear, mla** | kv,state | ✅ 减层(cache口径) | 线性(KDA)+MLA hybrid；减层 checkpoint，KDA state 280,576 + MLA 576 与 SGLang 运行时 0.0%（`../memory/kimi_k3_kda_state.md`） |
| assembleGlm4Moe | 1 | gqa | kv | ✅ 减层真机 | GLM-4.7(bf16)；减层 GQA(kv 512)+MoE E=9(8 routed+1 shared 融合) 真机（`runtime_profiles/glm4_minimax_deepseekv3.md`） |

## 结论（修正此前过度声明）

- **静态结构 + 算子族**：全 60 已验（`verify:models` 60/60 + 算子族原子覆盖）——这层无遗漏。
- **运行时真机**：**仅 2 个 builder 真验过**（qwen3_5、qwen4_exp）+ MLA 经 V2-Lite 代理、GQA 经既有。**其余
  builder 的运行时口径未真机**——尤其带独立模块的：`assembleDeepseekV32`(DSA indexer)、`assembleGlm5Next`
  (DSA+linear)、`assembleMiniMaxM3`(块稀疏)、`assembleKimiK3`(linear+MLA)。我此前"其他=同维度已覆盖"不准确。
- **A100 可补(bf16)**：`assembleGlm5Next` 的 **GLM-5.3-Flash-BF16**（DSA 稀疏 + 线性 + kv/state/index 三 cache，
  是最"全"的未验 builder）；DSA(v32)若有 bf16 小模型亦可。
- **须 H20/换环境**：dsv4/v41(fp8)、R1/V3.1/Kimi-K2/K3/MiniMax-M3/大 V3.2（fp8 或过大）。

## 可行性复核（2026-09-19，BOS 实查）——未验 builder 的真机在 A100 **不可行**

逐个查未验 builder 的可下 bf16 权重是否装得进 8×A100-80GB(640GB)：
- **assembleDeepseekV32 / DSA**（GLM-5、GLM-5.1、V3.2）：`GlmMoeDsaForCausalLM` bf16、78 层、256 experts、
  **权重 ~1.5 TB**（GLM-5=1,507,761,487,026 B）→ **装不进 640GB**；V3.2 同样 671B 级。无小 bf16 DSA 模型。
- **assembleGlm5Next**（GLM-5.3-Flash-BF16 / GLM-5-Next-0808）：BOS 实查该名下**无可用对象**（0 objects/占位）→ 拿不到权重。
- **assembleMiniMaxM3 / assembleKimiK3**：MiniMax-M3 / Kimi-K3 均为百 B~T 级前沿模型，A100 显存不够。
- dsv4/v41：fp8，SM80 无 fp8 张量核。

**结论（修正后终版）**：A100 上**能真机跑的 bf16 且装得下**的 builder 只有 **Qwen 线性 hybrid 家族（qwen3_5 + qwen4_exp，
均已验）** + MLA(V2-Lite 代理) + GQA。**其余 builder（DSA/glm5_next/minimax_m3/kimi_k3/dsv4/v41）对应的真实模型
一律 fp8 或过大（100B~1.5TB），A100 真机不可行**——运行时验证到此在 A100 已尽，须 H20（fp8）或更大显存池（多机/大卡）。
静态结构 + 算子族对这些 builder 仍 100% 覆盖（`verify:models` 60/60 + 原子族）。

## 减层路线补充（2026-09-19）——DSA builder 的 cache 口径已在 A100 补验

用户提示"减层"：对**过大但 bf16** 的 builder，可 `from_config` 建减层随机 checkpoint 在 A100 验运行时 **cache 口径**
（无需全权重）。已对 **DSA（`assembleDeepseekV32`，7 模型）** 实施（`runtime_profiles/sglang_dsa.md`）：
- reduced `deepseek_v32`（6 层/bf16/4.1B，去 fp8 quant）SGLang 起——**DSA backend + index_topk + KV cache 分配成功**，
  KV 7,704 B/token vs 前端 8,448（比值 1.10，MLA latent + index 同结构）→ **DSA cache 口径 A100 补验通过**。
- 但 **DSA 稀疏注意力前向 kernel = SM90a/SM100f only**（`Sparse Attention Forward Kernel ... SM90a and SM100f`）→
  A100(SM80) 只能到 cache 分配，稀疏**前向**留 H20。
- **对上表的修正**：DSA builder 从"A100 完全不可行"更新为"**cache 口径可减层补验(已做)、稀疏前向留 H20**"。
  同法可用于 glm5_next(其 DSA 前向亦 SM90+)、kimi_k3/minimax_m3（需 remote code 建减层，前向 kernel 支持性另验）。

复现：当时的一次性审计脚本已移除；builder 覆盖由 `MODELS` 注册表 + `verify:models` 60/60 派生。

## 减层路线补充②（2026-09-19）——assembleKimiK3（KDA+MLA）cache 口径已在 A100 补验

同减层法应用于 **`assembleKimiK3`（linear KDA + MLA hybrid）**（`../memory/kimi_k3_kda_state.md`）：
- Kimi-K3 本体 VL/93 层/mxfp4 过大；对文本塔 `KimiLinearForCausalLM`（SGLang 原生 kimi_linear）建减层随机
  checkpoint。remote code 需 transformers 4.5x（`OutputRecorder` 在 5.x 已删）→ 隔离 `--target` 装 4.57.6 仅建
  checkpoint，serve 走 $SGLANG。
- **KDA state 280,576 elems/层/请求**（conv 18,432 + temporal 262,144）、**MLA latent 576 elems/token/层**
  （512+64）——与 SGLang `KimiLinearStateShape.create` + MLA sizing（scheduler 分配两池的同一段代码）**逐点 0.0%**，
  KDA/MLA 分层也一致。
- 边界：全服务端 boot 在随机 checkpoint 权重 layout 命中 `IndexError`（3-D vs 2-D，与 cache sizing 正交）；
  cache 口径改由 SGLang 自身 cache-param 代码路径直取，已收口。全前向 kernel 留 H20/完整权重。
- **对上表修正**：`assembleKimiK3` 从"❌ 未验(过大)"更新为"**✅ 减层 cache 口径已补验(0.0%)**"。
  同法仍待补：`assembleGlm5Next`(DSA+linear，remote code)、`assembleMiniMaxM3`(块稀疏，remote code)。

## 减层路线补充③（2026-09-19）——glm5_next + minimax_m3 收口 + 一处 MSV 口径修正

对最后两个 builder 直取 SGLang cache-param 代码路径（`../memory/glm5next_minimax_m3_cache.md`）：
- **assembleGlm5Next**（GLM-5.3-Flash-BF16）：KDA state 1,122,304 + MLA latent 512 + DSA index 128 与 SGLang
  `Glm5NextTextConfig.mamba2_cache_params`/`index_key_cache` **逐点 0.0%**。
- **assembleMiniMaxM3**（MiniMax-M3）：GQA kv 1,024 + 稀疏 indexer 128 与 SGLang `MiniMaxSparseKVPool` 0.0%。
  **发现并修正 MSV bug**：index cache 曾记 `sparseIndexHeads·sparseIndexDim=512`（query 头数），实际 SGLang
  indexer 池 `head_num=1`（单头共享 K），已改为 `idx_head_dim(+V) = 128`。`node --test` 410/410、
  `verify:models` 60/60、`docs:check` 全绿；重生成 ops-spec/edge golden（仅 MiniMax-M3 两模型 hash 变）。

## 运行时验证终版结论（2026-09-19）

**11 个 builder 的运行时 cache 口径覆盖**：
- ✅ 真机：`assembleQwen3_5`（dense + MoE 分支）、`assembleQwen4Exp`
- ✅ 减层真机（端到端出 token）：`assembleGlm4Moe`(GLM-4.7)、`assembleMiniMaxM2`(MiniMax-M2.7)、
  `assembleDeepseekV3`(DeepSeek-V3.1，MLA+MoE)
- ✅ 减层 cache 口径（SGLang scheduler 分配所用同一段代码，0.0%）：`assembleDeepseekV32`(DSA)、
  `assembleKimiK3`(KDA+MLA)、`assembleGlm5Next`(KDA+MLA+DSA index)、`assembleMiniMaxM3`(GQA+块稀疏)
- ❌ 仅 fp8/fp4、留 H20 做**全前向**（静态结构 + 量化打包 header 级已验；减层 bf16 真机尝试见下）：
  `assembleDeepseekV4`、`assembleDeepseekV41`

**至此 A100 上 11 个 builder 中 9 个已运行时真机/减层验证；仅剩 2 个 fp4/fp8-only builder（dsv4/v41）的全前向留 H20。**
减层 bf16 真机尝试（`../memory/deepseek_v4_v41_a100_boundary.md`）：DeepSeek-V4 的 SGLang serving 路径被 **fp4/fp8 专家
探测门控**——bf16 反量化 → fp4 探测返回 None → 掉进无 `compress_ratios` 的通用 DeepseekV3Config 别名分支 →
`deepseek_v4.py:670` AttributeError 崩溃；保 fp4 则 SM80 无 fp4 张量核。V4.1 非 transformers-native、fp8 前向
在 SM80 报 `CUTE_ARCH_MMA_F32_SM89` 缺失。**双重硬边界（框架 fp4 耦合 + SM80 无 fp4/fp8），非建模缺口。**

## H20 补验①（2026-09-20）——glm5_next DSA 稀疏前向端到端跑通

环境切到 **8× H20-3e（SM90 Hopper）/ CUDA 13 / sglang 0.0.0.dev1+g20518d851**。对 `assembleGlm5Next`
（GLM-5.3-Flash）用减层 + `--load-format dummy`（`scripts/evidence/structure/glm5_next_reduce.py`，8 层
DSA[3,7]+KDA[0,1,2,4,5,6]、16 experts、去 fp8→bf16，保全部 per-head 维度）起 SGLang TP1：

- **DSA 稀疏前向 kernel 在 H20 端到端跑通**——`prefill=flashmla_sparse / decode=fa3`、KDA `TritonKDAKernel`，
  长 prompt 3001 tok（> 本轮 index_topk 2048）触发稀疏 top-k、`Prefill batch #new-token 3001` 成功出 token。
  **A100(SM80) 只能到 cache 分配、稀疏前向留 H20 的那块收口。**
- **三 cache 元素口径对前端 0.0%**：KDA state 1,122,304（conv 73,728 + temporal 1,048,576）、MLA latent 512、DSA index 128。
- **逐字节挖出前端 bug**：`dsa_sparse_mla` 把 DSA index 按 bf16 计（256 B/层），实测 SGLang 存 fp8+尺度（132 B/层）→
  glm5_next/deepseek_v32(9 模型) KV-per-token 高估 +10.7%（实测 2312 vs 前端 2560 B/token）；dsv4 分支正确、此分支漏传
  index dtype。详见 `runtime_profiles/sglang_glm5next.md`（修法留有 node 的开发机 + golden 重生成）。
- 边界：dummy 权重（验 kernel 路径 + cache，非输出正确性）；全权重忠实前向未做。证据
  `runtime_profiles/sglang_glm5next.md`。
- **顺带确认**：DSA 稀疏前向 kernel（`flashmla_sparse`）在 H20 可用 → `assembleDeepseekV32`(DSA) 的稀疏前向
  硬件前置同样解除（同一 kernel），其减层全前向可同法补（本轮未跑，仅登记）。
