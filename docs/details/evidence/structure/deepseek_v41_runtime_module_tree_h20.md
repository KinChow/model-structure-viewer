# DeepSeek-V4.1-Flash 运行时模块树对账（R1，2026-09-21 H20 现跑 meta 构造）

> 复现：H20 `10.98.95.16` 容器 `dsv41_zzj_deploy`，参考栈 `/ssd4/models/inference/model.py`。
> `torch.set_default_device("meta")` 全量构造（不占显存、不触发 GEMM），`Transformer(ModelArgs(**config.json), tokenizer)`
> 遍历 `named_modules()` 得每模块 `{path, class, weight_shapes, params}`。脚本
> `scripts/evidence/structure/deepseek_v41_runtime_module_tree.py`。

## 背景：为何 R1 需要非 transformers 的后端真值

`msv verify` 走 transformers-meta，但 `deepseek_v41` 在 transformers 里不被识别
（`AutoConfig.from_pretrained failed: ... does not recognize this architecture`）。这正是"第 60 模型"
静态对账无法产出后端 evidence 的原因。本项用参考栈 meta 构造补这一后端真值。

## 取到的运行时模块树（meta，全量 config）

`model_args`: `n_layers=40, n_mtp_layers=3, engram_layer_ids=[1,14], dspark_target_layer_ids=[37,38,39], vision_n_layers=32`；
共 64,114 个 module（canonical 去重后 82 个唯一模块种类）。关键结构均在位：

- 主干注意力 `attn`（`wq_a/wq_b/q_norm/wkv/kv_norm/wo_a/wo_b`），压缩 `attn.compressor`（`norm/wkv/wgate`）+ `attn.indexer`（`wq_b/wk/k_norm/weights_proj`）。
- MoE `ffn.gate/experts.{w1,w2,w3}/shared_experts.{w1,w2,w3}`。
- engram 层 1/14：`Engram`（`q_weight/k_weight`）+ `ParallelEngramEmbedding`。
- DSpark 草稿 `mtp.0/1/2`：`DSparkAttention`、`DSparkMarkovHead`（`embed/head`）、`DSparkConfidenceHead`（`proj [1,5376]`）、`mtp.0.main_proj [5120,15360]`。
- 权重 fp8 打包：每 `Linear` 带 `weight` + `scale`（如 `attn.wq_a.weight [1280,5120] + scale [40,160]`）。

## 三桶 diff（compare_structure.diff_module_evidence，对 MSV deepseek_v41 图）

### 初次（无 R1 规则）：残留 only_transformers=48 / only_msv=5 / mismatches=2

残留全部是"命名词汇差 + 融合/粒度差 + vision + fp8 scale"，非结构错误。参考栈是 DeepSeek 官方 model.py 命名与
未融合粒度（attn/ffn/wq_a/wkv、per-expert experts.w1/w2/w3、compressor.{norm,wkv,wgate}），而 MSV 前端图是
HF/vLLM 词汇 + 融合/算子分解表示（self_attn.fused_wqa_wkv、mlp.expert_mlp、mhc_* 前后归一算子）。

### 收口（叠加 R1 局部规则）：unclassified = 0/0/0（structurally_consistent）

脚本 `deepseek_v41_r1_reconcile.py` 在共享契约之外局部叠加 R1 专用 renaming + known_divergences
（不改 canonical_path_contract.json、不影响其余 59 模型），并把后端 numel 口径对齐 MSV（只计逻辑 weight、剔除 fp8 scale/bias）：

`classified={renaming:61, nonparam_drop:240, fold_frontend_suffixes:3, known_divergences:97}`，
`only_transformers=0, only_msv=0, mismatches=0`。每条 R1 规则都对应真实差、带 reason：

- 1:1 改名（renaming）：attn↔self_attn、ffn↔mlp、wq_b↔q_proj、gate↔router、vision↔visual、aligner↔projector、embed↔embed_tokens、head↔lm_head、shared_experts w1/w2/w3↔gate/down/up_proj。
- 融合差：MSV 把 wq_a+wkv 融成 fused_wqa_wkv、per-expert experts.* 融成 expert_mlp、SWA 计算为 dsv4_swa_attention 算子。
- 粒度差：MSV 把 compressor/indexer 建成单算子（参考栈有 compressor.{norm,wkv,wgate}、indexer.{wk,k_norm} 子模块）；前后归一为 mhc_* 算子（参考栈 attn_norm/ffn_norm）；ViT 为粗粒度算子（参考栈 per-block 树）；engram_hash 非参数缓冲。
- class 词汇差：Gate↔router logits、Expert↔MLP、Aligner↔Projector（同模块、类名不同源）。
- fp8 打包口径：参考栈 numel 含 scale/bias，MSV 只计逻辑权重——对齐到 weight 后逐模块 params 一致（如 q_proj 41,943,040、wo_b 41,943,040、indexer.q_proj 5,242,880、engram.embed 98,305,579,008）。

复现：`node ../scripts/verify-builtin-models.mjs --dump-graphs /tmp/msv_graphs` 后
`.venv/bin/python scripts/evidence/structure/deepseek_v41_r1_reconcile.py --graph /tmp/msv_graphs/deepseek-ai__DeepSeek-V4.1-Flash.graph.json`（退出码 0）。
后端真值固化在 `_fixtures/deepseek_v41_runtime_modules_reduced.json`（md5 7796a521...）。

## 结论

- R1 收口：MSV deepseek_v41 组网与参考栈运行时模块树三桶零 unclassified——凡 1:1 模块 class/shape/params（逻辑权重口径）全一致，
  其余差异都是 MSV 有意的融合/算子分解/命名/多模态/量化打包表示，逐条带 reason 归入 known_divergences（与其余 59 模型对账同款方法论）。
- 结构覆盖坐实：主干 CSA2 压缩/index、MoE+shared、engram[1,14]、DSpark mtp[37/38/39]（DSparkAttention/DSparkMarkovHead/DSparkConfidenceHead）、vision 均在位。
- 口径边界：这是结构/形状对账（meta 构造、无前向数值）；fp4/engram/DSpark 的运行时数值（accept/显存/吞吐）见 R2，fp4 精确逐字节需 Blackwell（见 R1 fp4 段）。
