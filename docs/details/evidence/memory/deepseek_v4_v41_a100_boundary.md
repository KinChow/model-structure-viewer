# DeepSeek-V4 / V4.1 减层 bf16 真机尝试 —— A100 硬边界确认（fp4-gated serving path）

用户"把内置模型都跑完"：尝试对最后 2 个 fp8/fp4-only builder（`assembleDeepseekV4`、`assembleDeepseekV41`）
建减层 **bf16** checkpoint 在 A100 真机跑（目标至少到 cache 分配，如 V3.2 DSA）。**结论：A100 不可行，硬边界**。

## 尝试与失败链（deepseek_v4，transformers-native）

- 建减层 bf16 checkpoint 成功：`build_families_tiny.py deepseek_v4` → `DeepseekV4Config` L=6、79.5M、
  compress_ratios=[0,0,4,128,4,128]、CSA2 + index(head 128) + hash 层 + mHC(hc_mult=4) + SWA(128) + MoE(8+1)。
- SGLang serve **失败**，根因链（代码取证）：
  1. `configs/deepseek_v4.py:37-44` `uses_deepseek_v4_fp4_fused_moe`：探测 routed-expert safetensors dtype，
     命中 `U8/I8/F4`→fp4 路径、`F8_E4M3`→fp8 路径；**我的 bf16 专家 → 返回 None**（日志
     `Unexpected routed-expert safetensors dtype=BF16 for DeepSeek V4`）。
  2. fp4 探测 None → config 走通用 `_DeepseekV4ConfigAlias`（`utils/hf_transformers/common.py:193`，
     **DeepseekV3Config 子类，不含 `compress_ratios`**）而非 `DeepSeekV4Config` dataclass（含 compress_ratios）。
  3. `models/deepseek_v4.py:670` `config.compress_ratios[layer_id]` → **AttributeError: no attribute
     'compress_ratios'** → Scheduler 崩溃。
- **即 DeepSeek-V4 的 SGLang serving 路径被 fp4/fp8 专家探测门控**：去量化成 bf16 会掉进不支持 CSA2 的通用配置分支。
  而真机保 fp4 专家 → A100 **SM80 无 fp4 张量核**，前向仍跑不了（且 fp4 indexer/CSA2 稀疏前向须 SM90a/SM100f）。

## deepseek_v41

- `DeepseekV41ForCausalLM` / model_type `deepseek_v41`：**非 transformers-native**（`native=False`），
  内置 config 无 auto_map；本体参考栈在 `$MODELS/DeepSeek/DeepSeek-V4.1-Flash/inference/model.py`（remote code），
  此前实测其 fp8 前向在 A100 报 `SM89_16x8x32_F32E4M3E4M3F32_TN ... without CUTE_ARCH_MMA_F32_SM89_ENABLED`
  （SM80 无 fp8 MMA）。SGLang 无独立 `deepseek_v41.py` EntryClass。→ 同 fp4/fp8 硬边界。

## 结论（诚实）

- **DeepSeek-V4 / V4.1 无法在 A100 真机跑**，双重硬边界：(1) SGLang V4 serving 路径 fp4/fp8 专家探测门控——
  bf16 反量化掉进无 compress_ratios 的通用分支；(2) SM80 无 fp4/fp8 张量核（CSA2 fp4 indexer + 稀疏前向须
  SM90a/SM100f/Blackwell）。**非建模缺口，是硬件 + 框架量化耦合**。
- 这 2 个 builder 的**静态结构（../structure/per_model_reconcile.md 逐张量）、cache 口径（MSV）、量化打包（fp4/fp8 header 级，
  `../structure/quant_packing.md`）、KV 字节模型（V4.1=890 / V4-Flash=3,514，`deepseek_v41_csa2_kv_bytes.md`）**
  均已收口；仅**运行时全前向留 H20/Blackwell**。

复现：`build_families_tiny.py deepseek_v4`（建减层 bf16）→ SGLang serve 触发上述 fp4-gated 失败链。
