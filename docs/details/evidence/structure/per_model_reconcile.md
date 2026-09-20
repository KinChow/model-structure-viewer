# 逐模型结构对账（60 内置模型 vs transformers 真值）

- 复现脚本：`scripts/evidence/structure/reconcile_reduced.py`（原始 60 份 `*.diff.json` 可由此重生，不入库）。
- 真值来源：各模型 HF config 经 `AutoConfig`/`AutoModel` 构造 → 逐张量 name/shape 与 MSV 前端 spec 对账。
- 在机环境（2026-09-17）：A100-SXM4-80GB、CUDA 13.0、torch 2.13.0+cu130、transformers 5.12.1、node 20.18.1；`verify:models` 60/60。

## 结论

| 结果 | 数量 | 说明 |
|---|---|---|
| 结构一致（零残差） | 59 / 60 | `structurally_consistent=true`、`residual_count=0`、`mismatches=[]` |
| transformers 不支持 | 1 / 60 | DeepSeek-V4.1-Flash：transformers 5.12.1 `AutoConfig` 不识别 `deepseek_v41`（非结构错误，见下） |

59 个 transformers-native 模型逐张量对账**零残差**：`only_transformers` / `only_msv` / `mismatches` 全空。

## 差异分类口径（每份 diff 的 `classified`）

- `renaming`：命名差（同一张量不同名），对账时归一。
- `nonparam_drop`：transformers 侧非参数 buffer（如 rotary inv_freq），MSV 不建模。
- `fold_frontend_suffixes`：MSV 前端把 fused 权重按后缀折叠（如 qkv → q/k/v）。
- `known_divergences`：已登记的口径差异（不计入残差）。

以上四类均为**已解释差异**，不构成结构残差。

## DeepSeek-V4.1-Flash 的 1 例说明（非结构错误）

`error: AutoConfig.from_pretrained ... model type deepseek_v41 ... Transformers does not recognize`。即该模型非 transformers-native，无法经 AutoConfig 走本对账通路。其结构正确性改由参考栈证据覆盖：
- 结构/张量：`structure/deepseek_v41_tensor_identity.md`、`structure/deepseek_v41_module_tree.md`
- KV 字节：`memory/deepseek_v41_csa2_kv_bytes.md`
- 硬件边界：`memory/deepseek_v4_v41_a100_boundary.md`
