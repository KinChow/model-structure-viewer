# NV-5 · DeepSeek-V4.1-Flash config + safetensors index 逐层实证（2026-09-17）

> 策略：config + safetensors index/header 只读元数据，**不下权重区**。后端 transformers 无法构造
> `deepseek_v41`（`auto_map:null`、无 transformers config/model 类，仅 DeepSeek 自研 `inference/` 栈），
> 故用 **config 作真值 + HF index 交叉核实** 做结构对账。

## HF 仓库文件（`deepseek-ai/DeepSeek-V4.1-Flash`，88 文件）

- `config.json`：`model_type=deepseek_v41`、`architectures=[DeepseekV41ForCausalLM]`、**`auto_map: null`**。
- `inference/`：`model.py / engram.py / vision.py / kernel.py / convert.py / generate.py …`（自研推理栈，
  非 transformers `PreTrainedModel`/`configuration_*.py`）。
- `model.safetensors.index.json`（张量清单，无权重区）、`tokenizer*`、`encoding/`。
- 结论：**升级/最新 transformers 也构造不了**（最新可安装 5.17.0 注册表只到 `deepseek_v4`，无
  `deepseek_v41`；且无 remote code config 类可 `trust_remote_code`）。后端 meta 构造维持受阻边界。

## index 逐层实证（`model.safetensors.index.json` ↔ config，全部一致）

| 项 | index 真值 | config 声明 | 结论 |
|---|---|---|---|
| 张量数 | 96085 | header-truth `tensor_count=96085` | ✓ |
| total_size | 510,286,023,000 (~510GB) | header-truth `parameterTotal≈508.18B` | ✓（fp8 混合一致） |
| 文本层 | 40（layers.0..39） | `num_hidden_layers=40` | ✓ |
| MTP | 3（mtp.0/1/2） | `num_nextn_predict_layers=3` | ✓ |
| vision blocks | 32（vision.blocks.0..31） | `vision.num_hidden_layers=32` | ✓ |
| routed experts | 384/层 | `n_routed_experts=384` | ✓ |
| **compressor 层** | **[2,8,14,20]** | `kv_source_layer_ids=[2,8,14,20]` | ✓ 精确 |
| **indexer 层** | **[2,8,14,20,24,28,32,36]** | `index_source_layer_ids=[2,8,14,20,24,28,32,36]` | ✓ 精确 |

层内张量叶（例 layer 38）：`attn.{wq_a,wq_b,wkv,wo_a,wo_b,q_norm,kv_norm,attn_sink}` +
`ffn.{gate,shared_experts.w1/w2/w3}` + `hc_{attn,ffn}_{base,fn,scale}`，含 fp8 `.scale`——与 V4 MLA +
超连接口径一致。

## 结论：解决 NV-5「compressor/indexer 层位」并修前端保真差

- **真值确认**：跨层 KV/index 复用——compressor 权重只在 `kv_source_layer_ids`、indexer 只在
  `index_source_layer_ids`；其余 `compress_ratio>0` 层复用 source 层，不自带权重。
- **前端旧实现（错）**：按 `compress_ratio>1` 启发式把 compressor 摆到层 2–19，且 indexer 一个都不摆。
- **修复**：`normalize.js` 读入 `kvSourceLayerIds`/`indexSourceLayerIds`；`ops/index.js`
  `deepseekV4AttentionOperatorSpecs` 的 compressor/indexer 改为按 source 层摆放（缺省退回启发式，
  V4-Flash/Pro 行为不变）。**修复后前端 compressor=[2,8,14,20]、indexer=[2,8,14,20,24,28,32,36]，
  与 index/config 逐位一致**。
- **回归**：V4-Flash/Pro/Vision-Exp 对账仍 0 残留；前端 `node --test` 410 pass、`docs:check` 全一致、
  `verify:models` 60/60。

## 逐张量恒等式（range-read 48 分片 safetensors 头部 ↔ header-truth.json，全对）

对全部 48 个 `model-*-of-00048.safetensors` **只 range-read 头部 JSON**（不下权重数据区），逐张量聚合
dtype/shape，与 builtin `header-truth.json` 对账（产物
[`v41_header_tensor_identity.json`](v41_header_tensor_identity.json)）：

| 项 | real（真实头部聚合） | header-truth | 结论 |
|---|---|---|---|
| tensor_count | 96085 | 96085 | ✓ |
| parameterTotal | 508,182,659,298 | 508,182,659,298 | ✓ 逐元素零差 |
| mtp_tensor_count | 2401 | 2401 | ✓ |
| BF16 | 1,976,441,856 | 1,976,441,856 | ✓ |
| F32 | 42,307,282 | 42,307,282 | ✓ |
| F8_E4M3 | 204,015,223,296 | 204,015,223,296 | ✓ |
| F8_E8M0 | 23,563,015,184 | 23,563,015,184 | ✓ |
| I8 | 278,585,671,680 | 278,585,671,680 | ✓ |

结论：manual `header-truth.json`（前端 V4.1-Flash 参数真值源）与真实 checkpoint 头部**逐张量、逐 dtype
零差**，权重逐张量恒等式（header 层）闭合。

## 仍待真实权重/GPU 推理（NV-5 剩余，本次未做）

- engram（第 1/14 层 n-gram 门控写回）与 DSpark 投机头运行时（接受率/显存/吞吐，需真实推理）；
- 后端 transformers 构造需 `deepseek_v41` 框架支持或补齐 remote code（`configuration_deepseek_v41.py`+`auto_map`）。
