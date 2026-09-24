# 模型结构最终审计快照

- 日期：2026-09-24
- 工作区：本地 `main`
- Graph IR：`version = 2`、`schema_version = 2`
- 覆盖范围：60 个内置条目、16 个结构家族
- 外部对照：Raschka LLM Architecture Gallery、官方模型卡、发布
  Transformers/vLLM/SGLang 实现、safetensors index/header

## 审计口径

Gallery 只用于家族、注意力类型和关键机制的交叉检查，不替代发布实现
或 checkpoint truth。最终判断分开记录：

1. **架构意图**：论文、技术报告、官方卡片和 Gallery；
2. **发布实装**：官方 forward、source-ref、config；
3. **权重归属**：safetensors header/index 或 skeleton truth；
4. **用户可见结构**：Graph IR、折叠视图、实际 Chrome 页面；
5. **成本边界**：静态公式、驻留归属和明确的 unknown。

因此“60/60 可生成图”不等于“60/60 已完成 GPU 或逐张量验证”。

## 当前已闭合的结构问题

### Kimi-K3

- AttnRes 已按 attention 前、MLP 前和末端输出聚合表达；
- NoPE 不再生成文本 MLA 执行 RoPE，视觉塔仍保留二维 RoPE；
- SiTU-GLU 已作为独立激活和计数公式，覆盖 dense、shared expert、
  routed expert；
- MLA output gate 区分真实 `g_proj` 与 `sigmoid × attention output`，
  并接入 `g_proj → output_gate ← attention → o_proj`；
- 门控输出宽度按 value width，不复用 query width。

### GLM-5.2 / GLM-5.3 IndexShare

- 21 个 compute 层、57 个 reuse 层；
- 4 个发布变体均有 19 个跨层 reuse 关系；
- reuse 层不实例化独立 indexer 权重；
- index source、index alias、主 MLA KV 路径分别建模；
- Gallery 审计脚本现已识别 `index_reuse` 目标，不再产生假阴性。

### DeepSeek-V4

- compressor 的 KV projection、gate、position bias、window reduction、
  norm、compression RoPE 已形成真实数据流；
- C4 indexer 与 compressor 的 index-control 关系已表达；
- ratio 0/4/128、视觉变体、DSpark/MTP 边界分别处理；
- 当前审计中不存在孤立 compressor operator。

### 多模态入口

当前公共入口保持两路汇合：

```text
image → vision tower → merger/projector ─┐
text  → token embedding ─────────────────┴→ multimodal fusion → language backbone
```

视觉输出不再作为 embedding lookup 的输入。不同家族仍保留自身的
placeholder、scatter、replace 或 image-span 语义，不统一伪装为 token
拼接。

### GLM-5.3-Flash MTP truth

- MTP tail layer 按发布 manifest 作为标准 decoder；
- 不虚构本地 `embed_tokens`、独立 `shared_head.head` 或主干 mHC；
- 288 个 routed experts 仍只显示一个 fused expert leaf；
- folded skeleton 可按 aggregate path 保留完整重复 tensor metadata；
- FP8/BF16 两种 manifest 的 expert tensors 分别为 1,728/864，均可聚合
  绑定到 fused leaf；
- 该结果证明 checkpoint truth 归属闭合，不证明 GPU fused kernel 的具体
  物化方式。

## 当前可复现观察

运行：

```bash
node scripts/evidence/structure/audit-gallery-semantics.mjs
```

当前观察应满足：

| 项目 | 结果 |
|---|---:|
| 内置条目 | 60 |
| 视觉条目 | 39 |
| GLM-5.2/5.2-FP8/5.3/5.3-BF16 IndexShare reuse | 每个 19 |
| 上述四个变体跨层 index-reuse 边 | 每个 19 |
| Kimi-K3 孤立 MLA gate | 0 |
| DeepSeek-V4 孤立 compressor | 0 |
| Kimi-K3 SiTU 语义 | `situ_glu:situ`、`fused_moe_mlp:situ` |

## 验证门禁

当前已通过：

- 前端单测：608/608；
- 后端 pytest：184/184；
- 内置模型验证：60/60；
- `docs:check`；
- production build；
- Chrome 定向结构验收：7 passed、1 skipped；
- Chrome 桌面内置模型扫描：60/60。

Chrome 扫描中 39 个视觉条目的 roofline `data-bound=unknown` 是已知的
融合搬运未知，不是页面空白、结构加载失败或错误地归零。

## 仍然必须保留的边界

以下项目不能由静态 Graph IR 或 Gallery 观察替代：

1. DeepSeek-V4 fused compressor 的 GPU kernel、量化 scale traffic 和
   跨请求 cache 追加 prefill；
2. 所有 FP8/GPTQ/MXFP8/Base 变体的 packed logical shape、scale ownership
   和 activation quantization 的逐模块核对；
3. 多模态 placeholder/scatter/image-span 的真实搬运量；
4. MTP/专家 fused kernel 的 GPU 物化与真实性能；
5. 未具备可靠 header/forward 证据的逐模块 truth。

这些项目继续标记为 `unknown` 或“未实测”，不为满足“所有费用有限”而
填零，也不把 config-only 探针当作生产权重证明。

## 本快照的复现边界

本文件是当前代码和审计脚本的状态快照。每次修改架构 builder、truth
binding、布局或成本公式后，必须重新运行前端、后端、60 模型和 Chrome
门禁，并重新审阅受影响模型的结构差异；不能只更新 golden。
