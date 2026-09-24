# Flash-Next Gated Residual：独立证据与待修复契约

状态：**GR/PLE/MTP 宽流路径及最终收束已实施，本批门禁通过。**
2026-09-24，本地 main 原基线 `3b42763fc347f1be255f87c9433dde32d884cbff`。
本页不覆盖/替换上一批 QSA 验收。

### 实施增量（2026-09-24）

- GR 读/写拆分：四路宽流输入/输出、逐分支 norm、elementwise 读门、标量写门及
  原始宽流＋本块输出＋写门的三路依赖。读复合节点拥有计算，真实 norm/down/up/
  inject 子节点拥有权重；写回无权重，不再叠加普通 residual_add。
- PLE 产生宽流增量，再与原始宽流相加；MTP 的 `fc_hidden` 逐分支复用同一
  `[2560,2560]` 权重，embedding 投影广播四路后相加，输入来自主干最终宽流。
  对照 pinned vLLM MTP forward（文件 SHA/URL 见 sources.json）。
- 官方参考 `Qwen4ExpTextModel.forward` 的末端是只读 GR mixer 直接接语言输出。
  两个发布权重 index 的全量 tensor-name map 都不存在独立的
  `model.language_model.norm.weight`，但包含 mixer 的三个真实权重；因此只对该
  GR 家族移除错误的独立 final norm，其他家族保留。索引证明的是权重路径缺失，
  联合前向才构成拓扑证据；不是仅凭 config 缺字段推断。
- Graph IR 保持 v2；`hc_phase`、delayed-combine 的说明保留。读门、写门和
  宽流写回在数学图中显式展示，执行实现可重排 combine，但不允许改变来源。
- 按报告 §2.2 Eq(30–34) 修正分组 RMSNorm、SiLU/read、branch mean、write
  gate/广播动作。融合边界流量仅为理论逻辑读写，不宣称具体 HBM 或 GPU 实测。

既有 shard82 safetensors header（两变体各四条 GR 张量）冻结在
`frontend/src/structure/models/__fixtures__/qwen-flash-next-gr-header.json`；
测试同时检验 raw tensors 和 skeleton truth 精确绑定。未重新下载完整权重。

### 当前验收与差异（工作负载 B=1 S=2051）

- 本批早期红测 0/5；最终定向 GR/QSA/形状 **20/20**。在移除假 final norm
  后重跑前端完整单测 **539/539**、后端 **184/184**、模型基础
  **60/60**，docs check、构建及原则检查均通过。完整日志见本地
  `artifacts/architecture-repair/C5-gr/*-final2.log` 及 `*-final3.log`。
- 当前 final-norm 版本相对 `3b42763`：两发布变体在 config/artifacts 路径
  均为图 1065/956 → 1538/1595（节点/边），模板参数
  179996864384 → 179996861824（恰少一份 2560 元素的伪 norm），
  prefill/decode MAC 保持 14435321724928/284520931328；前填 vector 增
  1558387314、SFU 增 2169958，解码 vector 增 1752614、SFU 增 1058。
- 原版 config-only 模板容量减 5120 B，FP8 config-only 推导容量也减
  5120 B；生产 artifacts 以 header 总量计费，原版保持 359999963128 B，
  FP8 保持 185502232570 B。该生产总量恒定**不代表**模板中伪 norm 正确。
  `artifacts/architecture-repair/C5-gr/reconcile-final.json` 保存逐节点两相位差异；
  它是理论动作/逻辑流量对账，不是 benchmark。
- final norm 移除后，原版/FP8 各桌面/移动端真实浏览器 **4/4**。
  展开后的 GR 读/写、PLE、MTP 及最终 mixer→lm_head SVG 连线、节点选择、
  公式、收起再展开、搜索、中英切换、成本面板、SVG 导出及 pageerror 均已检查。
  `artifacts/architecture-repair/C5-gr/browser-final2/` 有截图和导出；
  `browser-final2.log` 为运行记录。

golden 已在审阅实际节点/边差异后仅更新这两条变体；其它 58 条 hash 不变。
**这只是 Flash-Next GR 批次验收**，不是 60 个条目的架构审计完成。

## 证据与限制

精确 URL、读取时间、SHA256 见同目录 `qwen_flash_next_gr_sources.json`。
本轮重新下载图库和原版、FP8 两份 pinned 模型卡；报告使用此前下载的 pinned
PDF 并重新校验 SHA256。新的 PDF 下载超时，截断文件标记为 `.partial`，
不能作为正文证据。Transformers raw 请求也超时；以下前向定位是已入库参考
实现，不冒称本轮重新下载验证。未下载权重，未引入推理框架依赖。

独立架构依据为官方报告 §2.2 Eq(30–34)，不是 builder 或配置导出：

- 每条残差流独立 RMSNorm；4 条流分别保留 2560 通道，总宽 10240。
- Eq(31)：down 投影后除以 4、SiLU、up 投影、sigmoid，产生逐分支逐通道读门。
- Eq(32)：读门乘归一化后的流，沿 4 分支平均，输出 2560 通道。
- Eq(33)：从归一化宽流计算独立的每分支标量写门 `2*sigmoid(Ww*x/4)`。
- Eq(34)：块输出乘对应写门，写回原始（不是归一化后的）分支残差。
- 无残差流混合矩阵、无 Sinkhorn；GR 不能套用 mHC。
- 末端收束只读，无块输出写回，不能创建写门参数/计算。

已入库 `Qwen4ExpTextGatedResidual` 1001–1030 行与上述方程一致；
`Qwen4ExpTextDecoderLayer.forward` 1280–1310 行显式执行 attention/MLP 两次写回。
`TextModel.forward` 1480–1500 行将 embedding 重复为四路，末端 mixer 收为单路。
`Qwen4ExpTextRMSNorm` 150–174 行确认每分支 rsqrt 和 `(1+weight)`。
模型卡另确认 4 分支、rank=320，但不能替代上述机制证据。

## 已复现的问题

1. config-only 和 artifacts，两种变体的 decoder 输入/输出及 GR 输入均为
   2560，应保留 10240 的残差状态。现有普通 residual_add 不表达标量写门广播。
2. GR 内只有一个算子叶，没有可追踪的读门、写门、原始宽流和门控写入依赖。
3. 成本 extractor 的 grouped norm 将 hidden=4H、tokens=T 当作一组；
   应分为 T×4 个宽 H 的组，权重仍只有 4H，不能顺便复制容量。
4. 现有公式在最终 read-only mixer 仍计算 combine add；常规 GR 则遗漏写门
   sigmoid、缩放、广播乘及读分支平均等动作。与此同时层外仍计普通 residual
   add：不能只追加算子而保留原有重复/遗漏的计费。
5. `state_handoff` 和 `mlp_combine_mix` 表达某执行实现的 delayed combine。
   这种重排不能替代数学图中的原始宽流、对应写门和当前块输出三路依赖。

## 红测

文件：`frontend/src/structure/models/qwenFlashNextGatedResidual.test.js`。
执行：

```sh
# repository root
node --test frontend/src/structure/models/qwenFlashNextGatedResidual.test.js
```

产物 `artifacts/architecture-repair/C5-gr/red-mechanism.log`：**0 passed / 5 failed**。
四条模型路径分别先在 2560 vs 10240 的状态宽度断言失败。小尺寸 final-read
用例 T=2,H=3,nr=4,r=2 的期望 SFU=64，实际58，准确暴露少计6次分组 rsqrt。
其 MAC=96、权重字节=120 已通过；不能因这两项一致就宣称公式正确。
门控连线断言目前在宽度失败之后，尚未执行到，修宽后必须继续通过。

## 约束与遗留

Graph IR v2 与原始 canonical 路径不变；独立复合父子计费和权重容量所有者已用
小尺寸机制测试。PLE/MTP 的输入为多路，形状检查验证准确投影和广播宽度，
不全局豁免末维不等。`complete_blocks` 等 QSA 语义不在本批更改。

仍未证明所有量化 packed 逻辑形状与全模型参数完全一致；生产 header 总量
不能代替逐模块的完整 checkpoint truth。真实 GPU 性能不在本批验收范围。
