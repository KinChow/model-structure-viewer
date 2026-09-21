# Framework accounting 本地验收

验收时间：2026-09-22 Asia/Shanghai（2026-09-21 UTC）。

## 范围与版本

- 仓库：`/Users/zhouzijian01/Desktop/workspace/code/kinchow/model-structure-viewer`。
- 分支：`main`；本轮起点 `d63720f`；受测代码 `9e757d2`。
- 本地 remote-tracking `origin/main` 为
  `5043e9815a259f883b40202e908873ed9bfa856b`；本轮没有 fetch，
  此 SHA 不代表重新核验过的远端最新状态。
- 本轮只创建本地提交，没有 push、PR、服务部署或远端 GPU 操作。
- 原有前端摸排记录保留，新增修复闭环记录，未覆写历史实测结论。

| 本地提交 | 内容 |
|---|---|
| `9909da3` | 发布时间快照与 catalog 完整性校验 |
| `143ac21` | framework profile、唯一 cache pool 账本与 Fit/PD 接线 |
| `a625758` | vLLM effective plan、fusion 优先级、state dtype 一致性 |
| `9e757d2` | 框架/容量/发布日期定向场景和全量 Chrome 扫描 |

最后的文档提交仅归档本文、日志、截图及公式边界，不改变受测代码。

## 验证结果

除 pytest / 原则校验外，以下命令均在 `frontend/` 执行。

| 命令 | 结果 | 原始记录 |
|---|---|---|
| `node --test` | **452 passed，0 failed，0 skipped** | [node-test.log](node-test.log) |
| `.venv/bin/pytest -q`（仓库根目录） | **183 passed** | [pytest.log](pytest.log) |
| `npm run verify:models` | **60/60** | [verify-models.log](verify-models.log) |
| `npm run docs:check` | 通过 | [docs-check.log](docs-check.log) |
| `npm run build` | 通过；存在 Vite 大于 500 kB 的 chunk 提示 | [build.log](build.log) |
| `npm run test:e2e -- --project=desktop-chrome` | **27 passed**，5.9 分钟 | [desktop-chrome.log](desktop-chrome.log) |
| `npm run test:e2e -- --project=mobile-chrome` | **22 passed，5 skipped**，2.8 分钟 | [mobile-chrome.log](mobile-chrome.log) |
| `bash scripts/check_principles.sh`（仓库根目录） | 通过；家族名文件数 6/6，legacy root 活引用 0 | [principles.log](principles.log) |

### 浏览器口径

- 实际浏览器为 **Google Chrome 153.0.8010.50**，Playwright `channel: chrome`，
  headless；不是仅检查代码，也不是使用缺省 Chromium 替代 Chrome。
- 桌面视口 1440×1000；移动端为同一 Chrome 的 Pixel 7 设备模拟，
  **不是物理 Android 手机测试**。
- 两个项目均完整执行新增的 7 个测试，包括各自的 **60/60 模型扫描**。
- 移动端的 5 个 skip 是原有桌面专属测试：逐模型展开、桌面对比画布、
  TP 输入交互、宽屏布局、Inspector 高视口裁剪。新增移动全量扫描没有跳过。
- 测试 fixture 检查未捕获页面异常和对 MSV 后端的意外依赖。
  built-in 场景主动阻断 HF/ModelScope 网络，以本地快照作确定性输入。
- 所有模型检查图可见、成本非 NaN/Infinity、Roofline bound 非 unknown、
  默认 plan 无 invalid、KV 归属总和一致及页面无横向溢出。
  这不等于所有模型的所有可选并行计划和所有硬件后端都已验收。
- 测试结束后 4173 端口没有遗留监听服务。

逐模型结果：
[桌面 60 模型](desktop-all-models.json) /
[移动 60 模型](mobile-all-models.json)。
其中 KV 四项顺序为 `main / draft / shared / total`，单位为 bytes；
全部是软件公式的计算结果，不是 GPU 实测值。

### 定向场景

1. DeepSeek-V4.1-Flash 在 Provider 列表显示 **2026-09-10**，并位于
   DeepSeek 最新排序首项。来源是 Hub 仓库 `createdAt` 快照，不冒充官方公告日期。
2. dense / MoE 的 vLLM TP4 合法；MoE EP 开启再关闭，不遗留注入的 `moeTp=1`。
3. Qwen3.5-4B 的大上下文包含草稿驻留内存；总量超过 80 GiB 时显示 no-fit，
   Max Context 小于请求上下文。该超长输入仅作容量公式边界测试，
   不声明模型支持此上下文长度。
4. SGLang shared-expert fusion 默认关闭；显式开启改变通信量，关闭恢复。
5. DSpark 三个 profile 的 main/draft/shared/total KV 关系一致；
   runtime profile 的独立草稿窗口有界，neutral 保留完整上下文上界。
6. DSA 的显式 dtype 不受默认 KV bytes/element fallback 控件覆盖。

截图：
[桌面日期](desktop-release-metadata.png) /
[移动日期](mobile-release-metadata.png) /
[桌面 no-fit](desktop-draft-no-fit.png) /
[移动 no-fit](mobile-draft-no-fit.png)。

## 公式结论与未完成的实测验收

完整公式和固定源码版本见
[framework accounting](../../docs/details/framework_accounting.md)。

重要修正：共享 full-to-SWA 映射不等于共享 KV 存储。所审计的上游 vLLM /
SGLang DSpark 分配独立草稿 SWA/ring，因此不能无条件设 `draftKv=0`；
只有明确的同一 `cache_pool_id` 才去重。当前 runtime 公式是 BF16 逻辑窗口上界，
不是对具体 backend 的分页、打包、reserve slots 的精确预测。

本轮**没有新增 GPU 实测**，不能宣称全部 profile 已与真机逐字节对齐。
页对齐/保留槽、speculative verification headroom、backend packing、
压缩器状态、CUDA/workspace/scratch 和逐张量草稿权重归属仍是 evidence gap。
这些差异必须按精确 runtime 版本及配置逐项核对，不能用经验系数消除。
历史 H20 记录只是已有证据；未将其实测显存、吞吐或延迟写入 UI/公式。
