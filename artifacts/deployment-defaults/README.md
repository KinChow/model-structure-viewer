# 单机部署默认策略验收

## 范围

- 仓库：`/Users/zhouzijian01/Desktop/workspace/code/kinchow/model-structure-viewer`
- 分支：本地 `main`，功能起点 `f1bb907`；此目录随功能一起本地提交。
- 未 fetch、push、创建 PR；没有新增远端 GPU 操作或 serving 服务。
- 这是理论容量的初始配置策略，不是 runtime 部署认证，也不是吞吐优化器。

## 根因与实现

原来物理拓扑固定为单机 8 卡，但逻辑计划始终从 TP1 开始，两者没有联动。
`CostSummary`、架构图和 P/D 共用 `DetailWorkspace` 的部署状态，因此无需
修改 Graph IR 或模型 Builder。

1. `aggregateModelMemory` 从原 `aggregateCost` 提取内存入口，复用权重精度、
   framework cache accounting，不另写公式或经验修正系数。
2. `recommendSingleNodePlan` 只在 TP1/2/4/8 四个固定档位内，以权威逐卡投影
   选择最小理论适配档位；PP/EP/DP 均为 1。权重下界只用于提前排除不可能适配的档位。
3. 拓扑默认一台 8 卡机器，摘要额外显示策略实际使用卡数。小模型可以只用
   1/2/4 卡；TP8 仍不足就保留 no-fit，不悄悄扩到多机。
4. `useDeploymentDefaults` 统一派生自动配置，只存储手动覆盖。选新模型重置推荐；
   换硬件时自动配置更新，手动配置保留。后台 checkpoint 参数总量不参与模型 key。
5. 仅 focus/blur 而未编辑不能进入手动模式；“恢复默认部署”返回自动模式。
6. 基准固定 batch=1、2048 tokens、模型自身权重精度和选定 framework profile。
   编辑实际负载时不会偷偷扩卡，Cost/Fit 继续按照实际负载计算。
7. 用户主动选择 PD 时，P/D 默认各一台独立机器，手动计划独立保存。

成熟方案来源与未覆盖的 runtime 条件见
[`parallel_protocol.md`](../../docs/details/parallel_protocol.md)。

## 最终验证

前端命令在 `frontend/` 执行，其余在仓库根目录执行。

| 项目 | 结果 | 记录 |
|---|---|---|
| `npm test` / `node --test` | 460 passed，无 skip | `node-test.log` |
| `.venv/bin/python -m pytest -q` | 最终完整重跑 183 passed | `pytest-final.log` |
| `npm run verify:models` | 60/60 | `verify-models.log` |
| `npm run docs:check` | 通过 | `docs-check.log` |
| `npm run build` | 通过，保留已有大 chunk 警告 | `build.log` |
| `bash scripts/check_principles.sh` | 通过 | `principles.log` |
| i18n catalog 校验 | 8 passed | `i18n.log` |
| 桌面 Chrome 全量扫描 | 6 批全部通过，60 个不同模型 | `desktop-60.json` |
| 桌面 Chrome 新增定向场景 | 4 passed，无重试 | `desktop-targeted.log` |
| 移动 Chrome accounting 测试文件 | 16 passed，无重试；含 60/60 扫描 | `mobile-chrome-final.log`、`mobile-60.json` |

不是全部旧 `viewer.spec.js` 的最终完整重跑：本轮桌面验收覆盖新增定向场景和
所有模型成本/图扫描；移动端完整执行 `framework-accounting.spec.js`。

### 浏览器与覆盖口径

- Playwright 使用真实安装的 Google Chrome（`channel: chrome`），headless。
  收尾本机查询版本为 `153.0.8010.53`；移动项目是 Pixel 7 设备模拟，
  不冒充物理 Android 手机。
- 两端逐模型清单均有 60 个唯一模型 ID；图可见、成本无 NaN/Infinity、
  KV 四项相加一致、无无效默认计划、无横向溢出、未捕获页面异常为零。
- 默认 A100 80GB 下两端档位分布一致：TP1=22、TP2=2、TP4=5、TP8=31。
  **采用 TP8 不代表该模型已经适配**；超过单机容量的模型仍保留 no-fit。
- Qwen3.5-4B 默认 TP1；Qwen3.5-27B 在 A100 为 TP1、切 L40S 后为 TP2；
  Qwen3.5-122B-A10B 默认 TP4；DeepSeek-V4.1-Flash 默认 TP8。
- 新增定向用例覆盖无编辑 blur、切硬件来回、手动保留/恢复、
  P/D 默认节点和独立策略、通过模型抽屉切模型后恢复推荐。
- 原有 accounting 场景覆盖 vLLM TP4/EP、SGLang fusion、DSpark pool、
  DSA fallback、大上下文 draft KV 与 Fit。超长负载测试额外确认不会自动扩 TP。
- 截图：`desktop-defaults.png`、`mobile-defaults.png`。
- 测试完成后 4173 没有残留监听。

### 保留初次失败，不覆盖为“全程一次通过”

- 原先整段 60 模型单用例在第 49 个模型处触及 600 秒总预算。
  已拆成六个各 10 模型用例，仍检查完整 catalog，保持真实可见性断言。
- 等待 catalog 之前直接 `evaluateAll` 会读到空数组；现在先断言 60 个 option 已到达。
- `beforeEach` 原先受 30 秒限制，test body 内延长预算不会覆盖准备阶段。
  将场景 setup 预算设为 90 秒后，移动文件完整重跑 16/16。
  早期记录在 `mobile-chrome.log` 和 `mobile-concurrent-attempt.log`；
  后者在与重型 Node 测试争用资源时被主动中断，不作为验收证据。
- Python 首轮为 182 passed / 1 failed（inline BERT API 返回 500），
  单独复跑 1 passed，最后完整复跑 183 passed。没有为此改动后端或绕过断言。
  初次 HTTP 500 的内部原因未确证，不能直接声称是 worker 超时。

## 明确的边界

没有新 GPU 实测；既有 runtime evidence gap 不因本次默认策略而消失。
page reserve、backend packing、workspace、具体 kernel/TP shape 支持及真实吞吐
仍须针对精确模型/框架/硬件配置验证。本次没有补偿系数、模型下载或远端服务重启。
