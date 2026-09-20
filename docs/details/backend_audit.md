# 后端生产化硬化审计（backend，2026-09-17，只读审计，未部署）

> 后端定位（README §项目边界）：本地开发 / 可信内网工具，默认 `127.0.0.1`。以下 5 项是"暴露给
> 不可信用户前必须先落实"的硬化项。本次为**只读架构审计**，未改代码；触发（真正对外部署）未到。

## 五项现状（证据 file:line）

| # | 项 | 现状 | 优先级 | 关键证据 |
|---|---|---|---|---|
| 1 | 路径约束 | **缺失** | P0（任意文件读取） | `resolve/local_cache.py:230` `resolve_config_path` 仅 `Path(config_path).expanduser()`，不校验是否在 `model_root` 内；`api.py:92` `/api/local/config` 直收 `config_path` 无校验 |
| 2 | remote code 沙箱 | **缺失** | P0（任意代码执行） | `structure/introspect.py:51` 硬编码 `AutoModel.from_config(..., trust_remote_code=True)`；`resolve/resolver.py:215` 的 `auto_fetch_remote_code` 只控制**是否下载**、不控制**是否执行** |
| 3 | 鉴权 | **缺失** | P0（无访问控制） | `api.py:37` 仅 CORS 中间件，无任何 auth 依赖；所有 `/api/*` 开放；`api.py:74` `POST /api/settings` 任意客户端可改写进程级全局配置 |
| 4 | 限流/大小/超时/CORS | **部分** | P1（DoS/内存耗尽） | 已有：CORS 白名单（`api.py:39`，仅 5173）、worker 超时（`service.py:426`，默认 90s）。缺：HTTP 层超时（`cli.py:205` uvicorn 无超时）、请求体大小上限、rate-limit |
| 5 | 日志脱敏 | **部分** | P2（信息泄露） | 已有：第三方 stdout/stderr 隔离（`service.py:396` `_suppress_third_party_output`）。缺：异常 detail 回带完整路径（`errors`→`api.py:48` `{"detail": str(exc)}`；`local_cache.py:236` `Config file not found: <path>`）、`/api/health` 回带 `model_root` |

## 结论

- 与 README 声明一致：后端是本地/可信内网工具，5 项中 **3 项缺失（P0）+ 2 项部分**，属**设计如此**，非回归。
- **不改默认行为**：结构对账 / V4.1 实证 的 verify 路径依赖 `trust_remote_code=True` + 任意 `config` 路径；直接强制这些硬化会破坏本地开发与既有验证流。故硬化应做成**按 production/untrusted 开关的 opt-in（默认关，本地行为不变）**。

## 暴露公网前的最小加固（建议，opt-in、默认关）

- **P0-1 路径约束**：`resolved.resolve().relative_to(model_root.resolve())` 越界即 `ConfigError`；仅在 untrusted 模式强制。
- **P0-2 remote code 开关**：新增 `settings.trust_remote_code`（本地默认 True 兼容；untrusted 默认 False，命中即拒构造而非静默降级）。
- **P0-3 鉴权**：untrusted 模式下 `Depends` 校验 API Key/Token；并把 `POST /api/settings` 在该模式下改为只读或下线。
- **P1**：uvicorn/HTTP 超时、请求体大小上限、rate-limit 中间件。
- **P2**：异常 detail 脱敏（不回带绝对路径/堆栈），`/api/health` 不回带 `model_root`。

## 状态

只读审计完成；**硬化实现待"真正对外部署"触发**（触发未到不动工）。若需现在落 opt-in 默认关的 P0 守卫（不改本地行为），另行确认后实施并补单测。
