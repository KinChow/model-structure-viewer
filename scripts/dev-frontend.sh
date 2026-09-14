#!/usr/bin/env bash
# 从仓库根启动 Vite 前端。
# 用法: bash scripts/dev-frontend.sh [vite args...]
# 例:   bash scripts/dev-frontend.sh --host 127.0.0.1 --port 5173
# 打开 http://localhost:5173 ；/api 由 Vite 转发到 127.0.0.1:8000。
set -euo pipefail
cd "$(dirname "$0")/.."

if ! command -v npm >/dev/null 2>&1; then
  echo "error: 需要 npm。先安装 Node.js。" >&2
  exit 1
fi

if [ ! -d frontend/node_modules ]; then
  echo "error: frontend/node_modules 不存在。先运行: cd frontend && npm install" >&2
  exit 1
fi

cd frontend
exec npm run dev -- "$@"
