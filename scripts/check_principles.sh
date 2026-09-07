#!/usr/bin/env bash
# 原则护栏 —— docs/principles.md 的机械检查（refactor_plan.md W0）。
# 用法: bash scripts/check_principles.sh
# 退出码非 0 即违反。基线只允许下降，不允许上涨。
set -u
cd "$(dirname "$0")/.."

FAIL=0

# ---------- §8.1 家族名硬编码的非测试文件数只允许下降 ----------
# 完整 pattern：含下划线/驼峰变体与独立 "kimi"。
# 注意：早年 review 用窄 pattern 得出 11，本护栏以完整 pattern 为准。
# 基线（2026-09-07，W0 测得）= 16。W0.5 / W3 / W4.5 完成后应下调此数。
FAMILY_PATTERN='kimi|qwen4_?exp|qwen3_?5|glm5_?next|glm4_?moe|minimax_?m2|minimax_?m3|deepseek_?v32|deepseek_?v4|glm_?moe_?dsa'
FAMILY_BASELINE=16

FAMILY_FILES=$(grep -rliE "$FAMILY_PATTERN" frontend/src --include='*.js' --include='*.jsx' \
  | grep -v '\.test\.' | grep -v '__tests__' | sort)
FAMILY_COUNT=$(printf '%s\n' "$FAMILY_FILES" | grep -c .)

echo "§8.1 家族名硬编码文件数: ${FAMILY_COUNT} / 基线 ${FAMILY_BASELINE}"
[ -n "$FAMILY_FILES" ] && printf '%s\n' "$FAMILY_FILES" | sed 's/^/  - /'

if [ "$FAMILY_COUNT" -gt "$FAMILY_BASELINE" ]; then
  echo "✗ §8.1 违反：超过基线。新增家族只允许改一处数据文件（docs/principles.md §8.1）。"
  FAIL=1
fi

# ---------- §3.2 cost/ 禁止显示名参与数值计算 ----------
# 分派只允许基于 type / attributes.operator_id / 结构化 attributes。
# 允许清单（全部计划在 W5 清空，清空后本节收紧为全禁）：
#   cost/compute.js   —— 现存显示名分派链（W5 改查表后删除本条）
#   cost/comm.js      —— node?.id || node?.name 路径兜底（W5 收紧为仅 id）
#   cost/parallel.js  —— 同上
COST_ALLOWED='cost/compute\.js|cost/comm\.js|cost/parallel\.js'
COST_HITS=$(grep -rnE 'node\??\.name' frontend/src/cost --include='*.js' \
  | grep -v '\.test\.' | grep -vE "$COST_ALLOWED" || true)

if [ -n "$COST_HITS" ]; then
  echo "✗ §3.2 违反：cost/ 下新增显示名参与计算："
  printf '%s\n' "$COST_HITS" | sed 's/^/  /'
  FAIL=1
else
  echo "§3.2 cost/ 显示名检查: 通过（允许清单 3 个文件，W5 清空）"
fi

exit $FAIL
