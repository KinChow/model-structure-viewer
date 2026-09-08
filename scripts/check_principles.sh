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
# 度量口径（W1 修订）：剥除纯注释行（注释不可能构成分派）；豁免
# structure/formulas/index.js —— 它是 §8.1 认可的"一处数据文件"
# （operatorId → 公式 → counts），explanation 中的模型名是条目文档而非分派。
# plan.js 同理豁免：家族探测自 normalizeConfig 逐字搬迁（W3-C），待配方表接管后随 archs/ 下调。
# 基线（2026-09-07，W0 测得）= 16。W0.5 / W3 / W4.5 / W5（compute.js 与
# extractor.js legacy 镜像删除）完成后应下调此数。
FAMILY_PATTERN='kimi|qwen4_?exp|qwen3_?5|glm5_?next|glm4_?moe|minimax_?m2|minimax_?m3|deepseek_?v32|deepseek_?v4|glm_?moe_?dsa'
FAMILY_BASELINE=16

FAMILY_COUNT=0
FAMILY_FILES=""
for f in $(grep -rliE "$FAMILY_PATTERN" frontend/src --include='*.js' --include='*.jsx' \
    | grep -v '\.test\.' | grep -v '__tests__' | grep -v 'structure/formulas/index.js' | grep -v 'model_executor/plan.js' | sort); do
  n=$(grep -iE "$FAMILY_PATTERN" "$f" | grep -cvE '^[[:space:]]*(//|\*|/\*)')
  if [ "$n" -gt 0 ]; then
    FAMILY_COUNT=$((FAMILY_COUNT + 1))
    FAMILY_FILES="$FAMILY_FILES$f
"
  fi
done

echo "§8.1 家族名硬编码文件数: ${FAMILY_COUNT} / 基线 ${FAMILY_BASELINE}"
[ -n "$FAMILY_FILES" ] && printf '%s' "$FAMILY_FILES" | sed 's/^/  - /'

if [ "$FAMILY_COUNT" -gt "$FAMILY_BASELINE" ]; then
  echo "✗ §8.1 违反：超过基线。新增家族只允许改一处数据文件（docs/principles.md §8.1）。"
  FAIL=1
fi

# ---------- §3.2 cost/ 禁止显示名参与数值计算 ----------
# 分派只允许基于 type / attributes.operator_id / 结构化 attributes。
# W5-1（2026-09-08）起全禁：显示名一律不参与成本计算。
COST_HITS=$(grep -rnE 'node\??\.name' frontend/src/cost --include='*.js' \
  | grep -v '\.test\.' || true)

if [ -n "$COST_HITS" ]; then
  echo "✗ §3.2 违反：cost/ 下新增显示名参与计算："
  printf '%s\n' "$COST_HITS" | sed 's/^/  /'
  FAIL=1
else
  echo "§3.2 cost/ 显示名检查: 通过（全禁，无允许清单）"
fi

# ---------- §3.1 公式注册表完整性：每条目终止于 counts（无白名单） ----------
COUNTS_CHECK=$(node --input-type=module -e "
import { FORMULAS } from './frontend/src/structure/formulas/index.js';
const missing = Object.entries(FORMULAS).filter(([, v]) => typeof v.counts !== 'function').map(([k]) => k);
if (missing.length) { console.log('✗ §3.1 违反：以下条目未接线 counts：' + missing.join(', ')); process.exit(1); }
console.log('§3.1 公式注册表: ' + Object.keys(FORMULAS).length + ' 条全部终止于 counts');
") || FAIL=1
[ -n "$COUNTS_CHECK" ] && echo "$COUNTS_CHECK"

# ---------- §3.1b 运行时接线判据（M11-P0-7，registry completeness） ----------
# M11 四路审计：护栏原判据只查"注册表挂了 counts 函数"，而 extractor 有
# 手搓 switch（31 条提前 return）——注册表上 31 个 counts 引用是被护栏
# 认证过的死代码。新判据：每条目必须可从 extractor 分派到达——手搓
# case、ctxBuilder 键、或显式豁免清单（豁免必须在此登记，禁止沉默）。
REACH_CHECK=$(node --input-type=module -e "
import fs from 'node:fs';
import { FORMULAS } from './frontend/src/structure/formulas/index.js';
const src = fs.readFileSync('./frontend/src/structure/formulas/extractor.js', 'utf8');
const switchCases = new Set([...src.matchAll(/case \"([a-z0-9_]+)\":/g)].map((m) => m[1]));
const cbStart = src.indexOf('ctxBuilders');
const cbBlock = cbStart === -1 ? '' : src.slice(cbStart, src.indexOf('\n}', cbStart));
const ctxKeys = new Set([...cbBlock.matchAll(/([a-z0-9_]+):\s*\(/g)].map((m) => m[1]));
const RUNTIME_EXCEPTIONS = new Set([]);
const unreachable = Object.keys(FORMULAS).filter((k) => !switchCases.has(k) && !ctxKeys.has(k) && !RUNTIME_EXCEPTIONS.has(k));
if (unreachable.length) {
  console.log('✗ §3.1b 违反：以下条目运行时不可达（无手搓 case、无 ctxBuilder、无豁免登记）：' + unreachable.join(', '));
  process.exit(1);
}
const viaSwitch = Object.keys(FORMULAS).filter((k) => switchCases.has(k)).length;
const viaCtx = Object.keys(FORMULAS).filter((k) => !switchCases.has(k) && ctxKeys.has(k)).length;
console.log('§3.1b 运行时接线: ' + Object.keys(FORMULAS).length + ' 条可达（手搓 ' + viaSwitch + ' / ctxBuilder ' + viaCtx + ' / 豁免 ' + RUNTIME_EXCEPTIONS.size + '）');
") || FAIL=1
[ -n "$REACH_CHECK" ] && echo "$REACH_CHECK"

exit $FAIL
