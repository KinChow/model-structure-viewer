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
# formulas/modules.js 同类豁免（W5）：命中全在 source.ref / notes 的出处标注里，
# 是证据链而非分派逻辑。
# W5（2026-09-09）：plan.js 的豁免已**摘掉** —— 它的 modelTypeProbe.includes 全部
# 换成 config 字段判据，四个无字段判据的配方位迁到 structure/archs/index.js 的
# ARCH_RECIPES。豁免随之转移给 archs/index.js（§8.1 认可的「一处数据文件」：
# 模型 → 配方声明表，key 是 architectures[0] 原字符串，不做子串匹配）。
# 基线沿革：W0（2026-09-07）测得 16 → W5（2026-09-09）删 resolveArchitecture 的
# 家族名子串兜底得 15 → plan.js 的 modelTypeProbe 全换成 config 字段判据、四个
# 无字段判据的配方位迁进 archs/ARCH_RECIPES 后得 14（plan.js 只剩 attention kind
# 的**标签** "qwen35_full"，是值不是分派）。
FAMILY_PATTERN='kimi|qwen4_?exp|qwen3_?5|glm5_?next|glm4_?moe|minimax_?m2|minimax_?m3|deepseek_?v32|deepseek_?v4|glm_?moe_?dsa'
FAMILY_BASELINE=14

FAMILY_COUNT=0
FAMILY_FILES=""
for f in $(grep -rliE "$FAMILY_PATTERN" frontend/src --include='*.js' --include='*.jsx' \
    | grep -v '\.test\.' | grep -v '__tests__' | grep -v 'structure/formulas/index.js' | grep -v 'structure/formulas/modules.js' | grep -v 'structure/archs/index.js' | sort); do
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

# ---------- §3.1d 公式来源标注（M11-P2-4，principles §3.5 / MAINTENANCE 3c） ----------
# 三级体系（一等 aten 锚点 / 二等 modeling 对照 / 三等分解声明）逐条落到
# formulas/index.js 的 `// ref:` 注释。计数随注册表条目数走：ref 行数 ≥ 条目数。
FORMULAS_FILE=frontend/src/structure/formulas/index.js
REF_COUNT=$(grep -c "ref:" "$FORMULAS_FILE" || true)
ENTRY_COUNT=$(node --input-type=module -e "
import { FORMULAS } from './$FORMULAS_FILE';
console.log(Object.keys(FORMULAS).length);
")
if [ "$REF_COUNT" -lt "$ENTRY_COUNT" ]; then
  echo "✗ §3.1d 违反：来源标注不足——ref: 注释 ${REF_COUNT} 行 < 注册表 ${ENTRY_COUNT} 条（principles §3.5：无来源注释不予合入）。"
  FAIL=1
else
  echo "§3.1d 来源标注: ${REF_COUNT} / ${ENTRY_COUNT} 条（≥ 条目数）"
fi

# ---------- §3.5b 文档不得新增 /tmp 取证引用（棘轮，只许下降） ----------
# operators_reference.md 里还有一批历史 `/tmp/m11-formulas/*` 引用：那是早期会话的
# 临时取证文件，多数已不在盘（文中已标注），结论都已内联到各节。探针本身已收回仓库
#（gen-operators-reference.mjs 的 --json）。这条棘轮只做一件事：**不许再新增**。
# 新证据的去处 = models/<org>/<id>/ 证据库（fetch-evidence.mjs + manifest），
# 或者「跑生成器就能复现」。
TMP_CITE_BASELINE=15
TMP_CITE_COUNT=$(grep -c "/tmp/m11-formulas" docs/details/operators_reference.md || true)
if [ "$TMP_CITE_COUNT" -gt "$TMP_CITE_BASELINE" ]; then
  echo "✗ §3.5b 违反：/tmp 取证引用 ${TMP_CITE_COUNT} 处 > 基线 ${TMP_CITE_BASELINE}——新证据请落 models/<org>/<id>/ 证据库或改成可复现命令。"
  FAIL=1
else
  echo "§3.5b /tmp 取证引用: ${TMP_CITE_COUNT} / 基线 ${TMP_CITE_BASELINE}（只许下降）"
fi

# ---------- P0：legacy root 复活棘轮（步骤 7 收口，基线 0） ----------
# P8（2026-09-10）删除 root 与 graph-to-tree projection 后，`.root` 活引用必须
# 保持为零：selectors.graphViewNode/graphRoot 是**按需树视图出口**（root 从协议
# 层降级为视图层），任何人把它的输出写回 structure 就会造出第二个 root
# （P8 报告未决取舍 5：靠约定维持的清零会在第一个人偷懒时失效，故机械化）。
# 豁免：`\.root\b` 天然不匹配 root_id（下划线无词边界，契约字段安全）；注释行
# 不计（与 §8.1 同口径：注释不构成事实）。
ROOT_BASELINE=0
ROOT_HITS=$(grep -rnE '\.root\b' frontend/src src --include='*.js' --include='*.jsx' --include='*.py' 2>/dev/null \
  | grep -vE ':[0-9]+:[[:space:]]*(//|#|\*|/\*)' \
  | grep -vE 'args\.root|model_root|--root' || true)
ROOT_COUNT=$(printf '%s' "$ROOT_HITS" | grep -c . || true)
if [ "$ROOT_COUNT" -gt "$ROOT_BASELINE" ]; then
  echo "✗ P0 违反：legacy root 活引用 ${ROOT_COUNT} 处 > 基线 ${ROOT_BASELINE}——root 已退役（执行路线步骤 7），树视图只能经 selectors.graphViewNode 按需构造，禁止写回 structure。"
  printf '%s\n' "$ROOT_HITS" | sed 's/^/  /'
  FAIL=1
else
  echo "P0 legacy root 活引用: ${ROOT_COUNT} / 基线 ${ROOT_BASELINE}（棘轮，保持 0）"
fi

exit $FAIL
