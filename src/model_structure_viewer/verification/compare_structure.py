"""结构对账：transformers per-module evidence ↔ 前端 msv Graph 三分类 diff。

P7/步骤 6（tasks.md Task 7.3）。旧版只有 summary 两键比较——
``compare_structure_summary`` 是 catalog 门禁的历史入口
（tests/test_verification_compare.py 在用，生产零调用），原样保留；
对账主体是 ``diff_module_evidence`` 三分类：

- ``only_transformers``：后端 meta 构造有、前端 Graph 无的模块路径；
- ``only_msv``：前端 Graph 有、后端无的模块路径；
- ``mismatch``：路径命中但 class / shape 不一致（见 ``_module_mismatches``）。

路径规范化的契约精神（§6.4）：前后端各持一套路径 resolver——前端
graphTruth.js:5 ``canonicalModulePath``，后端本文件
``canonical_reconciliation_path``。不共享代码、不写前端逻辑的 Python
复制品；两侧行为由契约样例 fixture 锁定：
tests/fixtures/canonical_path_contract.json（前后端共享样例、不共享代码）。
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def compare_structure_summary(
    *,
    predicted: dict[str, Any],
    reference: dict[str, Any],
) -> dict[str, Any]:
    """Compare the high-level summary fields that gate catalog verification."""
    predicted_summary = predicted.get("summary") or {}
    reference_summary = reference.get("summary") or {}
    errors: list[str] = []
    warnings: list[str] = []

    for key in ("canonical_architecture", "text_layers"):
        predicted_value = predicted_summary.get(key)
        reference_value = reference_summary.get(key)
        if predicted_value != reference_value:
            errors.append(f"{key} mismatch: predicted={predicted_value!r} reference={reference_value!r}")

    return {
        "status": "failed" if errors else "passed",
        "errors": errors,
        "warnings": warnings,
    }


_RULES_FIXTURE = Path(__file__).parent / "fixtures" / "canonical_path_contract.json"


def load_reconciliation_rules() -> dict[str, Any]:
    """读包内契约 fixture 的 reconciliation_rules（P0-2）。

    规则是**生产逻辑输入**（triage 决定 diff 分桶），所以随包分发而非留在
    tests/；路径对样部分仍是前后端共享样例、不共享代码（§6.4）。
    """
    try:
        return json.loads(_RULES_FIXTURE.read_text(encoding="utf-8")).get("reconciliation_rules") or {}
    except (OSError, ValueError):
        return {}


# wrapper 段集合：与前端 graphTruth.js:1 PATH_WRAPPERS 对样（fixture 锚定，
# 不共享代码）。剥的是**前导**包装段，与前端 canonicalModulePath 同语义。
_WRAPPERS = {"model", "language_model"}
# 首段别名：后端 ModuleList/容器命名 ↔ 前端模板段名。与前端
# canonicalModulePath 的首段改名集合同款，只登记 fixture 样例覆盖到的条目；
# 前端新增别名时先补 fixture 样例，再谈规则。
_FIRST_SEGMENT_ALIASES = {"layers": "decoder", "visual": "vision_tower", "vision": "vision_tower"}


def canonical_reconciliation_path(path: Any) -> str:
    """把两侧模块路径折到同一键空间：剥 root/包装段、首段别名、折叠实例段。

    折叠实例段 = 纯数字索引（``layers.0``，实例序号）与后端折叠伪段
    （``groupN``/``patternN``，fold.py:47,117 的产物）——对账比的是模块
    种类，实例序号与折叠伪段不参与键。整树节点（后端 ``root``、前端
    ``model``）规范化为空串，由调用方排除出 diff。
    """
    segments = [segment for segment in str(path or "").split(".") if segment]
    if segments and segments[0] == "root":
        segments = segments[1:]
    while segments and segments[0] in _WRAPPERS:
        segments = segments[1:]
    if segments:
        segments[0] = _FIRST_SEGMENT_ALIASES.get(segments[0], segments[0])
    return ".".join(segment for segment in segments if not _is_folded_segment(segment))


def _is_folded_segment(segment: str) -> bool:
    if segment.isdigit():
        return True
    for prefix in ("group", "pattern"):
        if segment.startswith(prefix) and segment[len(prefix) :].isdigit():
            return True
    return False


def diff_module_evidence(
    *,
    transformers_modules: list[dict[str, Any]],
    msv_graph: Any,
    rules: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """三分类 diff。输入后端 evidence 模块列表与前端 Graph。

    ``msv_graph`` 收两形态：前端 Graph 整体（``{"nodes": [...]}``）或裸节点
    列表（tasks.md 7.3 的节点列表契约）。键空间为
    ``canonical_reconciliation_path``；空键不入 diff——整树节点的差异由
    summary.backbone_class ↔ canonical_architecture 承载，不在这里制造
    命名噪音。同键重复（折叠组头 + 模式组头折到同键）保留首见。

    P0-2 triage（规则词汇取自成熟方案，fixture 驱动）：
    - ``renaming``：vLLM WeightsMapper 的段级改名（linear_attn↔self_attn）；
    - **nonparam_drop**：前端无 ``weightMatrices`` 声明的计算叶整体排除出
      only_msv——P5「声明即权重归属」的直接推论：无声明 = 无 checkpoint
      对应物（scores/softmax/rope/split 等纯计算叶），对应 vLLM WeightsMapper
      的 None-drop 语义；
    - ``fold_frontend_suffixes``：前端两级 norm 粒度（norm 容器 + rmsnorm 算子
      叶）上提到父键参与对账——HF ``MergeModulelist`` 的反向同款；
    - ``known_divergences``：ignore-list（逐条带出处的真粒度/构造差异：
      MTP 构造差异、tied lm_head、GDN 融合拆分粒度、vision 层级），显式列出
      ≠ 静默吞——输出按 ``classified`` 分桶，``unclassified`` 为空才算
      ``structurally_consistent``。
    """
    renaming = [(str(rule.get("backend", "")), str(rule.get("frontend", "")))
                for rule in (rules or {}).get("renaming", [])]

    def _apply_renaming(key: str) -> str:
        if not renaming:
            return key
        segments = [renaming_map.get(segment, segment) for segment in key.split(".")]
        return ".".join(segments)

    renaming_map = dict(renaming)
    fold_suffixes = set((rules or {}).get("fold_frontend_suffixes", []))
    divergences = [
        (str(rule.get("pattern", "")), str(rule.get("reason", "")),
         tuple(rule.get("apply_to", ("only_transformers", "only_msv", "mismatches"))))
        for rule in (rules or {}).get("known_divergences", [])
    ]

    backend_by_key: dict[str, dict[str, Any]] = {}
    renamed_hits = 0
    for module in transformers_modules or []:
        key = canonical_reconciliation_path(module.get("path"))
        if not key:
            continue
        renamed = _apply_renaming(key) != key
        key = _apply_renaming(key)
        if renamed:
            renamed_hits += 1
        backend_by_key.setdefault(key, module)
    frontend_by_key: dict[str, dict[str, Any]] = {}
    # 分类计数桶在前端循环前定义（循环内计 nonparam_drop/fold）。
    classified: dict[str, int] = {"renaming": renamed_hits, "nonparam_drop": 0, "fold_frontend_suffixes": 0, "known_divergences": 0}
    for node in _msv_nodes(msv_graph):
        path = _msv_field(node, "canonical_id") or _msv_field(node, "id")
        key = canonical_reconciliation_path(path)
        if not key:
            continue
        # nonparam_drop：**仅限算子叶**（type=operator）——无 weightMatrices 声明的
        # 计算叶 = 纯计算过程（P5 推论：无声明 = 无 checkpoint 对应物）。容器
        # 模块节点（type=module）本就无声明，必须参与对账（它们是模块树的骨干）。
        declaration = _msv_field(node, "weightMatrices")
        is_operator = str(node.get("type") or "").lower() == "operator"
        if is_operator and not (isinstance(declaration, list) and declaration):
            classified["nonparam_drop"] += 1
            continue
        # fold_frontend_suffixes：两级 norm 粒度上提——末段是 fold 后缀且父键
        # 存在后端对手方时，以父键参与对账（class/shape 比较对象是后端单级模块）。
        folded = False
        if fold_suffixes:
            parent_key = key.rsplit(".", 1)[0] if "." in key else ""
            if key.rsplit(".", 1)[-1] in fold_suffixes and parent_key:
                key = parent_key
                folded = True
        if folded and key in backend_by_key:
            classified["fold_frontend_suffixes"] += 1
        entry = dict(node)
        entry["_folded_frontend_leaf"] = folded
        frontend_by_key.setdefault(key, entry)

    unclassified: list[str] = []

    def _classify(side_keys: set[str], bucket: str, label: str) -> list[str]:
        remaining: list[str] = []
        for key in sorted(side_keys):
            if any(re.search(pattern, key) and bucket in apply_to
                   for pattern, _reason, apply_to in divergences):
                classified["known_divergences"] += 1
                continue
            unclassified.append(f"{label}:{key}")
            remaining.append(key)
        return remaining

    only_transformers = _classify(backend_by_key.keys() - frontend_by_key.keys(), "only_transformers", "backend")
    only_msv = _classify(frontend_by_key.keys() - backend_by_key.keys(), "only_msv", "frontend")

    mismatches: list[dict[str, Any]] = []
    unclassified_mismatches: list[dict[str, Any]] = []
    for key in sorted(backend_by_key.keys() & frontend_by_key.keys()):
        for mismatch in _module_mismatches(key, backend_by_key[key], frontend_by_key[key]):
            if any(re.search(pattern, key) and "mismatches" in apply_to
                   for pattern, _reason, apply_to in divergences):
                classified["known_divergences"] += 1
                continue
            unclassified_mismatches.append(mismatch)
    mismatches = unclassified_mismatches
    return {
        "only_transformers": only_transformers,
        "only_msv": only_msv,
        "mismatches": mismatches,
        "classified": classified,
    }


def _module_mismatches(key: str, backend: dict[str, Any], frontend: dict[str, Any]) -> list[dict[str, Any]]:
    """单模块的 class / shape 两类不一致判定。任一侧字段缺 None 即不参与。"""
    mismatches: list[dict[str, Any]] = []

    # class：后端是真实 torch 类名（Qwen3_5RMSNorm），前端是模板/算子标签
    # （RMSNorm、linear）。候选 = 前端 class + operator_id 任一命中即一致。
    backend_class = backend.get("class")
    frontend_class = _msv_field(frontend, "class")
    candidates = [
        value
        for value in (frontend_class, _msv_field(frontend, "operator_id"))
        if isinstance(value, str) and value
    ]
    if backend_class and candidates and not _is_generic_container(backend_class):
        if not any(_classes_consistent(backend_class, candidate) for candidate in candidates):
            mismatches.append(
                {"path": key, "kind": "class", "transformers": backend_class, "msv": frontend_class or candidates[0]}
            )

    # shape：只比 weight_shapes——meta 构造不跑 forward，introspect 不产 I/O
    # 数值（introspect.py:179-204 只填 params/weight_shapes/dtype），前端节点
    # 的 input/output_shape 在后端无对手方。任一侧 weight_shapes 为 None 即
    # 整段跳过，不伪造数值。
    backend_shapes = backend.get("weight_shapes")
    frontend_shapes = _msv_field(frontend, "weight_shapes")
    if (
        isinstance(backend_shapes, dict)
        and backend_shapes
        and isinstance(frontend_shapes, dict)
        and frontend_shapes
    ):
        common = [name for name in backend_shapes if name in frontend_shapes]
        if not common:
            # 同名模块权重名完全不相交：结构对不上，整组上报。
            mismatches.append(
                {"path": key, "kind": "shape", "transformers": backend_shapes, "msv": frontend_shapes}
            )
        for name in common:
            if not _dims_consistent(backend_shapes[name], frontend_shapes[name]):
                mismatches.append(
                    {
                        "path": key,
                        "kind": "shape",
                        "transformers": {name: backend_shapes[name]},
                        "msv": {name: frontend_shapes[name]},
                    }
                )
    return mismatches


_GENERIC_CONTAINERS = {"modulelist", "moduledict", "sequential"}


def _is_generic_container(backend_class: str) -> bool:
    """torch 通用容器类名不含结构语义（ModuleList 只是个列表），与前端模板
    容器命名（DecoderStack 等）不可比，class 检查跳过。"""
    return backend_class.lower() in _GENERIC_CONTAINERS


def _classes_consistent(backend_class: str, frontend_label: str) -> bool:
    """class 词汇两侧不同源：后缀容忍 + 大小写不敏感。前端通用标签通常是
    后端专有类名的尾部（RMSNorm ⊂ Qwen3_5RMSNorm）；两端都对不上才算
    mismatch——那是真实的命名/形态分歧信号，不静默吞掉。"""
    # P0-2：先剥下划线/空格再比后缀——torch 类名的下划线版本段
    #（Qwen3_5RMSNorm）与前端 CamelCase 标签（GemmaRMSNorm）经 _ 归零后
    # 才落在同一字母序列上（qwen35rmsnorm ⊃ gemmarmsnorm）。同款手法 =
    # transformers 对 architectures 的 ForCausalLM 后缀剥离。
    backend = re.sub(r"[^a-z0-9]", "", backend_class.lower())
    frontend = re.sub(r"[^a-z0-9]", "", frontend_label.lower())
    return backend == frontend or backend.endswith(frontend) or frontend.endswith(backend)


def _dims_consistent(backend_dims: Any, frontend_dims: Any) -> bool:
    """正维语义比较：逐维相等才一致；非正维（-1/-2 batch/seq 占位）不参与；
    秩不同即不一致。非列表形态视为不可比，不参与（不伪造数值）。"""
    if not isinstance(backend_dims, list) or not isinstance(frontend_dims, list):
        return True
    if len(backend_dims) != len(frontend_dims):
        return False
    for backend_dim, frontend_dim in zip(backend_dims, frontend_dims):
        if (
            isinstance(backend_dim, int)
            and isinstance(frontend_dim, int)
            and backend_dim > 0
            and frontend_dim > 0
            and backend_dim != frontend_dim
        ):
            return False
    return True


def _msv_nodes(msv_graph: Any) -> list[dict[str, Any]]:
    if isinstance(msv_graph, dict):
        nodes = msv_graph.get("nodes")
    elif isinstance(msv_graph, list):
        nodes = msv_graph
    else:
        nodes = None
    return [node for node in nodes or [] if isinstance(node, dict)]


def _msv_field(node: dict[str, Any], name: str) -> Any:
    """前端节点把 class/operator_id 放在 attributes 里（moduleSpec 第 4 参），
    顶层同名字段优先——兼容任务契约里的扁平节点形态。"""
    value = node.get(name)
    if value is not None:
        return value
    attributes = node.get("attributes")
    if isinstance(attributes, dict):
        return attributes.get(name)
    return None
