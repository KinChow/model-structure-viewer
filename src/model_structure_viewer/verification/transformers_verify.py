from __future__ import annotations

from pathlib import Path
from typing import Any

from ..schemas import ModelStructure, VerifyEvidence, VerifyEvidenceDiff, VerifyResponse
from ..structure.recovery import MetaRecoveryError, MetaRecoveryOutcome, build_meta_model_with_recovery
from .compare_structure import diff_module_evidence, load_reconciliation_rules
from .summary import minimal_summary


def verify_transformers_structure(
    config: dict[str, Any],
    *,
    source: dict[str, Any],
    local_dir: Path | str | None = None,
    msv_graph: Any = None,
) -> VerifyResponse:
    """Validate that Transformers can build the model on meta tensors.

    This is intentionally strict. It never returns config-derived structure for
    failed model construction, because that would hide unsupported Transformers
    behavior.
    """
    source_info = dict(source)
    try:
        outcome = build_meta_model_with_recovery(config, source=source_info, local_dir=local_dir)
        return _passed(outcome, source_info, msv_graph=msv_graph)
    except MetaRecoveryError as exc:
        return VerifyResponse(
            ok=False,
            status="failed",
            strategy="transformers-meta",
            source=source_info,
            model_id=source_info.get("model_id"),
            summary=minimal_summary(config),
            diagnostics=exc.diagnostics,
            error=str(exc),
        )
    except Exception as exc:  # noqa: BLE001 - verification reports errors instead of raising
        return VerifyResponse(
            ok=False,
            status="failed",
            strategy="transformers-meta",
            source=source_info,
            model_id=source_info.get("model_id"),
            summary=minimal_summary(config),
            diagnostics={
                "failure_kind": "unknown",
                "error_type": type(exc).__name__,
            },
            error=f"{type(exc).__name__}: {exc}",
        )


def _passed(outcome: MetaRecoveryOutcome, source: dict[str, Any], *, msv_graph: Any = None) -> VerifyResponse:
    summary = dict(outcome.structure.summary)
    diagnostics = dict(outcome.diagnostics or outcome.structure.source.get("diagnostics") or {})
    if outcome.recovery_kind == "none":
        diagnostics.setdefault("backbone_class", summary.get("backbone_class"))
    else:
        summary["strategy"] = _verify_summary_strategy(outcome.recovery_kind)
    return VerifyResponse(
        ok=True,
        status="passed",
        strategy="transformers-meta",
        model_id=source.get("model_id"),
        source=source,
        summary=summary,
        diagnostics=diagnostics,
        evidence=_build_evidence(outcome, msv_graph=msv_graph),
    )


def _build_evidence(outcome: MetaRecoveryOutcome, *, msv_graph: Any) -> VerifyEvidence:
    """构造通过后把 per-module evidence 带出并按需对账（Task 7.1/7.2）。

    旧版 _passed() 只取 summary、introspect 图整份丢弃；现在 evidence.modules
    是折叠图的逐模块搬运（体积可控：collapse_graph 已把同构子树折成
    repeat=N 的单节点）。msv_graph 缺省时 diff 三分类为空并注明未对账，
    structurally_consistent 置 None（未知）而非 False——未对账≠不一致。
    """
    modules = extract_evidence_modules(outcome.structure)
    if msv_graph is None:
        diff = VerifyEvidenceDiff(note="msv_graph not provided")
        structurally_consistent = None
    else:
        rules = load_reconciliation_rules()
        diff = VerifyEvidenceDiff(**diff_module_evidence(transformers_modules=modules, msv_graph=msv_graph, rules=rules))
        # P0-2：triage 后的口径——renaming/fold/known_divergences 已分类的项不算
        # 不一致；only_transformers/only_msv 输出的只剩 unclassified。
        structurally_consistent = not (diff.only_transformers or diff.only_msv or diff.mismatches)
    return VerifyEvidence(
        modules=modules,
        diff=diff,
        summary={
            # 本函数只在 meta 构造通过后调用；构造失败走 failed 分支，evidence=None。
            "constructed": True,
            "structurally_consistent": structurally_consistent,
            "module_count": len(modules),
        },
    )


def extract_evidence_modules(structure: ModelStructure) -> list[dict[str, Any]]:
    """从折叠后的 graph 收 per-module evidence（Task 7.1 字段清单）。

    path 取 canonical_id：折叠后 graph 节点 id 变位置路径（graph.py:118-161
    materialize 时按序号重编），canonical_id 保留 introspect 的原始 module
    path。字段取舍：I/O shape 不带——meta 构造不跑 forward，introspect 不产
    I/O 数值，带出来只能是 None 壳（不伪造数值）；tensor_names 不带——体积
    大头且对账不消费。repeat 例外于任务六字段清单：折叠节点代表 N 个实例，
    缺了它对账方会把 group 误读为单实例（可空 int，体积可忽略）。
    """
    graph = structure.graph
    if graph is None:
        return []
    modules: list[dict[str, Any]] = []
    for node in graph.nodes:
        modules.append(
            {
                "path": node.canonical_id or node.id,
                "class": node.attributes.get("class") if isinstance(node.attributes, dict) else None,
                "params": node.params,
                "weight_shapes": node.weight_shapes,
                "dtype": node.dtype,
                "value_source": node.value_source,
                "repeat": node.repeat,
            }
        )
    return modules


def _verify_summary_strategy(recovery_kind: str) -> str:
    if recovery_kind == "repair":
        return "repaired-transformers-meta"
    if recovery_kind in {"attention", "repair_attention"}:
        return "attention-normalized-transformers-meta"
    if recovery_kind in {"kimi", "repair_kimi"}:
        return "tie-weights-compatible-transformers-meta"
    if recovery_kind in {"kimi_remote_code", "repair_kimi_remote_code"}:
        return "kimi-remote-code-compatible-transformers-meta"
    return "transformers-meta"
