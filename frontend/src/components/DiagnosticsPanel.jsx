import { checkpointTruthModel, diagnosticsModel } from "../cost/ui.js";
import { formatQuantity } from "../formatters.js";

/**
 * 诊断面板（W6-1，§4.4）：真值绑定与成本覆盖的诚实性上界面。
 * 数据全部来自 structure.source（strategy + graphTruth 诊断），不做任何推断。
 * M11-P1-5：checkpoint 真值获取失败与 endpoint 静默切换（HF→ModelScope）上界面。
 */
export default function DiagnosticsPanel({ structure, language = "zh" }) {
  const english = language === "en";
  const diagnostics = structure?.source?.diagnostics || null;
  if (!diagnostics) return null;
  const model = diagnosticsModel(diagnostics, { english });
  const truth = checkpointTruthModel(structure?.source, { english });
  if (!model.banner && model.gapCount === 0 && model.ambiguousCount === 0 && model.unsupportedCount === 0 && model.warningCount === 0 && !truth.show) return null;
  return <section className="diagnostics-panel" aria-label={english ? "Diagnostics" : "诊断"}>
    {model.unsupportedCount > 0 && <div className="diagnostics-banner diagnostics-banner-error" data-unsupported="1">{english ? "Unsupported structure" : "不支持的结构"}：{model.unsupported.map((entry) => entry.message).join(" ")}</div>}
    {truth.show && <div className={`diagnostics-banner${truth.tone === "error" ? " diagnostics-banner-error" : " diagnostics-banner-warn"}`} data-truth-source="1">{truth.headline}{truth.error && <span> · {truth.error}</span>}{truth.meta && <span> · {truth.meta}</span>}</div>}
    {model.banner && <div className={`diagnostics-banner${model.banner.skeleton ? " diagnostics-banner-warn" : ""}`}>{model.banner.text}</div>}
    {model.warningCount > 0 && <details className="diagnostics-group diagnostics-group-warn"><summary>{english ? `Structure warnings (${model.warningCount})` : `结构告警（${model.warningCount}）`}</summary><ul>{model.warnings.map((entry) => <li key={entry.code}>{entry.code}: {entry.message}</li>)}</ul></details>}
    {model.gapCount > 0 && <details className="diagnostics-group"><summary>{english ? `Checkpoint modules not declared by template (${model.gapCount})` : `checkpoint 有而模板未声明的模块（${model.gapCount}）`}</summary><ul>{model.gaps.slice(0, 50).map((gap) => <li key={gap}>{gap}</li>)}</ul>{model.gaps.length > 50 && <p className="diagnostics-more">{english ? `…and ${model.gaps.length - 50} more` : `…另有 ${model.gaps.length - 50} 项`}</p>}</details>}
    {model.ambiguousCount > 0 && <details className="diagnostics-group diagnostics-group-error"><summary>{english ? `Ambiguous truth bindings (${model.ambiguousCount})` : `真值绑定歧义（${model.ambiguousCount}）——绑定已放弃，需检查映射表`}</summary><ul>{model.ambiguous.slice(0, 50).map((entry, index) => <li key={`${entry.template}.${index}`}>{entry.template} ⇐ {entry.candidates?.join(", ")}</li>)}</ul></details>}
    {diagnostics.total_tensors != null && <p className="diagnostics-meta">{english ? `Bound ${diagnostics.graph_bound_tensors ?? diagnostics.bound_tensors ?? 0} / ${diagnostics.total_tensors} tensors` : `已绑定 ${diagnostics.graph_bound_tensors ?? diagnostics.bound_tensors ?? 0} / ${diagnostics.total_tensors} 张量`}{Number.isFinite(diagnostics.parameter_total) ? ` · ${formatQuantity(diagnostics.parameter_total)} params` : ""}</p>}
  </section>;
}
