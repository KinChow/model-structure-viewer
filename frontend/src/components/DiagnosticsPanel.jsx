import { checkpointTruthModel, diagnosticsModel } from "../cost/ui.js";
import { formatQuantity } from "../formatters.js";

/** 前端结构诊断：展示配置、checkpoint 元数据与成本覆盖，不调用开发验证服务。 */
export default function DiagnosticsPanel({ structure, language = "zh" }) {
  const english = language === "en";
  const diagnostics = structure?.source?.diagnostics || null;
  const model = diagnostics ? diagnosticsModel(diagnostics, { english }) : null;
  const truth = checkpointTruthModel(structure?.source, { english });
  const boundTensors = diagnostics?.graph_bound_tensors ?? diagnostics?.bound_tensors ?? 0;
  const tensorSummary = diagnostics?.total_tensors == null
    ? null
    : boundTensors > 0
      ? (english ? `Bound ${boundTensors} / ${diagnostics.total_tensors} tensors` : `已绑定 ${boundTensors} / ${diagnostics.total_tensors} 张量`)
      : (english ? `Aggregate header truth: ${diagnostics.total_tensors} tensors` : `聚合 header 真值：${diagnostics.total_tensors} 张量`);
  const hasDiagnostics = Boolean(model && (model.banner || model.gapCount || model.ambiguousCount || model.unsupportedCount || model.warningCount || truth.show));
  if (!hasDiagnostics && !truth.show && !tensorSummary) return null;

  return <section className="diagnostics-panel" aria-label={english ? "Diagnostics" : "诊断"}>
    {model?.unsupportedCount > 0 && <div className="diagnostics-banner diagnostics-banner-error" data-unsupported="1">{english ? "Unsupported structure" : "不支持的结构"}：{model.unsupported.map((entry) => entry.message).join(" ")}</div>}
    {truth.show && <div className={`diagnostics-banner${truth.tone === "error" ? " diagnostics-banner-error" : " diagnostics-banner-warn"}`} data-truth-source="1">{truth.headline}{truth.error && <span> · {truth.error}</span>}{truth.meta && <span> · {truth.meta}</span>}</div>}
    {model?.banner && <div className={`diagnostics-banner${model.banner.skeleton ? " diagnostics-banner-warn" : ""}`}>{model.banner.text}</div>}
    {model?.warningCount > 0 && <details className="diagnostics-group diagnostics-group-warn"><summary>{english ? `Structure warnings (${model.warningCount})` : `结构告警（${model.warningCount}）`}</summary><ul>{model.warnings.map((entry) => <li key={entry.code}>{entry.code}: {entry.message}</li>)}</ul></details>}
    {model?.gapCount > 0 && <details className="diagnostics-group"><summary>{english ? `Checkpoint modules not declared by template (${model.gapCount})` : `checkpoint 有而模板未声明的模块（${model.gapCount}）`}</summary><ul>{model.gaps.slice(0, 50).map((gap) => <li key={gap}>{gap}</li>)}</ul>{model.gaps.length > 50 && <p className="diagnostics-more">{english ? `…and ${model.gaps.length - 50} more` : `…另有 ${model.gaps.length - 50} 项`}</p>}</details>}
    {model?.ambiguousCount > 0 && <details className="diagnostics-group diagnostics-group-error"><summary>{english ? `Ambiguous truth bindings (${model.ambiguousCount})` : `真值绑定歧义（${model.ambiguousCount}）——绑定已放弃，需检查映射表`}</summary><ul>{model.ambiguous.slice(0, 50).map((entry, index) => <li key={`${entry.template}.${index}`}>{entry.template} ⇐ {entry.candidates?.join(", ")}</li>)}</ul></details>}
    {tensorSummary && <p className="diagnostics-meta">{tensorSummary}{Number.isFinite(diagnostics.parameter_total) ? ` · ${formatQuantity(diagnostics.parameter_total)} params` : ""}</p>}
  </section>;
}
