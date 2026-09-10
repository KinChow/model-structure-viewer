import { checkpointTruthModel, diagnosticsModel, verifyEvidenceModel } from "../cost/ui.js";
import { formatQuantity } from "../formatters.js";

/**
 * 诊断面板（W6-1，§4.4 / §6.3）：真值绑定、成本覆盖与 transformers 结构对账。
 * 数据来自 structure.source 与 /api/verify 响应，不做推断。
 */
export default function DiagnosticsPanel({
  structure,
  language = "zh",
  verifyResult = null,
  verifyLoading = false,
  verifyError = "",
  onVerify,
}) {
  const english = language === "en";
  const diagnostics = structure?.source?.diagnostics || null;
  const model = diagnostics ? diagnosticsModel(diagnostics, { english }) : null;
  const truth = checkpointTruthModel(structure?.source, { english });
  const verify = verifyEvidenceModel(verifyResult, { english });
  const hasDiagnostics = Boolean(model && (model.banner || model.gapCount || model.ambiguousCount || model.unsupportedCount || model.warningCount || truth.show));
  if (!hasDiagnostics && !onVerify && !verify.show && !verifyError) return null;

  return <section className="diagnostics-panel" aria-label={english ? "Diagnostics" : "诊断"}>
    {onVerify && <div className="diagnostics-verify">
      <button type="button" onClick={onVerify} disabled={verifyLoading} data-verify="1">
        {verifyLoading ? (english ? "Verifying…" : "校验中…") : (english ? "Verify with Transformers" : "用 Transformers 校验")}
      </button>
      {verifyError && <span className="diagnostics-verify-error">{verifyError}</span>}
    </div>}
    {verify.show && <div className={`diagnostics-banner${verify.tone === "error" ? " diagnostics-banner-error" : verify.tone === "ok" ? " diagnostics-banner-ok" : " diagnostics-banner-warn"}`} data-verify-result="1">
      {english ? "Transformers verify" : "Transformers 校验"}：{verify.headline}
      {verify.note && <span> · {verify.note}</span>}
      {verify.moduleCount > 0 && <span> · {english ? `${verify.moduleCount} modules` : `${verify.moduleCount} 个模块`}</span>}
    </div>}
    {verify.show && verify.classifiedEntries.length > 0 && <p className="diagnostics-meta" data-verify-classified="1">
      {english ? "Classified noise" : "已分类噪声"}：{verify.classifiedEntries.map(([bucket, count]) => `${bucket} ${count}`).join(" · ")}
    </p>}
    {verify.show && verify.residualCount > 0 && <details className="diagnostics-group diagnostics-group-error" data-verify-diff="1">
      <summary>{english ? `Unclassified differences (${verify.residualCount})` : `未分类残余（${verify.residualCount}）`}</summary>
      <ul>
        {verify.onlyTransformers.slice(0, 30).map((path) => <li key={`tf-${path}`}>{english ? "only transformers" : "仅 transformers"}: {path}</li>)}
        {verify.onlyMsv.slice(0, 30).map((path) => <li key={`msv-${path}`}>{english ? "only msv" : "仅 msv"}: {path}</li>)}
        {verify.mismatches.slice(0, 30).map((entry, index) => <li key={`mm-${entry.path}-${index}`}>{entry.kind} {entry.path}: transformers={String(entry.transformers)} · msv={String(entry.msv)}</li>)}
      </ul>
      {verify.residualCount > 30 && <p className="diagnostics-more">{english ? `…and more residual rows` : `…另有更多残余`}</p>}
    </details>}
    {model?.unsupportedCount > 0 && <div className="diagnostics-banner diagnostics-banner-error" data-unsupported="1">{english ? "Unsupported structure" : "不支持的结构"}：{model.unsupported.map((entry) => entry.message).join(" ")}</div>}
    {truth.show && <div className={`diagnostics-banner${truth.tone === "error" ? " diagnostics-banner-error" : " diagnostics-banner-warn"}`} data-truth-source="1">{truth.headline}{truth.error && <span> · {truth.error}</span>}{truth.meta && <span> · {truth.meta}</span>}</div>}
    {model?.banner && <div className={`diagnostics-banner${model.banner.skeleton ? " diagnostics-banner-warn" : ""}`}>{model.banner.text}</div>}
    {model?.warningCount > 0 && <details className="diagnostics-group diagnostics-group-warn"><summary>{english ? `Structure warnings (${model.warningCount})` : `结构告警（${model.warningCount}）`}</summary><ul>{model.warnings.map((entry) => <li key={entry.code}>{entry.code}: {entry.message}</li>)}</ul></details>}
    {model?.gapCount > 0 && <details className="diagnostics-group"><summary>{english ? `Checkpoint modules not declared by template (${model.gapCount})` : `checkpoint 有而模板未声明的模块（${model.gapCount}）`}</summary><ul>{model.gaps.slice(0, 50).map((gap) => <li key={gap}>{gap}</li>)}</ul>{model.gaps.length > 50 && <p className="diagnostics-more">{english ? `…and ${model.gaps.length - 50} more` : `…另有 ${model.gaps.length - 50} 项`}</p>}</details>}
    {model?.ambiguousCount > 0 && <details className="diagnostics-group diagnostics-group-error"><summary>{english ? `Ambiguous truth bindings (${model.ambiguousCount})` : `真值绑定歧义（${model.ambiguousCount}）——绑定已放弃，需检查映射表`}</summary><ul>{model.ambiguous.slice(0, 50).map((entry, index) => <li key={`${entry.template}.${index}`}>{entry.template} ⇐ {entry.candidates?.join(", ")}</li>)}</ul></details>}
    {diagnostics?.total_tensors != null && <p className="diagnostics-meta">{english ? `Bound ${diagnostics.graph_bound_tensors ?? diagnostics.bound_tensors ?? 0} / ${diagnostics.total_tensors} tensors` : `已绑定 ${diagnostics.graph_bound_tensors ?? diagnostics.bound_tensors ?? 0} / ${diagnostics.total_tensors} 张量`}{Number.isFinite(diagnostics.parameter_total) ? ` · ${formatQuantity(diagnostics.parameter_total)} params` : ""}</p>}
  </section>;
}
