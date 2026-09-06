import { structureStatus } from "../diagnostics";
import { normalizeConfig } from "../structure/config/normalize.js";
import { derivedWeightParameters } from "../cost/derivedWeights.js";
import { formatCount } from "../formatters.js";

function dtypeBreakdown(byDtype) {
  if (!byDtype || typeof byDtype !== "object") return null;
  return Object.entries(byDtype)
    .map(([dtype, count]) => `${dtype} ${formatCount(count)}`)
    .join(" · ");
}

function SummaryChips({ structure, sourceLabel, language = "zh" }) {
  const summary = structure?.summary || {};
  const status = structureStatus(structure);
  const derivedCandidate = summary.parameters_total == null && structure?.extra_config
    ? derivedWeightParameters(normalizeConfig(structure.extra_config))
    : null;
  const derivedParameters = derivedCandidate > 0 ? derivedCandidate : null;
  const parameterTotal = summary.parameters_total ?? derivedParameters;
  const paramsTitle = dtypeBreakdown(summary.parameters_by_dtype) || (derivedParameters != null ? (language === "en" ? "derived from model config" : "由模型配置推导") : undefined);
  const modelId = structure?.source?.model_id || "";
  const provider = modelId.includes("/") ? modelId.split("/")[0] : null;
  const english = language === "en";
  const label = english ? { model: "Model", provider: "Provider", modelType: "Model type", architecture: "Architecture", layers: "Layers", hidden: "Hidden", heads: "Heads", experts: "Experts", context: "Context", params: "Params", source: "Source", status: "Status" } : { model: "模型", provider: "Provider", modelType: "Model type", architecture: "架构", layers: "层数", hidden: "Hidden Size", heads: "Heads", experts: "Experts", context: "Context", params: "Params", source: "来源", status: "状态" };
  const chips = [
    [label.model, modelId || summary.model_family || summary.model_type],
    [label.provider, provider],
    [label.modelType, summary.model_type],
    [label.architecture, summary.architecture],
    [label.layers, summary.text_layers],
    [label.hidden, summary.hidden_size],
    [label.heads, summary.num_attention_heads],
    [label.experts, summary.num_local_experts ?? summary.n_routed_experts],
    [label.context, summary.max_position_embeddings],
    [label.params, formatCount(parameterTotal), derivedParameters != null ? "derived" : "truth", paramsTitle],
    [label.source, sourceLabel],
    [label.status, status.label, status.tone, status.detail],
  ];
  return (
    <div className="summary-chips">
      {chips.map(([label, value, tone, title], index) => (
        <span className={`chip ${tone || ""} ${index === 0 ? "identity" : ""} ${index === 1 ? "identity" : ""} ${index === 9 || index === 11 ? "primary" : ""}`.trim()} key={label} title={title || undefined}>
          <b>{label}</b>
          <span className="chip-value">{value ?? "-"}</span>
        </span>
      ))}
    </div>
  );
}

export default SummaryChips;
