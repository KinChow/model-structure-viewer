import { structureStatus } from "../diagnostics";

function formatCount(n) {
  if (n == null || !Number.isFinite(n)) return null;
  if (n >= 1e9) return `${(n / 1e9).toFixed(1)}B`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}K`;
  return String(n);
}

function dtypeBreakdown(byDtype) {
  if (!byDtype || typeof byDtype !== "object") return null;
  return Object.entries(byDtype)
    .map(([dtype, count]) => `${dtype} ${formatCount(count)}`)
    .join(" · ");
}

function SummaryChips({ structure, sourceLabel, language = "zh" }) {
  const summary = structure?.summary || {};
  const status = structureStatus(structure);
  const paramsTitle = dtypeBreakdown(summary.parameters_by_dtype);
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
    [label.params, formatCount(summary.parameters_total), "truth", paramsTitle],
    [label.source, sourceLabel],
    [label.status, status.label, status.tone, status.detail],
  ];
  return (
    <div className="summary-chips">
      {chips.map(([label, value, tone, title]) => (
        <span className={`chip ${tone || ""}`.trim()} key={label} title={title || undefined}>
          <b>{label}</b>
          {value ?? "-"}
        </span>
      ))}
    </div>
  );
}

export default SummaryChips;
