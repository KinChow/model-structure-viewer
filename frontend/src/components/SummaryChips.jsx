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

function SummaryChips({ structure, sourceLabel }) {
  const summary = structure?.summary || {};
  const status = structureStatus(structure);
  const paramsTitle = dtypeBreakdown(summary.parameters_by_dtype);
  const chips = [
    ["Model", summary.model_family || summary.model_type],
    ["Architecture", summary.architecture],
    ["Layers", summary.text_layers],
    ["Hidden", summary.hidden_size],
    ["Heads", summary.num_attention_heads],
    ["Experts", summary.num_local_experts ?? summary.n_routed_experts],
    ["Context", summary.max_position_embeddings],
    ["Params", formatCount(summary.parameters_total), "truth", paramsTitle],
    ["Source", sourceLabel],
    ["Status", status.label, status.tone, status.detail],
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
