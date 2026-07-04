import { structureStatus } from "../diagnostics";

function SummaryChips({ structure, sourceLabel }) {
  const summary = structure?.summary || {};
  const status = structureStatus(structure);
  const chips = [
    ["Model", summary.model_family || summary.model_type],
    ["Architecture", summary.architecture],
    ["Layers", summary.text_layers],
    ["Hidden", summary.hidden_size],
    ["Heads", summary.num_attention_heads],
    ["Experts", summary.num_local_experts ?? summary.n_routed_experts],
    ["Context", summary.max_position_embeddings],
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
