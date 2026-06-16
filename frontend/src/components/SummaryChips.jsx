function SummaryChips({ structure, sourceLabel }) {
  const summary = structure?.summary || {};
  const chips = [
    ["Model", summary.model_family || summary.model_type],
    ["Architecture", summary.architecture],
    ["Layers", summary.text_layers],
    ["Hidden", summary.hidden_size],
    ["Heads", summary.num_attention_heads],
    ["Experts", summary.num_local_experts],
    ["Context", summary.max_position_embeddings],
    ["Source", sourceLabel],
  ];
  return (
    <div className="summary-chips">
      {chips.map(([label, value]) => (
        <span className="chip" key={label}>
          <b>{label}</b>
          {value ?? "-"}
        </span>
      ))}
    </div>
  );
}

export default SummaryChips;
