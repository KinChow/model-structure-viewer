const STRATEGY_LABELS = {
  "frontend-architecture-template": ["Frontend template", "ok", "Config-driven frontend structure"],
  "meta-introspect": ["Meta introspect", "ok", "Live module tree"],
  "repaired-meta-introspect": ["Meta introspect", "ok", "Live module tree repaired"],
  "config-fallback": ["Config fallback", "warn", "Config-derived structure"],
  "budget-config-fallback": ["Config fallback", "warn", "Resource budget guard"],
  "worker-config-fallback": ["Config fallback", "warn", "Worker fallback"],
};

export function structureStatus(structure) {
  const summary = structure?.summary || {};
  const diagnostics = structure?.source?.diagnostics || {};
  const strategy = summary.strategy || structure?.source?.strategy;
  const [label, tone, defaultDetail] = STRATEGY_LABELS[strategy] || [
    "Not loaded",
    "neutral",
    "No structure generated",
  ];
  return {
    label,
    tone,
    detail: detailFor(strategy, summary, diagnostics, defaultDetail),
  };
}

function detailFor(strategy, summary, diagnostics, defaultDetail) {
  if (strategy === "budget-config-fallback") {
    const budget = diagnostics.budget || {};
    const parts = [
      budget.layers ? `layers ${budget.layers}` : null,
      budget.hidden_size ? `hidden ${budget.hidden_size}` : null,
      budget.experts ? `experts ${budget.experts}` : null,
    ].filter(Boolean);
    return parts.length > 0 ? `Budget exceeded: ${parts.join(", ")}` : defaultDetail;
  }
  if (strategy === "worker-config-fallback") {
    return diagnostics.failure_kind || summary.fallback_reason || defaultDetail;
  }
  if (strategy === "config-fallback") {
    return diagnostics.failure_kind || summary.fallback_reason || defaultDetail;
  }
  return defaultDetail;
}
