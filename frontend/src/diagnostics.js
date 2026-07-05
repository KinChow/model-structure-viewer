const STRATEGY_LABELS = {
  "frontend-architecture-template": ["Frontend template", "ok", "Config-driven frontend structure"],
  "meta-introspect": ["Meta introspect", "ok", "Live module tree"],
  "repaired-meta-introspect": ["Meta introspect", "ok", "Live module tree repaired"],
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
    detail: detailFor(strategy, diagnostics, defaultDetail),
  };
}

function detailFor(strategy, diagnostics, defaultDetail) {
  if (strategy === "repaired-meta-introspect" && diagnostics.repair_strategy) {
    return `Repaired by ${diagnostics.repair_strategy}`;
  }
  return defaultDetail;
}
