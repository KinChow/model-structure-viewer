const STRATEGY_LABELS = {
  "frontend-architecture-template": ["Frontend template", "ok", "Config-driven frontend structure"],
  "skeleton-truth": ["Checkpoint 骨架真值", "truth", "Checkpoint-derived module tree"],
  "template+truth": ["模板 + checkpoint 真值", "truth", "Template semantics with checkpoint values"],
  "meta-introspect": ["Meta introspect", "ok", "Live module tree"],
  "repaired-meta-introspect": ["Meta introspect", "ok", "Live module tree repaired"],
};

export function structureStatus(structure) {
  const summary = structure?.summary || {};
  const diagnostics = structure?.source?.diagnostics || {};
  const strategy = summary.strategy || structure?.source?.strategy;
  const checkpointStatus = structure?.source?.checkpoint_truth;
  const [label, tone, defaultDetail] = STRATEGY_LABELS[strategy] || [
    "Not loaded",
    "neutral",
    "No structure generated",
  ];
  return {
    label,
    tone,
    detail: checkpointDetail(
      detailFor(strategy, diagnostics, defaultDetail),
      checkpointStatus,
    ),
  };
}

function detailFor(strategy, diagnostics, defaultDetail) {
  if (strategy === "repaired-meta-introspect" && diagnostics.repair_strategy) {
    return `Repaired by ${diagnostics.repair_strategy}`;
  }
  return defaultDetail;
}

function checkpointDetail(detail, status) {
  if (status === "unavailable") return `${detail}; checkpoint metadata unavailable, using config-only structure`;
  if (status === "empty") return `${detail}; no safetensors metadata found`;
  return detail;
}
