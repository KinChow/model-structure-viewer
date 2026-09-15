import { t } from "./i18n/format.js";

const STRATEGY_TONES = {
  "frontend-architecture-template": "ok",
  "skeleton-truth": "truth",
  "template+truth": "truth",
  "template+header-truth": "truth",
  "header-truth": "truth",
  "meta-introspect": "ok",
  "repaired-meta-introspect": "ok",
};

export function structureStatus(structure, language = "zh") {
  const summary = structure?.summary || {};
  const diagnostics = structure?.source?.diagnostics || {};
  const strategy = summary.strategy || structure?.source?.strategy;
  const checkpointStatus = structure?.source?.checkpoint_truth;
  const known = Boolean(STRATEGY_TONES[strategy]);
  const key = known ? strategy : "fallback";
  const label = t(language, `status.${key}.label`);
  const tone = STRATEGY_TONES[strategy] || "neutral";
  const defaultDetail = t(language, `status.${key}.detail`);
  return {
    label,
    tone,
    detail: checkpointDetail(
      language,
      detailFor(language, strategy, diagnostics, defaultDetail),
      checkpointStatus,
    ),
  };
}

function detailFor(language, strategy, diagnostics, defaultDetail) {
  if (strategy === "repaired-meta-introspect" && diagnostics.repair_strategy) {
    return t(language, "status.repairedBy", { strategy: diagnostics.repair_strategy });
  }
  return defaultDetail;
}

function checkpointDetail(language, detail, status) {
  if (status === "unavailable") return t(language, "status.checkpointUnavailable", { detail });
  if (status === "empty") return t(language, "status.checkpointEmpty", { detail });
  return detail;
}
