export function modelDisplayName(entry) {
  return entry.displayName || String(entry.modelId || "").split("/").pop() || entry.modelId;
}

export function formatReleaseTime(releaseTime, language = "zh") {
  if (!releaseTime) return "";
  const date = new Date(releaseTime);
  if (Number.isNaN(date.getTime())) return releaseTime;
  return new Intl.DateTimeFormat(language === "en" ? "en-US" : "zh-CN", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  }).format(date);
}

export function sortModelsByReleaseTime(entries) {
  return entries
    .map((entry, index) => ({ entry, index, time: entry.releaseTime ? Date.parse(entry.releaseTime) : Number.NaN }))
    .sort((a, b) => {
      const aHasTime = Number.isFinite(a.time);
      const bHasTime = Number.isFinite(b.time);
      if (aHasTime !== bHasTime) return aHasTime ? -1 : 1;
      if (aHasTime && a.time !== b.time) return b.time - a.time;
      return a.index - b.index;
    })
    .map(({ entry }) => entry);
}

export function sortModelsByName(entries) {
  return [...entries].sort((a, b) => modelDisplayName(a).localeCompare(modelDisplayName(b)));
}
