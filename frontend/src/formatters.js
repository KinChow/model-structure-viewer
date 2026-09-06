export function formatCount(value, { largeDigits = 1 } = {}) {
  if (value == null || !Number.isFinite(value)) return null;
  if (value >= 1e9) return `${(value / 1e9).toFixed(largeDigits)}B`;
  if (value >= 1e6) return `${(value / 1e6).toFixed(largeDigits)}M`;
  if (value >= 1e3) return `${(value / 1e3).toFixed(1)}K`;
  return String(value);
}

export function formatBytes(value, { includeKib = false } = {}) {
  if (!Number.isFinite(value)) return "-";
  if (value >= 1024 ** 3) return `${(value / 1024 ** 3).toFixed(2)} GiB`;
  if (value >= 1024 ** 2) return `${(value / 1024 ** 2).toFixed(1)} MiB`;
  if (includeKib && value >= 1024) return `${(value / 1024).toFixed(1)} KiB`;
  return `${Math.round(value)} B`;
}

export function formatMetric(value) {
  if (!Number.isFinite(value)) return null;
  if (value >= 1e12) return `${(value / 1e12).toFixed(2)}T`;
  if (value >= 1e9) return `${(value / 1e9).toFixed(2)}G`;
  if (value >= 1e6) return `${(value / 1e6).toFixed(1)}M`;
  return `${Math.round(value)}`;
}

export function formatQuantity(value) {
  if (!Number.isFinite(value)) return "-";
  if (value >= 1e12) return `${(value / 1e12).toFixed(2)} T`;
  if (value >= 1e9) return `${(value / 1e9).toFixed(2)} G`;
  if (value >= 1e6) return `${(value / 1e6).toFixed(1)} M`;
  return `${Math.round(value)}`;
}

export function formatMacs(value) {
  if (!Number.isFinite(value)) return "-";
  if (value >= 1e12) return `${(value / 1e12).toFixed(2)} T`;
  if (value >= 1e9) return `${(value / 1e9).toFixed(2)} G`;
  return `${(value / 1e6).toFixed(1)} M`;
}

export function formatSeconds(value) {
  if (!Number.isFinite(value)) return "-";
  return value >= 1 ? `${value.toFixed(2)} s` : `${(value * 1000).toFixed(2)} ms`;
}

export function formatRate(value) {
  if (!Number.isFinite(value)) return "-";
  if (value >= 1e12) return `${(value / 1e12).toFixed(2)} TB/s`;
  if (value >= 1e9) return `${(value / 1e9).toFixed(1)} GB/s`;
  return `${(value / 1e6).toFixed(1)} MB/s`;
}
