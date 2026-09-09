// chipValidation.js —— 芯片条目基本形状校验（M11.5 子项 2：自 coverage.js 平移）。
// 斩断 coverage ↔ public 的加载环：本模块零 import，public.js（数据表）与
// coverage.js（覆盖判定）各自单向依赖这里，环消失。
// 单位量级健全性对照（10 倍偏差警告）留在 coverage.js，PUBLIC_CHIPS 由其直接 import。

const CONFIDENCE_VALUES = new Set(["official", "vendor-marketing", "community", "local"]);

/** 校验公开或本地芯片条目的基本形状；缺失规格返回错误，不自动填值。 */
export function validateChipEntry(chip) {
  const errors = [];
  if (!chip || typeof chip !== "object") return ["芯片条目必须是对象"];
  if (!chip.id) errors.push("缺少 id");
  if (!chip.vendor) errors.push("缺少 vendor");
  if (!chip.name) errors.push("缺少 name");
  if (!chip.source) errors.push("缺少 source");
  if (chip.confidence && !CONFIDENCE_VALUES.has(chip.confidence)) errors.push(`未知 confidence：${chip.confidence}`);
  return errors;
}

export { CONFIDENCE_VALUES };
