import { PUBLIC_CHIPS } from "./public.js";
import { validateChipEntry } from "./coverage.js";

// 本地覆盖只用于用户自己的非公开规格，不进入仓库发布的公开目录。
// 覆盖采用字段级深合并，避免只修改一个 dtype 时丢失其他公开字段。

function mergeChip(base, override) {
  return {
    ...base,
    ...override,
    peak_flops: { ...(base?.peak_flops || {}), ...(override?.peak_flops || {}) },
    interconnect: {
      ...(base?.interconnect || {}),
      ...(override?.interconnect || {}),
      intra_node: { ...(base?.interconnect?.intra_node || {}), ...(override?.interconnect?.intra_node || {}) },
      inter_node: { ...(base?.interconnect?.inter_node || {}), ...(override?.interconnect?.inter_node || {}) },
    },
    field_sources: { ...(base?.field_sources || {}), ...(override?.field_sources || {}) },
    notes: override?.notes || base?.notes,
  };
}

export function mergeChipCatalog(publicChips = PUBLIC_CHIPS, localChips = []) {
  const localById = new Map((Array.isArray(localChips) ? localChips : []).filter((chip) => chip?.id).map((chip) => [chip.id, chip]));
  const merged = publicChips.map((chip) => mergeChip(chip, localById.get(chip.id)));
  const publicIds = new Set(publicChips.map((chip) => chip.id));
  for (const chip of localChips || []) if (chip?.id && !publicIds.has(chip.id)) merged.push({ ...chip, confidence: chip.confidence || "local" });
  return merged;
}

/** 从静态资源路径加载可选本地覆盖；文件不存在时按空覆盖处理。 */
export async function loadLocalChipOverrides({ url = "/chips.local.json", fetchImpl = fetch } = {}) {
  const response = await fetchImpl(url);
  if (response.status === 404) return [];
  if (!response.ok) throw new Error(`本地芯片配置 HTTP ${response.status}`);
  const payload = await response.json();
  const chips = Array.isArray(payload) ? payload : payload?.chips;
  if (!Array.isArray(chips)) throw new Error("本地芯片配置必须是数组或 {chips: []}");
  const errors = chips.flatMap((chip) => validateChipEntry(chip));
  if (errors.length > 0) throw new Error(`本地芯片配置无效：${errors.join("；")}`);
  return chips;
}
