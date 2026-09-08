// edgeStyle.js —— 边的三轴展示（W6-2，§2.2）：宽度 / evidence 类名 / hover 提示。
// edgePresentation 是唯一决策点（OCP：新 evidence 类型只改 PRESENTATION 表），
// 渲染层（ReactFlow 边组件）只消费视图对象，不再自行拼样式逻辑。

function knownElements(shape) {
  if (!Array.isArray(shape) || shape.length === 0) return 0;
  // 非正维（batch/sequence 等动态位）跳过而非归零——归零会使含动态维的
  // 形状永远拿不到张量感知宽度（旧链遗留缺陷，W6-2 修复）。
  return shape.reduce((total, value) => {
    const dimension = Number(value);
    return Number.isFinite(dimension) && dimension > 0 ? total * dimension : total;
  }, 1);
}

// evidence → 展示语义。declared = builder 声明的事实边；module-order = 兄弟顺序
// 推断；shape-match = 输出形状匹配推测（内置模型已声明化后仅未来模块出现）。
const PRESENTATION = {
  declared: { hintZh: "builder 声明的数据流边", hintEn: "builder-declared dataflow edge" },
  "module-order": { hintZh: "此边由兄弟顺序推断（非声明数据流）", hintEn: "inferred from sibling order (not declared dataflow)" },
  "shape-match": { hintZh: "此边由输出形状匹配推测", hintEn: "inferred from output shape matching" },
};

/**
 * @returns {{width: number, className: string, hint: string, evidence: string}}
 */
export function edgePresentation(edge, source, { english = false } = {}) {
  const evidence = String(edge?.evidence || "declared");
  const meta = PRESENTATION[evidence] || PRESENTATION.declared;
  let width;
  if (evidence === "module-order") {
    width = 1.5; // §2.2：推断边弱化层级——声明边 > 形状匹配 > 兄弟顺序
  } else if (evidence === "shape-match") {
    width = 1.6;
  } else {
    const elements = knownElements(source?.node?.output_shape || source?.output_shape);
    // 下限 1.8：保证声明边恒大于 shape-match(1.6)/module-order(1.5)
    width = Math.max(1.8, Math.min(2.8, 1.4 + Math.max(0, (Math.log10(elements) - 3.5) * 0.45)));
  }
  return {
    width,
    className: evidence === "declared" ? "" : ` ${evidence}`,
    hint: english ? meta.hintEn : meta.hintZh,
    evidence,
  };
}

/** 兼容出口：仅宽度（既有调用方）。 */
export function edgeStrokeWidth(edge, source) {
  return edgePresentation(edge, source).width;
}
