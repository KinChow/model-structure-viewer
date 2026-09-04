// 比较两张芯片 lens 的节点 bound，输出发生翻转的路径。

export function boundFlips(primary = {}, secondary = {}) {
  const paths = new Set([...Object.keys(primary), ...Object.keys(secondary)]);
  return [...paths]
    .filter((path) => (primary[path]?.bound || "unknown") !== (secondary[path]?.bound || "unknown"))
    .map((path) => ({ path, primary: primary[path]?.bound || "unknown", secondary: secondary[path]?.bound || "unknown" }));
}
