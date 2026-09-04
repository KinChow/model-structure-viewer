// 来源：GNN 101 的 formula ↔ visualization 交互；收集公式与图节点的双向路径绑定。

export function collectFormulaLinks(root) {
  const links = [];
  function visit(node, path) {
    const formulaId = node?.attributes?.formula_id;
    if (formulaId) links.push({ path, formulaId, explanation: node.attributes.explanation || "" });
    node?.children?.forEach((child, index) => visit(child, `${path}.${index}`));
  }
  if (root) visit(root, "root");
  return links;
}
