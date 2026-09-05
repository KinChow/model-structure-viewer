// 图节点悬停关系判定：同一路径、祖先或后代属于同一局部执行子图。

export function isPathRelated(path, hoveredPath) {
  if (!hoveredPath) return false;
  return path === hoveredPath || path.startsWith(`${hoveredPath}.`) || hoveredPath.startsWith(`${path}.`);
}

export function isEdgeRelated(sourcePath, targetPath, hoveredPath) {
  return isPathRelated(sourcePath, hoveredPath) || isPathRelated(targetPath, hoveredPath);
}

export function isGraphEdgeRelated(sourcePath, targetPath, focusedPath) {
  return isEdgeRelated(sourcePath, targetPath, focusedPath)
    || sourcePath === focusedPath
    || targetPath === focusedPath;
}

export function relatedDataflowEdgeIds(edges, focusedPath) {
  if (!focusedPath) return new Set();
  const relatedNodes = new Set([focusedPath]);
  const relatedEdges = new Set();
  let changed = true;
  while (changed) {
    changed = false;
    edges.forEach((edge) => {
      if (edge.kind !== "dataflow") return;
      if (!isPathRelated(edge.source, focusedPath) && !isPathRelated(edge.target, focusedPath)
        && !relatedNodes.has(edge.source) && !relatedNodes.has(edge.target)) return;
      if (!relatedEdges.has(edge.id)) {
        relatedEdges.add(edge.id);
        changed = true;
      }
      if (!relatedNodes.has(edge.source)) {
        relatedNodes.add(edge.source);
        changed = true;
      }
      if (!relatedNodes.has(edge.target)) {
        relatedNodes.add(edge.target);
        changed = true;
      }
    });
  }
  return relatedEdges;
}
