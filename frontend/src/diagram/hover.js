// 图节点悬停关系判定：同一路径、祖先或后代属于同一局部执行子图。

export function isPathRelated(path, hoveredPath) {
  if (!hoveredPath) return false;
  return path === hoveredPath || path.startsWith(`${hoveredPath}.`) || hoveredPath.startsWith(`${path}.`);
}

export function isEdgeRelated(sourcePath, targetPath, hoveredPath) {
  return isPathRelated(sourcePath, hoveredPath) || isPathRelated(targetPath, hoveredPath);
}
