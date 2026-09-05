import { metaForNode, typeClass } from "./meta.js";

export const NODE_WIDTH = 220;
const NODE_HEIGHTS = [56, 76, 96];
const NODE_GAP_Y = 18;
const NODE_GAP_X = 60;
const LAYOUT_TOP = 28;
const LAYOUT_LEFT = 28;

export function layoutDiagram(root, expandedGroups) {
  const expanded = expandedGroups instanceof Set ? expandedGroups : new Set();
  const items = [];

  function measure(node, depth, path) {
    const metaLines = metaForNode(node);
    const height = NODE_HEIGHTS[Math.min(metaLines.length + 1, NODE_HEIGHTS.length - 1)];
    const isCollapsible = node.children?.length > 0;
    const isExpanded = isCollapsible ? expanded.has(path) : true;
    const item = {
      node,
      path,
      depth,
      children: [],
      width: NODE_WIDTH,
      height,
      typeClass: typeClass(node.type),
      repeat: node.repeat,
      fullName: node.name,
      displayName: node.repeat > 1 && String(node.type).includes("layer")
        ? "Decoder layer group"
        : node.name,
      metaLines,
      isCollapsible,
      isExpanded,
    };
    items.push(item);
    if (isCollapsible && isExpanded) {
      item.childItems = (node.children || []).map((child, index) => measure(child, depth + 1, `${path}.${index}`));
      item.children = item.childItems.map((child) => child.path);
    }
    const childHeight = item.childItems?.length
      ? item.childItems.reduce((sum, child) => sum + child.subtreeHeight, 0) + (item.childItems.length - 1) * NODE_GAP_Y
      : 0;
    item.subtreeHeight = Math.max(height, childHeight);
    return item;
  }

  const tree = measure(root, 0, "root");
  function place(item, x, y) {
    item.x = x;
    item.y = y + (item.subtreeHeight - item.height) / 2;
    let childY = y;
    (item.childItems || []).forEach((child) => {
      place(child, x + NODE_WIDTH + NODE_GAP_X, childY);
      childY += child.subtreeHeight + NODE_GAP_Y;
    });
  }
  place(tree, LAYOUT_LEFT, LAYOUT_TOP);

  items.forEach((item) => {
    if (!item.isCollapsible || !item.isExpanded) return;
    const descendants = items.filter((candidate) => candidate.path.startsWith(`${item.path}.`));
    if (descendants.length === 0) return;
    const frameItems = [item, ...descendants];
    const left = Math.min(...frameItems.map((candidate) => candidate.x)) - 14;
    const top = Math.min(...frameItems.map((candidate) => candidate.y)) - 22;
    const right = Math.max(...frameItems.map((candidate) => candidate.x + candidate.width)) + 14;
    const bottom = Math.max(...frameItems.map((candidate) => candidate.y + candidate.height)) + 14;
    item.containerFrame = {
      x: left,
      y: top,
      width: right - left,
      height: bottom - top,
      label: [
        item.displayName,
        item.repeat > 1 ? `×${item.repeat}` : null,
        item.node?.attributes?.range || null,
      ].filter(Boolean).join(" · "),
      depth: item.depth,
    };
  });
  return items;
}

/**
 * Convert the visible hierarchy into a graph view model. Analysis code keeps
 * the original tree paths; the canvas consumes these independent collections.
 */
export function layoutGraph(root, expandedGroups) {
  const items = layoutDiagram(root, expandedGroups);
  const stageForPath = (path) => {
    const firstChild = path.split(".")[1];
    const child = firstChild == null ? root : root?.children?.[Number(firstChild)];
    const type = String(child?.type || "model");
    if (type.includes("embedding") || type.includes("vision")) return "input";
    if (type.includes("projector")) return "representation";
    if (type.includes("decoder") || type.includes("layer")) return "decoder";
    if (type.includes("output") || type.includes("head")) return "output";
    return "model";
  };
  const nodes = items.map((item) => ({ ...item, stage: stageForPath(item.path), children: undefined, childItems: undefined }));
  const structureEdges = items.flatMap((item) => item.children.map((target) => ({
    id: `${item.path}->${target}`,
    source: item.path,
    target,
    kind: "structure",
  })));
  const orderedPairs = new Set(items.flatMap((item) => {
    const children = item.childItems || [];
    return children.slice(0, -1).map((source, index) => `${source.path}=>${children[index + 1].path}`);
  }));
  const moduleOrderEdges = items.flatMap((item) => {
    const children = item.childItems || [];
    return children.slice(0, -1).map((source, index) => ({
      id: `${source.path}~${children[index + 1].path}`,
      source: source.path,
      target: children[index + 1].path,
      kind: "dataflow",
      evidence: "module-order",
    }));
  });
  const dataflowEdges = items.flatMap((item) => {
    const operators = item.childItems?.filter((child) => child.node?.type === "operator") || [];
    const edges = [];
    operators.forEach((source) => {
      if (!source.node?.output_shape) return;
      const target = operators.find((candidate) => {
        if (source.path === candidate.path || !candidate.node?.input_shape) return false;
        if (candidate.path <= source.path) return false;
        return JSON.stringify(source.node.output_shape) === JSON.stringify(candidate.node.input_shape);
      });
      if (target) {
        edges.push({
          id: `${source.path}=>${target.path}`,
          source: source.path,
          target: target.path,
          kind: "dataflow",
          evidence: orderedPairs.has(`${source.path}=>${target.path}`) ? "module-order" : undefined,
        });
      }
    });
    return edges;
  });
  const topLevelPaths = items
    .filter((item) => item.path.split(".").length === 2)
    .sort((left, right) => Number(left.path.split(".")[1]) - Number(right.path.split(".")[1]))
    .map((item) => item.path);
  const dataflowPairs = new Set(dataflowEdges.map((edge) => `${edge.source}=>${edge.target}`));
  const missingOrderEdges = moduleOrderEdges.filter((edge) => !dataflowPairs.has(`${edge.source}=>${edge.target}`));
  const edges = [...structureEdges, ...dataflowEdges, ...missingOrderEdges];
  // Keep the synchronous state graph-first while ELK is loading or unavailable.
  // Top-level modules form columns; their visible operators stack inside each column.
  const moduleIndex = new Map(topLevelPaths.map((path, index) => [path, index]));
  const moduleRows = new Map();
  const graphNodes = nodes.map((node) => {
    if (node.path === "root") return { ...node, x: LAYOUT_LEFT, y: LAYOUT_TOP + 48 };
    const topLevelPath = node.path.split(".").slice(0, 2).join(".");
    const column = moduleIndex.get(topLevelPath) ?? 0;
    const row = moduleRows.get(topLevelPath) || 0;
    moduleRows.set(topLevelPath, row + 1);
    return {
      ...node,
      x: LAYOUT_LEFT + 300 + column * 300,
      y: LAYOUT_TOP + row * (node.height + NODE_GAP_Y),
    };
  });
  return { nodes: graphNodes, edges, containerFrames: [], layoutReady: false };
}
