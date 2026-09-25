/**
 * Layout-only projection.
 *
 * Graph IR remains the source of truth. This projection adds virtual compound
 * parents and lane metadata for ELK; virtual nodes must never be consumed by
 * cost, checkpoint, export, or structure APIs.
 */

const VIRTUAL_ROOT = "__layout_root__";
const LANES = ["input", "main", "auxiliary"];

function nodeType(node) {
  return String(node?.node?.type || node?.type || "").toLowerCase();
}

function laneFor(node) {
  const type = nodeType(node);
  if (type.includes("mtp") || type.includes("dspark") || type.includes("draft")) return "auxiliary";
  if (
    type.includes("embedding")
    || type.includes("vision")
    || type.includes("projector")
    || type.includes("merger")
    || type.includes("encoder")
    || type.includes("fusion")
  ) return "input";
  return "main";
}

function parentPath(path) {
  const index = path.lastIndexOf(".");
  return index < 0 ? null : path.slice(0, index);
}

/**
 * Build a layout-only hierarchy while preserving every Graph IR node path.
 *
 * `parentByPath` is consumed only by layout construction. `laneByPath` is
 * diagnostic metadata and intentionally does not modify Graph IR nodes.
 */
export function buildLayoutProjection(graph) {
  const nodes = graph?.nodes || [];
  const nodeByPath = new Map(nodes.map((node) => [node.path || node.id, node]));
  const topLevel = nodes
    .filter((node) => (node.path || node.id).split(".").length === 2)
    .sort((a, b) => Number((a.path || a.id).split(".")[1]) - Number((b.path || b.id).split(".")[1]));
  const virtualNodes = [{
    id: VIRTUAL_ROOT,
    virtual: true,
    kind: "layout-root",
    lane: null,
  }];
  const parentByPath = new Map();
  const layoutParentByPath = new Map();
  const layoutChildrenByParent = new Map();
  const laneByPath = new Map();
  const laneIds = new Map(LANES.map((lane) => [lane, `${VIRTUAL_ROOT}.${lane}`]));
  for (const lane of LANES) {
    virtualNodes.push({
      id: laneIds.get(lane),
      parent_id: VIRTUAL_ROOT,
      virtual: true,
      kind: "layout-lane",
      lane,
    });
  }
  for (const node of nodes) {
    const path = node.path || node.id;
    if (path === "root") continue;
    const parent = parentPath(path);
    if (parent && parent !== "root") parentByPath.set(path, parent);
  }
  for (const node of topLevel) {
    const path = node.path || node.id;
    const lane = laneFor(node);
    laneByPath.set(path, lane);
    layoutParentByPath.set(path, laneIds.get(lane));
  }
  // The model frame remains the visible owner of the virtual lanes. The
  // virtual root is only a projection namespace; it is never rendered.
  for (const lane of LANES) layoutParentByPath.set(laneIds.get(lane), "root");
  layoutParentByPath.set("root", null);
  // Descendants retain their Graph IR hierarchy. Their top-level ancestor
  // inherits the lane solely for layout constraints and diagnostics.
  for (const node of nodes) {
    const path = node.path || node.id;
    if (path === "root" || topLevel.some((item) => (item.path || item.id) === path)) continue;
    let top = path;
    while (top && top.split(".").length > 2) top = parentPath(top);
    if (top && laneByPath.has(top)) laneByPath.set(path, laneByPath.get(top));
    if (path !== "root" && !layoutParentByPath.has(path)) {
      layoutParentByPath.set(path, parentPath(path));
    }
  }
  for (const node of topLevel) {
    const path = node.path || node.id;
    layoutParentByPath.set(path, laneIds.get(laneByPath.get(path)));
  }
  for (const node of nodes) {
    const path = node.path || node.id;
    if (path === "root") continue;
    const parent = layoutParentByPath.get(path);
    if (!parent) continue;
    const children = layoutChildrenByParent.get(parent) || [];
    children.push(path);
    layoutChildrenByParent.set(parent, children);
  }
  for (const lane of LANES) {
    const laneId = laneIds.get(lane);
    const rootChildren = layoutChildrenByParent.get("root") || [];
    if (!rootChildren.includes(laneId)) rootChildren.push(laneId);
    layoutChildrenByParent.set("root", rootChildren);
    layoutChildrenByParent.set(laneId, layoutChildrenByParent.get(laneId) || []);
  }
  return {
    version: 1,
    graphVersion: graph.version || graph.graphVersion || 2,
    root_id: VIRTUAL_ROOT,
    virtualNodes,
    parentByPath,
    layoutParentByPath,
    layoutChildrenByParent,
    laneByPath,
    laneIds,
    nodeByPath,
  };
}

export { LANES, VIRTUAL_ROOT };
