let elkPromise;

function getElk() {
  if (!elkPromise) {
    elkPromise = (typeof Worker === "undefined"
      ? import("elkjs/lib/elk.bundled.js")
      : import("elkjs/lib/elk-api.js")).then(({ default: Elk }) => {
      if (typeof Worker === "undefined") return new Elk();
      return new Elk({
        workerFactory: () => new Worker(new URL("elkjs/lib/elk-worker.min.js", import.meta.url), { type: "classic" }),
      });
    });
  }
  return elkPromise;
}
const BASE_LAYOUT = {
  "elk.algorithm": "layered",
  "elk.layered.spacing.nodeNodeBetweenLayers": "44",
  "elk.spacing.nodeNode": "24",
  // 让 ELK 顺带做正交连线路由：布局后每条边带 sections（含 bendPoints），
  // 渲染层据此画避开节点的正交折线，替代渲染时的自由贝塞尔（消除“甩弧”）。
  "elk.edgeRouting": "ORTHOGONAL",
  // 连线美观：边与节点、边与边留出间距，避免线贴着节点或彼此重叠。
  //（不开 mergeEdges：它会改动边拓扑/计数，收益有限却增加不确定性。）
  "elk.spacing.edgeNode": "20",
  "elk.spacing.edgeEdge": "12",
  "elk.layered.spacing.edgeNodeBetweenLayers": "20",
  "elk.layered.spacing.edgeEdgeBetweenLayers": "12",
};

const SEMANTIC_LAYOUT = {
  ...BASE_LAYOUT,
  // Keep every operation at the longest dependency distance from the
  // module inputs. This prevents a side input such as MLA's value projection
  // from being compressed into the middle of the main q/k path.
  "elk.layered.layering.strategy": "LONGEST_PATH",
};

function parentPath(path) {
  const index = path.lastIndexOf(".");
  return index < 0 ? null : path.slice(0, index);
}

function directChildren(node, nodeByPath) {
  if (!node.isExpanded) return [];
  return (node.node?.children || [])
    .map((_, index) => nodeByPath.get(`${node.path}.${index}`))
    .filter(Boolean);
}

function layoutHeight(node) {
  return node.isCollapsible && node.isExpanded ? 28 : node.height;
}

/**
 * Compound graph layout: top-level modules read left-to-right while module
 * internals read top-to-bottom, keeping the canvas graph-first and readable.
 */
export async function layoutGraphWithElk(graph) {
  const elk = await getElk();
  const nodeByPath = new Map(graph.nodes.map((node) => [node.path, node]));
  const directEdges = (path, allowedIds) => graph.edges
    .filter((edge) => edge.kind === "dataflow"
      && parentPath(edge.source) === path && parentPath(edge.target) === path
      && (!allowedIds || (allowedIds.has(edge.source) && allowedIds.has(edge.target))))
    .map((edge) => ({ id: edge.id, source: edge.source, target: edge.target }));
  // 端口约束（成熟布局器让连线好看的关键）：竖向容器里边从子节点**底部中点出、顶部中点入**，
  // 横向容器里从**右侧中点出、左侧中点入**——与渲染层 Handle 位置一致，消除“从边角斜甩”。
  const portSides = (direction) => (direction === "RIGHT"
    ? { in: "WEST", out: "EAST" }
    : { in: "NORTH", out: "SOUTH" });
  const portId = (path, dir) => `${path}::${dir}`;
  const elkEdge = (edge) => ({ id: edge.id, sources: [portId(edge.source, "out")], targets: [portId(edge.target, "in")] });
  function attachPorts(shape, childPath, sides) {
    shape.ports = [
      { id: portId(childPath, "in"), layoutOptions: { "elk.port.side": sides.in } },
      { id: portId(childPath, "out"), layoutOptions: { "elk.port.side": sides.out } },
    ];
    shape.layoutOptions = {
      ...(shape.layoutOptions || {}),
      "elk.portConstraints": "FIXED_SIDE",
      "elk.portAlignment.default": "CENTER",
    };
    return shape;
  }

  function makeShape(node, depth) {
    const allChildren = directChildren(node, nodeByPath);
    const children = allChildren;
    const childIds = new Set(children.map((child) => child.path));
    if (children.length === 0) return { id: node.path, width: node.width, height: layoutHeight(node) };
    // 语义流布局只信任 builder 声明的边；semantic-flow 已随 legacySemanticEdges 退役。
    const semanticFlow = graph.edges.some((edge) => edge.evidence === "declared"
      && parentPath(edge.source) === node.path && parentPath(edge.target) === node.path);
    const rawEdges = directEdges(node.path, childIds);
    const inputIds = new Set(children
      .filter((child) => !rawEdges.some((edge) => edge.target === child.path))
      .map((child) => child.path));
    const direction = depth === 0 ? "RIGHT" : "DOWN";
    const sides = portSides(direction);
    // 端口约束只用于**嵌套竖向容器**（模块内部，也是我们消费 ELK 路由的地方）。
    // 顶层（RIGHT）保留原有无端口布局：其主干含草稿旁挂重排等精细逻辑，且其边不消费
    // ELK 路由——加端口反而会扰动分层（embed/lm_head 曾因此重叠）。
    const usePorts = direction === "DOWN";
    // 合成顺序边强制相邻子节点竖向排布。但只在**该相邻对没有真实边**时补：
    // 若已有真实内部边（折叠层组间的 module-order 边）还补一条同端点 __order__ 边，
    // ELK 会当两条平行边分别路由——其一绕行，正交消费后成「Z 字」。反过来，module-order
    // 也可能漏边（如 final_norm→lm_head 缺失），此时仍需合成边约束，否则该节点会散落到
    // 第 0 层与他人重叠。故按「缺失的相邻对」精确补齐。
    const realPairs = new Set(rawEdges.map((edge) => `${edge.source}=>${edge.target}`));
    const orderRaw = semanticFlow
      ? []
      : children.slice(0, -1)
        .map((child, index) => ({
          id: `__order__${node.path}__${index}`,
          source: child.path,
          target: children[index + 1].path,
        }))
        .filter((edge) => !realPairs.has(`${edge.source}=>${edge.target}`));
    return {
      id: node.path,
      layoutOptions: {
        ...(semanticFlow ? SEMANTIC_LAYOUT : BASE_LAYOUT),
        // Keep the model's top-level modules in a readable pipeline. Once a
        // module is opened, its implementation is a vertical sibling flow.
        "elk.direction": direction,
        // 内边距把"绘制时容器边框相对 ELK shape 的外扩量"（左右各 16 / 顶 22 / 底 16）
        // 预先并入 ELK 测量的盒子：left/right 24→40、top 32→54、bottom 24→40。随后
        // frame 贴着 shape 绘制（不再外扩），使 ELK 测量的盒子 = 实际绘制的盒子，
        // 相邻模块间距（nodeNodeBetweenLayers=44）在折叠/展开态下都稳定，不再变窄。
        "elk.padding": "[top=54,left=40,bottom=40,right=40]",
      },
      children: children.map((child) => {
        const shape = makeShape(child, depth + 1);
        if (usePorts) attachPorts(shape, child.path, sides);
        if (semanticFlow && inputIds.has(child.path)) {
          shape.layoutOptions = {
            ...(shape.layoutOptions || {}),
            // ELK's FIRST constraint is the explicit contract for module
            // inputs; LONGEST_PATH keeps the remaining graph well layered.
            "elk.layered.layering.layerConstraint": "FIRST",
          };
        }
        return shape;
      }),
      edges: [...rawEdges, ...orderRaw].map((e) => (usePorts
        ? elkEdge(e)
        : { id: e.id, sources: [e.source], targets: [e.target] })),
    };
  }

  // 模型即最外层复合节点；lm_head / 输出头作为 model 的普通顶层子节点归入容器内
  // （HF/vLLM：lm_head 是 XxxForCausalLM 的直接成员，不是模型外的独立模块）。
  const root = nodeByPath.get("root");
  const modelShape = root ? makeShape(root, 0) : null;
  const layoutRoot = root
    ? {
      id: "__graph_root__",
      layoutOptions: { ...BASE_LAYOUT, "elk.direction": "RIGHT", "elk.padding": "32" },
      children: [modelShape],
      edges: [],
    }
    : { id: "__graph_root__", layoutOptions: { ...BASE_LAYOUT, "elk.direction": "RIGHT" }, children: [] };

  const result = await elk.layout(layoutRoot);
  // ELK centers short siblings against a large expanded compound node. That
  // is technically valid, but it pushes embedding/norm/head far below the
  // container headers and makes the top-level execution chain look broken.
  // Keep the root pipeline on one baseline; nested containers retain ELK's
  // own placement.
  const modelLayout = result.id === "root" ? result : result.children?.find((child) => child.id === "root");
  if (modelLayout?.children?.length) {
    // 草稿分支（MTP / DSpark）在拓扑上是旁挂节点：与主干共享 decoder 输入却不回流
    // final norm / lm_head，ELK 会把它与主干同层节点纵向错开。若把它一并拉到主干
    // baseline，就会和 final norm 压在同一坐标（node overlap）。只拉平主干节点，
    // 草稿分支保留 ELK 计算的纵向偏移，旁挂在主干下方。
    const isDraftBranch = (child) => {
      const node = nodeByPath.get(child.id);
      const type = String(node?.node?.type || node?.typeClass || "").toLowerCase();
      return type === "mtp" || type === "dspark";
    };
    const trunk = modelLayout.children.filter((child) => !isDraftBranch(child));
    const drafts = modelLayout.children.filter(isDraftBranch);
    const baseline = trunk.length
      ? Math.min(...trunk.map((child) => child.y || 0))
      : Math.min(...modelLayout.children.map((child) => child.y || 0));
    for (const child of trunk) child.y = baseline;
    // 草稿分支（MTP / DSpark）是旁挂节点：ELK 会把它排在与主干同层节点相同的列里
    //（MTP 落 lm_head 列、DSpark 落 final norm 列），纵向本来错开、无重叠。上面把
    // 主干统一拉到 baseline 后，同列的主干节点被上移，若草稿仍停在 ELK 旧 y 就会与
    // 之相撞（node overlap）。且草稿本身展开时会变高、x 位移，无法靠"同列 x 相等"
    // 稳定判定。改为把所有草稿统一落到主干整体下方的独立行带：草稿保留 ELK 的横向
    // 位置（横跨 decoder→lm_head 区间），纵向排到主干最低点之下，逐个堆叠。这样无论
    // 主干或草稿是否展开都不会与主干重叠，语义上仍是"旁挂主干下方"。
    const DRAFT_BAND_GAP = 24;
    const DRAFT_ROW_GAP = 24;
    const trunkBottom = trunk.length
      ? Math.max(...trunk.map((child) => baseline + (child.height || 0)))
      : baseline;
    let draftTop = trunkBottom + DRAFT_BAND_GAP;
    for (const draft of drafts) {
      draft.y = draftTop;
      draftTop += (draft.height || 0) + DRAFT_ROW_GAP;
    }
    // 上面手动把草稿挪到主干下方独立行带后，ELK 早先为 model 容器算的高度仍基于草稿
    // 旁挂在主干同层（更紧凑）时的布局，并未覆盖被下移、且展开后更高的草稿子树，
    // 导致草稿溢出 model 容器框底部（展示层越界）。这里按重定位后的所有直接子节点
    // 重新撑高 model 容器 shape.height，使随后 walk() 生成的 frame 自洽包住全部子树。
    const MODEL_BOTTOM_PAD = 32;
    const childrenBottom = Math.max(
      ...modelLayout.children.map((child) => (child.y || 0) + (child.height || 0)),
    );
    modelLayout.height = Math.max(modelLayout.height || 0, childrenBottom + MODEL_BOTTOM_PAD);
  }
  const positions = new Map();
  const groupFrames = [];
  // ELK 正交路由折点（绝对画布坐标）：edgeId → [{x,y}...]。仅收集**非 root 容器**内的边——
  // root 直属子节点的 y 在下方被手动重排（草稿旁挂/主干对齐），其 root 级边的 ELK 路由会失真，
  // 故这些边回退到渲染层的 smart/贝塞尔；嵌套模块内部边不受重排影响（重排只平移容器整体）。
  const edgeBends = new Map();
  function walk(shape, offsetX = 0, offsetY = 0) {
    const x = offsetX + (shape.x || 0);
    const y = offsetY + (shape.y || 0);
    if (shape.id !== "__graph_root__") positions.set(shape.id, { x, y });
    if (shape.id !== "__graph_root__" && shape.id !== "root" && Array.isArray(shape.edges)) {
      for (const edge of shape.edges) {
        if (typeof edge.id !== "string" || edge.id.startsWith("__order__")) continue;
        const bends = (edge.sections || []).flatMap((section) => section.bendPoints || []);
        // sections 坐标相对于边所属容器（= 本 shape）原点，加上容器绝对偏移即画布绝对坐标。
        edgeBends.set(edge.id, bends.map((p) => ({ x: x + p.x, y: y + p.y })));
      }
    }
    if (shape.id !== "__graph_root__" && shape.children?.length && nodeByPath.has(shape.id)) {
      const node = nodeByPath.get(shape.id);
      groupFrames.push({
        id: shape.id,
        // frame 贴合 ELK shape 绘制：外扩量已并入上面的 elk.padding，避免边框凸向邻居
        // 压缩展开态下的模块间距。子节点相对 frame 的位置、frame 外框尺寸与此前一致。
        x,
        y,
        width: shape.width,
        height: shape.height,
        label: node.path === "root"
          ? "model"
          : `${node.displayName} · ${node.node?.type || "module"}${node.repeat > 1 ? ` · ×${node.repeat}` : ""}`,
        classLabel: node.path === "root" ? (node.node?.attributes?.class || node.node?.name || null) : null,
        depth: node.depth,
        edgeAnchorOffset: node.depth === 1 ? 70 : null,
        kind: "graph-group",
      });
    }
    for (const child of shape.children || []) walk(child, x, y);
  }
  walk(result);

  return {
    ...graph,
    layoutReady: true,
    nodes: graph.nodes.map((node) => ({ ...node, ...(positions.get(node.path) || {}) })),
    edges: graph.edges.map((edge) => (
      edgeBends.has(edge.id) ? { ...edge, bendPoints: edgeBends.get(edge.id) } : { ...edge }
    )),
    containerFrames: groupFrames,
  };
}
