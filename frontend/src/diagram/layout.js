import { metaForNode, typeClass } from "./meta.js";

export const NODE_WIDTH = 260;
const NODE_HEIGHTS = [64, 84, 108];
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

function semanticEdges(item) {
  if (!item?.childItems?.length) return null;
  const type = String(item.node?.type || "").toLowerCase();
  const name = String(item.node?.name || "").toLowerCase();
  const children = item.childItems;
  const find = (pattern) => children.find((child) => pattern.test(String(child.node?.name || "").toLowerCase()));

  if (type === "mlp" || /(^|\b)mlp(\b|$)/.test(name)) {
    const gate = find(/gate\s*(projection|proj)|gate_proj/);
    const up = find(/up\s*(projection|proj)|up_proj/);
    const activation = find(/swiglu|activation/);
    const down = find(/down\s*(projection|proj)|down_proj/);
    const edges = [];
    const add = (source, target) => {
      if (!source || !target || source.path === target.path) return;
      edges.push({
        id: `${source.path}=>${target.path}`,
        source: source.path,
        target: target.path,
        kind: "dataflow",
        evidence: "semantic-flow",
      });
    };
    add(gate, activation);
    add(up, activation);
    add(activation, down);
    return edges.length >= 2 ? edges : null;
  }

  if (type === "moe" || /(^|\b)(moe|mixture.?of.?experts)(\b|$)/.test(name)) {
    const router = find(/router|logits/);
    const topk = find(/top.?k|expert routing|routing/);
    const dispatch = find(/dispatch/);
    const expert = find(/expert.*(mlp|swiglu|feed.?forward)|expert mlp/);
    const combine = find(/combine|scatter/);
    const latentDown = find(/latent down projection/);
    const latentNorm = find(/latent rmsnorm/);
    const latentUp = find(/latent up projection/);
    const sharedAdd = find(/shared expert branch add/);
    const sharedMlp = find(/shared expert mlp/);
    const sharedGate = find(/shared expert gate/);
    const edges = [];
    const add = (source, target) => {
      if (!source || !target || source.path === target.path) return;
      edges.push({
        id: `${source.path}=>${target.path}`,
        source: source.path,
        target: target.path,
        kind: "dataflow",
        evidence: "semantic-flow",
      });
    };
    if (find(/hash expert routing/) && !topk) {
      add(router, dispatch);
      add(dispatch, expert);
      add(expert, combine);
      add(combine, sharedAdd);
      add(sharedMlp, sharedAdd);
      return edges.length >= 3 ? edges : null;
    }
    if (latentDown && latentNorm && latentUp && sharedAdd && sharedMlp) {
      add(router, topk);
      add(topk, dispatch);
      add(latentDown, dispatch);
      add(dispatch, expert);
      add(expert, combine);
      add(combine, latentNorm);
      add(latentNorm, latentUp);
      add(latentUp, sharedAdd);
      add(sharedMlp, sharedAdd);
      return edges.length >= 8 ? edges : null;
    }
    if (sharedAdd && sharedMlp) {
      add(router, topk);
      add(topk, dispatch);
      add(dispatch, expert);
      add(expert, combine);
      add(combine, sharedAdd);
      add(sharedMlp, sharedAdd);
      add(sharedGate, sharedAdd);
      return edges.length >= 5 ? edges : null;
    }
    add(router, topk);
    add(topk, dispatch);
    add(dispatch, expert);
    add(topk, combine);
    add(expert, combine);
    return edges.length >= 3 ? edges : null;
  }

  if (item.node?.attributes?.attention_kind === "linear" || /linear attention/.test(name)) {
    const qkv = find(/qkv|q.?k.?v/);
    const gate = find(/output gate|in_proj_z|gate projection/);
    const decay = find(/decay|in_proj_b/);
    const conv = find(/short conv/);
    const state = find(/state update|linear attention state/);
    const outputGate = find(/output gate/);
    const output = find(/output projection|out_proj/);
    const qwenInputs = children.filter((child) => /in_proj_(qkv|z|b|a)/.test(`${child.node?.id || ""} ${child.node?.name || ""}`.toLowerCase()));
    const kimiQkv = children.filter((child) => /(?:^|\.)(q|k|v)_proj$/.test(String(child.node?.id || "").toLowerCase()) || /^(q|k|v) projection$/.test(String(child.node?.name || "").toLowerCase()));
    const glmFused = find(/fused qkvbfg_a|qkvbfg_a projection/);
    const glmSplit = find(/qkvbfg_a split/);
    const glmQkvConv = children.filter((child) => /[qkv] causal short convolution/.test(String(child.node?.name || "").toLowerCase()));
    const glmState = find(/gated delta recurrent state/);
    const glmOutputNorm = find(/gated rmsnorm/);
    const edges = [];
    const add = (source, target) => {
      if (!source || !target || source.path === target.path) return;
      edges.push({ id: `${source.path}=>${target.path}`, source: source.path, target: target.path, kind: "dataflow", evidence: "semantic-flow" });
    };
    const qwenNorm = find(/gated rmsnorm|^norm$/);
    const canonicalQkv = find(/^qkv projection$/);
    const canonicalQkvzSplit = find(/qkvz split/);
    const canonicalBeta = find(/^beta projection$/);
    const canonicalDecay = find(/forget\/decay gate projection/);
    const canonicalConv = find(/qkv causal short convolution/);
    const canonicalState = find(/kda recurrent state/);
    const canonicalGateNorm = find(/gated rmsnorm/);
    if (canonicalQkv && canonicalBeta && canonicalDecay && canonicalConv && canonicalState && canonicalGateNorm) {
      add(canonicalQkv, canonicalQkvzSplit || canonicalConv);
      if (canonicalQkvzSplit) add(canonicalQkvzSplit, canonicalConv);
      add(canonicalConv, canonicalState);
      add(canonicalBeta, canonicalState);
      add(canonicalDecay, canonicalState);
      add(canonicalState, canonicalGateNorm);
      add(canonicalGateNorm, output);
      return edges.length >= 6 ? edges : null;
    }
    if (glmFused && glmSplit && glmQkvConv.length === 3 && glmState && glmOutputNorm) {
      add(glmFused, glmSplit);
      glmQkvConv.forEach((branch) => add(glmSplit, branch));
      glmQkvConv.forEach((branch) => add(branch, glmState));
      add(find(/forget gate projection/), glmState);
      add(find(/A_log decay parameter/), glmState);
      add(find(/dt bias parameter/), glmState);
      add(glmState, glmOutputNorm);
      add(find(/output gate projection/), glmOutputNorm);
      add(glmOutputNorm, output);
      return edges.length >= 7 ? edges : null;
    }
    if (qwenInputs.length >= 2 && qwenNorm) {
      qwenInputs.slice(0, -1).forEach((source, index) => add(source, qwenInputs[index + 1]));
      add(find(/in_proj_a/), conv);
      add(conv, qwenNorm);
      add(qwenNorm, output);
      return edges.length >= 3 ? edges : null;
    }
    if (kimiQkv.length >= 3) {
      kimiQkv.forEach((source) => add(source, conv));
      add(conv, state);
      add(find(/decay projection|b_proj/), state);
      add(state, outputGate || gate);
      add(outputGate || gate, output);
      return edges.length >= 3 ? edges : null;
    }
    add(qkv, conv);
    add(decay, state);
    add(conv, state);
    add(state, outputGate || gate);
    add(outputGate || gate, output);
    return edges.length >= 3 ? edges : null;
  }

  if (item.node?.attributes?.attention_kind === "qsa" || /qsa attention/.test(name)) {
    const dsaQueryDown = find(/query down projection/);
    if (dsaQueryDown) {
      const qNorm = find(/query latent rmsnorm/);
      const qUp = find(/query up projection/);
      const kvDown = find(/kv compression projection/);
      const kvSplit = find(/kv latent and rope split/);
      const kvNorm = find(/kv latent rmsnorm/);
      const kvUp = find(/kv expansion projection/);
      const rope = find(/rotary|rope/);
      const indexQ = find(/indexer query projection/);
      const indexWK = find(/indexer key and weight projection/);
      const indexNorm = find(/indexer key rmsnorm/);
      const indexer = find(/dsa indexer/);
      const sparse = find(/dsa sparse mla attention/);
      const output = find(/output projection|out_proj/);
      const edges = [];
      const add = (source, target) => {
        if (!source || !target || source.path === target.path) return;
        edges.push({ id: `${source.path}=>${target.path}`, source: source.path, target: target.path, kind: "dataflow", evidence: "semantic-flow" });
      };
      add(dsaQueryDown, qNorm);
      add(qNorm, qUp);
      add(qUp, rope);
      add(kvDown, kvNorm);
      add(kvDown, kvSplit);
      add(kvSplit, kvNorm);
      add(kvSplit, rope);
      add(kvNorm, kvUp);
      add(kvUp, rope);
      add(qNorm, indexQ);
      add(indexQ, indexer);
      add(indexWK, indexNorm);
      add(indexNorm, indexer);
      add(indexer, sparse);
      add(rope, sparse);
      add(sparse, output);
      return edges.length >= 8 ? edges : null;
    }
    const qkv = find(/qkv/);
    const qNorm = find(/q attention norm|q_norm/);
    const kNorm = find(/k attention norm|k_norm/);
    const rope = find(/rotary|rope/);
    const indexer = find(/qsa indexer|indexer/);
    const sparse = find(/qsa sparse|sparse attention/);
    const output = find(/output projection|out_proj/);
    const edges = [];
    const add = (source, target) => {
      if (!source || !target || source.path === target.path) return;
      edges.push({ id: `${source.path}=>${target.path}`, source: source.path, target: target.path, kind: "dataflow", evidence: "semantic-flow" });
    };
    add(qkv, qNorm);
    add(qkv, kNorm);
    add(qNorm, rope);
    add(kNorm, rope);
    add(indexer, sparse);
    add(rope, sparse);
    add(sparse, output);
    return edges.length >= 3 ? edges : null;
  }

  if (item.node?.attributes?.attention_kind === "dsv4") {
    const fused = find(/fused q\/kv projection/);
    const split = find(/q\/kv latent split/);
    const qNorm = find(/query latent rmsnorm/);
    const kvNorm = find(/kv latent rmsnorm/);
    const qProj = find(/query expansion projection/);
    const rope = find(/query\/kv rotary/);
    const compressor = find(/compressed kv\/state compressor/);
    const indexer = find(/indexer$/);
    const attention = find(/sparse mla attention|compressed mla attention|sliding-window mqa/);
    const inverseRope = find(/inverse output rotary/);
    const woA = find(/output low-rank projection/);
    const woB = find(/output hidden projection/);
    const edges = [];
    const add = (source, target) => {
      if (!source || !target || source.path === target.path) return;
      edges.push({ id: `${source.path}=>${target.path}`, source: source.path, target: target.path, kind: "dataflow", evidence: "semantic-flow" });
    };
    add(fused, split);
    add(split, qNorm);
    add(split, kvNorm);
    add(qNorm, qProj);
    add(qProj, rope);
    add(kvNorm, rope);
    add(compressor, attention);
    add(indexer, attention);
    add(rope, attention);
    add(attention, inverseRope);
    add(inverseRope, woA);
    add(woA, woB);
    return edges.length >= 6 ? edges : null;
  }

  if (item.node?.attributes?.attention_kind === "qwen35_full") {
    const fused = find(/fused qkv \+ attention gate projection/);
    const split = find(/qkv \+ gate split/);
    const qNorm = find(/q attention .*rmsnorm/);
    const kNorm = find(/k attention .*rmsnorm/);
    const rope = find(/rotary|rope/);
    const scores = find(/attention scores/);
    const probabilities = find(/attention probabilities|softmax/);
    const context = find(/weighted value/);
    const gate = find(/attention output gate/);
    const output = find(/output projection|out_proj/);
    const edges = [];
    const add = (source, target) => {
      if (!source || !target || source.path === target.path) return;
      edges.push({ id: `${source.path}=>${target.path}`, source: source.path, target: target.path, kind: "dataflow", evidence: "semantic-flow" });
    };
    add(fused, split);
    add(split, qNorm);
    add(split, kNorm);
    add(qNorm, rope);
    add(kNorm, rope);
    add(rope, scores);
    add(scores, probabilities);
    add(probabilities, context);
    add(context, gate);
    add(gate, output);
    return edges.length >= 7 ? edges : null;
  }

  if (type !== "attention" && !/(^|\b)(mla|multi.?head|attention)(\b|$)/.test(name)) return null;
  const q = find(/(^|\b)q\s*(projection|proj)\b|q_proj|query/);
  const k = find(/(^|\b)k\s*(projection|proj)\b|k_proj|key/);
  const v = find(/(^|\b)v\s*(projection|proj)\b|v_proj|value/);
  const rotary = find(/rotary|rope|position/);
  const scores = find(/attention\s*scores?|qk|score/);
  const probabilities = find(/attention\s*probabilities?|probabilities|softmax/);
  const weighted = find(/weighted\s*value|value\s*matmul|av/);
  const output = find(/output\s*projection|o\s*projection|out_proj/);
  const edges = [];
  const add = (source, target) => {
    if (!source || !target || source.path === target.path) return;
    edges.push({
      id: `${source.path}=>${target.path}`,
      source: source.path,
      target: target.path,
      kind: "dataflow",
      evidence: "semantic-flow",
    });
  };
  add(q, rotary);
  add(k, rotary);
  add(rotary, scores);
  add(scores, probabilities);
  add(probabilities, weighted);
  add(v, weighted);
  add(weighted, output);
  return edges.length >= 3 ? edges : null;
}

function isOutputNode(node) {
  const type = String(node?.node?.type || "").toLowerCase();
  const name = String(node?.node?.name || "").toLowerCase();
  return type === "output" || type === "head" || /(^|[._ -])(lm[_ -]?head|classifier|score)$/.test(name);
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
  const semanticParents = new Set(items.filter((item) => semanticEdges(item)).map((item) => item.path));
  const orderedPairs = new Set(items.flatMap((item) => {
    const children = item.childItems || [];
    return children.slice(0, -1).map((source, index) => `${source.path}=>${children[index + 1].path}`);
  }));
  const moduleOrderEdges = items.flatMap((item) => {
    if (semanticParents.has(item.path)) return [];
    const children = item.childItems || [];
    return children.slice(0, -1).map((source, index) => {
      const target = children[index + 1];
      // modelmap treats lm_head/classifier as an output sibling of the model
      // container. Preserve that semantic boundary instead of connecting the
      // last backbone module directly to the output head.
      const edgeSource = isOutputNode(target) ? item.path : source.path;
      return {
        id: `${edgeSource}~${target.path}`,
        source: edgeSource,
        target: target.path,
        kind: "dataflow",
        evidence: "module-order",
      };
    });
  });
  const dataflowEdges = items.flatMap((item) => {
    const semantic = semanticEdges(item);
    if (semantic) return semantic;
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
  const edges = [...dataflowEdges, ...missingOrderEdges];
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
