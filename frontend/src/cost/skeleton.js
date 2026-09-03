// skeleton.js —— safetensors key 建 trie → 精确含参模块树（v4 核心，纯函数，无网络/无 torch）。
//
// 依据 evolution_design.md §4.2(b2)：safetensors key 即 state_dict key 即 nn.Module 路径，
// 去掉参数名、按 "." 切分建 trie，即可得到任何模型的精确含参模块树。
//
// 边界说明：
// - 数字路径段（layers.0 / experts.3）识别为 ModuleList，折叠为 repeat 节点（结构一致才折叠）。
// - 同模块多参数（weight / qweight / qzeros / scales / bias）聚合成同一节点的 weight_shapes。
// - node.params = 本节点自有参数的元素计数（Σ prod(shape)），不含子树；
//   子字节量化张量按存储元素计数（真实参数量换算交给 @huggingface/hub，见 cost/weights.js）。
// - 参数量换算必须用库，这里只做结构建树与元素计数，不处理打包容器。

/**
 * @typedef {{name: string, dtype: string, shape: number[]}} TensorEntry
 * @typedef {{
 *   id: string, name: string, type: "module"|"list", repeat?: number,
 *   params: number, weight_shapes: Record<string, number[]>,
 *   weight_dtypes: Record<string, string>, dtype: string,
 *   tensor_names: string[], children: SkeletonNode[],
 * }} SkeletonNode
 */

/** 把一个完整张量名拆成 [模块路径段..., 参数名]。 */
function splitTensorName(name) {
  const parts = name.split(".");
  return { moduleParts: parts.slice(0, -1), paramName: parts[parts.length - 1] };
}

function createTrieNode() {
  return { children: new Map(), tensors: new Map() };
}

function buildTrie(tensors) {
  const root = createTrieNode();
  for (const t of tensors) {
    const { moduleParts, paramName } = splitTensorName(t.name);
    let node = root;
    for (const part of moduleParts) {
      if (!node.children.has(part)) node.children.set(part, createTrieNode());
      node = node.children.get(part);
    }
    node.tensors.set(paramName, { dtype: t.dtype, shape: t.shape });
  }
  return root;
}

/** 两个子树结构是否完全一致（张量参数名/shape/dtype + 子节点名与递归结构）。 */
function sameSubtree(a, b) {
  if (a.tensors.size !== b.tensors.size) return false;
  for (const [param, ta] of a.tensors) {
    const tb = b.tensors.get(param);
    if (!tb || tb.dtype !== ta.dtype || !shapesEqual(tb.shape, ta.shape)) return false;
  }
  if (a.children.size !== b.children.size) return false;
  for (const [name, ca] of a.children) {
    const cb = b.children.get(name);
    if (!cb || !sameSubtree(ca, cb)) return false;
  }
  return true;
}

function shapesEqual(a, b) {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}

/** 子节点键是否为 0..n-1 的连续数字段（ModuleList 特征）。 */
function numericSegmentInfo(childKeys) {
  if (childKeys.length < 2) return null;
  if (!childKeys.every((k) => /^\d+$/.test(k))) return null;
  const nums = childKeys.map(Number).sort((x, y) => x - y);
  for (let i = 0; i < nums.length; i++) if (nums[i] !== i) return null;
  return { count: nums.length };
}

/** 数字段折叠的一致性校验：各 index 子树必须同构，否则不折叠、保留展开。 */
function segmentFoldsCleanly(node) {
  const childKeys = [...node.children.keys()];
  const info = numericSegmentInfo(childKeys);
  if (!info) return false;
  const rep = node.children.get("0");
  for (const k of childKeys) {
    if (!sameSubtree(node.children.get(k), rep)) return false;
  }
  return true;
}

function dominantDtype(node) {
  let best = null;
  let bestCount = -1;
  for (const { dtype, shape } of node.tensors.values()) {
    const count = shape.reduce((a, b) => a * b, 1);
    if (count > bestCount) {
      bestCount = count;
      best = dtype;
    }
  }
  return best;
}

function convertNode(trieNode, path) {
  const id = path.join(".");
  const name = path[path.length - 1] ?? "root";
  const children = [...trieNode.children.entries()]
    .sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0))
    .map(([k, child]) => convertNode(child, [...path, k]));

  // 数字段折叠：连续 0..n-1 且各 index 同构 → repeat 节点，children 取 index 0 的子树
  const childKeys = [...trieNode.children.keys()];
  if (segmentFoldsCleanly(trieNode)) {
    const count = numericSegmentInfo(childKeys).count;
    const repNode = convertNode(trieNode.children.get("0"), [...path, "0"]);
    return {
      id,
      name,
      type: "list",
      repeat: count,
      params: 0,
      weight_shapes: {},
      weight_dtypes: {},
      dtype: null,
      tensor_names: [],
      children: [repNode],
    };
  }

  const weight_shapes = {};
  const weight_dtypes = {};
  const tensor_names = [];
  let params = 0;
  for (const [param, { dtype, shape }] of [...trieNode.tensors.entries()].sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0))) {
    weight_shapes[param] = shape;
    weight_dtypes[param] = dtype;
    tensor_names.push(`${id}.${param}`);
    params += shape.reduce((a, b) => a * b, 1);
  }

  return {
    id,
    name,
    type: "module",
    params,
    weight_shapes,
    weight_dtypes,
    dtype: dominantDtype(trieNode),
    tensor_names,
    children,
  };
}

/**
 * @param {TensorEntry[]} tensors 归一化后的张量列表（含 dtype/shape）
 * @returns {SkeletonNode} 折叠后的含参模块树
 */
export function buildSkeleton(tensors) {
  const trie = buildTrie(tensors);
  const root = convertNode(trie, []);
  // 单一顶层段（如 model）时去掉空路径包装，直接以该段为根
  if (root.children.length === 1 && root.tensor_names.length === 0 && root.children[0].type === "module") {
    return root.children[0];
  }
  return root;
}
