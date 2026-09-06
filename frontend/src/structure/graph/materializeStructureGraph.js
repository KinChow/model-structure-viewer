import { materializeDeclaredEdges } from "./declaredEdges.js";

function legacySemanticEdges(item) {
  if (!item?.childItems?.length) return null;
  const declared = materializeDeclaredEdges(item);
  if (declared) return declared;
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

  if (item.node?.attributes?.model_variant === "minimax_m3_vl") {
    const fused = find(/fused qkv \+ index projection|qkv projection/);
    const split = find(/main\/index qkv split|qkv split/);
    const qNorm = find(/^q gemma rmsnorm$/);
    const kNorm = find(/^k gemma rmsnorm$/);
    const rope = find(/^partial rotary position embedding$/);
    const scores = find(/attention scores/);
    const softmax = find(/attention probabilities|softmax/);
    const context = find(/weighted value/);
    const indexQNorm = find(/index q gemma rmsnorm/);
    const indexKNorm = find(/index k gemma rmsnorm/);
    const indexRope = find(/index partial rotary/);
    const indexer = find(/minimax m3 block indexer/);
    const sparse = find(/minimax m3 block-sparse gqa/);
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
    if (indexer) {
      add(split, indexQNorm);
      add(split, indexKNorm);
      add(indexQNorm, indexRope);
      add(indexKNorm, indexRope);
      add(indexRope, indexer);
      add(rope, sparse);
      add(indexer, sparse);
      add(sparse, output);
    } else {
      add(rope, scores);
      add(scores, softmax);
      add(softmax, context);
      add(context, output);
    }
    return edges.length >= 6 ? edges : null;
  }

  if (["minimax_m2", "glm4_moe"].includes(item.node?.attributes?.model_variant)) {
    const fused = find(/fused qkv projection/);
    const split = find(/qkv split/);
    const qNorm = find(/^q rmsnorm$/);
    const kNorm = find(/^k rmsnorm$/);
    const rope = find(/partial rotary/);
    const scores = find(/attention scores/);
    const softmax = find(/attention probabilities|softmax/);
    const context = find(/weighted value/);
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
    add(scores, softmax);
    add(softmax, context);
    add(context, output);
    return edges.length >= 7 ? edges : null;
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
  const mlaQueryDown = find(/query down projection/);
  const mlaQueryNorm = find(/query latent rmsnorm/);
  const mlaQueryUp = find(/query up projection/);
  const mlaKvDown = find(/kv compression projection/);
  const mlaKvSplit = find(/kv latent and rope split/);
  const mlaKvNorm = find(/kv latent rmsnorm/);
  const mlaKvUp = find(/kv expansion projection/);
  if (mlaQueryDown && mlaQueryNorm && mlaQueryUp && mlaKvDown && mlaKvSplit && mlaKvNorm && mlaKvUp) {
    const rotary = find(/rotary|rope/);
    const scores = find(/attention scores|latent attention scores/);
    const probabilities = find(/attention probabilities|softmax/);
    const weighted = find(/weighted value/);
    const output = find(/output projection|out_proj/);
    const outputGate = find(/mla output gate/);
    const edges = [];
    const add = (source, target) => {
      if (!source || !target || source.path === target.path) return;
      edges.push({ id: `${source.path}=>${target.path}`, source: source.path, target: target.path, kind: "dataflow", evidence: "semantic-flow" });
    };
    add(mlaQueryDown, mlaQueryNorm);
    add(mlaQueryNorm, mlaQueryUp);
    add(mlaKvDown, mlaKvSplit);
    add(mlaKvSplit, mlaKvNorm);
    add(mlaKvNorm, mlaKvUp);
    add(mlaQueryUp, rotary);
    add(mlaKvUp, rotary);
    add(rotary, scores);
    add(scores, probabilities);
    add(probabilities, weighted);
    add(weighted, outputGate || output);
    add(outputGate, output);
    return edges.length >= 8 ? edges : null;
  }
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

// Production graph materialization is declaration-only. The former
// display-name matcher remains isolated for historical comparisons while
// callers migrate their fixtures/builders to dataflow_edges.
function semanticEdges(item) {
  return materializeDeclaredEdges(item);
}


function isOutputNode(item) {
  const type = String(item?.node?.type || "").toLowerCase();
  const name = String(item?.node?.name || "").toLowerCase();
  return type === "output" || type === "head" || /(^|[._ -])(lm[_ -]?head|classifier|score)$/.test(name);
}

function flattenTree(root) {
  const items = [];
  function visit(node, path, parentId = null) {
    const item = { node, path, parentId, childItems: [] };
    items.push(item);
    item.childItems = (node?.children || []).map((child, index) =>
      visit(child, `${path}.${index}`, path));
    return item;
  }
  if (root) visit(root, "root");
  return items;
}

export function materializeStructureGraph(root) {
  const items = flattenTree(root);
  const semanticParents = new Set(items.filter((item) => semanticEdges(item)).map((item) => item.path));
  const orderedPairs = new Set(items.flatMap((item) =>
    item.childItems.slice(0, -1).map((source, index) =>
      `${source.path}=>${item.childItems[index + 1].path}`)));

  const moduleOrderEdges = items.flatMap((item) => {
    if (semanticParents.has(item.path)) return [];
    return item.childItems.slice(0, -1).map((source, index) => {
      const target = item.childItems[index + 1];
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
    const operators = item.childItems.filter((child) => child.node?.type === "operator");
    const edges = [];
    for (const source of operators) {
      if (!source.node?.output_shape) continue;
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
    }
    return edges;
  });

  const dataflowPairs = new Set(dataflowEdges.map((edge) => `${edge.source}=>${edge.target}`));
  return {
    version: 2,
    schema_version: 2,
    root_id: "root",
    nodes: items.map((item) => ({
      id: item.path,
      canonical_id: item.node?.id || item.path,
      module_id: item.node?.id || null,
      parent_id: item.parentId,
      order: item.parentId == null ? 0 : Number(item.path.split(".").at(-1)),
      name: item.node?.name || "",
      type: item.node?.type || "module",
      repeat: item.node?.repeat ?? null,
      attributes: item.node?.attributes || {},
      source_fields: item.node?.source_fields || [],
      confidence: item.node?.confidence || "high",
      params: item.node?.params ?? null,
      weight_shapes: item.node?.weight_shapes || null,
      dtype: item.node?.dtype || null,
      input_shape: item.node?.input_shape || null,
      output_shape: item.node?.output_shape || null,
      value_source: item.node?.value_source || null,
      tensor_names: item.node?.tensor_names || null,
    })),
    edges: [
      ...dataflowEdges,
      ...moduleOrderEdges.filter((edge) => !dataflowPairs.has(`${edge.source}=>${edge.target}`)),
    ].map((edge) => {
      const sourceNode = items.find((item) => item.path === edge.source)?.node;
      const targetNode = items.find((item) => item.path === edge.target)?.node;
      return {
        ...edge,
        source_canonical_id: sourceNode?.id || edge.source,
        target_canonical_id: targetNode?.id || edge.target,
      };
    }),
  };
}
