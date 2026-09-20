// nv_tp_projection.mjs —— 前端并行策略（TP）投影 dump：projectPlan/kvBytesPerCard/ringAllReduceBytes。
// 供与真实 SGLang TP=2 每卡 weight/KV 对账（线 B）。复用生产 API（cost/parallel.js、cost/comm.js）。
import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { buildStructureFromConfig } from "../frontend/src/structure/buildStructure.js";
import { normalizeConfig } from "../frontend/src/structure/config/normalize.js";
import { projectPlan, kvBytesPerCard } from "../frontend/src/cost/parallel.js";
import { ringAllReduceBytes } from "../frontend/src/cost/comm.js";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const cfgPath = process.argv[2] || "/ssd2/models/Qwen/Qwen3-0.6B/config.json";
const raw = JSON.parse(await fs.readFile(cfgPath, "utf8"));
const cfg = normalizeConfig(raw);
const graph = buildStructureFromConfig(raw, { modelId: "Qwen/Qwen3-0.6B", source: "nv-tp" }).graph;

// 名义总量：权重按 graph 自然量；KV 按 layers×kvHeads×headDim×2(K+V)×2B × seq
const SEQ = 4096;
const perTokKv = cfg.layers * (cfg.kvHeads || cfg.attentionHeads) * cfg.headDim * 2 * 2;
const totalKv = perTokKv * SEQ;

const out = { model: "Qwen/Qwen3-0.6B", config: { layers: cfg.layers, hidden: cfg.hiddenSize, attnHeads: cfg.attentionHeads, kvHeads: cfg.kvHeads, headDim: cfg.headDim, ffn: cfg.intermediateSize }, perTokenKvBytes: perTokKv, plans: [] };
for (const tp of [1, 2, 4, 8]) {
  const plan = { tp, pp: 1, dp: 1 };
  const proj = projectPlan({ graph, kvBytes: totalKv, config: cfg, plan });
  const weightPerCard = proj.stages.reduce((s, st) => s + st.weightBytes, 0);
  const kv = kvBytesPerCard(totalKv, cfg, plan);
  // 每层两次 all-reduce（o_proj + down_proj 后），B=1,T=SEQ,H=hidden,bf16
  const allreduce = ringAllReduceBytes({ batch: 1, tokens: SEQ, hidden: cfg.hiddenSize, bytesPerElement: 2, tp, operations: 2 }) * cfg.layers;
  out.plans.push({
    tp,
    weightBytesPerCard: weightPerCard,
    weightPerCardRatioVsTp1: null,
    kvShardFactor: kv.shardFactor,
    kvBytesPerCard: kv.bytes,
    allReduceBytesTotal: allreduce,
  });
}
const base = out.plans[0].weightBytesPerCard;
for (const p of out.plans) p.weightPerCardRatioVsTp1 = base ? p.weightBytesPerCard / base : null;

const dir = path.join(repoRoot, "docs/details/nv_evidence/nv2/tp");
await fs.mkdir(dir, { recursive: true });
await fs.writeFile(path.join(dir, "frontend_tp_projection.json"), JSON.stringify(out, null, 2));
for (const p of out.plans) {
  console.log(`tp=${p.tp}: weight/card=${(p.weightBytesPerCard / 1e6).toFixed(1)}MB (×${p.weightPerCardRatioVsTp1.toFixed(3)}) kvShard=${p.kvShardFactor} kv/card=${(p.kvBytesPerCard / 1e9).toFixed(2)}GB allreduce=${(p.allReduceBytesTotal / 1e6).toFixed(1)}MB`);
}
