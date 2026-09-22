// 从已归档的真实 shape/spec 重放公式对账，不起服务、不修改历史工件，
// 不把 runtime 字节注入产品。运行：node scripts/evidence/runtime/reconcile-accounting.mjs
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { buildStructureFromConfig } from "../../../frontend/src/structure/buildStructure.js";
import { normalizeConfig } from "../../../frontend/src/structure/config/normalize.js";
import { buildCostAccounting, bytesPerDtype } from "../../../frontend/src/cost/memory.js";

const root = fileURLToPath(new URL("../../../", import.meta.url));
// Python 标准库只读取白名单成员，不 extractall，也不执行归档内的代码。
const extract = String.raw`
import json, tarfile
base = "artifacts/framework-runtime-validation/20260921/"
with tarfile.open(base + "a100-evidence-final2.tar.gz") as t:
    config = json.load(t.extractfile("configs/a100-qwen35.json"))
    records = [json.loads(line) for line in t.extractfile("a100-sglang-qwen35-mtp/capture-178.jsonl")]
    mamba = next(e["allocation"] for e in records if e["event"] == "MambaPool.__init__")
    ssm = mamba["mamba_cache"]["intermediate_ssm"]
    conv = mamba["_intermediate_conv_window_phys"]
    # 同一进程的相同 storage 只计一次；重叠 as_strided view 不再相加。
    storage = {v["storage_ptr"]: v["storage_bytes"] for v in [ssm, *conv]}
    scratch = {"config": config, "draftTokens": ssm["shape"][2],
               "effectiveStateSlots": ssm["shape"][1] - 1,
               "ssmBytes": ssm["bytes"], "convBytes": sum(v["storage_bytes"] for v in conv),
               "uniqueBytes": sum(storage.values())}
with tarfile.open(base + "h20-evidence-final.tar.gz") as t:
    config = json.load(t.extractfile("configs/h20-glm5-reduced.json"))
    records = [json.loads(line) for line in t.extractfile("h20-vllm-glm5-dsa-kda/capture-1380.jsonl")]
    cache = next(e["allocation"]["kv_cache_config"] for e in records
                 if e["event"] == "GPUModelRunner.initialize_kv_cache")
    specs = [s for g in cache["kv_cache_groups"]
             for s in g["kv_cache_spec"].get("kv_cache_specs", {}).values()
             if s.get("cache_role") in ("indexer", "sparse")]
    dsa = {"config": config, "specs": specs}
print(json.dumps({"scratch": scratch, "dsa": dsa}))
`;
const evidence = JSON.parse(execFileSync("python3", ["-c", extract], { cwd: root, encoding: "utf8" }));
const accountingFor = (raw, frameworkProfile, speculative = {}) => buildCostAccounting({
  graph: buildStructureFromConfig(raw, { frameworkProfile }).graph,
  config: normalizeConfig(raw), frameworkProfile, speculative,
});
const { scratch, dsa } = evidence;
const speculative = { enabled: true, draftTokens: scratch.draftTokens, stateSlots: scratch.effectiveStateSlots };
const sglang = accountingFor(scratch.config, "sglang", speculative);
assert.equal(sglang.totalSpeculativeStateBytes, scratch.uniqueBytes);
assert.equal(sglang.draft.speculativeStateBytes, 0);
assert.equal(accountingFor(scratch.config, "sglang", {
  ...speculative, disaggregationMode: "prefill",
}).totalSpeculativeStateBytes, 0);
const runtimeGrowth = dsa.specs.reduce((sum, spec) => sum
  + spec.num_kv_heads * spec.head_size * bytesPerDtype(spec.dtype) / spec.tokens_per_state, 0);
const vllm = accountingFor(dsa.config, "vllm");
assert.equal(vllm.main.kvBytesPerToken, runtimeGrowth);
console.log(JSON.stringify({
  validation: "replay of archived runtime tensor metadata, not a new GPU run",
  sglang: {
    draftTokens: scratch.draftTokens, effectiveStateSlots: scratch.effectiveStateSlots,
    runtimeSsmBytes: scratch.ssmBytes, runtimeConvUniqueBytes: scratch.convBytes,
    runtimeScratchBytes: scratch.uniqueBytes,
    estimatedScratchBytes: sglang.totalSpeculativeStateBytes,
    match: true,
  },
  vllm: {
    runtimeDsaAndMlaBytesPerToken: runtimeGrowth,
    estimatedBytesPerToken: vllm.main.kvBytesPerToken,
    match: true,
  },
}, null, 2));
