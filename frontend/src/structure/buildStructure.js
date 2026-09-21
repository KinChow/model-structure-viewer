import { normalizeConfig } from "./config/normalize.js";
import { resolveArchitecture } from "./registry/resolveArchitecture.js";
import { buildNetwork } from "./models/index.js";
import { createStructureIr } from "./ir/createStructureIr.js";
import { materializeModelStructure } from "./materializers/modelStructure.js";
import { mtpTensorCount } from "../cost/weights.js";

// config 只声明「意图」，checkpoint 才是「实装真相」：num_nextn_predict_layers 等
// MTP 声明字段可能领先于权重（MiniMax-M3 / M2.7 config 声明 MTP，但发布权重里没有
// 任何 MTP 张量，vLLM/SGLang 亦未实装）。取 safetensors header 真值，供组网按实装
// 抑制幻影 MTP —— 对标 vLLM load_weights「扫 checkpoint key，不看 config.use_mtp」。
// 无真值（离线 golden / 取证失败）时返回 undefined，组网回退 config 行为（向后兼容）。
function checkpointMtpTensorCountFromTruth(truth, hiddenLayers) {
  if (!truth) return undefined;
  if (Number.isFinite(truth.mtp_tensor_count)) return truth.mtp_tensor_count;
  if (Array.isArray(truth.tensors)) return mtpTensorCount(truth.tensors, { hiddenLayers });
  return undefined;
}

// 框架预设（frameworkProfile）：把 vLLM≠SGLang 的运行时分叉做成可切换默认。
// 目前控制线性 recurrent(ssm/temporal) state dtype——vLLM 缺省 bf16(模型 dtype)、
// SGLang 缺省 fp32；neutral 保持"按 config、缺省 fp32"的既有行为。
// 优先级铁律：config 显式声明恒胜（仅在 config 未声明时才套预设默认）。
export const FRAMEWORK_PROFILES = ["neutral", "sglang", "vllm"];
const FRAMEWORK_SSM_DTYPE = { vllm: "bfloat16", sglang: "float32" };
export function applyFrameworkProfile(normalized, frameworkProfile) {
  if (!frameworkProfile || frameworkProfile === "neutral") return;
  const ssm = FRAMEWORK_SSM_DTYPE[frameworkProfile];
  // config 未显式声明 ssm dtype 时，才采用框架预设默认（config 显式恒胜）。
  if (ssm && normalized.mambaSsmDtype == null) normalized.mambaSsmDtype = ssm;
}

export function buildStructureFromConfig(config, options = {}) {
  const normalized = normalizeConfig(config);
  applyFrameworkProfile(normalized, options.frameworkProfile);
  const checkpointMtpTensorCount = checkpointMtpTensorCountFromTruth(options.truth, normalized.layers);
  if (checkpointMtpTensorCount !== undefined) {
    normalized.checkpointMtpTensorCount = checkpointMtpTensorCount;
  }
  const resolved = resolveArchitecture(normalized, options);
  const network = buildNetwork(resolved, normalized);
  const ir = createStructureIr({ network, normalized, resolved, options });
  return materializeModelStructure(ir);
}

export function buildStructureFromArtifacts(artifacts, options = {}) {
  return buildStructureFromConfig(artifacts.config, {
    modelId: artifacts.modelId,
    revision: artifacts.revision,
    source: artifacts.source?.kind || "model config",
    truth: artifacts.checkpointTruth,
    checkpointTruthStatus: artifacts.checkpointTruthStatus,
    checkpointTruthMethod: artifacts.checkpointTruth?.method || null,
    checkpointTruthError: artifacts.checkpointTruthError,
    configEndpoint: artifacts.configEndpoint,
    checkpointTruthEndpoint: artifacts.checkpointTruthEndpoint,
    sourceRef: artifacts.sourceRef || null,
    frameworkProfile: options.frameworkProfile,
  });
}
