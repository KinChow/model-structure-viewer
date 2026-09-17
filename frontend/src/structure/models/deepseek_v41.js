// 对标随附 model.py（DeepSeek-V4.1）：主干 = V4 组网（dsv4 注意力 + sqrtsoftplus
// routed MoE + MHC + DSpark 投机头 + 视觉塔），差异只有两处：
//   1. 无 hash 层（config 无 num_hash_layers → numHashLayers=0，全部 sqrtsoftplus 路由）；
//   2. 每层入口按 engramLayerIds 注入 Engram（n-gram 哈希记忆），由 decoderLayer 挂接。
// 因此直接复用 assembleDeepseekV4，engram 走 normalize.engramLayerIds 驱动的 layer 钩子。
export { assembleDeepseekV4 as assembleDeepseekV41 } from "./deepseek_v4.js";
