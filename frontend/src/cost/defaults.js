export const DEFAULT_PLAN = Object.freeze({ tp: 1, pp: 1, ep: 1, dp: 1, attnMode: "tp" });
export const DEFAULT_COMPARE_PLAN = Object.freeze({ tp: 2, ep: 1, attnMode: "tp" });
export const DEFAULT_LOADS = Object.freeze({
  prefill: Object.freeze({ batch: 1, sequence: 2048, chunked: false, chunkSize: 8192 }),
  decode: Object.freeze({ batch: 1, sequence: 2048 }),
});
export const DEFAULT_NODES = Object.freeze({ centralized: 1, prefill: 1, decode: 2 });
