// Qwen3.8-Flash-Next report §2.2 Eq(30–34). Logical fused-boundary
// traffic, not a claim about materialized intermediates or kernel HBM reads.
// The read owns norm/down/up and (optionally) write-gate prediction weights.
// The separate write owns only broadcast multiply + residual add.
export function gatedResidualCounts({ tokens: T, hidden: H, streams: N, lowrank: R,
  b, useCombine = true, stage = "read" }) {
  const W = N * H, E = T * W;
  if (stage === "write") return {
    matrix: 0, vector: 2 * E, sfu: 0,
    bytes: { weights: 0, actIn: T * (W + H + N) * b, actOut: E * b },
  };
  // Normalization: square, H-1 sum, mean scale, epsilon, rsqrt,
  // normalize, affine. Precompute (1+gamma) once per invocation.
  // Read: scale low-rank preactivation, SiLU, sigmoid, gated mean.
  // Write-gate prediction: scale, sigmoid, scale by two.
  return {
    matrix: T * (2 * W * R + (useCombine ? N * W : 0)),
    vector: T * (4 * W + N) + W + 2 * T * R + 2 * E
      + (useCombine ? 2 * T * N : 0),
    sfu: T * N + 2 * T * R + 2 * E + (useCombine ? 2 * T * N : 0),
    bytes: {
      weights: (W + 2 * W * R + (useCombine ? N * W : 0)) * b,
      actIn: E * b,
      actOut: T * (H + (useCombine ? N : 0)) * b,
    },
  };
}

// Independent primitive decomposition (not derived from the closed form).
export function gatedResidualDecomposition(p) {
  const { tokens: T, hidden: H, streams: N, lowrank: R, b } = p;
  const W = N * H, E = T * W;
  const op = (atom, elements, extra = {}) => ({ atom, args: { elements, bytesPerElement: b, ...extra } });
  const mm = (k, n) => ({ atom: "matmul", args: { m: T, k, n, rhs: "weight", bytesPerElement: b } });
  if (p.stage === "write") return [op("mul", E), op("add", E)];
  return [
    op("mul", E), op("reduce_sum", E, { groups: T * N }),
    op("mul", T * N), op("add", T * N), op("rsqrt", T * N),
    op("mul", E), op("add", W), op("mul", E, { weightElements: W }),
    mm(W, R), op("mul", T * R), op("silu", T * R), mm(R, W),
    op("sigmoid", E), op("mul", E),
    op("reduce_sum", E, { groups: T * H }), op("mul", T * H),
    ...(p.useCombine === false ? [] : [
      mm(W, N), op("mul", T * N), op("sigmoid", T * N), op("mul", T * N),
    ]),
  ];
}
