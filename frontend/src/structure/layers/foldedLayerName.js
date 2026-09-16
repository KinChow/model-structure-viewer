// 折叠重复层的命名：对标 PyTorch `nn.ModuleList.__repr__` 的压缩约定
// （torch/nn/modules/container.py：单块 `(i)`，多块 `(start-end): N x Block`）。
// 聚合层节点代表一整段被 compactRanges 合并的连续层，其 name 必须体现区间，
// 而不是只写起始层号，否则画布/Inspector/面包屑/搜索都会误显示成单层。
// 单一实现，供 decoderStack / vision 等所有折叠层构造点复用，避免各写一套再次跑偏。
export function foldedLayerName(start, end, label) {
  return start === end ? `${start} (${label})` : `${start}\u2013${end} (${label})`;
}
