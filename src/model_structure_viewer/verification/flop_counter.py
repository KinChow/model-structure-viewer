"""矩阵系 FLOP 抽查：msv MACs × 2 ↔ PyTorch FlopCounterMode。

原则 §6.3 / §3.1：flop_counter 只数 mm/bmm/addmm/convolution/SDPA。
msv ``counts.matrix`` 存 MAC；aten 公式含 2×，对账时 msv_macs * 2 == torch FLOPs。

本模块只提供独立算子夹具（Linear / BMM / Conv1d），**不**对 catalog 整模型
跑 forward——catalog 只有 config，没有可执行权重。整模型抽查要等真实
checkpoint + 输入，不能空 stub。

ref: torch.utils.flop_counter.flop_registry
     frontend/src/structure/operators/formulas/counts.js linearCounts / attentionCounts
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch.utils.flop_counter import FlopCounterMode


def torch_flops(fn) -> int:
    """跑一次 ``fn``，返回 FlopCounterMode 全局 FLOP 合计。"""
    flop = FlopCounterMode(display=False)
    with flop:
        fn()
    counts = flop.get_flop_counts().get("Global") or {}
    return int(sum(counts.values()))


def linear_macs(*, tokens: int, out: int, inn: int) -> int:
    """msv linearCounts.matrix = T · out · in（MAC）。"""
    return tokens * out * inn


def bmm_macs(*, heads: int, query_tokens: int, key_tokens: int, head_dim: int) -> int:
    """一组 scores BMM 的 MAC：H · T · S · D。aten.bmm FLOPs = 2 × MAC。"""
    return heads * query_tokens * key_tokens * head_dim


def conv1d_macs(*, tokens: int, channels: int, kernel: int, out_channels: int | None = None) -> int:
    """msv causalShortConvCounts.matrix = T · C · K（深度可分 / 每通道核）。

    标准 ``nn.Conv1d(C, C, K, groups=C)`` 与这条同形；普通 Conv1d(Cin, Cout, K)
    是 Cin·Cout·K·L，不是 msv 短卷积口径——对账夹具用 groups=C。
    """
    del out_channels
    return tokens * channels * kernel


def compare_linear(*, tokens: int = 3, inn: int = 4, out: int = 8) -> dict[str, Any]:
    """nn.Linear 无 bias：aten.mm FLOPs = 2 · T · out · in。"""
    weight = torch.randn(out, inn)
    x = torch.randn(tokens, inn)
    torch_total = torch_flops(lambda: x @ weight.T)
    msv = linear_macs(tokens=tokens, out=out, inn=inn)
    return _report("linear", torch_total, msv)


def compare_bmm(
    *,
    heads: int = 2,
    query_tokens: int = 4,
    key_tokens: int = 8,
    head_dim: int = 16,
) -> dict[str, Any]:
    """scores BMM：aten.bmm FLOPs = 2 · H · T · S · D。"""
    q = torch.randn(heads, query_tokens, head_dim)
    k = torch.randn(heads, head_dim, key_tokens)
    torch_total = torch_flops(lambda: torch.bmm(q, k))
    msv = bmm_macs(heads=heads, query_tokens=query_tokens, key_tokens=key_tokens, head_dim=head_dim)
    return _report("bmm", torch_total, msv)


def compare_conv1d(*, tokens: int = 8, channels: int = 4, kernel: int = 3) -> dict[str, Any]:
    """深度可分短卷积：aten.convolution FLOPs = 2 · T · C · K。

    因果短卷积实现是左 pad ``kernel-1`` 再 Conv1d，输出长度 = tokens。
    ``padding=kernel-1`` 两侧 pad 会把输出拉到 T+K-1，FLOP 对不上。
    """
    conv = nn.Conv1d(channels, channels, kernel_size=kernel, groups=channels, bias=False)
    x = torch.randn(1, channels, tokens)
    padded = torch.nn.functional.pad(x, (kernel - 1, 0))
    torch_total = torch_flops(lambda: conv(padded))
    msv = conv1d_macs(tokens=tokens, channels=channels, kernel=kernel)
    return _report("conv1d", torch_total, msv)


def _report(kind: str, torch_flops_total: int, msv_macs: int) -> dict[str, Any]:
    expected = msv_macs * 2
    return {
        "kind": kind,
        "torch_flops": torch_flops_total,
        "msv_macs": msv_macs,
        "msv_flops": expected,
        "ok": torch_flops_total == expected,
    }
