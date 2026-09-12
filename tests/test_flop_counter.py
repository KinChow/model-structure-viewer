"""FlopCounterMode 矩阵抽查：msv MAC × 2 == torch FLOPs。

原则 §6.3：只对 flop_registry 覆盖的 mm/bmm/convolution 对账。
softmax / rmsnorm / rope 明确不对账（torch 记 0）。
"""
from model_structure_viewer.verification.flop_counter import (
    compare_bmm,
    compare_conv1d,
    compare_linear,
)


def test_flop_counter_linear_matches_msv_macs():
    report = compare_linear(tokens=3, inn=4, out=8)
    assert report["ok"], report
    assert report["msv_macs"] == 3 * 8 * 4
    assert report["torch_flops"] == 192


def test_flop_counter_bmm_matches_msv_macs():
    report = compare_bmm(heads=2, query_tokens=4, key_tokens=8, head_dim=16)
    assert report["ok"], report
    assert report["msv_macs"] == 2 * 4 * 8 * 16
    assert report["torch_flops"] == 2048


def test_flop_counter_depthwise_conv1d_matches_msv_macs():
    report = compare_conv1d(tokens=8, channels=4, kernel=3)
    assert report["ok"], report
    assert report["msv_macs"] == 8 * 4 * 3
