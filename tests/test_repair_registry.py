from model_structure_viewer.structure.repair import IntrospectionFailureKind, RepairContext
from model_structure_viewer.structure.repair.registry import find_strategy


def _context(config, failure_kind, message="failed"):
    return RepairContext(
        config=config,
        source={},
        local_dir=None,
        failure_kind=failure_kind,
        original_error=message,
    )


def test_finds_deepseek_import_compat_strategy():
    context = _context(
        {"model_type": "deepseek_v3", "architectures": ["DeepseekV3ForCausalLM"]},
        IntrospectionFailureKind.REMOTE_IMPORT_COMPAT,
        "cannot import name 'is_torch_fx_available'",
    )

    strategy = find_strategy(context)

    assert strategy is not None
    assert strategy.name == "deepseek_import_compat"


def test_finds_minimax_config_adapter_strategy():
    context = _context(
        {"model_type": "minimax_m3", "architectures": ["MiniMaxM3ForCausalLM"]},
        IntrospectionFailureKind.CONFIG_FIELD_MISSING,
        "object has no attribute 'temporal_patch_size'",
    )

    strategy = find_strategy(context)

    assert strategy is not None
    assert strategy.name == "minimax_config_adapter"


def test_returns_none_for_unmatched_model():
    context = _context(
        {"model_type": "llama", "architectures": ["LlamaForCausalLM"]},
        IntrospectionFailureKind.REMOTE_IMPORT_COMPAT,
        "cannot import name 'is_torch_fx_available'",
    )

    assert find_strategy(context) is None
