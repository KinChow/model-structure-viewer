from model_structure_viewer.structure.repair import IntrospectionFailureKind, RepairContext
from model_structure_viewer.structure.repair.strategies.deepseek_import_compat import DeepSeekImportCompatStrategy
from model_structure_viewer.structure.repair.strategies.minimax_config_adapter import MiniMaxConfigAdapterStrategy


def _context(config, failure_kind, message, source=None):
    return RepairContext(
        config=config,
        source=source or {},
        local_dir=None,
        failure_kind=failure_kind,
        original_error=message,
    )


def test_deepseek_strategy_matches_only_known_import_compat_error():
    strategy = DeepSeekImportCompatStrategy()
    context = _context(
        {"model_type": "deepseek_v3", "architectures": ["DeepseekV3ForCausalLM"]},
        IntrospectionFailureKind.REMOTE_IMPORT_COMPAT,
        "cannot import name 'is_torch_fx_available'",
    )

    assert strategy.matches(context) is True


def test_deepseek_strategy_rejects_non_deepseek_model():
    strategy = DeepSeekImportCompatStrategy()
    context = _context(
        {"model_type": "llama", "architectures": ["LlamaForCausalLM"]},
        IntrospectionFailureKind.REMOTE_IMPORT_COMPAT,
        "cannot import name 'is_torch_fx_available'",
    )

    assert strategy.matches(context) is False


def test_deepseek_strategy_matches_kimi_remote_code_using_deepseek_module():
    strategy = DeepSeekImportCompatStrategy()
    context = _context(
        {
            "model_type": "kimi_k25",
            "architectures": ["KimiK25ForConditionalGeneration"],
            "auto_map": {"AutoModel": "modeling_deepseek.DeepseekV3Model"},
        },
        IntrospectionFailureKind.REMOTE_IMPORT_COMPAT,
        "cannot import name 'is_torch_fx_available'",
        source={"model_id": "moonshotai/Kimi-K2.5"},
    )

    assert strategy.matches(context) is True


def test_deepseek_strategy_matches_kimi_with_deepseek_helper_file(tmp_path):
    strategy = DeepSeekImportCompatStrategy()
    (tmp_path / "modeling_deepseek.py").write_text("# helper\n", encoding="utf-8")
    context = RepairContext(
        config={
            "model_type": "kimi_k25",
            "architectures": ["KimiK25ForConditionalGeneration"],
            "auto_map": {"AutoModel": "modeling_kimi_k25.KimiK25ForConditionalGeneration"},
        },
        source={"model_id": "moonshotai/Kimi-K2.5"},
        local_dir=tmp_path,
        failure_kind=IntrospectionFailureKind.REMOTE_IMPORT_COMPAT,
        original_error="cannot import name 'is_torch_fx_available'",
    )

    assert strategy.matches(context) is True


def test_deepseek_strategy_reports_diagnostics_without_mutating_config():
    strategy = DeepSeekImportCompatStrategy()
    config = {"model_type": "deepseek_v3", "architectures": ["DeepseekV3ForCausalLM"]}
    context = _context(
        config,
        IntrospectionFailureKind.REMOTE_IMPORT_COMPAT,
        "cannot import name 'is_torch_fx_available'",
    )

    result = strategy.apply(context)

    assert result.config == config
    assert result.config is not config
    assert result.runtime_patch is not None
    assert result.runtime_patch.name == "deepseek_torch_fx_compat"
    assert result.diagnostics == {
        "repair_strategy": "deepseek_import_compat",
        "repair_status": "prepared",
        "compat_symbol": "is_torch_fx_available",
        "runtime_patch": "deepseek_torch_fx_compat",
    }


def test_minimax_strategy_matches_temporal_patch_size_error():
    strategy = MiniMaxConfigAdapterStrategy()
    context = _context(
        {"model_type": "minimax_m3", "architectures": ["MiniMaxM3ForCausalLM"]},
        IntrospectionFailureKind.CONFIG_FIELD_MISSING,
        "object has no attribute 'temporal_patch_size'",
    )

    assert strategy.matches(context) is True


def test_minimax_strategy_copies_nested_temporal_patch_size_to_top_level():
    strategy = MiniMaxConfigAdapterStrategy()
    config = {
        "model_type": "minimax_m3",
        "architectures": ["MiniMaxM3ForCausalLM"],
        "img_token_compression_config": {"temporal_patch_size": 4},
    }
    context = _context(
        config,
        IntrospectionFailureKind.CONFIG_FIELD_MISSING,
        "object has no attribute 'temporal_patch_size'",
    )

    result = strategy.apply(context)

    assert "temporal_patch_size" not in config
    assert result.config["temporal_patch_size"] == 4
    assert result.config_overrides == {"temporal_patch_size": 4}
    assert result.config_normalizer is not None
    assert result.config_normalizer.name == "minimax_config_normalizer"
    assert result.diagnostics == {
        "repair_strategy": "minimax_config_adapter",
        "repair_status": "prepared",
        "patched_fields": ["temporal_patch_size"],
        "config_overrides": ["temporal_patch_size"],
        "config_normalizer": "minimax_config_normalizer",
    }


def test_minimax_strategy_reports_no_patch_when_nested_value_missing():
    strategy = MiniMaxConfigAdapterStrategy()
    context = _context(
        {"model_type": "minimax_m3", "architectures": ["MiniMaxM3ForCausalLM"]},
        IntrospectionFailureKind.CONFIG_FIELD_MISSING,
        "object has no attribute 'temporal_patch_size'",
    )

    result = strategy.apply(context)

    assert "temporal_patch_size" not in result.config
    assert result.config_overrides == {}
    assert result.diagnostics == {
        "repair_strategy": "minimax_config_adapter",
        "repair_status": "skipped",
        "reason": "temporal_patch_size_not_found",
    }


def test_minimax_strategy_derives_m2_rope_parameters():
    strategy = MiniMaxConfigAdapterStrategy()
    config = {
        "model_type": "minimax_m2",
        "architectures": ["MiniMaxM2ForCausalLM"],
        "rope_theta": 5000000,
    }
    context = _context(
        config,
        IntrospectionFailureKind.CONFIG_FIELD_MISSING,
        "object has no attribute 'rope_parameters'",
    )

    result = strategy.apply(context)

    assert "rope_parameters" not in config
    assert result.config["rope_parameters"] == {
        "rope_type": "default",
        "rope_theta": 5000000,
    }
    assert result.config_overrides == {
        "rope_parameters": {
            "rope_type": "default",
            "rope_theta": 5000000,
        }
    }
    assert result.config_normalizer is not None
    assert result.config_normalizer.name == "minimax_m2_config_normalizer"
    assert result.diagnostics == {
        "repair_strategy": "minimax_config_adapter",
        "repair_status": "prepared",
        "patched_fields": ["rope_parameters"],
        "config_overrides": ["rope_parameters"],
        "config_normalizer": "minimax_m2_config_normalizer",
    }
