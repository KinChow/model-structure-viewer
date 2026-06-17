from model_structure_viewer.errors import IntrospectionError
from model_structure_viewer.structure.repair import IntrospectionFailureKind, classify_introspection_error


def test_classifies_remote_code_unavailable():
    error = IntrospectionError(
        "Config requires custom remote code (auto_map) but no local model directory is available."
    )

    assert classify_introspection_error(error) == IntrospectionFailureKind.REMOTE_CODE_UNAVAILABLE


def test_classifies_remote_import_compatibility_error():
    error = IntrospectionError(
        "AutoModel.from_config failed: cannot import name 'is_torch_fx_available' "
        "from 'transformers.utils.import_utils'"
    )

    assert classify_introspection_error(error) == IntrospectionFailureKind.REMOTE_IMPORT_COMPAT


def test_classifies_missing_config_field_error():
    error = IntrospectionError(
        "AutoModel.from_config failed: 'PreTrainedConfig' object has no attribute 'temporal_patch_size'"
    )

    assert classify_introspection_error(error) == IntrospectionFailureKind.CONFIG_FIELD_MISSING


def test_classifies_config_load_failure():
    error = IntrospectionError("AutoConfig.from_pretrained failed: invalid config.json")

    assert classify_introspection_error(error) == IntrospectionFailureKind.CONFIG_LOAD_FAILED


def test_classifies_unknown_error():
    error = IntrospectionError("something unexpected happened")

    assert classify_introspection_error(error) == IntrospectionFailureKind.UNKNOWN
