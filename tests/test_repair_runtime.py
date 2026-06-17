from model_structure_viewer.structure.repair.runtime import NoopRuntimePatch
from model_structure_viewer.structure.repair.strategies.deepseek_import_compat import (
    DeepSeekTorchFxCompatPatch,
)


def test_noop_runtime_patch_can_be_used_as_context_manager():
    patch = NoopRuntimePatch()

    with patch.activate():
        value = "active"

    assert patch.name == "noop"
    assert value == "active"


def test_deepseek_patch_adds_missing_torch_fx_symbol(monkeypatch):
    import transformers.utils.import_utils as import_utils

    monkeypatch.delattr(import_utils, "is_torch_fx_available", raising=False)
    patch = DeepSeekTorchFxCompatPatch()

    with patch.activate():
        assert import_utils.is_torch_fx_available() is False


def test_deepseek_patch_restores_original_torch_fx_symbol(monkeypatch):
    import transformers.utils.import_utils as import_utils

    original = lambda: True
    monkeypatch.setattr(import_utils, "is_torch_fx_available", original, raising=False)
    patch = DeepSeekTorchFxCompatPatch()

    with patch.activate():
        assert import_utils.is_torch_fx_available is original

    assert import_utils.is_torch_fx_available is original


def test_deepseek_patch_removes_symbol_when_it_created_it(monkeypatch):
    import transformers.utils.import_utils as import_utils

    monkeypatch.delattr(import_utils, "is_torch_fx_available", raising=False)
    patch = DeepSeekTorchFxCompatPatch()

    with patch.activate():
        assert hasattr(import_utils, "is_torch_fx_available")

    assert not hasattr(import_utils, "is_torch_fx_available")
