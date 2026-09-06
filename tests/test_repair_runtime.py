from contextlib import contextmanager

import pytest

from model_structure_viewer.structure.repair import RepairResult
from model_structure_viewer.structure.repair.runtime import NoopRuntimePatch
from model_structure_viewer.structure.repair.strategies.deepseek_import_compat import (
    DeepSeekTorchFxCompatPatch,
)
from model_structure_viewer.structure.repair.compat import (
    CompositeRuntimePatch,
    KimiRemoteCodeCompatPatch,
    is_kimi_output_recorder_import_error,
)


class _ContextPatch:
    def __init__(self, name, events, *, fail=False):
        self.name = name
        self.events = events
        self.fail = fail

    @contextmanager
    def activate(self):
        self.events.append(f"enter:{self.name}")
        if self.fail:
            raise RuntimeError(self.name)
        try:
            yield
        finally:
            self.events.append(f"exit:{self.name}")


def test_noop_runtime_patch_can_be_used_as_context_manager():
    patch = NoopRuntimePatch()

    with patch.activate():
        value = "active"

    assert patch.name == "noop"
    assert value == "active"


def test_composite_runtime_patch_unwinds_when_later_enter_fails():
    events = []
    patch = CompositeRuntimePatch(
        _ContextPatch("first", events),
        _ContextPatch("second", events, fail=True),
    )

    with pytest.raises(RuntimeError, match="second"):
        with patch.activate():
            raise AssertionError("unreachable")

    assert events == ["enter:first", "enter:second", "exit:first"]


def test_repair_result_defaults_without_config_normalizer():
    result = RepairResult(config={}, local_dir=None, strategy_name="demo")

    assert result.config_normalizer is None


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


def test_kimi_remote_code_patch_restores_output_recorder_alias(monkeypatch):
    import transformers.modeling_utils as modeling_utils
    import transformers.utils.generic as generic_utils

    monkeypatch.delattr(generic_utils, "OutputRecorder", raising=False)
    patch = KimiRemoteCodeCompatPatch()

    with patch.activate():
        assert generic_utils.OutputRecorder is modeling_utils.OutputRecorder

    assert not hasattr(generic_utils, "OutputRecorder")


def test_kimi_output_recorder_error_is_classified():
    error = ImportError(
        "cannot import name 'OutputRecorder' from 'transformers.utils.generic'"
    )
    assert is_kimi_output_recorder_import_error(error)
