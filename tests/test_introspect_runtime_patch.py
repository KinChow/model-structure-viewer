from contextlib import contextmanager
from types import SimpleNamespace

from model_structure_viewer.structure import introspect


class RecordingPatch:
    name = "recording"

    def __init__(self):
        self.active = False
        self.was_active_during_call = False

    @contextmanager
    def activate(self):
        self.active = True
        try:
            yield
        finally:
            self.active = False


def test_load_config_applies_overrides_to_local_config(monkeypatch, tmp_path):
    class AutoConfig:
        @staticmethod
        def from_pretrained(path, trust_remote_code):
            return SimpleNamespace(model_type="minimax_m3")

    loaded = introspect._load_config(
        AutoConfig,
        {"model_type": "minimax_m3"},
        tmp_path,
        config_overrides={"temporal_patch_size": 4},
    )

    assert loaded.temporal_patch_size == 4


def test_build_from_meta_model_activates_runtime_patch(monkeypatch):
    patch = RecordingPatch()

    class AutoConfig:
        @staticmethod
        def for_model(model_type, **kwargs):
            return SimpleNamespace(model_type=model_type)

    class AutoModel:
        @staticmethod
        def from_config(config, trust_remote_code):
            patch.was_active_during_call = patch.active
            return SimpleNamespace(named_children=lambda: [])

    @contextmanager
    def init_empty_weights():
        yield

    monkeypatch.setattr(introspect, "_import_introspection_deps", lambda: (AutoConfig, AutoModel, init_empty_weights))

    structure = introspect.build_from_meta_model(
        {"model_type": "demo"},
        source={},
        runtime_patch=patch,
    )

    assert patch.was_active_during_call is True
    assert patch.active is False
    assert structure.summary["strategy"] == "meta-introspect"
