from model_structure_viewer.service import _settings_for_payload
from model_structure_viewer.schemas import StructureRequest
from model_structure_viewer.settings import AppSettings


def test_structure_request_inherits_offline_and_cache_policy():
    base = AppSettings(offline=True, cache_policy="refresh")
    request = StructureRequest(source="auto", model_id="Org/Model")
    settings = _settings_for_payload(request, base)
    assert settings.offline is True
    assert settings.cache_policy == "refresh"


def test_modelscope_endpoint_selects_modelscope_defaults():
    base = AppSettings()
    request = StructureRequest(source="hf", model_id="Org/Model", endpoint="modelscope")
    settings = _settings_for_payload(request, base)
    assert settings.hf_endpoint == "https://www.modelscope.cn"
