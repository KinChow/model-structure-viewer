"""Backend hygiene invariants (M10 bypass D).

Guards two rules from docs/refactor_plan.md bypass D:
1. "无 assert 做入参校验" — asserts vanish under ``python -O``, so the backend
   source must contain zero ``assert`` statements; validation raises explicitly.
2. errors.py mapping coverage — every ViewerError subclass carries an
   ``http_status`` that the FastAPI handler in api.py turns into the response.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import model_structure_viewer
from model_structure_viewer import service
from model_structure_viewer.api import app, get_settings, set_settings
from model_structure_viewer.errors import (
    ConfigError,
    IntrospectionError,
    NotFoundError,
    RemoteError,
    ViewerError,
)
from model_structure_viewer.settings import AppSettings
from model_structure_viewer.structure.recovery import MetaRecoveryError

PACKAGE_DIR = Path(model_structure_viewer.__file__).resolve().parent
client = TestClient(app)


def _backend_sources() -> list[Path]:
    return sorted(PACKAGE_DIR.rglob("*.py"))


def test_backend_source_contains_no_assert_statements():
    # assert is stripped under `python -O`; any validation must be an explicit raise.
    offenders: list[str] = []
    for path in _backend_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Assert):
                offenders.append(f"{path.relative_to(PACKAGE_DIR)}:{node.lineno}")
    assert offenders == []


@pytest.mark.parametrize(
    ("error_type", "expected_status"),
    [
        (ViewerError, 500),
        (ConfigError, 400),
        (NotFoundError, 404),
        (RemoteError, 502),
        (IntrospectionError, 500),
        (MetaRecoveryError, 500),
    ],
)
def test_viewer_error_http_status_mapping(error_type: type[ViewerError], expected_status: int):
    assert error_type.http_status == expected_status


def _all_viewer_error_subclasses() -> list[type[ViewerError]]:
    seen: list[type[ViewerError]] = []
    stack = list(ViewerError.__subclasses__())
    while stack:
        error_type = stack.pop()
        seen.append(error_type)
        stack.extend(error_type.__subclasses__())
    return seen


def test_every_viewer_error_subclass_declares_http_status():
    subclasses = _all_viewer_error_subclasses()
    assert subclasses, "expected the ViewerError hierarchy to be non-empty"
    for error_type in subclasses:
        status = getattr(error_type, "http_status", None)
        assert isinstance(status, int) and 400 <= status <= 599, (
            f"{error_type.__name__} must declare an http_status in [400, 599] "
            "so api.py's handler can map every raise point"
        )


def test_handler_maps_config_error_to_400():
    service.clear_structure_cache()
    response = client.post(
        "/api/structure",
        json={"source": "config", "detail_level": "compressed"},
    )

    assert response.status_code == 400
    assert "config_json" in response.json()["detail"]


def test_handler_maps_not_found_error_to_404(tmp_path):
    original_settings = get_settings()
    try:
        set_settings(AppSettings(model_root=tmp_path, offline=True))
        response = client.get("/api/local/config?model_id=Missing/Model")
    finally:
        set_settings(original_settings)

    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()
