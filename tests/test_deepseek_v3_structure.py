"""Tests for unsupported local-only deepseek_v3 fixture behavior."""
import json
from pathlib import Path

import pytest

from model_structure_viewer.errors import IntrospectionError
from model_structure_viewer.structure import build_model_structure

FIXTURE = Path(__file__).parent / "fixtures" / "deepseek_v3" / "config.json"


def load_config():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_deepseek_v3_fixture_without_remote_code_raises():
    # The deepseek_v3 fixture has auto_map but no local modeling files,
    # so backend introspection must fail explicitly instead of fabricating a
    # config-derived structure.
    with pytest.raises(IntrospectionError, match="custom remote code"):
        build_model_structure(load_config(), source={"kind": "fixture"})
