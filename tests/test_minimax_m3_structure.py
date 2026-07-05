"""Tests for unsupported local-only MiniMax-M3 fixture behavior."""
import json
from pathlib import Path

import pytest

from model_structure_viewer.errors import IntrospectionError
from model_structure_viewer.structure import build_model_structure

FIXTURE = Path(__file__).parent / "fixtures" / "minimax_m3" / "config.json"


def load_config():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_minimax_m3_fixture_without_remote_code_raises():
    with pytest.raises(IntrospectionError, match="custom remote code"):
        build_model_structure(load_config(), source={"kind": "fixture"})
