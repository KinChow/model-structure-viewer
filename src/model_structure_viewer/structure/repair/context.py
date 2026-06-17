from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import IntrospectionFailureKind
from .runtime import ConfigNormalizer, RuntimePatch


@dataclass(frozen=True)
class RepairContext:
    config: dict[str, Any]
    source: dict[str, Any]
    local_dir: Path | None
    failure_kind: IntrospectionFailureKind
    original_error: str


@dataclass(frozen=True)
class RepairResult:
    config: dict[str, Any]
    local_dir: Path | None
    strategy_name: str
    diagnostics: dict[str, Any] = field(default_factory=dict)
    runtime_patch: RuntimePatch | None = None
    config_overrides: dict[str, Any] = field(default_factory=dict)
    config_normalizer: ConfigNormalizer | None = None
