"""Local filesystem cache for model configs.

Owns ``model_root``: scans the directory, computes per-model paths, loads /
writes ``config.json``. Has no notion of HTTP. Raises ``NotFoundError`` /
``ConfigError`` on local issues; never ``RemoteError``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..errors import ConfigError, NotFoundError
from ..schemas import ModelEntry


@dataclass
class ResolvedConfig:
    """Output of resolution: the parsed config plus provenance."""

    config: dict[str, Any]
    source: dict[str, Any]
    local_dir: Path | None = None


class LocalModelCache:
    def __init__(self, model_root: Path):
        self.model_root = model_root

    # ---- listing -------------------------------------------------------------------
    def list_local_models(self) -> list[ModelEntry]:
        root = self.model_root
        if not root.exists():
            return []
        entries: list[ModelEntry] = []
        for config_path in sorted(root.rglob("config.json")):
            try:
                rel_parent = config_path.parent.relative_to(root)
            except ValueError:
                continue
            if not rel_parent.parts:
                continue
            model_id = "/".join(rel_parent.parts)
            entries.append(
                ModelEntry(
                    model_id=model_id,
                    config_path=str(config_path),
                    has_readme=(config_path.parent / "README.md").exists(),
                    has_remote_config_code=any(config_path.parent.glob("configuration_*.py")),
                )
            )
        return entries

    # ---- path computation ----------------------------------------------------------
    def local_config_path(self, model_id: str) -> Path:
        parts = [part for part in model_id.split("/") if part]
        return self.model_root.joinpath(*parts, "config.json")

    # ---- resolution helpers --------------------------------------------------------
    def try_local_model(self, model_id: str, detail_level: str) -> ResolvedConfig | None:
        path = self.local_config_path(model_id)
        if not path.exists():
            return None
        return ResolvedConfig(
            config=self.load_json(path),
            source={
                "kind": "local cache",
                "model_id": model_id,
                "config_path": str(path),
                "detail_level": detail_level,
            },
            local_dir=path.parent,
        )

    def resolve_local_model(self, model_id: str, detail_level: str) -> ResolvedConfig:
        resolved = self.try_local_model(model_id, detail_level)
        if resolved is None:
            expected = self.local_config_path(model_id)
            raise NotFoundError(f"Local model config not found: {expected}")
        return resolved

    def resolve_config_path(self, config_path: str, detail_level: str) -> ResolvedConfig:
        path = Path(config_path).expanduser()
        if not path.exists():
            raise NotFoundError(f"Config file not found: {path}")
        return ResolvedConfig(
            config=self.load_json(path),
            source={"kind": "local file", "config_path": str(path), "detail_level": detail_level},
            local_dir=path.parent,
        )

    # ---- IO ------------------------------------------------------------------------
    @staticmethod
    def load_json(path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ConfigError(f"Config JSON must be an object: {path}")
        return payload

    @staticmethod
    def write_json(path: Path, payload: dict[str, Any]) -> None:
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
