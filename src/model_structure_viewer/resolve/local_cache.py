"""Local filesystem cache for model configs.

Owns ``model_root``: scans the directory, computes per-model paths, loads /
writes ``config.json``. Has no notion of HTTP. Raises ``NotFoundError`` /
``ConfigError`` on local issues; never ``RemoteError``.
"""
from __future__ import annotations

import json
import os
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
    def __init__(self, model_root: Path | str):
        self.model_root = Path(model_root).expanduser()

    # ---- listing -------------------------------------------------------------------
    def list_local_models(self) -> list[ModelEntry]:
        root = self.model_root
        if not root.exists():
            return []
        entries: list[ModelEntry] = []
        seen_paths: set[Path] = set()
        for config_path in _iter_candidate_json_files(root):
            try:
                rel_path = config_path.relative_to(root)
                rel_parent = config_path.parent.relative_to(root)
            except ValueError:
                continue
            if any(part.startswith(".") for part in rel_path.parts):
                continue
            if not _is_model_config_path(config_path):
                continue
            if not rel_parent.parts and config_path.name == "config.json":
                continue
            try:
                config = self.load_json(config_path)
            except ConfigError:
                continue
            if not _looks_like_model_config(config):
                continue
            resolved_path = config_path.resolve()
            if resolved_path in seen_paths:
                continue
            seen_paths.add(resolved_path)

            model_id = _model_id_for_path(root, config_path)
            load_by = "model_id" if config_path.name == "config.json" else "config_path"
            entries.append(
                ModelEntry(
                    model_id=model_id,
                    config_path=str(config_path),
                    has_readme=(config_path.parent / "README.md").exists(),
                    has_remote_config_code=any(config_path.parent.glob("configuration_*.py")),
                    load_by=load_by,
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


_EXCLUDED_JSON_NAMES = {
    "configuration.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
}
_EXCLUDED_PATH_PARTS = {"assets", "encoding", "inference"}
_MODEL_CONFIG_KEYS = {
    "architectures",
    "auto_map",
    "model_type",
    "text_config",
    "vision_config",
}
_LAYER_KEYS = {"num_hidden_layers", "num_layers", "n_layer", "n_layers"}
_WIDTH_KEYS = {
    "d_model",
    "dim",
    "hidden_size",
    "n_embd",
    "num_attention_heads",
    "n_heads",
}


def _iter_candidate_json_files(root: Path):
    for current_root, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(
            dirname
            for dirname in dirnames
            if not dirname.startswith(".") and dirname not in _EXCLUDED_PATH_PARTS
        )
        for filename in sorted(filenames):
            if filename.endswith(".json"):
                yield Path(current_root) / filename


def _is_model_config_path(path: Path) -> bool:
    if path.name in _EXCLUDED_JSON_NAMES:
        return False
    if path.name.endswith(".safetensors.index.json"):
        return False
    if _EXCLUDED_PATH_PARTS.intersection(path.parts):
        return False
    return path.suffix == ".json"


def _looks_like_model_config(config: dict[str, Any]) -> bool:
    if any(key in config for key in _MODEL_CONFIG_KEYS):
        return True
    return any(key in config for key in _LAYER_KEYS) and any(key in config for key in _WIDTH_KEYS)


def _model_id_for_path(root: Path, config_path: Path) -> str:
    rel_path = config_path.relative_to(root)
    if config_path.name == "config.json":
        return "/".join(rel_path.parent.parts)
    return "/".join((*rel_path.parent.parts, config_path.stem))
