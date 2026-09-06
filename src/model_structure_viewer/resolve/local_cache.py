"""Local filesystem cache for model configs.

Owns ``model_root``: scans the directory, computes per-model paths, loads /
writes ``config.json``. Has no notion of HTTP. Raises ``NotFoundError`` /
``ConfigError`` on local issues; never ``RemoteError``.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any
from urllib.parse import urlparse

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
        seen_model_ids: set[str] = set()
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
            seen_model_ids.add(model_id)
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
        entries.extend(self._list_remote_snapshots(seen_model_ids))
        return entries

    def _list_remote_snapshots(self, excluded_model_ids: set[str]) -> list[ModelEntry]:
        cache_root = self.model_root / ".msv-cache"
        if not cache_root.exists():
            return []
        latest: dict[str, tuple[int, Path]] = {}
        for ref_path in cache_root.rglob("refs/*.json"):
            try:
                ref = self.load_json(ref_path)
            except (ConfigError, OSError):
                continue
            model_id = ref.get("model_id")
            snapshot_id = ref.get("snapshot_id")
            if not isinstance(model_id, str) or not isinstance(snapshot_id, str):
                continue
            if model_id in excluded_model_ids:
                continue
            config_path = ref_path.parents[1] / "snapshots" / snapshot_id / "config.json"
            if not config_path.exists():
                continue
            modified = ref_path.stat().st_mtime_ns
            if model_id not in latest or modified > latest[model_id][0]:
                latest[model_id] = (modified, config_path)
        return [
            ModelEntry(
                model_id=model_id,
                config_path=str(config_path),
                has_readme=(config_path.parent / "README.md").exists(),
                has_remote_config_code=any(config_path.parent.glob("configuration_*.py")),
                load_by="config_path",
            )
            for model_id, (_, config_path) in sorted(latest.items())
        ]

    # ---- path computation ----------------------------------------------------------
    def local_config_path(self, model_id: str) -> Path:
        parts = model_id_parts(model_id)
        return self.model_root.joinpath(*parts, "config.json")

    def try_remote_snapshot(
        self,
        model_id: str,
        *,
        endpoint: str,
        revision: str,
        detail_level: str,
    ) -> ResolvedConfig | None:
        ref_path = self._remote_ref_path(model_id, endpoint, revision)
        if not ref_path.exists():
            return None
        try:
            ref = self.load_json(ref_path)
        except ConfigError:
            return None
        if ref.get("endpoint") != endpoint or ref.get("requested_revision") != revision:
            return None
        snapshot_id = ref.get("snapshot_id")
        if not isinstance(snapshot_id, str) or not snapshot_id:
            return None
        snapshot_dir = ref_path.parents[1] / "snapshots" / snapshot_id
        config_path = snapshot_dir / "config.json"
        if not config_path.exists():
            return None
        return ResolvedConfig(
            config=self.load_json(config_path),
            source={
                "kind": "hf cache",
                "model_id": model_id,
                "revision": revision,
                "resolved_revision": ref.get("resolved_revision"),
                "hf_endpoint": endpoint,
                "cache_path": str(config_path),
                "detail_level": detail_level,
            },
            local_dir=snapshot_dir,
        )

    def store_remote_snapshot(
        self,
        model_id: str,
        config: dict[str, Any],
        *,
        endpoint: str,
        revision: str,
        resolved_revision: str | None,
        detail_level: str,
    ) -> ResolvedConfig:
        repo_dir = self._remote_repo_dir(model_id, endpoint)
        snapshot_material = {
            "endpoint": endpoint,
            "requested_revision": revision,
            "resolved_revision": resolved_revision,
            "config": config,
        }
        snapshot_id = _stable_digest(snapshot_material)
        snapshot_dir = repo_dir / "snapshots" / snapshot_id
        config_path = snapshot_dir / "config.json"
        self.write_json(config_path, config)
        ref = {
            "endpoint": endpoint,
            "model_id": model_id,
            "requested_revision": revision,
            "resolved_revision": resolved_revision,
            "snapshot_id": snapshot_id,
        }
        self.write_json(self._remote_ref_path(model_id, endpoint, revision), ref)
        return ResolvedConfig(
            config=config,
            source={
                "kind": "hf remote",
                "model_id": model_id,
                "revision": revision,
                "resolved_revision": resolved_revision,
                "hf_endpoint": endpoint,
                "cache_path": str(config_path),
                "detail_level": detail_level,
            },
            local_dir=snapshot_dir,
        )

    def _remote_repo_dir(self, model_id: str, endpoint: str) -> Path:
        endpoint_key = _endpoint_cache_key(endpoint)
        return self.model_root.joinpath(".msv-cache", endpoint_key, *model_id_parts(model_id))

    def _remote_ref_path(self, model_id: str, endpoint: str, revision: str) -> Path:
        return self._remote_repo_dir(model_id, endpoint) / "refs" / f"{sha256(revision.encode('utf-8')).hexdigest()}.json"

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
        is_directory = path.is_dir()
        if is_directory:
            path = path / "config.json"
        if not path.exists():
            raise NotFoundError(f"Config file not found: {path}")
        return ResolvedConfig(
            config=self.load_json(path),
            source={"kind": "local directory" if is_directory else "local file", "config_path": str(path), "detail_level": detail_level},
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
        path.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
        temp_path: Path | None = None
        try:
            with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
                temp_path = Path(handle.name)
            os.replace(temp_path, path)
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)


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


def model_id_parts(model_id: str) -> list[str]:
    """Return safe repo path segments without allowing traversal."""
    if not isinstance(model_id, str) or not model_id.strip():
        raise ConfigError("model_id must not be empty")
    raw = model_id.strip()
    from_url = "://" in raw
    if from_url:
        parsed = urlparse(raw)
        if parsed.hostname not in {"huggingface.co", "www.huggingface.co", "modelscope.cn", "www.modelscope.cn"}:
            raise ConfigError("model URL must point to Hugging Face or ModelScope")
        raw = parsed.path
        if parsed.hostname in {"modelscope.cn", "www.modelscope.cn"}:
            raw = raw.removeprefix("/models/")
    parts = [part for part in raw.split("/") if part]
    if not parts or any(part in {".", ".."} or "\\" in part for part in parts):
        raise ConfigError("model_id must be a repository id, not a path traversal")
    return parts[:2] if from_url else parts


def _stable_digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return sha256(encoded.encode("utf-8")).hexdigest()


def _endpoint_cache_key(endpoint: str) -> str:
    host = urlparse(endpoint).hostname or "endpoint"
    safe_host = "".join(character if character.isalnum() or character in "-." else "-" for character in host)
    digest = sha256(endpoint.rstrip("/").encode("utf-8")).hexdigest()[:12]
    return f"{safe_host}-{digest}"
