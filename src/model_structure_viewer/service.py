from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
from collections import OrderedDict
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from hashlib import sha256
from multiprocessing import get_context
from pathlib import Path
from typing import Any

from .resolver import ModelSourceResolver
from .schemas import ModelStructure, StructureRequest, VerifyRequest, VerifyResponse
from .settings import AppSettings
from .structure import build_model_structure
from .verification.transformers_verify import verify_transformers_structure

_DEFAULT_STRUCTURE_CACHE_SIZE = 8
_DEFAULT_WORKER_TIMEOUT_SECONDS = 90.0
_CACHEABLE_METADATA_NAMES = {"config.json", "README.md"}
_CACHEABLE_METADATA_PREFIXES = ("configuration_", "modeling_", "tokenization_")
_STRUCTURE_CACHE: OrderedDict[str, ModelStructure] = OrderedDict()
_STRUCTURE_CACHE_LOCK = threading.Lock()


def build_structure_response(
    payload: StructureRequest,
    base_settings: AppSettings,
) -> ModelStructure:
    """Resolve a config per ``payload`` and build the structure tree.

    Shared by the FastAPI ``/api/structure`` route and the CLI ``inspect``
    command. Lives outside ``api`` so the CLI does not have to import the
    FastAPI app (which would trigger CORS / static-mount setup).
    """
    request_settings = base_settings.with_overrides(
        model_root=payload.model_root,
        hf_endpoint=payload.hf_endpoint,
        cache_policy=payload.cache_policy,
        offline=payload.offline,
        auto_fetch_remote_code=payload.auto_fetch_remote_code,
    )
    resolver = ModelSourceResolver(request_settings)
    resolved = resolver.resolve(
        source=payload.source,
        model_id=payload.model_id,
        config_path=payload.config_path,
        config_json=payload.config_json,
        revision=payload.revision,
        cache_policy=payload.cache_policy,
        detail_level=payload.detail_level,
    )
    cache_key = _structure_cache_key(payload, request_settings, resolved)
    cached = _get_cached_structure(cache_key)
    if cached is not None:
        return cached

    structure = _build_layered_structure(
        resolved.config,
        source=resolved.source,
        detail_level=payload.detail_level,
        local_dir=resolved.local_dir,
    )
    return _put_cached_structure(cache_key, structure)


def verify_structure_response(
    payload: VerifyRequest,
    base_settings: AppSettings,
) -> VerifyResponse:
    """Resolve a config and strictly validate Transformers meta construction."""
    request_settings = base_settings.with_overrides(
        model_root=payload.model_root,
        hf_endpoint=payload.hf_endpoint,
        cache_policy=payload.cache_policy,
        offline=payload.offline,
        auto_fetch_remote_code=payload.auto_fetch_remote_code,
    )
    resolver = ModelSourceResolver(request_settings)
    resolved = resolver.resolve(
        source=payload.source,
        model_id=payload.model_id,
        config_path=payload.config_path,
        config_json=payload.config_json,
        revision=payload.revision,
        cache_policy=payload.cache_policy,
        detail_level=payload.detail_level,
    )
    worker_result = _run_transformers_verify_worker(
        resolved.config,
        source=resolved.source,
        local_dir=resolved.local_dir,
        timeout_seconds=_worker_timeout_seconds(),
    )
    return VerifyResponse.model_validate(worker_result)


def clear_structure_cache() -> None:
    """Clear the in-process structure cache. Intended for tests and service resets."""
    with _STRUCTURE_CACHE_LOCK:
        _STRUCTURE_CACHE.clear()


def _get_cached_structure(cache_key: str) -> ModelStructure | None:
    with _STRUCTURE_CACHE_LOCK:
        cached = _STRUCTURE_CACHE.get(cache_key)
        if cached is None:
            return None
        _STRUCTURE_CACHE.move_to_end(cache_key)
        return cached


def _put_cached_structure(cache_key: str, structure: ModelStructure) -> ModelStructure:
    limit = _structure_cache_size()
    if limit <= 0:
        return structure
    with _STRUCTURE_CACHE_LOCK:
        _STRUCTURE_CACHE[cache_key] = structure
        _STRUCTURE_CACHE.move_to_end(cache_key)
        while len(_STRUCTURE_CACHE) > limit:
            _STRUCTURE_CACHE.popitem(last=False)
    return structure


def _structure_cache_size() -> int:
    raw = os.environ.get("MSV_STRUCTURE_CACHE_SIZE")
    if raw is None:
        return _DEFAULT_STRUCTURE_CACHE_SIZE
    try:
        return max(0, int(raw))
    except ValueError:
        return _DEFAULT_STRUCTURE_CACHE_SIZE


def _structure_cache_key(
    payload: StructureRequest,
    settings: AppSettings,
    resolved,
) -> str:
    material = {
        "payload": payload.model_dump(mode="json", exclude={"config_json"}),
        "config_hash": _stable_hash(resolved.config),
        "config_json_hash": _stable_hash(payload.config_json) if payload.config_json else None,
        "settings": {
            "model_root": str(settings.model_root),
            "hf_endpoint": settings.hf_endpoint,
            "offline": settings.offline,
            "auto_fetch_remote_code": settings.auto_fetch_remote_code,
        },
        "source": resolved.source,
        "local_dir": str(resolved.local_dir) if resolved.local_dir is not None else None,
        "metadata": _metadata_fingerprint(resolved.local_dir),
    }
    return _stable_hash(material)


def _stable_hash(value) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    return sha256(payload.encode("utf-8")).hexdigest()


def _metadata_fingerprint(local_dir: Path | None) -> list[dict[str, object]]:
    if local_dir is None or not local_dir.exists():
        return []
    fingerprints: list[dict[str, object]] = []
    for path in sorted(local_dir.iterdir()):
        if not path.is_file() or not _is_cacheable_metadata(path.name):
            continue
        stat = path.stat()
        fingerprints.append(
            {
                "name": path.name,
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    return fingerprints


def _is_cacheable_metadata(filename: str) -> bool:
    return filename in _CACHEABLE_METADATA_NAMES or (
        filename.endswith(".py") and filename.startswith(_CACHEABLE_METADATA_PREFIXES)
    )


def _build_layered_structure(
    config: dict[str, Any],
    *,
    source: dict[str, Any],
    detail_level: str,
    local_dir: Path | None,
) -> ModelStructure:
    worker_result = _run_introspection_worker(
        config,
        source=source,
        detail_level=detail_level,
        local_dir=local_dir,
        timeout_seconds=_worker_timeout_seconds(),
    )
    if worker_result.get("ok"):
        return ModelStructure.model_validate(worker_result["structure"])

    from .errors import IntrospectionError

    raise IntrospectionError(str(worker_result.get("message") or "introspection worker failed"))


def _run_introspection_worker(
    config: dict[str, Any],
    *,
    source: dict[str, Any],
    detail_level: str,
    local_dir: Path | None,
    timeout_seconds: float,
) -> dict[str, Any]:
    if _parse_bool(os.environ.get("MSV_DISABLE_STRUCTURE_WORKER", "0")):
        try:
            with _suppress_third_party_output():
                structure = build_model_structure(
                    config,
                    source=source,
                    detail_level=detail_level,
                    local_dir=local_dir,
                )
            return {"ok": True, "structure": structure.model_dump(mode="json")}
        except Exception as exc:  # noqa: BLE001 - worker contract mirrors subprocess failures
            return {
                "ok": False,
                "failure_kind": "worker_failed",
                "message": f"{type(exc).__name__}: {exc}",
                "error_type": type(exc).__name__,
            }

    with tempfile.TemporaryDirectory(prefix="msv-structure-worker-") as temp_dir:
        temp_path = Path(temp_dir)
        input_path = temp_path / "input.json"
        output_path = temp_path / "output.json"
        input_path.write_text(
            json.dumps(
                {
                    "config": config,
                    "source": source,
                    "detail_level": detail_level,
                    "local_dir": str(local_dir) if local_dir is not None else None,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        context = get_context("spawn")
        process = context.Process(target=_structure_worker_entrypoint, args=(input_path, output_path))
        process.start()
        process.join(timeout_seconds)

        if process.is_alive():
            process.terminate()
            process.join(2)
            if process.is_alive():
                process.kill()
                process.join()
            return {
                "ok": False,
                "failure_kind": "worker_timeout",
                "message": f"introspection worker timed out after {timeout_seconds:g}s",
                "timeout_seconds": timeout_seconds,
                "exit_code": process.exitcode,
            }

        if output_path.exists():
            try:
                return json.loads(output_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                return {
                    "ok": False,
                    "failure_kind": "worker_failed",
                    "message": f"worker returned invalid JSON: {exc}",
                    "error_type": type(exc).__name__,
                    "exit_code": process.exitcode,
                }

        failure_kind = "worker_killed" if process.exitcode and process.exitcode < 0 else "worker_failed"
        return {
            "ok": False,
            "failure_kind": failure_kind,
            "message": f"worker exited with code {process.exitcode}",
            "exit_code": process.exitcode,
        }


def _run_transformers_verify_worker(
    config: dict[str, Any],
    *,
    source: dict[str, Any],
    local_dir: Path | None,
    timeout_seconds: float,
) -> dict[str, Any]:
    if _parse_bool(os.environ.get("MSV_DISABLE_STRUCTURE_WORKER", "0")):
        with _suppress_third_party_output():
            return verify_transformers_structure(config, source=source, local_dir=local_dir).model_dump(mode="json")

    with tempfile.TemporaryDirectory(prefix="msv-verify-worker-") as temp_dir:
        temp_path = Path(temp_dir)
        input_path = temp_path / "input.json"
        output_path = temp_path / "output.json"
        input_path.write_text(
            json.dumps(
                {
                    "config": config,
                    "source": source,
                    "local_dir": str(local_dir) if local_dir is not None else None,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        context = get_context("spawn")
        process = context.Process(target=_verify_worker_entrypoint, args=(input_path, output_path))
        process.start()
        process.join(timeout_seconds)

        if process.is_alive():
            process.terminate()
            process.join(2)
            if process.is_alive():
                process.kill()
                process.join()
            return {
                "ok": False,
                "status": "failed",
                "strategy": "transformers-meta",
                "model_id": source.get("model_id"),
                "source": source,
                "summary": _verify_minimal_summary(config),
                "diagnostics": {
                    "failure_kind": "worker_timeout",
                    "worker_timeout_seconds": timeout_seconds,
                    "worker_exit_code": process.exitcode,
                },
                "error": f"transformers verification worker timed out after {timeout_seconds:g}s",
            }

        if output_path.exists():
            try:
                return json.loads(output_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                return {
                    "ok": False,
                    "status": "failed",
                    "strategy": "transformers-meta",
                    "model_id": source.get("model_id"),
                    "source": source,
                    "summary": _verify_minimal_summary(config),
                    "diagnostics": {
                        "failure_kind": "worker_failed",
                        "error_type": type(exc).__name__,
                        "worker_exit_code": process.exitcode,
                    },
                    "error": f"verification worker returned invalid JSON: {exc}",
                }

        failure_kind = "worker_killed" if process.exitcode and process.exitcode < 0 else "worker_failed"
        return {
            "ok": False,
            "status": "failed",
            "strategy": "transformers-meta",
            "model_id": source.get("model_id"),
            "source": source,
            "summary": _verify_minimal_summary(config),
            "diagnostics": {
                "failure_kind": failure_kind,
                "worker_exit_code": process.exitcode,
            },
            "error": f"verification worker exited with code {process.exitcode}",
        }


def _structure_worker_entrypoint(input_path: Path, output_path: Path) -> None:
    try:
        payload = json.loads(input_path.read_text(encoding="utf-8"))
        with _suppress_third_party_output():
            structure = build_model_structure(
                payload["config"],
                source=payload["source"],
                detail_level=payload["detail_level"],
                local_dir=payload.get("local_dir"),
            )
        result = {"ok": True, "structure": structure.model_dump(mode="json")}
    except Exception as exc:  # noqa: BLE001 - third-party model code can raise anything
        result = {
            "ok": False,
            "failure_kind": "worker_failed",
            "message": f"{type(exc).__name__}: {exc}",
            "error_type": type(exc).__name__,
        }
    output_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")


def _verify_worker_entrypoint(input_path: Path, output_path: Path) -> None:
    try:
        payload = json.loads(input_path.read_text(encoding="utf-8"))
        with _suppress_third_party_output():
            result = verify_transformers_structure(
                payload["config"],
                source=payload["source"],
                local_dir=payload.get("local_dir"),
            ).model_dump(mode="json")
    except Exception as exc:  # noqa: BLE001 - third-party model code can raise anything
        result = {
            "ok": False,
            "status": "failed",
            "strategy": "transformers-meta",
            "source": {},
            "summary": {},
            "diagnostics": {"failure_kind": "worker_failed", "error_type": type(exc).__name__},
            "error": f"{type(exc).__name__}: {exc}",
        }
    output_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")


@contextmanager
def _suppress_third_party_output():
    stdout_fd = os.dup(1)
    stderr_fd = os.dup(2)
    try:
        _flush_standard_streams()
        with (
            tempfile.TemporaryFile("w+", encoding="utf-8") as stdout_buffer,
            tempfile.TemporaryFile("w+", encoding="utf-8") as stderr_buffer,
        ):
            os.dup2(stdout_buffer.fileno(), 1)
            os.dup2(stderr_buffer.fileno(), 2)
            with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                yield
                _flush_standard_streams()
    finally:
        os.dup2(stdout_fd, 1)
        os.dup2(stderr_fd, 2)
        os.close(stdout_fd)
        os.close(stderr_fd)


def _flush_standard_streams() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:  # noqa: BLE001 - flushing should never break cleanup
            continue


def _verify_minimal_summary(config: dict[str, Any]) -> dict[str, Any]:
    architectures = config.get("architectures")
    architecture = architectures[0] if isinstance(architectures, list) and architectures else None
    return {
        "model_type": config.get("model_type"),
        "architecture": architecture,
    }


def _worker_timeout_seconds() -> float:
    raw = os.environ.get("MSV_STRUCTURE_WORKER_TIMEOUT_SECONDS")
    if raw is None:
        return _DEFAULT_WORKER_TIMEOUT_SECONDS
    try:
        return max(1.0, float(raw))
    except ValueError:
        return _DEFAULT_WORKER_TIMEOUT_SECONDS

def _parse_bool(value: str) -> bool:
    return value.lower() in {"1", "true", "yes", "on"}
