from __future__ import annotations

from .resolver import ModelSourceResolver
from .schemas import ModelStructure, StructureRequest
from .settings import AppSettings
from .structure import build_model_structure


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
    return build_model_structure(
        resolved.config,
        source=resolved.source,
        detail_level=payload.detail_level,
        local_dir=resolved.local_dir,
    )
