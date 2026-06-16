from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles

from .exporters import export_structure
from .resolver import ModelSourceResolver, SourceResolutionError
from .schemas import ExportRequest, SettingsPayload, StructureRequest
from .settings import AppSettings
from .structure import build_model_structure

settings = AppSettings.from_env()
app = FastAPI(title="Model Structure Viewer", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.get("/api/health")
def health() -> dict[str, object]:
    return {
        "ok": True,
        "model_root": str(settings.model_root),
        "hf_endpoint": settings.hf_endpoint,
        "cache_policy": settings.cache_policy,
        "offline": settings.offline,
        "auto_fetch_remote_code": settings.auto_fetch_remote_code,
    }


@app.get("/api/settings")
def get_settings() -> dict[str, object]:
    return health() | {"ok": True}


@app.post("/api/settings")
def update_settings(payload: SettingsPayload) -> dict[str, object]:
    global settings
    settings = settings.with_overrides(
        model_root=payload.model_root,
        hf_endpoint=payload.hf_endpoint,
        cache_policy=payload.cache_policy,
        offline=payload.offline,
        auto_fetch_remote_code=payload.auto_fetch_remote_code,
    )
    return get_settings()


@app.get("/api/models")
def list_models() -> list[dict[str, object]]:
    return [entry.model_dump() for entry in ModelSourceResolver(settings).list_local_models()]


@app.get("/api/hf/search")
def hf_search(q: str = Query(..., min_length=1), limit: int = 10) -> list[dict[str, object]]:
    try:
        return [entry.model_dump() for entry in ModelSourceResolver(settings).search_hf_models(q, limit=limit)]
    except SourceResolutionError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.get("/api/hf/config")
def hf_config(model_id: str, revision: str = "main") -> dict[str, object]:
    try:
        return ModelSourceResolver(settings).get_remote_config(model_id, revision=revision)
    except SourceResolutionError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/api/structure")
def structure(payload: StructureRequest) -> dict[str, object]:
    request_settings = settings.with_overrides(
        model_root=payload.model_root,
        hf_endpoint=payload.hf_endpoint,
        cache_policy=payload.cache_policy,
        offline=payload.offline,
        auto_fetch_remote_code=payload.auto_fetch_remote_code,
    )
    resolver = ModelSourceResolver(request_settings)
    try:
        resolved = resolver.resolve(
            source=payload.source,
            model_id=payload.model_id,
            config_path=payload.config_path,
            config_json=payload.config_json,
            revision=payload.revision,
            cache_policy=payload.cache_policy,
            detail_level=payload.detail_level,
        )
        model_structure = build_model_structure(
            resolved.config,
            source=resolved.source,
            detail_level=payload.detail_level,
            local_dir=resolved.local_dir,
        )
        return model_structure.model_dump()
    except SourceResolutionError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/export", response_class=PlainTextResponse)
def export(payload: ExportRequest) -> str:
    try:
        return export_structure(payload.structure, payload.format)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


frontend_dist = Path(__file__).resolve().parents[2] / "frontend" / "dist"
if frontend_dist.exists():
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")
