from __future__ import annotations

from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from .errors import ViewerError
from .exporters import export_structure
from .resolve.local_cache import LocalModelCache
from .resolver import ModelSourceResolver
from .schemas import ExportRequest, SettingsPayload, StructureRequest
from .service import build_structure_response
from .settings import AppSettings


__all__ = ["app", "build_structure_response", "get_settings", "set_settings"]


# Module-level holder so cli.cmd_serve can inject overrides before uvicorn starts,
# while request handlers receive the active settings via Depends(get_settings).
_settings_holder: dict[str, AppSettings] = {"current": AppSettings.from_env()}


def get_settings() -> AppSettings:
    return _settings_holder["current"]


def set_settings(new_settings: AppSettings) -> None:
    _settings_holder["current"] = new_settings


app = FastAPI(title="Model Structure Viewer", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.exception_handler(ViewerError)
async def _viewer_error_handler(_request: Request, exc: ViewerError) -> JSONResponse:
    return JSONResponse(status_code=exc.http_status, content={"detail": str(exc)})


def _settings_dict(s: AppSettings) -> dict[str, object]:
    return {
        "ok": True,
        "model_root": str(s.model_root),
        "hf_endpoint": s.hf_endpoint,
        "cache_policy": s.cache_policy,
        "offline": s.offline,
        "auto_fetch_remote_code": s.auto_fetch_remote_code,
    }


@app.get("/api/health")
def health(s: AppSettings = Depends(get_settings)) -> dict[str, object]:
    return _settings_dict(s)


@app.get("/api/settings")
def read_settings(s: AppSettings = Depends(get_settings)) -> dict[str, object]:
    return _settings_dict(s)


@app.post("/api/settings")
def update_settings(payload: SettingsPayload, s: AppSettings = Depends(get_settings)) -> dict[str, object]:
    new_settings = s.with_overrides(
        model_root=payload.model_root,
        hf_endpoint=payload.hf_endpoint,
        cache_policy=payload.cache_policy,
        offline=payload.offline,
        auto_fetch_remote_code=payload.auto_fetch_remote_code,
    )
    set_settings(new_settings)
    return _settings_dict(new_settings)


@app.get("/api/models")
def list_models(s: AppSettings = Depends(get_settings)) -> list[dict[str, object]]:
    return [entry.model_dump() for entry in ModelSourceResolver(s).list_local_models()]


@app.get("/api/local/config")
def local_config(
    model_id: str | None = None,
    config_path: str | None = None,
    s: AppSettings = Depends(get_settings),
) -> dict[str, object]:
    if not model_id and not config_path:
        raise HTTPException(status_code=400, detail="model_id or config_path is required")
    cache = LocalModelCache(s.model_root)
    if config_path:
        resolved = cache.resolve_config_path(config_path, detail_level="compressed")
    else:
        assert model_id is not None
        resolved = cache.resolve_local_model(model_id, detail_level="compressed")
    return {
        "model_id": model_id,
        "config": resolved.config,
        "source": resolved.source,
    }


@app.get("/api/hf/search")
def hf_search(
    q: str = Query(..., min_length=1),
    limit: int = 10,
    s: AppSettings = Depends(get_settings),
) -> list[dict[str, object]]:
    return [entry.model_dump() for entry in ModelSourceResolver(s).search_hf_models(q, limit=limit)]


@app.get("/api/hf/config")
def hf_config(
    model_id: str,
    revision: str = "main",
    s: AppSettings = Depends(get_settings),
) -> dict[str, object]:
    return ModelSourceResolver(s).get_remote_config(model_id, revision=revision)


@app.post("/api/structure")
def structure(
    payload: StructureRequest,
    s: AppSettings = Depends(get_settings),
) -> dict[str, object]:
    return build_structure_response(payload, s).model_dump()


@app.post("/api/export", response_class=PlainTextResponse)
def export(payload: ExportRequest) -> str:
    try:
        return export_structure(payload.structure, payload.format)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


frontend_dist = Path(__file__).resolve().parents[2] / "frontend" / "dist"
if frontend_dist.exists():
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")
