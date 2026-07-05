from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

SourceKind = Literal["auto", "local", "hf", "config"]
CachePolicy = Literal["prefer-local", "refresh", "offline"]
ExportFormat = Literal["json", "mermaid", "dot"]


class StructureNode(BaseModel):
    id: str
    name: str
    type: str
    repeat: int | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)
    source_fields: list[str] = Field(default_factory=list)
    confidence: str = "high"
    children: list["StructureNode"] = Field(default_factory=list)


class ModelStructure(BaseModel):
    summary: dict[str, Any] = Field(default_factory=dict)
    source: dict[str, Any] = Field(default_factory=dict)
    root: StructureNode
    extra_config: dict[str, Any] = Field(default_factory=dict)


class StructureRequest(BaseModel):
    source: SourceKind = "auto"
    model_id: str | None = None
    config_path: str | None = None
    config_json: dict[str, Any] | None = None
    revision: str = "main"
    cache_policy: CachePolicy = "prefer-local"
    detail_level: Literal["compressed", "expanded"] = "compressed"
    hf_endpoint: str | None = None
    model_root: str | None = None
    offline: bool = False
    auto_fetch_remote_code: bool | None = None


class VerifyRequest(StructureRequest):
    pass


class VerifyResponse(BaseModel):
    ok: bool
    status: Literal["passed", "failed", "skipped"]
    strategy: str = "transformers-meta"
    model_id: str | None = None
    source: dict[str, Any] = Field(default_factory=dict)
    summary: dict[str, Any] = Field(default_factory=dict)
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class ExportRequest(BaseModel):
    structure: ModelStructure
    format: ExportFormat = "json"


class SettingsPayload(BaseModel):
    model_root: str | None = None
    hf_endpoint: str | None = None
    cache_policy: CachePolicy | None = None
    offline: bool | None = None
    auto_fetch_remote_code: bool | None = None


class ModelEntry(BaseModel):
    model_id: str
    config_path: str
    has_readme: bool = False
    has_remote_config_code: bool = False
    load_by: Literal["model_id", "config_path"] = "model_id"


class HfSearchResult(BaseModel):
    model_id: str
    pipeline_tag: str | None = None
    tags: list[str] = Field(default_factory=list)
    downloads: int | None = None
    likes: int | None = None
