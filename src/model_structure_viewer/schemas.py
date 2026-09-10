from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

SourceKind = Literal["auto", "builtin", "local", "hf", "config"]
CachePolicy = Literal["prefer-local", "refresh", "offline"]
EndpointKind = Literal["huggingface", "modelscope"]
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
    # 结构节点扩展字段（None 表示未知）；图边位于 ModelStructure.graph。
    params: int | None = None  # 本节点自有参数（不含子树）
    weight_shapes: dict[str, list[int]] | None = None  # 数值形状，如 {"weight": [4096, 4096]}
    dtype: str | None = None  # 实际 dtype：BF16/F8_E4M3/I32…
    input_shape: list[int] | None = None  # 数值 I/O；batch/seq 用 -1 占位
    output_shape: list[int] | None = None
    value_source: str | None = None  # "checkpoint" | "derived" | "introspect"
    tensor_names: list[str] | None = None  # 绑定到本节点的 header 张量名


class StructureGraphNode(BaseModel):
    id: str
    canonical_id: str | None = None
    module_id: str | None = None
    parent_id: str | None = None
    order: int = 0
    name: str = ""
    type: str = "module"
    repeat: int | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)
    source_fields: list[str] = Field(default_factory=list)
    confidence: str = "high"
    params: int | None = None
    weight_shapes: dict[str, list[int]] | None = None
    dtype: str | None = None
    input_shape: list[int] | None = None
    output_shape: list[int] | None = None
    value_source: str | None = None
    tensor_names: list[str] | None = None


class StructureGraphEdge(BaseModel):
    id: str
    source: str
    target: str
    source_canonical_id: str | None = None
    target_canonical_id: str | None = None
    kind: str = "dataflow"
    evidence: str | None = None


class StructureGraph(BaseModel):
    version: int = 2
    schema_version: int = 2
    root_id: str = "root"
    nodes: list[StructureGraphNode] = Field(default_factory=list)
    edges: list[StructureGraphEdge] = Field(default_factory=list)


class ModelStructure(BaseModel):
    summary: dict[str, Any] = Field(default_factory=dict)
    source: dict[str, Any] = Field(default_factory=dict)
    # P7（步骤 7）：Graph IR 是唯一结构载荷（必填）——legacy root 可逆视图与
    # ensure_graph_primary 补投影校验器一并退役（协议执法由必填字段承担）。
    graph: StructureGraph
    extra_config: dict[str, Any] = Field(default_factory=dict)


class StructureRequest(BaseModel):
    source: SourceKind = "auto"
    model_id: str | None = None
    config_path: str | None = None
    config_json: dict[str, Any] | None = None
    revision: str = "main"
    cache_policy: CachePolicy | None = None
    detail_level: Literal["compressed", "expanded"] = "compressed"
    hf_endpoint: str | None = None
    model_root: str | None = None
    offline: bool | None = None
    auto_fetch_remote_code: bool | None = None
    endpoint: EndpointKind | None = None


class VerifyRequest(StructureRequest):
    # 对账上行载荷（P7/步骤 6）：前端 Graph（{"nodes": [...]} 整体或裸节点列表）。
    # 缺省时 /api/verify 仍返回 evidence.modules，diff 三分类为空并注明未对账。
    msv_graph: list[dict[str, Any]] | dict[str, Any] | None = None


class VerifyEvidenceMismatch(BaseModel):
    # kind: "class"（torch 类名 ↔ 前端模板标签分歧）| "shape"（weight_shapes 正维语义分歧）
    path: str
    kind: Literal["class", "shape"]
    transformers: Any = None
    msv: Any = None


class VerifyEvidenceDiff(BaseModel):
    # 路径键为 canonical_reconciliation_path 折叠后的种类键（非原始实例路径）。
    only_transformers: list[str] = Field(default_factory=list)
    only_msv: list[str] = Field(default_factory=list)
    mismatches: list[VerifyEvidenceMismatch] = Field(default_factory=list)
    # msv_graph 缺省时置 "msv_graph not provided"，三分类为空不等于对账通过。
    note: str | None = None
    # P0-2 triage 分桶计数：renaming / nonparam_drop / fold_frontend_suffixes /
    # known_divergences。only_* 与 mismatches 输出的是**未分类**残余。
    classified: dict[str, int] = Field(default_factory=dict)


class VerifyEvidence(BaseModel):
    # modules: per-module 证据 {path, class, params, weight_shapes, dtype, value_source, repeat}；
    # 形状类字段无值保持 None（不伪造数值，introspect.py:179-204 的原样搬运）。
    modules: list[dict[str, Any]] = Field(default_factory=list)
    diff: VerifyEvidenceDiff = Field(default_factory=VerifyEvidenceDiff)
    # summary 承载对账结论两态：constructed（meta 构造通过）与
    # structurally_consistent（diff 干净；未对账时 None 而非 False）。
    summary: dict[str, Any] = Field(default_factory=dict)


class VerifyResponse(BaseModel):
    ok: bool
    status: Literal["passed", "failed", "skipped"]
    strategy: str = "transformers-meta"
    model_id: str | None = None
    source: dict[str, Any] = Field(default_factory=dict)
    summary: dict[str, Any] = Field(default_factory=dict)
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    # 对账结论只进新增字段，status 的 passed/failed/skipped 契约不变
    # （构造通过 ≠ 结构一致，两态分立，tasks.md Task 7.2）。
    evidence: VerifyEvidence | None = None


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
