from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

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
    # Graph IR is the canonical payload. root remains a reversible legacy view.
    root: StructureNode | None = None
    graph: StructureGraph | None = None
    extra_config: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def ensure_graph_primary(self) -> "ModelStructure":
        if self.graph is None and self.root is None:
            raise ValueError("ModelStructure requires graph or root")
        # Local import avoids a schemas -> graph -> schemas import cycle during
        # module initialization; validation runs after both modules load.
        from .structure.graph import project_graph_to_tree

        if self.root is None:
            self.root = project_graph_to_tree(self.graph)
        return self


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
