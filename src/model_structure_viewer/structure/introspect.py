"""Plan A: instantiate the model on the meta device and walk nn.Module tree."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ..schemas import ModelStructure, StructureNode
from . import fold, semantics
from .summary import extract_summary, infer_model_family


class IntrospectionError(RuntimeError):
    """Raised when meta-model introspection cannot proceed."""


def build_from_meta_model(
    config: dict[str, Any],
    *,
    source: dict[str, Any],
    local_dir: Path | None = None,
) -> ModelStructure:
    """Construct a ModelStructure by walking the live nn.Module tree on meta device."""
    try:
        import torch  # noqa: F401  (ensures torch is importable before init_empty_weights)
        from accelerate import init_empty_weights
        from transformers import AutoConfig, AutoModel
    except ImportError as exc:
        raise IntrospectionError(f"Missing optional dependency: {exc}") from exc

    hf_config = _load_config(AutoConfig, config, local_dir)

    try:
        with init_empty_weights():
            model = AutoModel.from_config(hf_config, trust_remote_code=True)
    except Exception as exc:  # noqa: BLE001  - third-party can raise anything
        raise IntrospectionError(f"AutoModel.from_config failed: {exc}") from exc

    raw_root = _walk(model, attribute_name="", path="root")
    folded_root = fold.collapse(raw_root)

    family = infer_model_family(config) or type(model).__name__
    summary = extract_summary(
        config,
        model_family=family,
        confidence="high",
        extra={"backbone_class": type(model).__name__},
    )
    enriched_source = dict(source)
    enriched_source.setdefault("strategy", "meta-introspect")
    enriched_source["backbone_class"] = type(model).__name__
    return ModelStructure(
        summary=summary,
        source=enriched_source,
        root=folded_root,
        extra_config=_extra_config(config),
    )


def _load_config(AutoConfig: Any, config: dict[str, Any], local_dir: Path | None) -> Any:
    if local_dir is not None and local_dir.exists():
        try:
            return AutoConfig.from_pretrained(str(local_dir), trust_remote_code=True)
        except Exception as exc:  # noqa: BLE001
            raise IntrospectionError(f"AutoConfig.from_pretrained failed: {exc}") from exc

    model_type = config.get("model_type")
    if not model_type:
        raise IntrospectionError("Config missing model_type and no local directory available.")
    if config.get("auto_map"):
        raise IntrospectionError(
            "Config requires custom remote code (auto_map) but no local model directory is available."
        )
    try:
        return AutoConfig.for_model(model_type, **{k: v for k, v in config.items() if k != "model_type"})
    except Exception as exc:  # noqa: BLE001
        raise IntrospectionError(f"AutoConfig.for_model failed: {exc}") from exc


def _walk(module: Any, *, attribute_name: str, path: str) -> StructureNode:
    class_name = type(module).__name__
    node_type = semantics.classify(module)
    attrs = semantics.extract_attributes(module)
    attrs["class"] = class_name

    children: list[StructureNode] = []
    for name, child in module.named_children():
        child_path = f"{path}.{name}" if name else path
        children.append(_walk(child, attribute_name=name, path=child_path))

    display = semantics.display_name(attribute_name, module) if attribute_name else class_name
    return StructureNode(
        id=path,
        name=display,
        type=node_type,
        attributes=_clean(attrs),
        confidence="high",
        children=children,
    )


def _clean(values: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in values.items() if v is not None}


def _extra_config(config: dict[str, Any]) -> dict[str, Any]:
    skip = {
        "model_type",
        "architectures",
        "auto_map",
        "torch_dtype",
        "transformers_version",
        "text_config",
        "vision_config",
    }
    return {k: v for k, v in config.items() if k not in skip}
