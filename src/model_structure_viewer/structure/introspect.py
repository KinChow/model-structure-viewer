"""Plan A: instantiate the model on the meta device and walk nn.Module tree."""
from __future__ import annotations

import logging
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from ..errors import IntrospectionError
from ..schemas import ModelStructure, StructureNode
from . import fold, semantics
from .keys import make_extra_config
from .repair.runtime import ConfigNormalizer, RuntimePatch
from .summary import extract_summary, infer_model_family

_LOG = logging.getLogger(__name__)


__all__ = ["build_from_meta_model", "IntrospectionError"]


def build_from_meta_model(
    config: dict[str, Any],
    *,
    source: dict[str, Any],
    local_dir: Path | None = None,
    config_overrides: dict[str, Any] | None = None,
    runtime_patch: RuntimePatch | None = None,
    config_normalizer: ConfigNormalizer | None = None,
) -> ModelStructure:
    """Construct a ModelStructure by walking the live nn.Module tree on meta device."""
    try:
        AutoConfig, AutoModel, init_empty_weights = _import_introspection_deps()
    except ImportError as exc:
        raise IntrospectionError(f"Missing optional dependency: {exc}") from exc

    patch_context = runtime_patch.activate() if runtime_patch is not None else nullcontext()
    with patch_context:
        hf_config = _load_config(AutoConfig, config, local_dir, config_overrides=config_overrides)
        normalizer_diagnostics = _apply_config_normalizer(hf_config, config_normalizer)

        try:
            with init_empty_weights():
                model = AutoModel.from_config(hf_config, trust_remote_code=True)
        except Exception as exc:  # noqa: BLE001  - third-party can raise anything
            _LOG.info("AutoModel.from_config failed for %s: %s", config.get("model_type"), exc)
            raise IntrospectionError(f"AutoModel.from_config failed: {exc}") from exc

    raw_root = _walk(model, attribute_name="", path="root")
    folded_root = fold.collapse(raw_root)

    family = infer_model_family(config) or type(model).__name__
    summary = extract_summary(
        config,
        model_family=family,
        confidence="high",
        extra={"backbone_class": type(model).__name__},
        strategy="meta-introspect",
    )
    enriched_source = dict(source)
    enriched_source.setdefault("strategy", "meta-introspect")
    enriched_source["backbone_class"] = type(model).__name__
    if normalizer_diagnostics:
        diagnostics = dict(enriched_source.get("diagnostics") or {})
        diagnostics.update(normalizer_diagnostics)
        enriched_source["diagnostics"] = diagnostics
    return ModelStructure(
        summary=summary,
        source=enriched_source,
        root=folded_root,
        extra_config=make_extra_config(config),
    )


def _import_introspection_deps() -> tuple[Any, Any, Any]:
    import torch  # noqa: F401  (ensures torch is importable before init_empty_weights)
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModel

    return AutoConfig, AutoModel, init_empty_weights


def _load_config(
    AutoConfig: Any,
    config: dict[str, Any],
    local_dir: Path | None,
    *,
    config_overrides: dict[str, Any] | None = None,
) -> Any:
    if local_dir is not None and local_dir.exists():
        try:
            hf_config = AutoConfig.from_pretrained(str(local_dir), trust_remote_code=True)
            _apply_config_overrides(hf_config, config_overrides)
            return hf_config
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
        hf_config = AutoConfig.for_model(model_type, **{k: v for k, v in config.items() if k != "model_type"})
        _apply_config_overrides(hf_config, config_overrides)
        return hf_config
    except Exception as exc:  # noqa: BLE001
        raise IntrospectionError(f"AutoConfig.for_model failed: {exc}") from exc


def _apply_config_overrides(hf_config: Any, config_overrides: dict[str, Any] | None) -> None:
    for key, value in (config_overrides or {}).items():
        setattr(hf_config, key, value)


def _apply_config_normalizer(
    hf_config: Any,
    config_normalizer: ConfigNormalizer | None,
) -> dict[str, Any]:
    if config_normalizer is None:
        return {}
    return config_normalizer.normalize(hf_config)


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
        attributes=_drop_none(attrs),
        confidence="high",
        children=children,
    )


def _drop_none(values: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in values.items() if v is not None}
