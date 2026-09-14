"""Plan A: instantiate the model on the meta device and walk nn.Module tree."""
from __future__ import annotations

import logging
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from ..errors import IntrospectionError
from ..schemas import ModelStructure
from . import semantics
from .fold import _weight_shapes_key
from .graph import GraphDraft, collapse_graph
from .keys import make_extra_config
from .repair.runtime import ConfigNormalizer, RuntimePatch
from .source_ref import collect_source_ref
from .summary import extract_summary, infer_model_family

_LOG = logging.getLogger(__name__)


__all__ = ["build_from_meta_model"]


def build_from_meta_model(
    config: dict[str, Any],
    *,
    source: dict[str, Any],
    local_dir: Path | None = None,
    config_overrides: dict[str, Any] | None = None,
    runtime_patch: RuntimePatch | None = None,
    config_normalizer: ConfigNormalizer | None = None,
    collapse_repeated: bool = True,
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
                # Keep True here: V3.1/K2.5 auto_map AutoModel to Hub modeling
                # (ModuleList of experts). Config loading already skipped catalog
                # copies of in-tree files; the model class still comes from auto_map.
                model = AutoModel.from_config(hf_config, trust_remote_code=True)
        except Exception as exc:  # noqa: BLE001  - third-party can raise anything
            _LOG.info("AutoModel.from_config failed for %s: %s", config.get("model_type"), exc)
            raise IntrospectionError(f"AutoModel.from_config failed: {exc}") from exc

    graph = _build_graph_draft(model, collapse_repeated=collapse_repeated).finalize()
    if collapse_repeated:
        graph = collapse_graph(graph)

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
    # P7（步骤 7）：ModelStructure 只携带 Graph IR——root 补投影退役
    # （schemas.ModelStructure 的 graph 为唯一必需载荷）。
    return ModelStructure(
        summary=summary,
        source=enriched_source,
        graph=graph,
        extra_config=make_extra_config(config),
    )


def _import_introspection_deps() -> tuple[Any, Any, Any]:
    import torch  # noqa: F401  (ensures torch is importable before init_empty_weights)
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModel

    return AutoConfig, AutoModel, init_empty_weights


def _auto_map_dict(config: Any) -> dict[str, Any]:
    raw = config.get("auto_map") if isinstance(config, dict) else getattr(config, "auto_map", None)
    return raw if isinstance(raw, dict) else {}


def _trust_remote_code_for_config(config: Any) -> bool:
    """Prefer in-tree AutoConfig unless the checkpoint ships a paired remote model.

    MiniMax-M3: ``auto_map`` only lists AutoConfig, and the catalog file is a
    generated ``from ...modeling_rope_utils`` copy. transformers CONFIG_MAPPING
    already has ``minimax_m3_vl`` — do not execute that copy.

    DeepSeek-V3 / Kimi-K2: ``auto_map`` pairs AutoConfig with AutoModel
    (Hub ``modeling_deepseek.py`` still reads ``config.rope_theta``). The
    in-tree ``DeepseekV3Config`` dropped that attribute, so config and model
    must come from the same remote files.
    """
    auto_map = _auto_map_dict(config)
    if auto_map.get("AutoConfig") and auto_map.get("AutoModel"):
        return True
    model_type = config.get("model_type") if isinstance(config, dict) else getattr(config, "model_type", None)
    if not model_type:
        return bool(auto_map)
    try:
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING
    except ImportError:
        return True
    return model_type not in CONFIG_MAPPING


def _load_config(
    AutoConfig: Any,
    config: dict[str, Any],
    local_dir: Path | None,
    *,
    config_overrides: dict[str, Any] | None = None,
) -> Any:
    if local_dir is not None and (local_dir / "config.json").exists():
        try:
            hf_config = AutoConfig.from_pretrained(
                str(local_dir),
                trust_remote_code=_trust_remote_code_for_config(config),
            )
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


def _immediate_child_classes(module: Any) -> tuple[str, ...]:
    children = getattr(module, "_modules", None)
    if isinstance(children, dict):
        return tuple(type(child).__name__ for child in children.values() if child is not None)
    return tuple(type(child).__name__ for _, child in module.named_children())


def _module_iso_key(module: Any) -> tuple:
    """Own class + own weight shapes + immediate child classes.

    DeepSeek-V3 remote MoE is 256 identical DeepseekV3MLP under one ModuleList:
    experts share class and child layout, so one representative is enough.
    Decoder layers that swap MLP for MoE differ in child class names
    (DeepseekV3MLP vs DeepseekV3MoE). Read ``_modules`` so comparing a decoder
    layer does not recursively enter the expert ModuleList.
    """
    metadata = _direct_parameter_metadata(module, "")
    return (type(module).__name__, _weight_shapes_key(metadata.get("weight_shapes")), _immediate_child_classes(module))


def _mark_repeated_group(draft: GraphDraft, node_id: str, *, repeat: int, range_label: str, group_index: int) -> None:
    """Match fold.collapse consecutive-group output so a second collapse is a no-op."""
    node = draft._nodes_by_id[node_id]
    attributes = dict(node.attributes)
    attributes["range"] = range_label
    attributes.pop("source_ref", None)
    canonical = f"{node.canonical_id}.group{group_index}"
    updated = node.model_copy(update={
        "type": "layer-group",
        "repeat": repeat,
        "name": f"{node.name} x{repeat}",
        "attributes": attributes,
        "canonical_id": canonical,
        "module_id": canonical,
    })
    draft.replace_node(node_id, updated)


def _build_graph_draft(module: Any, *, collapse_repeated: bool = True) -> GraphDraft:
    draft = GraphDraft()

    def visit(current: Any, *, attribute_name: str, path: str, parent_id: str | None, order: int) -> None:
        class_name = type(current).__name__
        node_type = semantics.classify(current)
        attrs = _drop_none({
            **semantics.extract_attributes(current),
            "class": class_name,
            "source_ref": collect_source_ref(current),
        })
        display = semantics.display_name(attribute_name, current) if attribute_name else class_name
        metadata = _direct_parameter_metadata(current, path)
        draft.add_node(
            node_id=path,
            canonical_id=path,
            parent_id=parent_id,
            order=order,
            name=display,
            type=node_type,
            attributes=attrs,
            confidence="high",
            **metadata,
        )
        named = list(current.named_children())
        child_paths: list[str] = []
        if collapse_repeated and node_type == "module-list" and named:
            index = 0
            group_index = 0
            emitted_order = 0
            while index < len(named):
                name, child = named[index]
                key = _module_iso_key(child)
                end = index + 1
                while end < len(named) and _module_iso_key(named[end][1]) == key:
                    end += 1
                child_path = f"{path}.{name}" if name else path
                child_paths.append(child_path)
                visit(child, attribute_name=name, path=child_path, parent_id=path, order=emitted_order)
                run = end - index
                if run > 1:
                    last_name = named[end - 1][0]
                    range_label = f"{name}..{last_name}" if name != last_name else name
                    _mark_repeated_group(
                        draft,
                        child_path,
                        repeat=run,
                        range_label=range_label,
                        group_index=group_index,
                    )
                index = end
                group_index += 1
                emitted_order += 1
        else:
            for child_order, (name, child) in enumerate(named):
                child_path = f"{path}.{name}" if name else path
                child_paths.append(child_path)
                visit(child, attribute_name=name, path=child_path, parent_id=path, order=child_order)
        for source, target in zip(child_paths, child_paths[1:]):
            draft.add_dataflow(source, target)

    visit(module, attribute_name="", path="root", parent_id=None, order=0)
    return draft


def _drop_none(values: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in values.items() if v is not None}


def _direct_parameter_metadata(module: Any, path: str) -> dict[str, Any]:
    named_parameters = getattr(module, "named_parameters", None)
    if not callable(named_parameters):
        return {}
    shapes: dict[str, list[int]] = {}
    tensor_names: list[str] = []
    dtype_elements: dict[str, int] = {}
    params = 0
    for name, parameter in named_parameters(recurse=False):
        shape = list(parameter.shape)
        elements = parameter.numel()
        dtype = _dtype_name(parameter.dtype)
        shapes[name] = shape
        tensor_names.append(f"{path}.{name}")
        params += elements
        dtype_elements[dtype] = dtype_elements.get(dtype, 0) + elements
    if not shapes:
        return {}
    dominant_dtype = max(dtype_elements, key=dtype_elements.get)
    return {
        "params": params,
        "weight_shapes": shapes,
        "dtype": dominant_dtype,
        "value_source": "introspect",
        "tensor_names": tensor_names,
    }


def _dtype_name(dtype: Any) -> str:
    aliases = {
        "bfloat16": "BF16",
        "float16": "F16",
        "float32": "F32",
        "float64": "F64",
        "float8_e4m3fn": "F8_E4M3",
        "float8_e5m2": "F8_E5M2",
        "int8": "I8",
        "int16": "I16",
        "int32": "I32",
        "int64": "I64",
        "uint8": "U8",
        "bool": "BOOL",
    }
    name = str(dtype).removeprefix("torch.").lower()
    return aliases.get(name, name.upper())
