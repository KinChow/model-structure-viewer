from __future__ import annotations

from typing import Any

from ..context import RepairContext, RepairResult
from ..errors import IntrospectionFailureKind


class MiniMaxM2ConfigNormalizer:
    name = "minimax_m2_config_normalizer"

    def __init__(self, rope_parameters: dict[str, Any]):
        self.rope_parameters = rope_parameters

    def normalize(self, hf_config: Any) -> dict[str, Any]:
        normalized_fields: set[str] = set()
        normalized_targets: list[str] = []
        if not hasattr(hf_config, "rope_parameters"):
            setattr(hf_config, "rope_parameters", self.rope_parameters)
            normalized_fields.add("rope_parameters")
            normalized_targets.append("root.rope_parameters")
        return {
            "config_normalizer": self.name,
            "normalized_fields": sorted(normalized_fields),
            "normalized_targets": normalized_targets,
        }


class MiniMaxConfigNormalizer:
    name = "minimax_config_normalizer"

    def __init__(
        self,
        temporal_patch_size: Any,
        spatial_merge_size: Any | None = None,
        layer_types: list[str] | None = None,
        mlp_layer_types: list[str] | None = None,
        sparse_index_fields: dict[str, Any] | None = None,
        text_rope_parameters: dict[str, Any] | None = None,
    ):
        self.temporal_patch_size = temporal_patch_size
        self.spatial_merge_size = spatial_merge_size
        self.layer_types = layer_types
        self.mlp_layer_types = mlp_layer_types
        self.sparse_index_fields = sparse_index_fields or {}
        self.text_rope_parameters = text_rope_parameters

    def normalize(self, hf_config: Any) -> dict[str, Any]:
        normalized_fields: set[str] = set()
        normalized_targets: list[str] = []
        if not hasattr(hf_config, "temporal_patch_size"):
            setattr(hf_config, "temporal_patch_size", self.temporal_patch_size)
            normalized_fields.add("temporal_patch_size")
            normalized_targets.append("root")

        vision_config = getattr(hf_config, "vision_config", None)
        if vision_config is not None:
            if not hasattr(vision_config, "temporal_patch_size"):
                setattr(vision_config, "temporal_patch_size", self.temporal_patch_size)
                normalized_fields.add("temporal_patch_size")
                normalized_targets.append("vision_config")
            if not hasattr(vision_config, "rope_parameters") and hasattr(vision_config, "rope_theta"):
                setattr(vision_config, "rope_parameters", {"rope_theta": vision_config.rope_theta})
                normalized_fields.add("rope_parameters")
                normalized_targets.append("vision_config.rope_parameters")
            if self.spatial_merge_size is not None and not hasattr(vision_config, "spatial_merge_size"):
                setattr(vision_config, "spatial_merge_size", self.spatial_merge_size)
                normalized_fields.add("spatial_merge_size")
                normalized_targets.append("vision_config.spatial_merge_size")

        text_config = getattr(hf_config, "text_config", None)
        if text_config is not None:
            if not hasattr(text_config, "pad_token_id"):
                setattr(text_config, "pad_token_id", None)
                normalized_fields.add("pad_token_id")
                normalized_targets.append("text_config.pad_token_id")
            if not hasattr(text_config, "attention_dropout"):
                setattr(text_config, "attention_dropout", 0.0)
                normalized_fields.add("attention_dropout")
                normalized_targets.append("text_config.attention_dropout")
            if self.layer_types is not None and not hasattr(text_config, "layer_types"):
                setattr(text_config, "layer_types", self.layer_types)
                normalized_fields.add("layer_types")
                normalized_targets.append("text_config.layer_types")
            if self.mlp_layer_types is not None and not hasattr(text_config, "mlp_layer_types"):
                setattr(text_config, "mlp_layer_types", self.mlp_layer_types)
                normalized_fields.add("mlp_layer_types")
                normalized_targets.append("text_config.mlp_layer_types")
            for field, value in self.sparse_index_fields.items():
                if not hasattr(text_config, field):
                    setattr(text_config, field, value)
                    normalized_fields.add(field)
                    normalized_targets.append(f"text_config.{field}")
            if self.text_rope_parameters is not None and not hasattr(text_config, "rope_parameters"):
                setattr(text_config, "rope_parameters", self.text_rope_parameters)
                normalized_fields.add("rope_parameters")
                normalized_targets.append("text_config.rope_parameters")

        vision_config = getattr(hf_config, "vision_config", None)
        if (
            not hasattr(hf_config, "merged_hidden_size")
            and text_config is not None
            and vision_config is not None
            and hasattr(text_config, "hidden_size")
            and hasattr(vision_config, "spatial_merge_size")
        ):
            setattr(
                hf_config,
                "merged_hidden_size",
                text_config.hidden_size * (vision_config.spatial_merge_size**2),
            )
            normalized_fields.add("merged_hidden_size")
            normalized_targets.append("root.merged_hidden_size")

        return {
            "config_normalizer": self.name,
            "normalized_fields": sorted(normalized_fields),
            "normalized_targets": normalized_targets,
        }


class MiniMaxConfigAdapterStrategy:
    name = "minimax_config_adapter"

    def matches(self, context: RepairContext) -> bool:
        if context.failure_kind != IntrospectionFailureKind.CONFIG_FIELD_MISSING:
            return False
        if not _is_minimax(context):
            return False
        return "temporal_patch_size" in context.original_error or "rope_parameters" in context.original_error

    def apply(self, context: RepairContext) -> RepairResult:
        if "rope_parameters" in context.original_error and _is_minimax_m2(context):
            return _apply_minimax_m2_rope_parameters(context, strategy_name=self.name)

        patched = dict(context.config)
        temporal_patch_size = _find_nested_value(context.config, "temporal_patch_size")
        spatial_merge_size = _find_nested_value(context.config, "spatial_merge_size")
        layer_types = _derive_layer_types(context.config)
        mlp_layer_types = _derive_mlp_layer_types(context.config)
        sparse_index_fields = _derive_sparse_index_fields(context.config)
        text_rope_parameters = _derive_text_rope_parameters(context.config)
        if temporal_patch_size is None:
            return RepairResult(
                config=patched,
                local_dir=context.local_dir,
                strategy_name=self.name,
                diagnostics={
                    "repair_strategy": self.name,
                    "repair_status": "skipped",
                    "reason": "temporal_patch_size_not_found",
                },
            )
        patched["temporal_patch_size"] = temporal_patch_size
        return RepairResult(
            config=patched,
            local_dir=context.local_dir,
            strategy_name=self.name,
            diagnostics={
                "repair_strategy": self.name,
                "repair_status": "prepared",
                "patched_fields": ["temporal_patch_size"],
                "config_overrides": ["temporal_patch_size"],
                "config_normalizer": MiniMaxConfigNormalizer.name,
            },
            config_overrides={"temporal_patch_size": temporal_patch_size},
            config_normalizer=MiniMaxConfigNormalizer(
                temporal_patch_size,
                spatial_merge_size=spatial_merge_size,
                layer_types=layer_types,
                mlp_layer_types=mlp_layer_types,
                sparse_index_fields=sparse_index_fields,
                text_rope_parameters=text_rope_parameters,
            ),
        )


def _apply_minimax_m2_rope_parameters(context: RepairContext, *, strategy_name: str) -> RepairResult:
    patched = dict(context.config)
    rope_parameters = _derive_root_rope_parameters(context.config)
    if rope_parameters is None:
        return RepairResult(
            config=patched,
            local_dir=context.local_dir,
            strategy_name=strategy_name,
            diagnostics={
                "repair_strategy": strategy_name,
                "repair_status": "skipped",
                "reason": "rope_parameters_not_derived",
            },
        )
    patched["rope_parameters"] = rope_parameters
    return RepairResult(
        config=patched,
        local_dir=context.local_dir,
        strategy_name=strategy_name,
        diagnostics={
            "repair_strategy": strategy_name,
            "repair_status": "prepared",
            "patched_fields": ["rope_parameters"],
            "config_overrides": ["rope_parameters"],
            "config_normalizer": MiniMaxM2ConfigNormalizer.name,
        },
        config_overrides={"rope_parameters": rope_parameters},
        config_normalizer=MiniMaxM2ConfigNormalizer(rope_parameters),
    )


def _is_minimax(context: RepairContext) -> bool:
    values = [str(context.config.get("model_type", "")), str(context.source.get("model_id", ""))]
    values.extend(str(item) for item in context.config.get("architectures") or [])
    return any("minimax" in value.lower() for value in values)


def _is_minimax_m2(context: RepairContext) -> bool:
    values = [str(context.config.get("model_type", "")), str(context.source.get("model_id", ""))]
    values.extend(str(item) for item in context.config.get("architectures") or [])
    return any("minimax_m2" in value.lower() or "minimax-m2" in value.lower() for value in values)


def _find_nested_value(value: Any, key: str) -> Any:
    if not isinstance(value, dict):
        return None
    if key in value:
        return value[key]
    for nested in value.values():
        found = _find_nested_value(nested, key)
        if found is not None:
            return found
    return None


def _derive_root_rope_parameters(config: dict[str, Any]) -> dict[str, Any] | None:
    if isinstance(config.get("rope_parameters"), dict):
        return config["rope_parameters"]
    rope_theta = config.get("rope_theta")
    if rope_theta is None:
        return None
    parameters = {"rope_type": "default", "rope_theta": rope_theta}
    if "partial_rotary_factor" in config:
        parameters["partial_rotary_factor"] = config["partial_rotary_factor"]
    return parameters


def _derive_layer_types(config: dict[str, Any]) -> list[str] | None:
    text_config = config.get("text_config")
    if not isinstance(text_config, dict):
        return None
    if isinstance(text_config.get("layer_types"), list):
        return text_config["layer_types"]
    sparse_config = text_config.get("sparse_attention_config")
    if isinstance(sparse_config, dict) and isinstance(sparse_config.get("sparse_attention_freq"), list):
        return ["minimax_m3_sparse" if value else "full_attention" for value in sparse_config["sparse_attention_freq"]]
    num_hidden_layers = text_config.get("num_hidden_layers")
    if isinstance(num_hidden_layers, int):
        return ["full_attention"] * num_hidden_layers
    return None


def _derive_mlp_layer_types(config: dict[str, Any]) -> list[str] | None:
    text_config = config.get("text_config")
    if not isinstance(text_config, dict):
        return None
    if isinstance(text_config.get("mlp_layer_types"), list):
        return text_config["mlp_layer_types"]
    moe_layer_freq = text_config.get("moe_layer_freq")
    if isinstance(moe_layer_freq, list):
        return ["sparse" if value else "dense" for value in moe_layer_freq]
    num_hidden_layers = text_config.get("num_hidden_layers")
    if isinstance(num_hidden_layers, int):
        return ["sparse"] * num_hidden_layers
    return None


def _derive_sparse_index_fields(config: dict[str, Any]) -> dict[str, Any]:
    text_config = config.get("text_config")
    if not isinstance(text_config, dict):
        return {}
    sparse_config = text_config.get("sparse_attention_config")
    if not isinstance(sparse_config, dict):
        return {}
    mapping = {
        "index_n_heads": "sparse_num_index_heads",
        "index_head_dim": "sparse_index_dim",
        "index_block_size": "sparse_block_size",
        "index_topk_blocks": "sparse_topk_blocks",
        "index_local_blocks": "sparse_local_block",
    }
    return {field: sparse_config[legacy] for field, legacy in mapping.items() if legacy in sparse_config}


def _derive_text_rope_parameters(config: dict[str, Any]) -> dict[str, Any] | None:
    text_config = config.get("text_config")
    if not isinstance(text_config, dict):
        return None
    if isinstance(text_config.get("rope_parameters"), dict):
        return text_config["rope_parameters"]
    rope_theta = text_config.get("rope_theta")
    if rope_theta is None:
        return None
    parameters = {"rope_type": "default", "rope_theta": rope_theta}
    if "partial_rotary_factor" in text_config:
        parameters["partial_rotary_factor"] = text_config["partial_rotary_factor"]
    return parameters
