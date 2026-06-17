from types import SimpleNamespace

from model_structure_viewer.structure.repair.strategies.minimax_config_adapter import MiniMaxConfigNormalizer


def test_minimax_config_normalizer_sets_missing_top_level_temporal_patch_size():
    config = SimpleNamespace(vision_config=SimpleNamespace())
    normalizer = MiniMaxConfigNormalizer(temporal_patch_size=2)

    diagnostics = normalizer.normalize(config)

    assert config.temporal_patch_size == 2
    assert diagnostics["config_normalizer"] == "minimax_config_normalizer"
    assert diagnostics["normalized_fields"] == ["temporal_patch_size"]
    assert "root" in diagnostics["normalized_targets"]


def test_minimax_config_normalizer_sets_missing_vision_temporal_patch_size():
    vision_config = SimpleNamespace()
    config = SimpleNamespace(vision_config=vision_config)
    normalizer = MiniMaxConfigNormalizer(temporal_patch_size=2)

    diagnostics = normalizer.normalize(config)

    assert vision_config.temporal_patch_size == 2
    assert "vision_config" in diagnostics["normalized_targets"]


def test_minimax_config_normalizer_does_not_overwrite_existing_values():
    vision_config = SimpleNamespace(temporal_patch_size=8)
    config = SimpleNamespace(temporal_patch_size=4, vision_config=vision_config)
    normalizer = MiniMaxConfigNormalizer(temporal_patch_size=2)

    diagnostics = normalizer.normalize(config)

    assert config.temporal_patch_size == 4
    assert vision_config.temporal_patch_size == 8
    assert diagnostics["normalized_targets"] == []


def test_minimax_config_normalizer_sets_missing_vision_rope_parameters():
    vision_config = SimpleNamespace(rope_theta=10000.0)
    config = SimpleNamespace(vision_config=vision_config)
    normalizer = MiniMaxConfigNormalizer(temporal_patch_size=2, spatial_merge_size=2)

    diagnostics = normalizer.normalize(config)

    assert vision_config.rope_parameters == {"rope_theta": 10000.0}
    assert "vision_config.rope_parameters" in diagnostics["normalized_targets"]
    assert "rope_parameters" in diagnostics["normalized_fields"]


def test_minimax_config_normalizer_sets_missing_vision_spatial_merge_size():
    vision_config = SimpleNamespace()
    config = SimpleNamespace(vision_config=vision_config)
    normalizer = MiniMaxConfigNormalizer(temporal_patch_size=2, spatial_merge_size=2)

    diagnostics = normalizer.normalize(config)

    assert vision_config.spatial_merge_size == 2
    assert "vision_config.spatial_merge_size" in diagnostics["normalized_targets"]
    assert "spatial_merge_size" in diagnostics["normalized_fields"]


def test_minimax_config_normalizer_sets_missing_merged_hidden_size():
    text_config = SimpleNamespace(hidden_size=6144)
    vision_config = SimpleNamespace(spatial_merge_size=2)
    config = SimpleNamespace(text_config=text_config, vision_config=vision_config)
    normalizer = MiniMaxConfigNormalizer(temporal_patch_size=2, spatial_merge_size=2)

    diagnostics = normalizer.normalize(config)

    assert config.merged_hidden_size == 24576
    assert "root.merged_hidden_size" in diagnostics["normalized_targets"]
    assert "merged_hidden_size" in diagnostics["normalized_fields"]


def test_minimax_config_normalizer_sets_missing_text_pad_token_id_default():
    text_config = SimpleNamespace()
    config = SimpleNamespace(text_config=text_config)
    normalizer = MiniMaxConfigNormalizer(temporal_patch_size=2)

    diagnostics = normalizer.normalize(config)

    assert text_config.pad_token_id is None
    assert "text_config.pad_token_id" in diagnostics["normalized_targets"]
    assert "pad_token_id" in diagnostics["normalized_fields"]


def test_minimax_config_normalizer_sets_missing_text_attention_dropout_default():
    text_config = SimpleNamespace()
    config = SimpleNamespace(text_config=text_config)
    normalizer = MiniMaxConfigNormalizer(temporal_patch_size=2)

    diagnostics = normalizer.normalize(config)

    assert text_config.attention_dropout == 0.0
    assert "text_config.attention_dropout" in diagnostics["normalized_targets"]
    assert "attention_dropout" in diagnostics["normalized_fields"]


def test_minimax_config_normalizer_sets_missing_text_layer_types():
    text_config = SimpleNamespace()
    config = SimpleNamespace(text_config=text_config)
    normalizer = MiniMaxConfigNormalizer(
        temporal_patch_size=2,
        layer_types=["full_attention", "minimax_m3_sparse"],
    )

    diagnostics = normalizer.normalize(config)

    assert text_config.layer_types == ["full_attention", "minimax_m3_sparse"]
    assert "text_config.layer_types" in diagnostics["normalized_targets"]
    assert "layer_types" in diagnostics["normalized_fields"]


def test_minimax_config_normalizer_sets_missing_text_mlp_layer_types():
    text_config = SimpleNamespace()
    config = SimpleNamespace(text_config=text_config)
    normalizer = MiniMaxConfigNormalizer(
        temporal_patch_size=2,
        mlp_layer_types=["dense", "sparse"],
    )

    diagnostics = normalizer.normalize(config)

    assert text_config.mlp_layer_types == ["dense", "sparse"]
    assert "text_config.mlp_layer_types" in diagnostics["normalized_targets"]
    assert "mlp_layer_types" in diagnostics["normalized_fields"]


def test_minimax_config_normalizer_sets_missing_sparse_index_fields():
    text_config = SimpleNamespace()
    config = SimpleNamespace(text_config=text_config)
    normalizer = MiniMaxConfigNormalizer(
        temporal_patch_size=2,
        sparse_index_fields={
            "index_n_heads": 4,
            "index_head_dim": 128,
            "index_block_size": 128,
            "index_topk_blocks": 16,
            "index_local_blocks": 1,
        },
    )

    diagnostics = normalizer.normalize(config)

    assert text_config.index_n_heads == 4
    assert text_config.index_head_dim == 128
    assert text_config.index_block_size == 128
    assert text_config.index_topk_blocks == 16
    assert text_config.index_local_blocks == 1
    assert "text_config.index_head_dim" in diagnostics["normalized_targets"]
    assert "index_head_dim" in diagnostics["normalized_fields"]


def test_minimax_config_normalizer_sets_missing_text_rope_parameters():
    text_config = SimpleNamespace()
    config = SimpleNamespace(text_config=text_config)
    normalizer = MiniMaxConfigNormalizer(
        temporal_patch_size=2,
        text_rope_parameters={
            "rope_type": "default",
            "rope_theta": 5000000,
            "partial_rotary_factor": 0.5,
        },
    )

    diagnostics = normalizer.normalize(config)

    assert text_config.rope_parameters == {
        "rope_type": "default",
        "rope_theta": 5000000,
        "partial_rotary_factor": 0.5,
    }
    assert "text_config.rope_parameters" in diagnostics["normalized_targets"]
    assert "rope_parameters" in diagnostics["normalized_fields"]


def test_minimax_config_normalizer_handles_missing_vision_config():
    config = SimpleNamespace()
    normalizer = MiniMaxConfigNormalizer(temporal_patch_size=2)

    diagnostics = normalizer.normalize(config)

    assert config.temporal_patch_size == 2
    assert diagnostics["normalized_targets"] == ["root"]
