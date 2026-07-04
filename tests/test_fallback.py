"""Tests for structure.fallback.build_from_config."""
from model_structure_viewer.structure.fallback import build_from_config


def test_fallback_minimal_config():
    structure = build_from_config({"model_type": "mystery"}, source={"kind": "test"})
    assert structure.summary["confidence"] == "low"
    assert structure.summary.get("strategy") == "config-fallback"
    assert structure.root.children  # at least the synthetic config leaf
    assert structure.source["strategy"] == "config-fallback"


def test_fallback_text_and_vision_nested_children():
    config = {
        "model_type": "vlm",
        "architectures": ["VLMForCausalLM"],
        "text_config": {
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
        },
        "vision_config": {
            "hidden_size": 1024,
            "num_hidden_layers": 24,
            "patch_size": 14,
            "image_size": 336,
        },
    }
    structure = build_from_config(config, source={"kind": "test"})
    child_ids = {c.id for c in structure.root.children}
    assert "text" in child_ids
    assert "vision" in child_ids

    text = next(c for c in structure.root.children if c.id == "text")
    assert text.repeat == 32
    assert text.attributes.get("hidden_size") == 4096

    vision = next(c for c in structure.root.children if c.id == "vision")
    assert vision.attributes.get("patch_size") == 14
    assert vision.attributes.get("image_size") == 336


def test_fallback_nested_text_config_includes_decoder_layers():
    structure = build_from_config(
        {
            "model_type": "vlm",
            "architectures": ["VLMForCausalLM"],
            "text_config": {
                "hidden_size": 4096,
                "num_hidden_layers": 32,
                "num_attention_heads": 32,
            },
        },
        source={"kind": "test"},
    )

    text = next(c for c in structure.root.children if c.id == "text")
    decoder = next(c for c in text.children if c.id == "decoder")
    assert decoder.repeat == 32
    assert decoder.attributes["hidden_size"] == 4096


def test_fallback_pick_filters_unknown_keys():
    config = {
        "model_type": "x",
        "hidden_size": 16,
        "completely_unrelated_key": "ignored",
    }
    structure = build_from_config(config, source={"kind": "test"})
    attrs = structure.root.attributes
    assert "hidden_size" in attrs
    assert "completely_unrelated_key" not in attrs


def test_fallback_extra_config_strips_nested_towers():
    config = {
        "model_type": "x",
        "architectures": ["XForCausalLM"],
        "torch_dtype": "bfloat16",
        "text_config": {"hidden_size": 1},
        "vision_config": {"hidden_size": 2},
        "custom_field": {"keep": True},
    }
    structure = build_from_config(config, source={"kind": "test"})
    assert "text_config" not in structure.extra_config
    assert "vision_config" not in structure.extra_config
    assert "model_type" not in structure.extra_config
    assert structure.extra_config.get("custom_field") == {"keep": True}


def test_fallback_depth_limit_truncates_deep_nesting():
    inner = {"text_config": {"hidden_size": 1}}
    cfg = inner
    # Build a chain of 12 nested text_configs to exceed _MAX_NESTED_CONFIG_DEPTH=8.
    for _ in range(12):
        cfg = {"text_config": cfg}
    cfg["model_type"] = "deep"
    structure = build_from_config(cfg, source={"kind": "test"})

    # Walk the chain; depth must be bounded.
    node = structure.root
    depth = 0
    while node.children:
        text_children = [c for c in node.children if c.id == "text"]
        if not text_children:
            break
        node = text_children[0]
        depth += 1
    assert depth <= 8


def test_fallback_reason_propagates_to_summary_and_source():
    structure = build_from_config(
        {"model_type": "x"},
        source={"kind": "test"},
        fallback_reason="meta-instantiation-failed",
    )
    assert structure.summary.get("fallback_reason") == "meta-instantiation-failed"
    assert structure.source["fallback_reason"] == "meta-instantiation-failed"


def test_fallback_uses_inference_config_aliases_for_summary_and_decoder():
    structure = build_from_config(
        {
            "n_layers": 61,
            "dim": 7168,
            "n_heads": 128,
            "inter_dim": 18432,
            "vocab_size": 129280,
        },
        source={"kind": "test"},
    )

    assert structure.summary["text_layers"] == 61
    assert structure.summary["hidden_size"] == 7168
    assert structure.summary["num_attention_heads"] == 128
    assert structure.summary["intermediate_size"] == 18432
    decoder = next(child for child in structure.root.children if child.type == "decoder")
    assert decoder.repeat == 61
    assert decoder.attributes["n_layers"] == 61


def test_fallback_accepts_strategy_and_diagnostics_without_breaking_contract():
    structure = build_from_config(
        {"model_type": "x", "num_hidden_layers": 2},
        source={"kind": "test"},
        fallback_reason="meta-instantiation-failed",
        fallback_strategy="model-aware-fallback",
        diagnostics={"failure_kind": "model_init_failed"},
    )

    assert structure.summary["strategy"] == "model-aware-fallback"
    assert structure.source["strategy"] == "model-aware-fallback"
    assert structure.source["diagnostics"] == {"failure_kind": "model_init_failed"}
    assert structure.root.children
    assert structure.extra_config is not None
