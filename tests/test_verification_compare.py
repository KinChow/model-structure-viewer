from model_structure_viewer.verification.compare_structure import compare_structure_summary


def test_compare_structure_summary_passes_matching_layers_and_architecture():
    result = compare_structure_summary(
        predicted={
            "summary": {
                "canonical_architecture": "mla-moe-decoder",
                "text_layers": 61,
            }
        },
        reference={
            "summary": {
                "canonical_architecture": "mla-moe-decoder",
                "text_layers": 61,
            }
        },
    )

    assert result["status"] == "passed"
    assert result["errors"] == []


def test_compare_structure_summary_fails_layer_mismatch():
    result = compare_structure_summary(
        predicted={"summary": {"canonical_architecture": "mla-moe-decoder", "text_layers": 60}},
        reference={"summary": {"canonical_architecture": "mla-moe-decoder", "text_layers": 61}},
    )

    assert result["status"] == "failed"
    assert "text_layers mismatch" in result["errors"][0]
