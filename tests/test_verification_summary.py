from model_structure_viewer.verification.summary import minimal_summary


def test_minimal_summary_uses_the_first_declared_architecture():
    assert minimal_summary({
        "model_type": "example",
        "architectures": ["ExampleForCausalLM", "FallbackModel"],
    }) == {
        "model_type": "example",
        "architecture": "ExampleForCausalLM",
    }


def test_minimal_summary_handles_missing_architecture():
    assert minimal_summary({"model_type": "example"}) == {
        "model_type": "example",
        "architecture": None,
    }
