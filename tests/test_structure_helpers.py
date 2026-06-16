"""Unit tests for semantics, fold, fallback helpers."""
from model_structure_viewer.schemas import StructureNode
from model_structure_viewer.structure.fold import collapse
from model_structure_viewer.structure.fallback import build_from_config
from model_structure_viewer.structure import semantics


class _FakeModule:
    def __init__(self, class_name: str, **attrs):
        self.__class__ = type(class_name, (object,), {})
        for key, value in attrs.items():
            setattr(self, key, value)


def test_classify_attention_module():
    module = _FakeModule("DeepseekV3MLAAttention", num_attention_heads=128)
    assert semantics.classify(module) == "attention"
    attrs = semantics.extract_attributes(module)
    assert attrs.get("num_attention_heads") == 128
    assert attrs.get("kind") == "Multi-head Latent Attention"


def test_classify_moe_module():
    module = _FakeModule("MiniMaxSparseMoeBlock")
    assert semantics.classify(module) == "moe"


def test_classify_module_list():
    module = _FakeModule("ModuleList")
    assert semantics.classify(module) == "module-list"


def _layer(class_name: str, idx: int) -> StructureNode:
    return StructureNode(
        id=f"root.layers.{idx}",
        name=str(idx),
        type="module",
        attributes={"class": class_name},
    )


def test_fold_homogeneous_module_list():
    parent = StructureNode(
        id="root.layers",
        name="layers",
        type="module-list",
        children=[_layer("DecoderLayer", i) for i in range(4)],
    )
    folded = collapse(parent)
    assert len(folded.children) == 1
    assert folded.children[0].repeat == 4


def test_fold_heterogeneous_module_list_splits_groups():
    children = [_layer("DenseLayer", i) for i in range(3)] + [_layer("MoeLayer", i) for i in range(3, 9)]
    parent = StructureNode(id="root.layers", name="layers", type="module-list", children=children)
    folded = collapse(parent)
    assert len(folded.children) == 2
    assert folded.children[0].repeat == 3
    assert folded.children[1].repeat == 6


def test_fallback_handles_minimal_config():
    structure = build_from_config({"model_type": "mystery"}, source={"kind": "test"})
    assert structure.summary["confidence"] == "low"
    assert structure.root.children
