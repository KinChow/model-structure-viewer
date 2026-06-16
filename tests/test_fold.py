"""Tests for structure.fold.collapse."""
from model_structure_viewer.schemas import StructureNode
from model_structure_viewer.structure.fold import collapse


def _layer(class_name: str, idx: int) -> StructureNode:
    return StructureNode(
        id=f"root.layers.{idx}",
        name=str(idx),
        type="module",
        attributes={"class": class_name},
    )


def test_collapse_single_homogeneous_group():
    parent = StructureNode(
        id="root.layers",
        name="layers",
        type="module-list",
        children=[_layer("DecoderLayer", i) for i in range(6)],
    )
    folded = collapse(parent)
    assert len(folded.children) == 1
    assert folded.children[0].repeat == 6
    assert folded.children[0].type == "layer-group"
    assert folded.children[0].attributes["range"] == "0..5"


def test_collapse_heterogeneous_groups_split():
    children = [_layer("Dense", i) for i in range(2)] + [_layer("MoE", i) for i in range(2, 7)]
    parent = StructureNode(id="root.layers", name="layers", type="module-list", children=children)
    folded = collapse(parent)
    assert len(folded.children) == 2
    assert folded.children[0].repeat == 2
    assert folded.children[1].repeat == 5


def test_collapse_does_not_fold_different_classes():
    children = [_layer("A", 0), _layer("B", 1), _layer("A", 2)]
    parent = StructureNode(id="root.layers", name="layers", type="module-list", children=children)
    folded = collapse(parent)
    # Three distinct consecutive groups; nothing collapses since no two adjacent share class.
    assert len(folded.children) == 3
    for child in folded.children:
        assert child.type == "module"


def test_collapse_recurses_into_nested_module_list():
    inner = StructureNode(
        id="root.outer.0.inner",
        name="inner",
        type="module-list",
        children=[_layer("Block", i) for i in range(4)],
    )
    outer_child = StructureNode(
        id="root.outer.0",
        name="0",
        type="module",
        attributes={"class": "Outer"},
        children=[inner],
    )
    parent = StructureNode(
        id="root.outer", name="outer", type="module-list", children=[outer_child]
    )
    folded = collapse(parent)
    nested = folded.children[0].children[0]
    assert len(nested.children) == 1
    assert nested.children[0].repeat == 4


def test_collapse_preserves_singleton():
    parent = StructureNode(
        id="root.layers",
        name="layers",
        type="module-list",
        children=[_layer("Solo", 0)],
    )
    folded = collapse(parent)
    assert len(folded.children) == 1
    assert folded.children[0].type == "module"
    assert folded.children[0].repeat is None
