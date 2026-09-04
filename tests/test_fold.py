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


def test_collapse_repeated_layer_pattern_with_separator():
    children = (
        [_layer("GlmMoeDsaDecoderLayer", i) for i in range(3)]
        + [_layer("DenseTransitionLayer", 3)]
        + [_layer("GlmMoeDsaDecoderLayer", i) for i in range(4, 7)]
        + [_layer("DenseTransitionLayer", 7)]
    )
    parent = StructureNode(id="root.layers", name="layers", type="module-list", children=children)

    folded = collapse(parent)

    assert len(folded.children) == 1
    group = folded.children[0]
    assert group.type == "layer-pattern-group"
    assert group.repeat == 2
    assert group.attributes["pattern"] == "GlmMoeDsaDecoderLayer x3 + DenseTransitionLayer"
    assert group.attributes["range"] == "0..7"
    assert len(group.children) == 2
    assert group.children[0].repeat == 3
    assert group.children[1].attributes["class"] == "DenseTransitionLayer"


def test_collapse_repeated_layer_pattern_preserves_incomplete_tail():
    children = (
        [_layer("GlmMoeDsaDecoderLayer", i) for i in range(3)]
        + [_layer("DenseTransitionLayer", 3)]
        + [_layer("GlmMoeDsaDecoderLayer", i) for i in range(4, 7)]
        + [_layer("DenseTransitionLayer", 7)]
        + [_layer("GlmMoeDsaDecoderLayer", i) for i in range(8, 11)]
    )
    parent = StructureNode(id="root.layers", name="layers", type="module-list", children=children)

    folded = collapse(parent)

    assert len(folded.children) == 2
    pattern = folded.children[0]
    tail = folded.children[1]
    assert pattern.type == "layer-pattern-group"
    assert pattern.repeat == 2
    assert pattern.attributes["range"] == "0..7"
    assert tail.type == "layer-group"
    assert tail.repeat == 3
    assert tail.attributes["range"] == "8..10"


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


def _layer_with_shapes(class_name: str, idx: int, weight_shapes: dict):
    return StructureNode(
        id=f"root.layers.{idx}",
        name=str(idx),
        type="module",
        attributes={"class": class_name},
        weight_shapes=weight_shapes,
    )


def test_collapse_does_not_fold_same_class_different_weight_shapes():
    """G3：class 相同但中间维度（weight_shapes）不同 → 不折叠，防止误折叠混合层。"""
    dense = {"gate_proj.weight": [4096, 1024], "up_proj.weight": [4096, 1024], "down_proj.weight": [1024, 4096]}
    moe = {"gate_proj.weight": [16384, 1024], "up_proj.weight": [16384, 1024], "down_proj.weight": [1024, 16384]}
    children = [
        _layer_with_shapes("DecoderLayer", 0, dense),
        _layer_with_shapes("DecoderLayer", 1, dense),
        _layer_with_shapes("DecoderLayer", 2, moe),
        _layer_with_shapes("DecoderLayer", 3, moe),
    ]
    parent = StructureNode(id="root.layers", name="layers", type="module-list", children=children)

    folded = collapse(parent)

    assert len(folded.children) == 2
    assert folded.children[0].repeat == 2
    assert folded.children[1].repeat == 2
    # 两组 range 分开，不跨组误折叠
    assert folded.children[0].attributes["range"] == "0..1"
    assert folded.children[1].attributes["range"] == "2..3"


def test_collapse_folds_same_class_same_weight_shapes():
    """G3：class 与 weight_shapes 均相同的层正常折叠。"""
    shapes = {"gate_proj.weight": [12288, 4096], "down_proj.weight": [4096, 12288]}
    children = [_layer_with_shapes("DecoderLayer", i, dict(shapes)) for i in range(4)]
    parent = StructureNode(id="root.layers", name="layers", type="module-list", children=children)

    folded = collapse(parent)

    assert len(folded.children) == 1
    assert folded.children[0].repeat == 4


def test_collapse_weight_shapes_key_order_insensitive():
    """G3：weight_shapes 键顺序不同但内容相同 → 视为同构。"""
    a = {"gate_proj.weight": [12288, 4096], "up_proj.weight": [12288, 4096]}
    b = {"up_proj.weight": [12288, 4096], "gate_proj.weight": [12288, 4096]}
    children = [_layer_with_shapes("DecoderLayer", 0, a), _layer_with_shapes("DecoderLayer", 1, b)]
    parent = StructureNode(id="root.layers", name="layers", type="module-list", children=children)

    folded = collapse(parent)

    assert len(folded.children) == 1
    assert folded.children[0].repeat == 2
