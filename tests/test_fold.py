"""Tests for structure.fold.collapse（P7 步骤 7：夹具与断言走 Graph IR）。

旧树夹具已随树投影退役：用 GraphDraft 构图，折叠后按位序路径 id 断言，
语义身份在 canonical_id（含 group/pattern 后缀）。
"""
from model_structure_viewer.structure.fold import collapse
from model_structure_viewer.structure.graph import GraphDraft


def _graph(children_specs, parent_type="module-list"):
    """children_specs: list of (class_name, weight_shapes|None)。构造 root→layers→子层 图。"""
    draft = GraphDraft()
    draft.add_node(node_id="root", canonical_id="root", parent_id=None, order=0,
                   name="root", type="model")
    draft.add_node(node_id="root.layers", canonical_id="root.layers", parent_id="root", order=0,
                   name="layers", type=parent_type)
    for index, (class_name, weight_shapes) in enumerate(children_specs):
        draft.add_node(
            node_id=f"root.layers.{index}", canonical_id=f"root.layers.{index}",
            parent_id="root.layers", order=index, name=str(index), type="module",
            attributes={"class": class_name}, weight_shapes=weight_shapes,
        )
    return draft.finalize()


def _children_of(graph, parent_id):
    return [node for node in graph.nodes if node.parent_id == parent_id]


def test_collapse_single_homogeneous_group():
    folded = collapse(_graph([("DecoderLayer", None)] * 6))
    layers = next(node for node in folded.nodes if node.canonical_id == "root.layers")
    children = _children_of(folded, layers.id)
    assert len(children) == 1
    assert children[0].repeat == 6
    assert children[0].type == "layer-group"
    assert children[0].attributes["range"] == "0..5"


def test_collapse_heterogeneous_groups_split():
    folded = collapse(_graph([("Dense", None)] * 2 + [("MoE", None)] * 5))
    layers = next(node for node in folded.nodes if node.canonical_id == "root.layers")
    children = _children_of(folded, layers.id)
    assert len(children) == 2
    assert children[0].repeat == 2
    assert children[1].repeat == 5


def test_collapse_does_not_fold_different_classes():
    folded = collapse(_graph([("A", None), ("B", None), ("A", None)]))
    layers = next(node for node in folded.nodes if node.canonical_id == "root.layers")
    children = _children_of(folded, layers.id)
    # Three distinct consecutive groups; nothing collapses since no two adjacent share class.
    assert len(children) == 3
    for child in children:
        assert child.type == "module"


def test_collapse_repeated_layer_pattern_with_separator():
    folded = collapse(_graph(
        [("GlmMoeDsaDecoderLayer", None)] * 3
        + [("DenseTransitionLayer", None)]
        + [("GlmMoeDsaDecoderLayer", None)] * 3
        + [("DenseTransitionLayer", None)]
    ))
    layers = next(node for node in folded.nodes if node.canonical_id == "root.layers")
    children = _children_of(folded, layers.id)
    assert len(children) == 1
    group = children[0]
    assert group.type == "layer-pattern-group"
    assert group.repeat == 2
    assert group.attributes["pattern"] == "GlmMoeDsaDecoderLayer x3 + DenseTransitionLayer"
    assert group.attributes["range"] == "0..7"
    grand_children = _children_of(folded, group.id)
    assert len(grand_children) == 2
    assert grand_children[0].repeat == 3
    assert grand_children[1].attributes["class"] == "DenseTransitionLayer"


def test_collapse_repeated_layer_pattern_preserves_incomplete_tail():
    folded = collapse(_graph(
        [("GlmMoeDsaDecoderLayer", None)] * 3
        + [("DenseTransitionLayer", None)]
        + [("GlmMoeDsaDecoderLayer", None)] * 3
        + [("DenseTransitionLayer", None)]
        + [("GlmMoeDsaDecoderLayer", None)] * 3
    ))
    layers = next(node for node in folded.nodes if node.canonical_id == "root.layers")
    children = _children_of(folded, layers.id)
    assert len(children) == 2
    pattern, tail = children
    assert pattern.type == "layer-pattern-group"
    assert pattern.repeat == 2
    assert pattern.attributes["range"] == "0..7"
    assert tail.type == "layer-group"
    assert tail.repeat == 3
    assert tail.attributes["range"] == "8..10"


def test_collapse_recurses_into_nested_module_list():
    draft = GraphDraft()
    draft.add_node(node_id="root", canonical_id="root", parent_id=None, order=0,
                   name="root", type="model")
    draft.add_node(node_id="root.outer", canonical_id="root.outer", parent_id="root", order=0,
                   name="outer", type="module-list")
    draft.add_node(node_id="root.outer.0", canonical_id="root.outer.0", parent_id="root.outer", order=0,
                   name="0", type="module", attributes={"class": "Outer"})
    draft.add_node(node_id="root.outer.0.inner", canonical_id="root.outer.0.inner",
                   parent_id="root.outer.0", order=0, name="inner", type="module-list")
    for index in range(4):
        draft.add_node(node_id=f"root.outer.0.inner.{index}", canonical_id=f"root.outer.0.inner.{index}",
                       parent_id="root.outer.0.inner", order=index, name=str(index),
                       type="module", attributes={"class": "Block"})
    folded = collapse(draft.finalize())
    inner = next(node for node in folded.nodes if node.canonical_id == "root.outer.0.inner")
    nested = _children_of(folded, inner.id)
    assert len(nested) == 1
    assert nested[0].repeat == 4


def test_collapse_preserves_singleton():
    folded = collapse(_graph([("Solo", None)]))
    layers = next(node for node in folded.nodes if node.canonical_id == "root.layers")
    children = _children_of(folded, layers.id)
    assert len(children) == 1
    assert children[0].type == "module"
    assert children[0].repeat is None


def test_collapse_does_not_fold_same_class_different_weight_shapes():
    """G3：class 相同但中间维度（weight_shapes）不同 → 不折叠，防止误折叠混合层。"""
    dense = {"gate_proj.weight": [4096, 1024], "up_proj.weight": [4096, 1024], "down_proj.weight": [1024, 4096]}
    moe = {"gate_proj.weight": [16384, 1024], "up_proj.weight": [16384, 1024], "down_proj.weight": [1024, 16384]}
    folded = collapse(_graph([("DecoderLayer", dense), ("DecoderLayer", dense), ("DecoderLayer", moe), ("DecoderLayer", moe)]))
    layers = next(node for node in folded.nodes if node.canonical_id == "root.layers")
    children = _children_of(folded, layers.id)
    assert len(children) == 2
    assert children[0].repeat == 2
    assert children[1].repeat == 2
    # 两组 range 分开，不跨组误折叠
    assert children[0].attributes["range"] == "0..1"
    assert children[1].attributes["range"] == "2..3"


def test_collapse_folds_same_class_same_weight_shapes():
    """G3：class 与 weight_shapes 均相同的层正常折叠。"""
    shapes = {"gate_proj.weight": [12288, 4096], "down_proj.weight": [4096, 12288]}
    folded = collapse(_graph([("DecoderLayer", dict(shapes))] * 4))
    layers = next(node for node in folded.nodes if node.canonical_id == "root.layers")
    children = _children_of(folded, layers.id)
    assert len(children) == 1
    assert children[0].repeat == 4


def test_collapse_weight_shapes_key_order_insensitive():
    """G3：weight_shapes 键顺序不同但内容相同 → 视为同构。"""
    a = {"gate_proj.weight": [12288, 4096], "up_proj.weight": [12288, 4096]}
    b = {"up_proj.weight": [12288, 4096], "gate_proj.weight": [12288, 4096]}
    folded = collapse(_graph([("DecoderLayer", a), ("DecoderLayer", b)]))
    layers = next(node for node in folded.nodes if node.canonical_id == "root.layers")
    children = _children_of(folded, layers.id)
    assert len(children) == 1
    assert children[0].repeat == 2
