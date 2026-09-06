"""Unit tests for semantics and fold helpers."""
from model_structure_viewer.schemas import ModelStructure, StructureGraph, StructureGraphNode, StructureNode
from model_structure_viewer.structure.fold import collapse
from model_structure_viewer.structure import semantics
from model_structure_viewer.structure.introspect import _walk
from model_structure_viewer.structure.graph import materialize_structure_graph, project_graph_to_tree


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


def test_extract_linear_leaf_module_parameters():
    module = _FakeModule("Linear", in_features=7168, out_features=1536, bias=None)
    attrs = semantics.extract_attributes(module)
    assert attrs["in_features"] == 7168
    assert attrs["out_features"] == 1536
    assert attrs["bias"] is False


def test_extract_leaf_module_bias_presence():
    module = _FakeModule("Linear", in_features=1536, out_features=7168, bias=object())
    attrs = semantics.extract_attributes(module)
    assert attrs["bias"] is True


def test_extract_leaf_module_scalar_parameters():
    module = _FakeModule("Dropout", p=0.1, inplace=False)
    attrs = semantics.extract_attributes(module)
    assert attrs["p"] == 0.1
    assert attrs["inplace"] is False


def test_extract_leaf_module_tuple_parameters():
    module = _FakeModule("LayerNorm", normalized_shape=(7168,), eps=1e-6)
    attrs = semantics.extract_attributes(module)
    assert attrs["normalized_shape"] == [7168]
    assert attrs["eps"] == 1e-6


def test_extract_leaf_module_skips_complex_objects():
    module = _FakeModule("Conv2d", kernel_size=(3, 3), weight=object())
    attrs = semantics.extract_attributes(module)
    assert attrs["kernel_size"] == [3, 3]
    assert "weight" not in attrs


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


def test_introspection_shapes_prevent_folding_different_linear_layers():
    import torch

    raw = _walk(
        torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.Linear(8, 32)),
        attribute_name="",
        path="root",
    )
    folded = collapse(raw)

    assert len(folded.children) == 2
    assert folded.children[0].weight_shapes == {"weight": [16, 8], "bias": [16]}
    assert folded.children[1].weight_shapes == {"weight": [32, 8], "bias": [32]}
    assert folded.children[0].params == 16 * 8 + 16
    assert folded.children[0].dtype == "F32"
    assert folded.children[0].value_source == "introspect"


def test_structure_graph_materializes_stable_paths_and_edges():
    root = StructureNode(
        id="model",
        name="Model",
        type="model",
        children=[
            StructureNode(id="embed", name="Embed", type="embedding"),
            StructureNode(id="decoder", name="Decoder", type="decoder"),
            StructureNode(id="head", name="Head", type="output"),
        ],
    )

    graph = materialize_structure_graph(root)

    assert graph.version == 2
    assert graph.schema_version == 2
    assert graph.root_id == "root"
    assert [node.id for node in graph.nodes] == ["root", "root.0", "root.1", "root.2"]
    assert [node.canonical_id for node in graph.nodes] == ["model", "embed", "decoder", "head"]
    assert graph.nodes[1].name == "Embed"
    assert graph.nodes[1].order == 0
    assert graph.edges[0].source_canonical_id == "embed"
    assert graph.edges[0].target_canonical_id == "decoder"
    projected = project_graph_to_tree(graph)
    assert projected.id == "model"
    assert [child.id for child in projected.children] == ["embed", "decoder", "head"]
    assert [(edge.source, edge.target) for edge in graph.edges] == [
        ("root.0", "root.1"),
        ("root.1", "root.2"),
    ]


def test_model_structure_normalizes_root_and_graph_as_reversible_views():
    root_only = ModelStructure(root=StructureNode(id="model", name="Model", type="model"))
    assert root_only.graph is None

    graph_only = ModelStructure(graph=StructureGraph(
        nodes=[StructureGraphNode(id="root", canonical_id="model", name="Graph Model", type="model")],
    ))
    assert graph_only.root is not None
    assert graph_only.root.id == "model"
    assert graph_only.root.name == "Graph Model"
