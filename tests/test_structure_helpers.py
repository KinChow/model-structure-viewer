"""Unit tests for semantics and fold helpers."""
import pytest
from pydantic import ValidationError

from model_structure_viewer.schemas import ModelStructure, StructureGraph, StructureGraphNode
from model_structure_viewer.structure import semantics
from model_structure_viewer.structure.graph import GraphDraft, collapse_graph


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


# P7（步骤 7）：树夹具折叠用例（test_fold_homogeneous/heterogeneous）已迁往
# tests/test_fold.py 的 Graph IR 夹具——树投影退役后折叠只在图上发生。


def test_introspection_shapes_prevent_folding_different_linear_layers():
    import torch

    from model_structure_viewer.structure.introspect import _build_graph_draft

    # P7（步骤 7）：_walk 兼容树视图退役——改走图草稿 + 图原生折叠。
    graph = _build_graph_draft(
        torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.Linear(8, 32))
    ).finalize()
    folded = collapse_graph(graph)

    children = [node for node in folded.nodes if node.parent_id == folded.root_id]
    assert len(children) == 2
    assert children[0].weight_shapes == {"weight": [16, 8], "bias": [16]}
    assert children[1].weight_shapes == {"weight": [32, 8], "bias": [32]}
    assert children[0].params == 16 * 8 + 16
    assert children[0].dtype == "F32"
    assert children[0].value_source == "introspect"


def test_collapse_graph_rebuilds_stable_paths_and_edges():
    """P7（步骤 7）：materialize/project 双向视图退役——折叠边界的重建契约
    （位序路径 id、canonical 语义身份、module-order 边与 canonical 端点）
    由 collapse_graph 单向保证；本夹具无 module-list，折叠退化为纯重建。"""
    draft = GraphDraft()
    draft.add_node(node_id="root", canonical_id="model", parent_id=None, order=0,
                   name="Model", type="model")
    for index, (canonical, node_type, name) in enumerate([
        ("embed", "embedding", "Embed"),
        ("decoder", "decoder", "Decoder"),
        ("head", "output", "Head"),
    ]):
        draft.add_node(node_id=f"root.{index}", canonical_id=canonical, parent_id="root",
                       order=index, name=name, type=node_type)
    graph = draft.finalize()

    rebuilt = collapse_graph(graph)

    assert rebuilt.version == 2
    assert rebuilt.schema_version == 2
    assert rebuilt.root_id == "root"
    assert [node.id for node in rebuilt.nodes] == ["root", "root.0", "root.1", "root.2"]
    assert [node.canonical_id for node in rebuilt.nodes] == ["model", "embed", "decoder", "head"]
    assert rebuilt.nodes[1].name == "Embed"
    assert rebuilt.nodes[1].order == 0
    assert rebuilt.edges[0].source_canonical_id == "embed"
    assert rebuilt.edges[0].target_canonical_id == "decoder"
    assert [(edge.source, edge.target) for edge in rebuilt.edges] == [
        ("root.0", "root.1"),
        ("root.1", "root.2"),
    ]


def test_model_structure_requires_graph_as_only_payload():
    """P7（步骤 7）：graph 是唯一必需载荷——root 视图与补投影校验器退役。"""
    structure = ModelStructure(graph=StructureGraph(
        nodes=[StructureGraphNode(id="root", canonical_id="model", name="Graph Model", type="model")],
    ))
    assert structure.graph.nodes[0].canonical_id == "model"
    assert structure.graph.nodes[0].name == "Graph Model"

    with pytest.raises(ValidationError):
        ModelStructure(summary={}, source={})
