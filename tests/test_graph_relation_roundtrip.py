from model_structure_viewer.schemas import ModelStructure


def test_graph_relations_survive_model_serialization():
    payload = {
        "summary": {}, "source": {},
        "graph": {
            "nodes": [{"id": "a"}, {"id": "b"}],
            "edges": [{
                "id": "a=>b", "source": "a", "target": "b",
                "source_canonical_id": "layers.2.indexer",
                "target_canonical_id": "layers.3.index_reuse",
                "relation": "index-reuse", "label": "source layer 2",
            }],
        },
    }
    result = ModelStructure.model_validate(payload).model_dump()
    assert result["graph"]["version"] == 2
    assert result["graph"]["schema_version"] == 2
    edge = result["graph"]["edges"][0]
    assert edge["relation"] == "index-reuse"
    assert edge["label"] == "source layer 2"
    assert edge["source_canonical_id"] == "layers.2.indexer"
