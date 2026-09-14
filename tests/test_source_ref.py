from pathlib import Path
from types import SimpleNamespace

from model_structure_viewer.cli import cmd_dump_source_ref
from model_structure_viewer.schemas import VerifyEvidence, VerifyResponse
from model_structure_viewer.structure.source_ref import collect_source_ref


def test_collect_source_ref_inspect_failure_is_null(monkeypatch):
    class Demo:
        pass

    monkeypatch.setattr("inspect.getsourcefile", lambda cls: (_ for _ in ()).throw(OSError("no source")))
    assert collect_source_ref(Demo()) is None


def test_collect_source_ref_unknown_package_has_file_but_no_url():
    class LocalOp:
        pass

    result = collect_source_ref(LocalOp())
    assert result is not None
    assert result["url"] is None
    assert result["file"]
    assert result["class_name"] == "LocalOp"


def test_collect_source_ref_known_root_builds_github_url(monkeypatch, tmp_path):
    module_file = tmp_path / "modeling_demo.py"
    module_file.write_text("class DemoAttention:\n    pass\n", encoding="utf-8")

    class DemoAttention:
        pass

    DemoAttention.__module__ = "transformers.models.demo.modeling_demo"
    monkeypatch.setattr("inspect.getsourcefile", lambda cls: str(module_file))
    monkeypatch.setattr("inspect.getsourcelines", lambda cls: (["class DemoAttention:\n"], 12))
    monkeypatch.setattr(
        "model_structure_viewer.structure.source_ref._package_roots",
        lambda: [(str(tmp_path), "huggingface/transformers", "src/transformers/", "v4.40.0")],
    )
    ref = collect_source_ref(DemoAttention())
    assert ref["file"] == "src/transformers/modeling_demo.py"
    assert ref["line"] == 12
    assert ref["url"] == "https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/modeling_demo.py#L12"
    assert ref["class_name"] == "DemoAttention"


def test_cli_dump_source_ref_writes_catalog_json(tmp_path, monkeypatch):
    captured = {}

    def fake_verify(payload, settings):
        captured["payload"] = payload
        return VerifyResponse(
            ok=True,
            status="passed",
            strategy="transformers-meta",
            model_id=payload.model_id,
            evidence=VerifyEvidence(
                modules=[
                    {
                        "path": "root.model.norm",
                        "class": "RMSNorm",
                        "params": None,
                        "source_ref": {
                            "framework": "transformers",
                            "class_name": "RMSNorm",
                            "file": "src/transformers/models/llama/modeling_llama.py",
                            "line": 80,
                            "version": "4.40.0",
                            "url": "https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/models/llama/modeling_llama.py#L80",
                        },
                    }
                ]
            ),
        )

    monkeypatch.setattr("model_structure_viewer.cli.verify_structure_response", fake_verify)
    out = tmp_path / "source-ref.json"
    args = SimpleNamespace(
        model="Org/Demo",
        source="builtin",
        revision="main",
        cache_policy="prefer-local",
        out=str(out),
    )
    assert cmd_dump_source_ref(args, SimpleNamespace()) == 0
    payload = out.read_text(encoding="utf-8")
    assert "RMSNorm" in payload
    assert captured["payload"].model_id == "Org/Demo"
    assert Path(out).exists()
