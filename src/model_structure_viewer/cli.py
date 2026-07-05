from __future__ import annotations

import argparse
import json
import sys

from .exporters import export_structure
from .resolver import ModelSourceResolver, SourceResolutionError
from .schemas import StructureRequest, VerifyRequest
from .service import build_structure_response, verify_structure_response
from .settings import AppSettings


def add_common_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--root", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    parser.add_argument("--endpoint", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    parser.add_argument("--offline", action="store_true", default=argparse.SUPPRESS, help=argparse.SUPPRESS)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="msv", description="Model Structure Viewer")
    parser.add_argument("--root", default=None, help="Model root directory.")
    parser.add_argument("--endpoint", default=None, help="Hugging Face endpoint.")
    parser.add_argument("--offline", action="store_true", help="Disable remote HF access.")
    parser.add_argument(
        "--no-auto-fetch-remote-code",
        dest="no_auto_fetch_remote_code",
        action="store_true",
        help="Do not auto-download modeling_*.py / configuration_*.py from HF.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List local models.")
    add_common_options(list_parser)
    list_parser.set_defaults(func=cmd_list)

    search_parser = subparsers.add_parser("search", help="Search Hugging Face models.")
    add_common_options(search_parser)
    search_parser.add_argument("query")
    search_parser.add_argument("--limit", type=int, default=10)
    search_parser.set_defaults(func=cmd_search)

    inspect_parser = subparsers.add_parser("inspect", help="Build a model structure.")
    add_common_options(inspect_parser)
    inspect_parser.add_argument("--model", default=None, help="Model id, for example MiniMaxAI/MiniMax-M3.")
    inspect_parser.add_argument("--config", default=None, help="Path to config.json.")
    inspect_parser.add_argument("--source", choices=["auto", "builtin", "local", "hf", "config"], default="auto")
    inspect_parser.add_argument("--revision", default="main")
    inspect_parser.add_argument("--cache-policy", choices=["prefer-local", "refresh", "offline"], default="prefer-local")
    inspect_parser.add_argument("--format", choices=["json", "mermaid", "dot"], default="json")
    inspect_parser.add_argument("--detail-level", choices=["compressed", "expanded"], default="compressed")
    inspect_parser.set_defaults(func=cmd_inspect)

    verify_parser = subparsers.add_parser("verify", help="Validate Transformers meta-model construction.")
    add_common_options(verify_parser)
    verify_parser.add_argument("--model", default=None, help="Model id, for example MiniMaxAI/MiniMax-M3.")
    verify_parser.add_argument("--config", default=None, help="Path to config.json.")
    verify_parser.add_argument("--source", choices=["auto", "builtin", "local", "hf", "config"], default="auto")
    verify_parser.add_argument("--revision", default="main")
    verify_parser.add_argument("--cache-policy", choices=["prefer-local", "refresh", "offline"], default="prefer-local")
    verify_parser.add_argument("--format", choices=["json", "text"], default="json")
    verify_parser.set_defaults(func=cmd_verify)

    serve_parser = subparsers.add_parser("serve", help="Start the FastAPI server.")
    add_common_options(serve_parser)
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.set_defaults(func=cmd_serve)

    args = parser.parse_args(argv)
    settings = AppSettings.from_env().with_overrides(
        model_root=args.root,
        hf_endpoint=args.endpoint,
        offline=args.offline,
        auto_fetch_remote_code=False if getattr(args, "no_auto_fetch_remote_code", False) else None,
    )
    try:
        return args.func(args, settings)
    except SourceResolutionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def cmd_list(args: argparse.Namespace, settings: AppSettings) -> int:
    resolver = ModelSourceResolver(settings)
    for entry in resolver.list_local_models():
        print(f"{entry.model_id}\t{entry.config_path}")
    return 0


def cmd_search(args: argparse.Namespace, settings: AppSettings) -> int:
    resolver = ModelSourceResolver(settings)
    for item in resolver.search_hf_models(args.query, limit=args.limit):
        tag = item.pipeline_tag or "-"
        print(f"{item.model_id}\t{tag}\tdownloads={item.downloads or 0}\tlikes={item.likes or 0}")
    return 0


def cmd_inspect(args: argparse.Namespace, settings: AppSettings) -> int:
    config_json = None
    source = args.source
    if args.config and source == "config":
        with open(args.config, "r", encoding="utf-8") as handle:
            config_json = json.load(handle)
    if args.config and source == "auto" and not args.model:
        source = "local"
    payload = StructureRequest(
        source=source,
        model_id=args.model,
        config_path=args.config if source != "config" else None,
        config_json=config_json,
        revision=args.revision,
        cache_policy=args.cache_policy,
        detail_level=args.detail_level,
    )
    structure = build_structure_response(payload, settings)
    print(export_structure(structure, args.format), end="")
    return 0


def cmd_verify(args: argparse.Namespace, settings: AppSettings) -> int:
    config_json = None
    source = args.source
    if args.config and source == "config":
        with open(args.config, "r", encoding="utf-8") as handle:
            config_json = json.load(handle)
    if args.config and source == "auto" and not args.model:
        source = "local"
    payload = VerifyRequest(
        source=source,
        model_id=args.model,
        config_path=args.config if source != "config" else None,
        config_json=config_json,
        revision=args.revision,
        cache_policy=args.cache_policy,
    )
    result = verify_structure_response(payload, settings)
    if args.format == "text":
        model = result.model_id or args.model or args.config or "<config>"
        print(f"{result.status}\t{model}\tstrategy={result.strategy}\terror={result.error or '-'}")
    else:
        print(json.dumps(result.model_dump(), indent=2, ensure_ascii=False))
    return 0 if result.ok else 1


def cmd_serve(args: argparse.Namespace, settings: AppSettings) -> int:
    import uvicorn
    from . import api

    api.set_settings(settings)
    uvicorn.run(api.app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
