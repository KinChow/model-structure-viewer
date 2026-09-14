"""Named public model endpoints shared by API and resolution orchestration."""

from pathlib import Path

ENDPOINT_URLS = {
    "huggingface": "https://huggingface.co",
    "modelscope": "https://www.modelscope.cn",
}
DEFAULT_REVISIONS = {"huggingface": "main", "modelscope": "master"}


def endpoint_url(endpoint: str | None) -> str | None:
    return ENDPOINT_URLS.get(endpoint) if endpoint else None


def endpoint_revision(endpoint: str | None, revision: str | None) -> str:
    default = DEFAULT_REVISIONS.get(endpoint or "huggingface", "main")
    if not revision or (endpoint == "modelscope" and revision == "main"):
        return default
    return revision


def source_cache_key(*, repo_id: str, revision: str, cache_dir: str | Path) -> tuple[str, str, str]:
    """huggingface_hub snapshot 同构：repo_id + revision + cache_dir 唯一键。"""
    if not repo_id:
        raise ValueError("repo_id is required")
    return (repo_id, revision, str(Path(cache_dir).expanduser()))
