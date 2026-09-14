"""Named public model endpoints shared by API and resolution orchestration."""

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
