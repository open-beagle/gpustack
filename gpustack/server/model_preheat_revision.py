import re

from huggingface_hub import HfApi
from modelscope.hub.api import HubApi

from gpustack.worker.model_preheat.identity import normalize_source


class ModelPreheatRevisionResolutionError(ValueError):
    pass


_COMMIT_SHA = re.compile(r"^[0-9a-fA-F]{40}(?:[0-9a-fA-F]{24})?$")


def resolve_model_preheat_revision(
    source: str,
    model_id: str,
    requested_revision: str | None,
    *,
    token: str | None = None,
    hf_api_factory=HfApi,
    modelscope_api_factory=HubApi,
) -> str:
    try:
        normalized_source = normalize_source(source)
        if normalized_source == "huggingface":
            resolved = (
                hf_api_factory(token=token)
                .repo_info(repo_id=model_id, revision=requested_revision)
                .sha
            )
            if not isinstance(resolved, str) or not _COMMIT_SHA.fullmatch(resolved):
                raise ValueError("invalid_huggingface_commit")
            return resolved.lower()

        resolved = modelscope_api_factory().get_valid_revision(
            model_id, revision=requested_revision
        )
        if not _is_safe_modelscope_revision(resolved):
            raise ValueError("invalid_modelscope_revision")
        return resolved
    except Exception as exc:
        if isinstance(exc, ModelPreheatRevisionResolutionError):
            raise
        raise ModelPreheatRevisionResolutionError(
            "remote_revision_resolution_failed"
        ) from None


def _is_safe_modelscope_revision(value: object) -> bool:
    return (
        isinstance(value, str)
        and 0 < len(value) <= 256
        and value not in (".", "..")
        and not value.startswith("/")
        and "/" not in value
        and "\\" not in value
        and not any(ord(char) < 32 or ord(char) == 127 for char in value)
    )
