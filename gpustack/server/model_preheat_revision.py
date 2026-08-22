import hashlib
import json
import re

from huggingface_hub import HfApi
from modelscope.hub.api import HubApi
from modelscope_hub.api import HubApi as ModelScopeHubApi

from gpustack.worker.model_preheat.identity import encode_path, normalize_source


class ModelPreheatRevisionResolutionError(ValueError):
    pass


_COMMIT_SHA = re.compile(r"^[0-9a-fA-F]{40}(?:[0-9a-fA-F]{24})?$")
MODELSCOPE_FILELIST_REVISION_PREFIX = "modelscope-filelist-v1-"


def resolve_model_preheat_revision(
    source: str,
    model_id: str,
    requested_revision: str | None,
    *,
    token: str | None = None,
    hf_api_factory=HfApi,
    modelscope_api_factory=HubApi,
    modelscope_file_api_factory=ModelScopeHubApi,
) -> str:
    try:
        normalized_source = normalize_source(source)
        if isinstance(requested_revision, str) and _COMMIT_SHA.fullmatch(
            requested_revision
        ):
            return requested_revision.lower()
        if normalized_source == "huggingface":
            resolved = (
                hf_api_factory(token=token)
                .repo_info(repo_id=model_id, revision=requested_revision)
                .sha
            )
            if not isinstance(resolved, str) or not _COMMIT_SHA.fullmatch(resolved):
                raise ValueError("invalid_huggingface_commit")
            return resolved.lower()

        api = modelscope_api_factory()
        detail = None
        if hasattr(api, "get_valid_revision_detail"):
            detail = api.get_valid_revision_detail(
                model_id, revision=requested_revision
            )
            resolved = detail.get("Revision") if isinstance(detail, dict) else None
        else:
            resolved = api.get_valid_revision(model_id, revision=requested_revision)
        commit = _extract_modelscope_commit(detail) or (
            resolved
            if isinstance(resolved, str) and _COMMIT_SHA.fullmatch(resolved)
            else None
        )
        if commit is not None:
            return commit.lower()
        if not _is_safe_modelscope_revision(resolved):
            raise ValueError("invalid_modelscope_revision")
        return modelscope_filelist_revision(
            modelscope_file_api_factory().list_repo_files(
                model_id,
                "model",
                revision=resolved,
                recursive=True,
            )
        )
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
        and not value.lower().startswith("refs/heads/")
        and value not in (".", "..")
        and not value.startswith("/")
        and "/" not in value
        and "\\" not in value
        and not any(ord(char) < 32 or ord(char) == 127 for char in value)
    )


def is_modelscope_filelist_revision(revision: object) -> bool:
    return (
        isinstance(revision, str)
        and revision.startswith(MODELSCOPE_FILELIST_REVISION_PREFIX)
        and len(revision) == len(MODELSCOPE_FILELIST_REVISION_PREFIX) + 64
        and all(
            char in "0123456789abcdef"
            for char in revision[len(MODELSCOPE_FILELIST_REVISION_PREFIX) :]
        )
    )


def modelscope_upstream_revision(
    resolved_revision: str, requested_revision: str | None
) -> str | None:
    if is_modelscope_filelist_revision(resolved_revision):
        return requested_revision
    return resolved_revision


def _extract_modelscope_commit(value):
    if isinstance(value, dict):
        for key, item in value.items():
            normalized_key = re.sub(r"[^a-z0-9]", "", str(key).lower())
            if (
                normalized_key
                in {
                    "commit",
                    "commitid",
                    "commitsha",
                    "revisionid",
                    "sha",
                    "sha1",
                    "snapshotid",
                }
                and isinstance(item, str)
                and _COMMIT_SHA.fullmatch(item)
            ):
                return item
            nested = _extract_modelscope_commit(item)
            if nested is not None:
                return nested
    elif isinstance(value, (list, tuple)):
        for item in value:
            nested = _extract_modelscope_commit(item)
            if nested is not None:
                return nested
    return None


def modelscope_filelist_revision(rows) -> str:
    files = []
    for row in rows:
        if isinstance(row, dict):
            path = row.get("path") or row.get("Path")
            row_type = row.get("type") or row.get("Type")
            size = row.get("size", row.get("Size", 0))
            blob_id = row.get("blob_id") or row.get("BlobId") or row.get("Sha256")
            lfs = row.get("lfs") or row.get("Lfs")
        else:
            path = getattr(row, "path", None)
            row_type = getattr(row, "type", None)
            size = getattr(row, "size", 0)
            blob_id = getattr(row, "blob_id", None)
            lfs = getattr(row, "lfs", None)
        if not isinstance(path, str) or not path or row_type == "tree":
            continue
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise ValueError("invalid_modelscope_file_size")
        if blob_id is None and isinstance(lfs, dict):
            blob_id = lfs.get("sha256") or lfs.get("oid")
        if not isinstance(blob_id, str) or not blob_id:
            raise ValueError("invalid_modelscope_blob_id")
        files.append(
            {
                "blob_id": blob_id.lower(),
                "path": encode_path(path),
                "size": size,
            }
        )
    if not files:
        raise ValueError("empty_modelscope_filelist")
    payload = json.dumps(
        sorted(files, key=lambda item: item["path"]),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return MODELSCOPE_FILELIST_REVISION_PREFIX + hashlib.sha256(payload).hexdigest()
