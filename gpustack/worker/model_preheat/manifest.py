"""模型 Artifact 的 Manifest 协议。

统一 Artifact（设计文档第 8 节）的不可变字段：
schema_version / artifact_id / source / model_id / resolved_revision /
include_patterns / exclude_patterns / file_count / total_size / files。

`requested_revision` 属于任务请求身份，不写入统一 Artifact Manifest，
避免同一内容分别通过 `master` 和 Commit SHA 发布时产生语义冲突。
generation、`ready.json` 属于旧预热发布协议，仍由旧字段承载，
统一 Artifact 的 Manifest 不携带这些字段。
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote

from huggingface_hub.utils import filter_repo_objects

from gpustack.worker.model_preheat.identity import (
    ModelPreheatIdentity,
    ModelPreheatIdentityError,
    _canonical_sha256,
    decode_path,
    encode_path,
)


class ModelPreheatManifestError(ValueError):
    pass


MAX_MANIFEST_BYTES = 4 * 1024 * 1024
MAX_MANIFEST_FILES = 1024
MAX_MANIFEST_PATH_LENGTH = 1024
MAX_MANIFEST_TOTAL_SIZE = 1 << 50
_SHA256_HEX_CHARS = "0123456789abcdef"


def _is_sha256(value) -> bool:
    """严格判定 64 位小写十六进制 SHA-256（类型与取值都校验）。"""
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in _SHA256_HEX_CHARS for char in value)
    )


def _require_sha256(value, field: str):
    if not _is_sha256(value):
        raise ModelPreheatManifestError(f"invalid_sha256:{field}")
    return value


# 统一 Artifact Manifest 的固定字段集合，任何多余或缺失都视为非法。
ARTIFACT_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_id",
        "source",
        "model_id",
        "resolved_revision",
        "include_patterns",
        "exclude_patterns",
        "file_count",
        "total_size",
        "files",
    }
)


@dataclass(frozen=True)
class ManifestFile:
    path: str
    size: int
    sha256: str

    def __post_init__(self):
        object.__setattr__(self, "path", validate_manifest_path(self.path))
        if len(self.path) > MAX_MANIFEST_PATH_LENGTH:
            raise ModelPreheatManifestError("manifest_path_too_long")
        if not isinstance(self.size, int) or self.size < 0:
            raise ModelPreheatManifestError("invalid_file_size")
        object.__setattr__(self, "sha256", _require_sha256(self.sha256, "file"))

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "size": self.size,
        }


def compute_request_digest(
    source: str,
    model_id: str,
    requested_revision: str | None,
    include_patterns: tuple[str, ...] | list[str],
    exclude_patterns: tuple[str, ...] | list[str],
) -> str:
    """按设计文档第 7.1 节计算规范请求身份摘要。

    输入均为编码后的规范值；`main`/`master` 等移动 revision 只能作为
    `requested_revision` 参与请求身份，不能进入 Artifact 身份。
    """
    return _canonical_sha256(
        {
            "source": source,
            "model_id": model_id,
            "requested_revision": requested_revision,
            "include_patterns": sorted(include_patterns),
            "exclude_patterns": sorted(exclude_patterns),
        }
    )


def compute_artifact_id(
    source: str,
    model_id: str,
    resolved_revision: str,
    include_patterns: tuple[str, ...] | list[str],
    exclude_patterns: tuple[str, ...] | list[str],
    files: tuple[ManifestFile, ...] | list[ManifestFile],
) -> str:
    """按设计文档第 7.3 节计算 64 位小写 SHA-256 Artifact ID。

    使用规范化后的编码路径；`requested_revision`、时间、任务 ID、
    Worker ID 和上传者不参与摘要，因此同一内容通过 `master` 和
    Commit SHA 发布时必须得到相同 Artifact ID。
    """
    payload = {
        "source": source,
        "model_id": model_id,
        "resolved_revision": resolved_revision,
        "include_patterns": sorted(include_patterns),
        "exclude_patterns": sorted(exclude_patterns),
        "files": [
            {
                "path": file.path,
                "size": file.size,
                "sha256": file.sha256,
            }
            for file in sorted(files, key=lambda item: item.path)
        ],
    }
    return _canonical_sha256(payload)


@dataclass(frozen=True)
class ModelPreheatManifest:
    """统一 Artifact 的不可变 Manifest（设计文档第 8 节）。

    只承载 §8 的固定字段（含 schema_version）。generation、
    ready.json、cache_key、selection_digest、requested_revision
    都属于旧发布协议或请求身份，不进入本 Manifest。
    """

    identity: ModelPreheatIdentity
    files: tuple[ManifestFile, ...]
    exclude_patterns: tuple[str, ...] | list[str] = ()
    schema_version: int = 1

    def __post_init__(self):
        if (
            not isinstance(self.schema_version, int)
            or isinstance(self.schema_version, bool)
            or self.schema_version != 1
        ):
            raise ModelPreheatManifestError("unsupported_schema_version")
        if len(self.files) > MAX_MANIFEST_FILES:
            raise ModelPreheatManifestError("too_many_manifest_files")
        if sum(file.size for file in self.files) > MAX_MANIFEST_TOTAL_SIZE:
            raise ModelPreheatManifestError("manifest_total_size_too_large")
        paths = [file.path for file in self.files]
        if len(paths) != len(set(paths)):
            raise ModelPreheatManifestError("duplicate_path:files")
        try:
            exclude_patterns = tuple(
                sorted(encode_path(pattern) for pattern in self.exclude_patterns)
            )
        except ModelPreheatIdentityError as exc:
            raise ModelPreheatManifestError(str(exc)) from exc
        if len(exclude_patterns) != len(set(exclude_patterns)):
            raise ModelPreheatManifestError("duplicate_path:exclude_patterns")
        object.__setattr__(self, "exclude_patterns", exclude_patterns)
        object.__setattr__(self, "artifact_id", self._artifact_id())

    def _artifact_id(self) -> str:
        return compute_artifact_id(
            self.identity.source,
            self.identity.model_path,
            self.identity.revision_path,
            self.identity.file_patterns,
            self.exclude_patterns,
            self.files,
        )

    @property
    def total_size(self) -> int:
        return sum(file.size for file in self.files)

    def artifact_prefix(self, profile_prefix: str) -> str:
        """统一 Artifact 前缀：<prefix>/<source>/<organization>/<model>/<artifact_id>。

        不包含协议版本目录、resolved revision 路径段（已编码进
        artifact_id）、generation 或 requested_revision（设计文档第 6 节）。

        profile_prefix 必须显式传入：空串表示“无前缀”，非空时必须是
        安全的对象段。
        """
        return self.identity._join_object_name(
            profile_prefix,
            self.identity.source,
            self.identity.model_path,
            self.artifact_id,
        )

    def to_artifact_dict(self) -> dict:
        """设计文档第 8 节统一 Artifact Manifest 字段，集合必须精确。"""
        payload = {
            "schema_version": self.schema_version,
            "artifact_id": self.artifact_id,
            "source": self.identity.source,
            "model_id": self.identity.model_path,
            "resolved_revision": self.identity.revision_path,
            "include_patterns": list(self.identity.file_patterns),
            "exclude_patterns": list(self.exclude_patterns),
            "file_count": len(self.files),
            "total_size": self.total_size,
            "files": [file.to_dict() for file in self.files],
        }
        # 校验 artifact_id 与身份核心字段一致，防止构造后篡改。
        recomputed = compute_artifact_id(
            payload["source"],
            payload["model_id"],
            payload["resolved_revision"],
            tuple(payload["include_patterns"]),
            tuple(payload["exclude_patterns"]),
            self.files,
        )
        if recomputed != payload["artifact_id"]:
            raise ModelPreheatManifestError("artifact_id_mismatch")
        return payload

    def to_artifact_json_bytes(self) -> bytes:
        return json.dumps(
            self.to_artifact_dict(),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")


def build_model_preheat_manifest(
    root_dir: str | Path,
    identity: ModelPreheatIdentity,
    *,
    exclude_patterns: tuple[str, ...] | list[str] = (),
    cancel_callback=None,
    progress_callback=None,
) -> ModelPreheatManifest:
    root = Path(root_dir).resolve()
    if not root.is_dir():
        raise ModelPreheatManifestError("root_dir_not_found")

    try:
        decoded_exclude_patterns = tuple(
            decode_path(encode_path(pattern)) for pattern in exclude_patterns
        )
    except ModelPreheatIdentityError as exc:
        raise ModelPreheatManifestError(str(exc)) from exc

    decoded_include_patterns = []
    for pattern in identity.file_patterns:
        try:
            decoded_pattern = decode_path(pattern)
            encode_path(decoded_pattern)
        except ModelPreheatIdentityError as exc:
            raise ModelPreheatManifestError(str(exc)) from exc

        decoded_include_patterns.append(decoded_pattern)

    candidates = {}
    for path in root.rglob("*"):
        _run_cancel_callback(cancel_callback)
        resolved = path.resolve()
        if root not in resolved.parents and resolved != root:
            raise ModelPreheatManifestError("path_traversal")
        if resolved.is_file() and not path.is_symlink():
            candidates[resolved.relative_to(root).as_posix()] = resolved
    selected_paths = filter_repo_objects(
        candidates,
        allow_patterns=decoded_include_patterns or None,
        ignore_patterns=list(decoded_exclude_patterns) or None,
    )
    matched_paths = {candidates[relative_path] for relative_path in selected_paths}

    if not matched_paths:
        raise ModelPreheatManifestError("no_manifest_files")

    files = []
    completed_paths = []
    completed_size = 0
    total_size = sum(path.stat().st_size for path in matched_paths)
    for path in sorted(
        matched_paths, key=lambda item: item.relative_to(root).as_posix()
    ):
        _run_cancel_callback(cancel_callback)
        relative_path = path.relative_to(root).as_posix()
        try:
            encoded_path = encode_path(relative_path)
        except ModelPreheatIdentityError as exc:
            raise ModelPreheatManifestError(str(exc)) from exc
        files.append(
            ManifestFile(
                path=encoded_path,
                size=path.stat().st_size,
                sha256=_sha256_file(path, cancel_callback),
            )
        )
        completed_paths.append(relative_path)
        completed_size += path.stat().st_size
        if progress_callback is not None:
            progress_callback(tuple(completed_paths), completed_size, total_size)

    return ModelPreheatManifest(
        identity=identity,
        files=tuple(files),
        exclude_patterns=exclude_patterns,
    )


def parse_artifact_manifest(payload: dict) -> ModelPreheatManifest:
    """解析统一 Artifact Manifest JSON，字段集合必须精确匹配。

    旧协议的 Manifest 字段集合不同，无法伪装成统一 Artifact，
    因此这里严格拒绝任何多余或缺失字段。

    额外约束（设计文档第 8 节）：
    - ``schema_version`` 必须严格等于 1（不允许 0 / 2 / 字符串）。
    - 所有 ``sha256`` 字段必须严格为 64 位小写十六进制。
    - ``file_count`` 必须等于 ``len(files)``。
    """
    if not isinstance(payload, dict) or set(payload) != ARTIFACT_MANIFEST_FIELDS:
        raise ModelPreheatManifestError("s3_manifest_invalid")
    if (
        not isinstance(payload.get("schema_version"), int)
        or isinstance(payload.get("schema_version"), bool)
        or payload.get("schema_version") != 1
    ):
        raise ModelPreheatManifestError("unsupported_schema_version")
    files_payload = payload.get("files")
    if not isinstance(files_payload, list):
        raise ModelPreheatManifestError("s3_manifest_invalid")
    for item in files_payload:
        if not isinstance(item, dict):
            raise ModelPreheatManifestError("s3_manifest_invalid")
        if not _is_sha256(item.get("sha256")):
            raise ModelPreheatManifestError("s3_manifest_invalid")
        size = item.get("size")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise ModelPreheatManifestError("s3_manifest_invalid")
    file_count = payload.get("file_count")
    if (
        not isinstance(file_count, int)
        or isinstance(file_count, bool)
        or file_count != len(files_payload)
    ):
        raise ModelPreheatManifestError("s3_manifest_invalid")
    try:
        identity = ModelPreheatIdentity(
            source=payload["source"],
            model_id=decode_path(payload["model_id"]),
            revision=decode_path(payload["resolved_revision"]),
            file_patterns=tuple(
                decode_path(pattern) for pattern in payload["include_patterns"]
            ),
        )
        manifest = ModelPreheatManifest(
            identity=identity,
            files=tuple(
                ManifestFile(
                    path=file["path"],
                    size=file["size"],
                    sha256=file["sha256"],
                )
                for file in payload["files"]
            ),
            exclude_patterns=tuple(
                decode_path(pattern) for pattern in payload["exclude_patterns"]
            ),
            schema_version=payload["schema_version"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ModelPreheatManifestError("s3_manifest_invalid") from exc
    if payload != manifest.to_artifact_dict():
        raise ModelPreheatManifestError("s3_manifest_invalid")
    return manifest


def validate_manifest_path(path: str) -> str:
    if not isinstance(path, str) or path == "":
        raise ModelPreheatManifestError("empty_path")
    _reject_raw_manifest_path(path)

    decoded = decode_path(path)
    _reject_decoded_manifest_path(decoded)

    try:
        canonical = encode_path(decoded)
    except ModelPreheatIdentityError as exc:
        raise ModelPreheatManifestError(str(exc)) from exc
    if canonical != path:
        raise ModelPreheatManifestError("non_canonical_path")
    return canonical


def _reject_raw_manifest_path(path: str):
    if path.startswith("/") or "\\" in path:
        raise ModelPreheatManifestError("invalid_path")
    if any(ord(char) < 32 or ord(char) == 127 for char in path):
        raise ModelPreheatManifestError("control_char")
    segments = path.split("/")
    if any(segment == "" for segment in segments):
        raise ModelPreheatManifestError("empty_path_segment")
    if any(segment in (".", "..") for segment in segments):
        raise ModelPreheatManifestError("path_traversal")


def _reject_decoded_manifest_path(path: str):
    if path.startswith("/") or "\\" in path:
        raise ModelPreheatManifestError("invalid_path")
    if any(ord(char) < 32 or ord(char) == 127 for char in path):
        raise ModelPreheatManifestError("control_char")
    segments = path.split("/")
    if any(segment == "" for segment in segments):
        raise ModelPreheatManifestError("empty_path_segment")
    if any(unquote(segment) in (".", "..") for segment in segments):
        raise ModelPreheatManifestError("path_traversal")


def _run_cancel_callback(cancel_callback):
    if cancel_callback is not None:
        cancel_callback()


def _sha256_file(path: Path, cancel_callback=None) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            _run_cancel_callback(cancel_callback)
            digest.update(chunk)
    return digest.hexdigest()
